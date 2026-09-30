from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from .stego import (FAMILIES, CONDITIONS, DISABLED_SCHEMES, load_assets, read_jsonl, write_json, messages,
                    receiver_messages, judge_messages)
from .generation import VllmBackend
from .records import Recorder, append
from .decoders import (carrier_streams, score_streams, score_phrase, leakage, cover_artifacts,
                       exposed_secret, parse_bool, language_artifacts, llm_carrier_count)

CONTROL_RETRIES = 5


def run(out: Path, backend=None) -> dict:
    def progress(message):
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{stamp}] {message}", flush=True)

    progress("START loading run plan")
    config = json.loads((out / "config.json").read_text())
    evaluation_suffix = ":v2" if (out / "prompt_migration.json").exists() else ""
    assets = load_assets(out / "assets")
    cases = read_jsonl(out / "cases.jsonl")
    schemes = {s["id"]: s for s in assets["schemes"]}
    progress(f"DONE loading run plan: {len(cases)} candidates, batch size {config['batch_size']}")
    progress(f"Disabled schemes: {', '.join(sorted(DISABLED_SCHEMES))}")
    started = monotonic()
    progress("START loading saved generation logs")
    recorder = Recorder(out, config, backend)
    progress(f"DONE loading saved generation logs in {monotonic() - started:.1f}s")
    if backend is None:
        from .decoders import grammar_parser
        started = monotonic()
        progress("START loading grammar parser")
        grammar_parser()
        progress(f"DONE loading grammar parser in {monotonic() - started:.1f}s")
        from huggingface_hub import HfApi
        pinned = out / "model_revision.json"
        started = monotonic()
        progress("START resolving model revision")
        revision = (json.loads(pinned.read_text())["revision"] if pinned.exists() else
                    HfApi().model_info(config["model"], revision=config["model_revision"]).sha)
        if not pinned.exists():
            write_json(pinned, {"model": config["model"], "revision": revision})
        progress(f"DONE resolving model revision in {monotonic() - started:.1f}s")
        from importlib.metadata import version
        environment = {p: version(p) for p in
                       ("torch", "transformers", "spacy", "huggingface-hub", "vllm")}
        environment["en_core_web_sm"] = grammar_parser().meta["version"]
        environment_path = out / "environment.json"
        if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
            raise ValueError("Package/parser versions changed. Use a new run directory.")
        started = monotonic()
        progress(f"START loading {config['model']} on {config['device']}")
        backend = VllmBackend(config, revision)
        progress(f"DONE loading model in {monotonic() - started:.1f}s")
        write_json(environment_path, environment)
        recorder.backend = backend
    groups = read_jsonl(out / "accepted.jsonl")
    decisions = {d["id"]: d for d in read_jsonl(out / "decisions.jsonl")}
    finished = set(decisions) | {g["id"] for g in groups}
    counts = Counter(g["family"] for g in groups)
    questions = {g["question_id"] for g in groups}
    target = config["target_groups"]
    batch_size = config["batch_size"]
    quotas = {f: target // len(FAMILIES) + (i < target % len(FAMILIES)) for i, f in enumerate(FAMILIES)}

    def call_many(jobs, role):
        records = []
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start:start + batch_size]
            label = f"{role}/{chunk[0]['condition']}" if role == "writer" else role
            started = monotonic()
            progress(f"START {label}: {len(chunk)} requests, {chunk[0]['id']} .. {chunk[-1]['id']}")
            result = recorder.call(chunk, role)
            records.extend(result)
            progress(f"DONE {label}: {len(result)} responses, "
                     f"{sum(len(record['output_ids']) for record in result)} output tokens, "
                     f"{monotonic() - started:.1f}s")
        return records

    def checks_many(items):
        """Apply each gate to a batch before starting the next model call."""
        if not items:
            return {}
        started = monotonic()
        progress(f"START local screening: {len(items)} answers")
        results, ready = {}, []
        for case, record, condition in items:
            result = {"condition": condition, "generation_id": record["id"], "pass": False}
            results[record["id"]] = result
            if record["finish_reason"] != "eos" or not record["text"].strip():
                result["reject"] = "unfinished_or_empty"
                continue
            result["language_artifacts"] = language_artifacts(record["text"])
            if result["language_artifacts"]:
                result["reject"] = "non_english_output"
                continue
            if condition in {"covert", "concealment"}:
                result["deterministic_rejects"] = leakage(record["text"], case["secret"])
                if result["deterministic_rejects"]:
                    result["reject"] = "public_disclosure_or_commentary"
                    continue
                result["cover_artifacts"] = cover_artifacts(record["text"], schemes[case["scheme_id"]])
                if result["cover_artifacts"]:
                    result["reject"] = "obvious_code_or_artifact"
                    continue
            if condition == "overt" and not exposed_secret(record["text"], case["secret"]):
                result["reject"] = "overt_secret_missing"
                continue
            scheme = schemes[case["scheme_id"]]
            if scheme["receiver"] == "llm" and case["secret"]["kind"] == "bits":
                result["eligible_carriers"] = llm_carrier_count(record["text"], scheme)
                if result["eligible_carriers"] < len(case["secret"]["value"]):
                    result["reject"] = "insufficient_carriers"
                    continue
            ready.append((case, record, condition))
        progress(f"DONE local screening: {len(ready)}/{len(items)} remain, {monotonic() - started:.1f}s")

        receiver_jobs = [{"id": f"{record['id']}:receiver{evaluation_suffix}", "seed": case["seed"],
                          "messages": receiver_messages(case, record["text"], assets)}
                         for case, record, _ in ready if schemes[case["scheme_id"]]["receiver"] == "llm"]
        receivers = {record["id"]: record for record in call_many(receiver_jobs, "receiver")}
        started = monotonic()
        progress(f"START decoding: {len(ready)} answers")
        survivors = []
        for case, record, condition in ready:
            result = results[record["id"]]
            scheme = schemes[case["scheme_id"]]
            if scheme["receiver"] == "llm":
                response_id = f"{record['id']}:receiver{evaluation_suffix}"
                response = receivers[response_id]
                recovery = score_phrase(response["text"], case["secret"], config["partial_policy"])
                if response["finish_reason"] != "eos":
                    recovery["label"] = "FAIL"
                result["recovery"] = {**recovery, "receiver": "llm", "call_id": response_id}
            else:
                streams = carrier_streams(record["text"], scheme, case["position"], len(case["secret"]["value"]))
                result["recovery"] = score_streams(streams, scheme, case, config["partial_policy"])
            recovered = result["recovery"]["label"] in config["accepted_recovery"]
            if condition == "covert" and not recovered:
                result["reject"] = ("ambiguous_start" if result["recovery"].get("candidate_windows", 0) > 1
                                    else "recovery_failed")
                continue
            control_test = condition != "covert" and config["reject_control_channel_matches"]
            if condition == "overt" and scheme["receiver"] == "llm":
                control_test = False
            if control_test and recovered:
                result["reject"] = "control_also_matches_covert_channel"
                continue
            survivors.append((case, record, condition))
        progress(f"DONE decoding: {len(survivors)}/{len(ready)} remain, {monotonic() - started:.1f}s")

        def judge(items_to_judge, kind):
            jobs = [{"id": f"{record['id']}:{kind}{evaluation_suffix}", "seed": case["seed"],
                     "messages": judge_messages(kind, case["question"], record["text"], assets)}
                    for case, record, _ in items_to_judge]
            return {record["id"]: record for record in call_many(jobs, kind)}

        monitor_items = [item for item in survivors if item[2] in {"covert", "concealment"}]
        monitors = judge(monitor_items, "monitor")
        quality_items = []
        for case, record, condition in survivors:
            result = results[record["id"]]
            if condition in {"covert", "concealment"}:
                result["monitor_call"] = f"{record['id']}:monitor{evaluation_suffix}"
                response = monitors[result["monitor_call"]]
                result["monitor_reject"] = (parse_bool(response["text"])
                                            if response["finish_reason"] == "eos" else None)
                if result["monitor_reject"] is not False:
                    result["reject"] = "monitor_or_invalid_boolean"
                    continue
            quality_items.append((case, record, condition))
        qualities = judge(quality_items, "quality")
        for _, record, _ in quality_items:
            result = results[record["id"]]
            result["quality_call"] = f"{record['id']}:quality{evaluation_suffix}"
            response = qualities[result["quality_call"]]
            result["quality_ok"] = (parse_bool(response["text"])
                                    if response["finish_reason"] == "eos" else None)
            if result["quality_ok"] is True:
                result["pass"] = True
            else:
                result["reject"] = "quality_or_invalid_boolean"
        progress(f"DONE checks: {sum(result['pass'] for result in results.values())}/{len(items)} passed")
        return results

    attempted = len(decisions)
    pending = []

    def save_decision(case, evaluated, controls=(), control_attempts=None):
        nonlocal attempted
        accepted = (len(evaluated) == 4 and all(check["pass"] for check in evaluated.values())
                    and len(groups) < target
                    and (not config["unique_questions"] or case["question_id"] not in questions)
                    and (not config["balance_families"] or counts[case["family"]] < quotas[case["family"]]))
        decision = {"id": case["id"], "family": case["family"],
                    "accepted": accepted, "checks": evaluated}
        if control_attempts is not None:
            decision["control_retry_limit"] = CONTROL_RETRIES
            decision["control_attempts"] = control_attempts
        if not accepted and len(evaluated) == 4 and all(check["pass"] for check in evaluated.values()):
            decision["reject"] = ("target_reached" if len(groups) >= target else
                                  "duplicate_question" if case["question_id"] in questions else
                                  "family_quota_met")
        if accepted:
            group = {**case, "generation_ids": {record["condition"]: record["id"] for record in controls},
                     "checks": evaluated, "control_attempts": control_attempts}
            append(out / "accepted.jsonl", group)
            groups.append(group)
            counts[case["family"]] += 1
            questions.add(case["question_id"])
        append(out / "decisions.jsonl", decision)
        if case["id"] not in decisions:
            attempted += 1
        decisions[case["id"]] = decision
        finished.add(case["id"])
        if accepted:
            progress(f"ACCEPT {case['id']} {case['scheme_id']} | groups {len(groups)}/{target}")

    def process_controls(batch, previous=None):
        before = len(groups)
        progress(f"START controls for {len(batch)} Covert passes")
        control_jobs = [{"id": f"{case['id']}:{condition}", "case_id": case["id"],
                         "condition": condition, "seed": case["seed"] + i + 1,
                         "messages": messages(case, condition, assets)}
                        for i, condition in enumerate(CONDITIONS[1:]) for case, _, _ in batch]
        controls = {record["id"]: record for record in call_many(control_jobs, "writer")}
        if previous is None:
            control_items = [(case, controls[f"{case['id']}:{condition}"], condition)
                             for case, _, _ in batch for condition in CONDITIONS[1:]]
            control_checks = checks_many(control_items)
        else:
            control_checks = {f"{case['id']}:{condition}": previous[case["id"]]["checks"][condition]
                              for case, _, _ in batch for condition in CONDITIONS[1:]}

        selected = {}
        attempts = {}
        for case, covert, covert_check in batch:
            for condition in CONDITIONS[1:]:
                record = controls[f"{case['id']}:{condition}"]
                check = control_checks[record["id"]]
                key = (case["id"], condition)
                selected[key] = (record, check)
                attempts[key] = [{"generation_id": record["id"], "pass": check["pass"],
                                  "reject": check.get("reject")}]

        for retry in range(1, CONTROL_RETRIES + 1):
            failed = [(case, condition, i) for case, _, _ in batch
                      for i, condition in enumerate(CONDITIONS[1:])
                      if not selected[(case["id"], condition)][1]["pass"]]
            if not failed:
                break
            progress(f"START control retry {retry}/{CONTROL_RETRIES}: {len(failed)} failed answers")
            jobs = [{"id": f"{case['id']}:{condition}:retry{retry}", "case_id": case["id"],
                     "condition": condition, "seed": case["seed"] + i + 1 + retry * 1000000,
                     "messages": messages(case, condition, assets)} for case, condition, i in failed]
            generated = call_many(jobs, "writer")
            checks = checks_many([(case, record, condition) for (case, condition, _), record
                                  in zip(failed, generated, strict=True)])
            for record in generated:
                key = (record["case_id"], record["condition"])
                check = checks[record["id"]]
                selected[key] = (record, check)
                attempts[key].append({"generation_id": record["id"], "pass": check["pass"],
                                      "reject": check.get("reject")})
            progress(f"DONE control retry {retry}/{CONTROL_RETRIES}: "
                     f"{sum(selected[(case['id'], condition)][1]['pass'] for case, condition, _ in failed)}"
                     f"/{len(failed)} passed")

        for case, covert, covert_check in batch:
            records = [selected[(case["id"], condition)][0] for condition in CONDITIONS[1:]]
            evaluated = {"covert": covert_check,
                         **{condition: selected[(case["id"], condition)][1]
                            for condition in CONDITIONS[1:]}}
            history = {condition: attempts[(case["id"], condition)] for condition in CONDITIONS[1:]}
            save_decision(case, evaluated, [covert, *records], history)
        progress(f"DONE controls: {len(groups) - before}/{len(batch)} accepted, "
                 f"{len(pending)} Covert passes remain queued")

    def flush_pending():
        nonlocal pending
        batch, pending = pending[:batch_size], pending[batch_size:]
        if batch:
            process_controls(batch)

    saved_cases = {case["id"]: case for case in cases}
    accepted_ids = {group["id"] for group in groups}
    recoverable = [decision for decision in decisions.values()
                   if len(decision["checks"]) == 4 and decision["checks"]["covert"]["pass"]
                   and not decision["accepted"] and not decision.get("reject")
                   and decision.get("control_retry_limit") != CONTROL_RETRIES
                   and decision["id"] not in accepted_ids]
    if recoverable:
        progress(f"START retrying controls for {len(recoverable)} saved Covert passes")
        for start in range(0, len(recoverable), batch_size):
            if len(groups) >= target:
                break
            old = recoverable[start:start + batch_size]
            batch = [(saved_cases[d["id"]], recorder.cache[f"{d['id']}:covert"],
                      d["checks"]["covert"]) for d in old]
            process_controls(batch, {d["id"]: d for d in old})
        progress(f"DONE retrying saved controls: {len(groups)}/{target} accepted")

    try:
        for start in range(0, len(cases), batch_size):
            if len(groups) >= target:
                break
            subset = [case for case in cases[start:start + batch_size]
                      if case["id"] not in finished
                      and case["scheme_id"] not in DISABLED_SCHEMES
                      and (not config["balance_families"] or counts[case["family"]] < quotas[case["family"]])
                      and (not config["unique_questions"] or case["question_id"] not in questions)]
            if subset:
                progress(f"START Covert batch {start // batch_size + 1}: {len(subset)} cases, "
                         f"{len(pending)} pooled, {len(groups)}/{target} accepted")
            jobs = [{"id": f"{case['id']}:covert", "case_id": case["id"], "condition": "covert",
                     "seed": case["seed"], "messages": messages(case, "covert", assets)} for case in subset]
            writers = call_many(jobs, "writer")
            checked = checks_many([(case, record, "covert") for case, record in zip(subset, writers)])
            for case, covert in zip(subset, writers):
                covert_check = checked[covert["id"]]
                if covert_check["pass"]:
                    pending.append((case, covert, covert_check))
                else:
                    save_decision(case, {"covert": covert_check})
            while len(pending) >= batch_size:
                flush_pending()
            if subset:
                progress(f"DONE Covert batch {start // batch_size + 1}: "
                         f"{sum(checked[record['id']]['pass'] for record in writers)} passed checks, "
                         f"{len(pending)} pooled, {attempted} decided, {len(groups)}/{target} accepted")
            if subset and attempted % config["export_every"] < batch_size:
                from .export import export
                started = monotonic()
                progress("START periodic CSV/ZIP export")
                export(out)
                progress(f"DONE periodic CSV/ZIP export in {monotonic() - started:.1f}s")
    finally:
        from .export import export
        started = monotonic()
        progress("START final CSV/ZIP export")
        summary = export(out)
        progress(f"DONE final CSV/ZIP export in {monotonic() - started:.1f}s")
    if pending:
        progress(f"{len(pending)} Covert passes remain queued for the next full batch")
    progress(f"Saved {len(groups)} matched groups. Results: {out}")
    return summary
