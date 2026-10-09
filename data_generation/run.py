"""Collect matched answers, screening Covert batches before generating controls."""

import json
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from time import monotonic

from .decoders import (cover_artifacts, decode, language_artifacts, leakage,
                       parse_bool, verbatim_secret)
from .generation import VllmBackend
from .records import Recorder, append
from .stego import CONDITIONS, judge_messages, load_assets, messages, read_jsonl, write_json


def progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def answer_tokens(record: dict) -> int:
    ids = record["output_ids"]
    return len(ids) - record["answer_start"] - int(bool(ids and ids[-1] == record["eos_token_id"]))


def run(out: Path, per_scheme: int = 200, backend=None) -> dict:
    if per_scheme < 1:
        raise ValueError("per_scheme must be positive")
    config = json.loads((out / "config.json").read_text())
    assets = load_assets(out / "assets")
    cases = read_jsonl(out / "cases.jsonl")
    scheme_ids = [s["id"] for s in assets["schemes"]]
    batch_size = config["batch_size"]
    recorder = Recorder(out, config, backend)

    if backend is None:
        from huggingface_hub import HfApi

        revision_file = out / "model_revision.json"
        if revision_file.exists():
            revision = json.loads(revision_file.read_text())["revision"]
        else:
            revision = HfApi().model_info(config["model"], revision=config["model_revision"]).sha
            write_json(revision_file, {"model": config["model"], "revision": revision})
        started = monotonic()
        progress(f"START loading {config['model']}")
        recorder.backend = VllmBackend(config, revision)
        progress(f"DONE loading model in {monotonic() - started:.1f}s")
        from importlib.metadata import version
        write_json(out / "environment.json", {name: version(name) for name in
                   ("torch", "transformers", "huggingface-hub", "vllm")})

    matched = read_jsonl(out / "matched.jsonl")
    matched_ids = {group["id"] for group in matched}
    counts = Counter(group["scheme_id"] for group in matched)
    used_questions = {group["question_id"] for group in matched}
    decisions = {item["id"]: item for item in read_jsonl(out / "decisions.jsonl")}
    cases_by_id = {case["id"]: case for case in cases}
    pending = [(cases_by_id[item["id"]], recorder.cache[f"{item['id']}:covert"], item["check"])
               for item in decisions.values() if item["status"] == "covert_pass"
               and item["id"] not in matched_ids]
    progress(f"Loaded {len(cases)} cases, {len(matched)} matched groups, {len(pending)} pooled Covert passes")

    def target_reached() -> bool:
        return all(counts[scheme_id] >= per_scheme for scheme_id in scheme_ids)

    def call_many(jobs: list[dict], role: str) -> list[dict]:
        records = []
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start:start + batch_size]
            label = f"{role}/{chunk[0]['condition']}" if role == "writer" else role
            began = monotonic()
            progress(f"START {label}: {len(chunk)} requests, {chunk[0]['id']} .. {chunk[-1]['id']}")
            result = recorder.call(chunk, role)
            records.extend(result)
            progress(f"DONE {label}: {len(result)} responses, "
                     f"{sum(len(r['output_ids']) for r in result)} output tokens, "
                     f"{monotonic() - began:.1f}s")
        return records

    def check_many(items: list[tuple[dict, dict, str]]) -> dict:
        results, ready = {}, []
        for case, record, condition in items:
            check = {"condition": condition, "generation_id": record["id"], "pass": False}
            results[record["id"]] = check
            if record["finish_reason"] != "eos" or not record["text"].strip():
                check["reject"] = "local_screen"
                continue
            if condition == "overt":
                covert = recorder.cache[f"{case['id']}:covert"]
                if abs(answer_tokens(record) - answer_tokens(covert)) > 50:
                    check["reject"] = "length"
                    continue
            if language_artifacts(record["text"]):
                check["reject"] = "local_screen"
                continue
            if condition in {"covert", "concealment"}:
                if leakage(record["text"], case["secret"]):
                    check["reject"] = "local_screen"
                    continue
                if cover_artifacts(record["text"]):
                    check["reject"] = "local_screen"
                    continue
            if condition == "overt" and not verbatim_secret(record["text"], case["secret"]):
                check["reject"] = "local_screen"
                continue
            check["recovery"] = decode(record["text"], case)
            recovered = check["recovery"]["label"] == "FULL"
            if (condition == "covert" and not recovered) or (condition != "covert" and recovered):
                check["reject"] = "local_screen"
                continue
            ready.append((case, record, condition))
        progress(f"DONE local screening: {len(ready)}/{len(items)} remain")

        def judge(kind: str, selected: list[tuple[dict, dict, str]]) -> dict:
            jobs = [{"id": f"{record['id']}:{kind}", "seed": case["seed"],
                     "messages": judge_messages(kind, case["question"], record["text"], assets)}
                    for case, record, _ in selected]
            return {r["id"]: r for r in call_many(jobs, kind)}

        monitor_items = [item for item in ready if item[2] in {"covert", "concealment"}]
        monitors = judge("monitor", monitor_items)
        quality_items = []
        for case, record, condition in ready:
            check = results[record["id"]]
            if condition in {"covert", "concealment"}:
                response = monitors[f"{record['id']}:monitor"]
                check["monitor_reject"] = (parse_bool(response["text"])
                                           if response["finish_reason"] == "eos" else None)
                if check["monitor_reject"] is not False:
                    check["reject"] = "monitor"
                    continue
            quality_items.append((case, record, condition))
        qualities = judge("quality", quality_items)
        for _, record, _ in quality_items:
            check = results[record["id"]]
            response = qualities[f"{record['id']}:quality"]
            check["quality_ok"] = (parse_bool(response["text"])
                                   if response["finish_reason"] == "eos" else None)
            check["pass"] = check["quality_ok"] is True
            if not check["pass"]:
                check["reject"] = "quality"
        progress(f"DONE checks: {sum(check['pass'] for check in results.values())}/{len(items)} passed")
        return results

    def save_decision(case: dict, status: str, check: dict, **extra) -> None:
        item = {"id": case["id"], "scheme_id": case["scheme_id"], "status": status,
                "check": check, **extra}
        append(out / "decisions.jsonl", item)
        decisions[case["id"]] = item

    def process_controls(batch: list[tuple[dict, dict, dict]]) -> None:
        progress(f"START controls for {len(batch)} Covert passes")
        selected, attempts = {}, {}
        for retry in range(config["control_retries"] + 1):
            failed = [(case, condition, i) for i, condition in enumerate(CONDITIONS[1:])
                      for case, _, _ in batch
                      if retry == 0 or not selected[(case["id"], condition)][1]["pass"]]
            if not failed:
                break
            jobs = [{"id": f"{case['id']}:{condition}" + (f":retry{retry}" if retry else ""),
                     "condition": condition, "case_id": case["id"],
                     "seed": case["seed"] + i + 1 + retry * 1000000,
                     "messages": messages(case, condition, assets)}
                    for case, condition, i in failed]
            generated = call_many(jobs, "writer")
            checks = check_many([(case, record, condition) for (case, condition, _), record
                                 in zip(failed, generated, strict=True)])
            for record in generated:
                key = (record["case_id"], record["condition"])
                check = checks[record["id"]]
                selected[key] = (record, check)
                attempts.setdefault(key, []).append({"generation_id": record["id"],
                                                      "pass": check["pass"],
                                                      "reject": check.get("reject")})
            progress(f"DONE control attempt {retry + 1}: "
                     f"{sum(selected[(case['id'], condition)][1]['pass'] for case, condition, _ in failed)}"
                     f"/{len(failed)} passed")

        for case, covert, covert_check in batch:
            controls = {condition: selected[(case["id"], condition)] for condition in CONDITIONS[1:]}
            checks = {"covert": covert_check,
                      **{condition: item[1] for condition, item in controls.items()}}
            history = {condition: attempts[(case["id"], condition)] for condition in CONDITIONS[1:]}
            if all(check["pass"] for check in checks.values()) and case["question_id"] not in used_questions:
                group = {**case, "generation_ids": {"covert": covert["id"],
                         **{condition: item[0]["id"] for condition, item in controls.items()}},
                         "checks": checks, "control_attempts": history}
                append(out / "matched.jsonl", group)
                matched.append(group)
                matched_ids.add(case["id"])
                counts[case["scheme_id"]] += 1
                used_questions.add(case["question_id"])
                save_decision(case, "matched", covert_check, checks=checks)
                progress(f"MATCH {case['id']} {case['scheme_id']}: "
                         f"{counts[case['scheme_id']]}/{per_scheme}")
            else:
                save_decision(case, "rejected", covert_check, checks=checks,
                              reject="duplicate_question" if case["question_id"] in used_questions
                              else "control")

    def flush_pending() -> None:
        batch = pending[:batch_size]
        del pending[:batch_size]
        process_controls(batch)

    from .export import export
    try:
        while len(pending) >= batch_size and not target_reached():
            flush_pending()
        eligible = (case for case in cases
                    if case["id"] not in decisions and counts[case["scheme_id"]] < per_scheme
                    and case["question_id"] not in used_questions)
        batch_number = 0
        while not target_reached():
            subset = list(islice(eligible, batch_size))
            if not subset:
                break
            batch_number += 1
            progress(f"START Covert batch {batch_number}: {len(subset)} cases, "
                     f"{len(pending)} pooled, {sum(min(counts[s], per_scheme) for s in scheme_ids)}"
                     f"/{per_scheme * len(scheme_ids)} selected")
            jobs = [{"id": f"{case['id']}:covert", "case_id": case["id"],
                     "condition": "covert", "seed": case["seed"],
                     "messages": messages(case, "covert", assets)} for case in subset]
            writers = call_many(jobs, "writer")
            checked = check_many([(case, record, "covert") for case, record
                                  in zip(subset, writers, strict=True)])
            for case, record in zip(subset, writers, strict=True):
                check = checked[record["id"]]
                if check["pass"]:
                    save_decision(case, "covert_pass", check)
                    pending.append((case, record, check))
                else:
                    save_decision(case, "rejected", check, reject=check["reject"])
            while len(pending) >= batch_size and not target_reached():
                flush_pending()
            progress(f"DONE Covert batch {batch_number}: "
                     f"{sum(checked[r['id']]['pass'] for r in writers)} passed, "
                     f"{len(pending)} pooled")
    finally:
        summary = export(out, per_scheme)
    if pending:
        progress(f"{len(pending)} Covert passes remain queued")
    progress(f"Saved {summary['accepted_groups']} groups in {out}")
    if not target_reached():
        raise RuntimeError(f"Candidate plan ended before reaching {per_scheme} groups per scheme")
    return summary
