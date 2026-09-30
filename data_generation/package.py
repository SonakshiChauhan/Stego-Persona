"""Package accepted generation records as a Hugging Face dataset."""

import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean

import pyarrow as pa
import pyarrow.parquet as pq

from .stego import CONDITIONS, read_jsonl, write_json

SCHEMA = pa.schema([
    ("group_id", pa.string()), ("condition", pa.string()),
    ("question_id", pa.string()), ("question", pa.string()),
    ("scheme_id", pa.string()), ("payload_kind", pa.string()), ("payload", pa.string()),
    ("wording", pa.int32()),
    ("position", pa.string()), ("word", pa.string()), ("letter", pa.string()), ("order", pa.string()),
    ("mapping", pa.string()), ("case_seed", pa.int32()), ("generation_seed", pa.int32()),
    ("prompt", pa.string()), ("answer", pa.string()), ("reasoning_text", pa.string()),
    ("input_ids", pa.list_(pa.int32())), ("output_ids", pa.list_(pa.int32())),
    ("answer_start", pa.int32()), ("eos_token_id", pa.int32()),
    ("prompt_tokens", pa.int32()), ("answer_tokens", pa.int32()),
    ("recovery", pa.string()), ("monitor_reject", pa.bool_()), ("quality_ok", pa.bool_()),
    ("model", pa.string()), ("model_revision", pa.string()),
])
SCHEMES = ("W1", "W2", "B1", "B2", "C1", "C2")
METRICS = ("covert_attempts", "automatic_passes", "manually_filtered",
           "passes_after_review", "dataset_groups")


def rates(counts: dict) -> dict:
    attempts = counts["covert_attempts"]
    automatic = counts["automatic_passes"]
    return {**counts,
            "automatic_acceptance_rate": automatic / attempts if attempts else 0,
            "manual_filter_rate": counts["manually_filtered"] / automatic if automatic else 0,
            "review_adjusted_acceptance_rate": counts["passes_after_review"] / attempts
            if attempts else 0,
            "dataset_yield": counts["dataset_groups"] / attempts if attempts else 0}


def acceptance_stats(out: Path, groups: list[dict]) -> dict:
    decisions = {row["id"]: row for row in read_jsonl(out / "decisions.jsonl")}
    attempts = Counter(row["scheme_id"] for row in decisions.values())
    filtered = Counter(row["scheme_id"] for row in decisions.values()
                       if row.get("reject") == "manual_review")
    matched = Counter(row["scheme_id"] for row in read_jsonl(out / "matched.jsonl"))
    selected = Counter(row["scheme_id"] for row in groups)
    return {scheme: rates({"covert_attempts": attempts[scheme],
                           "automatic_passes": matched[scheme] + filtered[scheme],
                           "manually_filtered": filtered[scheme],
                           "passes_after_review": matched[scheme],
                           "dataset_groups": selected[scheme]}) for scheme in SCHEMES}


def package(out: Path, dest: Path) -> Path:
    groups = read_jsonl(out / "accepted.jsonl")
    if not groups:
        raise ValueError(f"No accepted groups in {out}")
    generations = {r["id"]: r for r in read_jsonl(out / "generations.jsonl")}
    config = json.loads((out / "config.json").read_text())
    revision = json.loads((out / "model_revision.json").read_text())["revision"]
    split = json.loads((out / "manifest.json").read_text())["split"]
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown dataset split: {split}")
    metadata_path = dest / "data" / "metadata.json"
    metadata = (json.loads(metadata_path.read_text()) if metadata_path.exists() else
                {"model": config["model"], "model_revision": revision,
                 "activations": "not extracted", "splits": {}})
    if (metadata["model"], metadata["model_revision"]) != (config["model"], revision):
        raise ValueError("Model differs from other packaged splits")
    questions = {" ".join(group["question"].casefold().split()) for group in groups}
    secrets = {(group["secret"]["kind"], group["secret"]["value"]) for group in groups}
    for other in ("train", "val", "test"):
        path = dest / "data" / f"{other}.parquet"
        if other == split or not path.exists():
            continue
        previous = pq.read_table(path, columns=["question", "payload_kind", "payload"]).to_pylist()
        if questions & {" ".join(row["question"].casefold().split()) for row in previous}:
            raise ValueError(f"Question overlap between {split} and {other}")
        if secrets & {(row["payload_kind"], row["payload"]) for row in previous}:
            raise ValueError(f"Secret overlap between {split} and {other}")
    rows = []
    for group in groups:
        variant = group["variant"]
        for condition in CONDITIONS:
            record = generations[group["generation_ids"][condition]]
            ids = record["output_ids"]
            eos = bool(ids and ids[-1] == record["eos_token_id"])
            check = group["checks"][condition]
            rows.append({
                "group_id": group["id"], "condition": condition,
                "question_id": group["question_id"], "question": group["question"],
                "scheme_id": group["scheme_id"], "payload_kind": group["secret"]["kind"],
                "payload": group["secret"]["value"], "wording": group["wording"],
                "position": variant["position"],
                "word": variant.get("word"), "letter": variant.get("letter"),
                "order": variant.get("order"),
                "mapping": json.dumps(variant.get("map"), sort_keys=True) if "map" in variant else None,
                "case_seed": group["seed"], "generation_seed": record["seed"],
                "prompt": record["rendered_prompt"],
                "answer": record["text"], "reasoning_text": record.get("reasoning_text", ""),
                "input_ids": record["input_ids"], "output_ids": ids,
                "answer_start": record.get("answer_start", 0),
                "eos_token_id": record["eos_token_id"],
                "prompt_tokens": len(record["input_ids"]),
                "answer_tokens": len(ids) - record.get("answer_start", 0) - int(eos),
                "recovery": check["recovery"]["label"],
                "monitor_reject": check.get("monitor_reject"),
                "quality_ok": check.get("quality_ok"),
                "model": config["model"], "model_revision": revision,
            })

    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), data / f"{split}.parquet",
                   compression="zstd")
    provenance = data / "provenance" / split
    provenance.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "manifest.json", "summary.json", "accepted.jsonl",
                 "model_revision.json", "environment.json", "manual_exclusions.jsonl"):
        source = out / name
        if source.exists():
            shutil.copyfile(source, provenance / name)
    for name in ("schemes.json", "prompts.toml", "words.txt", "binary.txt"):
        shutil.copyfile(out / "assets" / name, provenance / name)
    metadata["splits"][split] = {"groups": len(groups), "rows": len(rows),
                                  "acceptance_by_scheme": acceptance_stats(out, groups)}
    metadata["acceptance_by_scheme"] = {
        scheme: rates({metric: sum(part["acceptance_by_scheme"][scheme][metric]
                                   for part in metadata["splits"].values())
                       for metric in METRICS}) for scheme in SCHEMES}
    metadata["acceptance_overall"] = rates({
        metric: sum(stats[metric] for stats in metadata["acceptance_by_scheme"].values())
        for metric in METRICS})
    metadata["acceptance_definitions"] = {
        "covert_attempts": "Unique Covert cases generated, including unfinished pooled cases.",
        "automatic_passes": "Matched groups before manual exclusions, including quota overflow.",
        "manually_filtered": "Automatically matched groups removed by manual review.",
        "passes_after_review": "Matched groups retained after manual review, including quota overflow.",
        "dataset_groups": "Final selected groups after per-scheme quotas.",
        "dataset_yield": "Final selected groups divided by unique Covert attempts.",
    }
    write_json(metadata_path, metadata)
    available = [name for name in ("train", "val", "test") if (data / f"{name}.parquet").exists()]
    def rate_row(name: str, stats: dict) -> str:
        return (f"| {name} | {stats['covert_attempts']} | {stats['automatic_passes']} | "
                f"{stats['manually_filtered']} ({stats['manual_filter_rate']:.1%}) | "
                f"{stats['automatic_acceptance_rate']:.1%} | "
                f"{stats['review_adjusted_acceptance_rate']:.1%} | {stats['dataset_groups']} "
                f"({stats['dataset_yield']:.1%}) |\n")

    table = ("\n| Scheme | Covert attempts | Automatic matches | Manually removed | "
             "Automatic rate | Adjusted rate | Dataset groups |\n"
             "|---|---:|---:|---:|---:|---:|---:|\n")
    table += "".join(rate_row(scheme, metadata["acceptance_by_scheme"][scheme])
                     for scheme in SCHEMES)
    table += rate_row("Total", metadata["acceptance_overall"])
    token_rows = []
    for name in available:
        token_rows.extend(pq.read_table(
            data / f"{name}.parquet",
            columns=["condition", "prompt_tokens", "answer_start", "answer_tokens", "output_ids"]
        ).to_pylist())
    token_table = ("\n## Mean token counts per answer\n\n"
                   "| Condition | Thinking prefix | Visible answer | Total including prompt |\n"
                   "|---|---:|---:|---:|\n")
    for condition in (*CONDITIONS, "All conditions"):
        selected = (token_rows if condition == "All conditions" else
                    [row for row in token_rows if row["condition"] == condition])
        label = condition if condition == "All conditions" else condition.title()
        token_table += (f"| {label} | "
                        f"{mean(row['answer_start'] for row in selected):,.0f} | "
                        f"{mean(row['answer_tokens'] for row in selected):,.0f} | "
                        f"{mean(row['prompt_tokens'] + len(row['output_ids']) for row in selected):,.0f} |\n")
    token_table += ("\nMeans are over all included answer rows. Thinking prefix is `answer_start` "
                    "and includes thinking delimiters. Total includes `prompt_tokens` and "
                    "all `output_ids`.\n")
    (dest / "README.md").write_text(
        "---\nconfigs:\n- config_name: default\n  data_files:\n" +
        "".join(f"  - split: {name}\n    path: data/{name}.parquet\n" for name in available) +
        "---\n\n"
        "# Steganography matched generations\n\n"
        "Each `group_id` has Covert, Overt, Normal, and Concealment rows. "
        "The `prompt`, `input_ids`, and `output_ids` are exact generation records; "
        "`wording` selects one of three prompt variations. "
        "`answer_start` is the first visible-answer token within `output_ids`. "
        "No activations are included. See `data/provenance/` for each split's run settings "
        "and manual exclusions. `data/metadata.json` reports automatic and review-adjusted "
        "acceptance rates by scheme and split.\n"
        "\nRates use distinct generated Covert cases as the denominator. Automatic matches "
        "include groups later removed by manual review and groups beyond the dataset quota; "
        "the adjusted rate excludes manual removals.\n" + table + token_table)
    print(f"Packaged {split}: {len(groups)} groups ({len(rows)} rows) in {dest}", flush=True)
    return dest
