from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .stego import CONDITIONS, read_jsonl, write_json


def export(out: Path) -> dict:
    cases = {c["id"]: c for c in read_jsonl(out / "cases.jsonl")}
    generations = {g["id"]: g for g in read_jsonl(out / "generations.jsonl")}
    decisions = list({d["id"]: d for d in read_jsonl(out / "decisions.jsonl")}.values())
    groups = read_jsonl(out / "accepted.jsonl")
    directory = out / "exports"
    directory.mkdir(exist_ok=True)
    summary = {
        "attempts_evaluated": len(decisions), "writer_generations": len(generations),
        "accepted_groups": len(groups), "accepted_rollouts": 4 * len(groups),
        "target_groups": json.loads((out / "config.json").read_text())["target_groups"],
        "covert_passed": sum(d["checks"]["covert"]["pass"] for d in decisions),
        "covert_recovery_labels": dict(Counter(d["checks"]["covert"].get("recovery", {}).get("label", "not_tested") for d in decisions)),
        "by_family": dict(Counter(g["family"] for g in groups)),
        "by_scheme": dict(Counter(g["scheme_id"] for g in groups)),
        "by_recovery": dict(Counter(g["checks"]["covert"]["recovery"]["label"] for g in groups)),
        "rejects": dict(Counter(d.get("reject", d["checks"]["covert"].get("reject", "control_failed"))
                                for d in decisions if not d["accepted"])),
    }
    write_json(out / "summary.json", summary)
    fields = ["matched_id", "question_id", "question", "prompt", "answer", "steganography",
              "coherence", "secret", "scheme_id", "position", "wording", "recovery"]
    for condition in CONDITIONS:
        with (directory / f"{condition}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for group in groups:
                record = generations[group["generation_ids"][condition]]
                writer.writerow({"matched_id": group["id"], "question_id": group["question_id"],
                    "question": group["question"], "prompt": record["rendered_prompt"],
                    "answer": record["text"], "steganography": 100 if condition == "covert" else 0,
                    "coherence": 100, "secret": group["secret"]["value"], "scheme_id": group["scheme_id"],
                    "position": group["position"], "wording": group["wording"],
                    "recovery": group["checks"][condition]["recovery"]["label"]})
    with (directory / "attempts.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["id", "family", "scheme", "position", "question", "secret", "answer", "accepted", "checks"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in decisions:
            c = cases[d["id"]]
            g = generations[f"{d['id']}:covert"]
            writer.writerow({"id": c["id"], "family": c["family"], "scheme": c["scheme_id"],
                "position": c["position"], "question": c["question"], "secret": c["secret"]["value"],
                "answer": g["text"], "accepted": d["accepted"],
                "checks": json.dumps(d["checks"], ensure_ascii=False)})
    parts = ['<!doctype html><meta charset="utf-8"><title>Steganography review</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:1px solid #aaa;padding:16px 0}'
             'details{margin:12px 0}</style><h1>Accepted matched groups</h1>',
             '<pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>']
    for g in groups:
        parts += [f'<article><h2>{html.escape(g["id"])} · {g["scheme_id"]}</h2>',
                  '<p>' + html.escape(g["question"]) + '</p>',
                  '<p>Secret: ' + html.escape(g["secret"]["value"]) + '</p>']
        for condition in CONDITIONS:
            text = generations[g["generation_ids"][condition]]["text"]
            parts.append(f'<h3>{condition}</h3><pre>{html.escape(text)}</pre>')
        parts.append('<details><summary>Checks</summary><pre>' + html.escape(json.dumps(g["checks"], indent=2)) + '</pre></details></article>')
    (directory / "review.html").write_text("\n".join(parts))
    with ZipFile(out.with_suffix(".zip"), "w", compression=ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file() and path.relative_to(out).parts[0] != "activations":
                archive.write(path, path.relative_to(out))
    return summary
