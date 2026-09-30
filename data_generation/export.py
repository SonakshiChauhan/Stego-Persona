"""Export accepted groups for review and later activation extraction."""

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path

from .decoders import WORD, sentences
from .stego import CONDITIONS, load_assets, read_jsonl, scheme_rule, write_json, write_jsonl


def highlighted_answer(answer: str, group: dict) -> tuple[str, str]:
    carriers = group["checks"]["covert"]["recovery"]["window"]["carriers"]
    scheme = group["scheme_id"]
    if scheme == "W2":
        spans = [(match.start(), match.end()) for match in WORD.finditer(answer)]
    else:
        spans, cursor = [], 0
        for sentence in sentences(answer):
            pattern = r"\s+".join(map(re.escape, sentence.split()))
            match = re.search(pattern, answer[cursor:])
            if match is None:
                return html.escape(answer), "Could not locate the decoded sentences in the answer."
            start, end = cursor + match.start(), cursor + match.end()
            spans.append((start, end))
            cursor = end
    if not carriers or max(item["index"] for item in carriers) >= len(spans):
        return html.escape(answer), "Could not locate the decoded carriers in the answer."

    marks, trace = [], []
    for number, item in enumerate(carriers, 1):
        start, end = spans[item["index"]]
        unit = "word" if scheme == "W2" else "sentence"
        if scheme.startswith("C"):
            count = len(item["carrier"].split())
            evidence = f"{count} words → {item['symbol']}"
            css = "count"
        else:
            words = list(WORD.finditer(answer[start:end]))
            word = words[-1] if scheme == "W1" and group["variant"]["word"] == "last" else words[0]
            letter = group["variant"].get("letter", "first")
            start += word.start() if letter == "first" else word.end() - 1
            end = start + 1
            evidence = f"{answer[start].upper()} → {item['symbol']}"
            css = "letter"
        label = f"{number}. {unit} {item['index'] + 1}: {evidence}"
        marks.append((start, end, label, css))
        trace.append(f'<li>{html.escape(label)}</li>')

    parts, cursor = [], 0
    for start, end, label, css in marks:
        parts.append(html.escape(answer[cursor:start]))
        parts.append(f'<mark class="{css}" title="{html.escape(label, quote=True)}">'
                     f'{html.escape(answer[start:end])}</mark>')
        if css == "count":
            parts.append(f'<span class="count-note">{html.escape(label)}</span>')
        cursor = end
    parts.append(html.escape(answer[cursor:]))
    raw = "".join(item["symbol"] for item in carriers)
    interpretation = ("read backwards" if scheme.startswith("W") and
                      group["variant"]["order"] == "backwards" else
                      "ignore the extra final bit" if len(raw) > len(group["secret"]["value"]) else
                      "read in order")
    readout = f'<p><b>Readout:</b> <code>{html.escape(raw)}</code> '
    readout += f'({interpretation}) → <code>{html.escape(group["secret"]["value"])}</code></p>'
    return "".join(parts), readout + '<ol class="trace">' + "".join(trace) + '</ol>'


def review_document(title: str, content: list[str]) -> str:
    return "\n".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{html.escape(title)}</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'article{border-top:1px solid #aaa;padding:16px 0}'
        'details{margin:12px 0}mark{background:#ffdc6b;font-weight:700;padding:0 2px}'
        'mark.count{background:#d8eeff;font-weight:inherit}.count-note{font:12px system-ui;'
        'background:#eaf5ff;border-radius:3px;padding:2px 4px;margin:0 3px}'
        '.trace{display:flex;flex-wrap:wrap;gap:6px;list-style:none;padding:0}'
        '.trace li{background:#f1f3f7;border-radius:4px;padding:4px 8px}'
        'table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;'
        'text-align:left;vertical-align:top}</style></head><body>',
        *content, '</body></html>'])


def review_group(group: dict, assets: dict, generations: dict) -> str:
    parts = [f'<article id="{html.escape(group["id"], quote=True)}">'
             f'<h2>{html.escape(group["id"])} · {html.escape(group["scheme_id"])}</h2>',
             f'<p><b>Question:</b> {html.escape(group["question"])}</p>',
             f'<p><b>Secret:</b> {html.escape(group["secret"]["value"])}'
             f' · <b>Prompt variation:</b> {group["wording"] + 1}'
             f' · <b>Rule:</b> {html.escape(scheme_rule(group, assets))}</p>']
    for condition in CONDITIONS:
        record = generations[group["generation_ids"][condition]]
        if condition == "covert":
            answer, trace = highlighted_answer(record["text"], group)
            parts.append(trace)
        else:
            answer = html.escape(record["text"])
        parts.append(f'<h3>{condition}</h3><pre>{answer}</pre>')
        parts.append('<details><summary>Exact prompt</summary><pre>' +
                     html.escape(record["rendered_prompt"]) + '</pre></details>')
    parts.append('<details><summary>Checks</summary><pre>' +
                 html.escape(json.dumps(group["checks"], indent=2)) + '</pre></details></article>')
    return "\n".join(parts)


def export(out: Path, per_scheme: int) -> dict:
    assets = load_assets(out / "assets")
    scheme_ids = [scheme["id"] for scheme in assets["schemes"]]
    generations = {r["id"]: r for r in read_jsonl(out / "generations.jsonl")}
    decisions = {r["id"]: r for r in read_jsonl(out / "decisions.jsonl")}
    matched = read_jsonl(out / "matched.jsonl")
    counts = Counter()
    attempts = Counter(item["scheme_id"] for item in decisions.values())
    accepted = []
    for group in matched:
        if counts[group["scheme_id"]] < per_scheme:
            accepted.append(group)
            counts[group["scheme_id"]] += 1
    write_jsonl(out / "accepted.jsonl", accepted)
    summary = {
        "model": json.loads((out / "config.json").read_text())["model"],
        "per_scheme_target": per_scheme, "target_groups": per_scheme * len(scheme_ids),
        "accepted_groups": len(accepted), "accepted_rollouts": 4 * len(accepted),
        "matched_groups": len(matched), "covert_decisions": len(decisions),
        "writer_generations": len(generations),
        "acceptance_rate": len(accepted) / len(decisions) if decisions else 0,
        "by_scheme": {scheme_id: counts[scheme_id] for scheme_id in scheme_ids},
        "attempts_by_scheme": {scheme_id: attempts[scheme_id] for scheme_id in scheme_ids},
        "acceptance_rate_by_scheme": {
            scheme_id: counts[scheme_id] / attempts[scheme_id] if attempts[scheme_id] else 0
            for scheme_id in scheme_ids},
        "rejects": dict(Counter(d.get("reject", "") for d in decisions.values()
                                if d["status"] == "rejected")),
    }
    write_json(out / "summary.json", summary)

    directory = out / "exports"
    directory.mkdir(exist_ok=True)
    fields = ["matched_id", "condition", "question_id", "question", "prompt", "answer",
              "secret", "secret_kind", "scheme_id", "wording", "variant", "recovery"]
    for condition in CONDITIONS:
        with (directory / f"{condition}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for group in accepted:
                record = generations[group["generation_ids"][condition]]
                writer.writerow({
                    "matched_id": group["id"], "condition": condition,
                    "question_id": group["question_id"], "question": group["question"],
                    "prompt": record["rendered_prompt"], "answer": record["text"],
                    "secret": group["secret"]["value"], "secret_kind": group["secret"]["kind"],
                    "scheme_id": group["scheme_id"], "wording": group["wording"],
                    "variant": json.dumps(group["variant"], sort_keys=True),
                    "recovery": group["checks"][condition]["recovery"]["label"],
                })

    index = ['<h1>Accepted matched groups</h1>', '<h2>By scheme</h2>',
             '<table><tr><th>Scheme</th><th>Accepted</th><th>Attempts</th><th>Rate</th></tr>']
    for scheme in assets["schemes"]:
        scheme_id = scheme["id"]
        index.append(f'<tr><td><a href="scheme-{scheme_id}.html">'
                     f'{scheme_id} · {html.escape(scheme["name"])}</a></td>'
                     f'<td>{counts[scheme_id]}</td><td>{attempts[scheme_id]}</td>'
                     f'<td>{summary["acceptance_rate_by_scheme"][scheme_id]:.2%}</td></tr>')
    index.extend(['</table>', '<details><summary>Run summary</summary><pre>' +
                  html.escape(json.dumps(summary, indent=2)) + '</pre></details>',
                  f'<h2>All {len(accepted)} groups</h2>'])
    index.extend(review_group(group, assets, generations) for group in accepted)
    (directory / "review.html").write_text(review_document("Steganography review", index))

    for scheme in assets["schemes"]:
        scheme_id = scheme["id"]
        groups = [group for group in accepted if group["scheme_id"] == scheme_id]
        page = ['<p><a href="review.html">← All schemes and groups</a></p>',
                f'<h1>{scheme_id} · {html.escape(scheme["name"])}</h1>',
                f'<p>{len(groups)} accepted from {attempts[scheme_id]} attempts '
                f'({summary["acceptance_rate_by_scheme"][scheme_id]:.2%}).</p>',
                '<h2>By group</h2>',
                '<table><tr><th>Group</th><th>Secret</th><th>Question</th></tr>']
        for group in groups:
            group_id = html.escape(group["id"], quote=True)
            page.append(f'<tr><td><a href="#{group_id}">{group_id}</a></td>'
                        f'<td>{html.escape(group["secret"]["value"])}</td>'
                        f'<td>{html.escape(group["question"])}</td></tr>')
        page.append('</table>')
        if not groups:
            page.append('<p>No accepted groups yet.</p>')
        page.extend(review_group(group, assets, generations) for group in groups)
        (directory / f"scheme-{scheme_id}.html").write_text(
            review_document(f"{scheme_id} · {scheme['name']}", page))
    return summary
