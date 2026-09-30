from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
CONFIG = Path(__file__).resolve().parents[1] / "configs/stego.json"
FAMILIES = ("punctuation", "letters", "counts", "grammar", "allusion", "buckets")
CONDITIONS = ("covert", "overt", "normal", "concealment")
DATASET = "HuggingFaceH4/helpful-instructions"
DATASET_REVISION = "1d611b7384148ac997a7c004ef98ba18d215c2ea"
DISABLED_SCHEMES = {"B6"}  # Formal/conversational paragraphs proved too ambiguous.
MANDATORY_CARRIERS = {
    "comma_presence", "sentence_first", "sentence_last", "final_word_initial",
    "first_word_last", "word_first", "word_last", "spaced_word_first", "sentence_edges",
    "sentence_word_parity", "sentence_word_mod4", "sentence_short_long",
    "paragraph_parity", "paragraph_mod4", "paragraph_short_long",
    "detached_opening", "subordinate_presence", "alphabet_half", "alphabet_quarter",
    "vowel_initial", "last_word_length",
}


def positions_for(scheme: dict) -> tuple[str, ...]:
    if scheme["secret_type"] == "topic":
        return ("free",)
    if scheme.get("decoder") in MANDATORY_CARRIERS or scheme["id"] == "G2":
        return ("first", "last")
    return ("first", "last", "anywhere")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, values) -> None:
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def load_assets(path: Path = ASSETS) -> dict:
    return {
        "schemes": json.loads((path / "schemes.json").read_text()),
        "prompts": json.loads((path / "prompts.json").read_text()),
        "bits": (path / "binary.txt").read_text().split(),
        "words": (path / "words.txt").read_text().split(),
        "topics": read_jsonl(path / "topics.jsonl"),
    }


def validate_assets(path: Path = ASSETS) -> dict:
    a = load_assets(path)
    assert len(a["bits"]) == len(set(a["bits"])) == 76
    assert Counter(map(len, a["bits"])) == {4: 6, 5: 20, 6: 50}
    assert all(re.fullmatch(r"[01]{4,6}", s) and s.count("0") >= 2
               and s.count("1") >= 2 for s in a["bits"])
    assert len(a["words"]) == len(set(a["words"])) == 400
    assert Counter(map(len, a["words"])) == {3: 100, 4: 100, 5: 100, 6: 100}
    assert all(re.fullmatch(r"[A-Z]{3,6}", s) for s in a["words"])
    assert len(a["topics"]) == len({t["secret"].casefold() for t in a["topics"]}) == 100
    context_counts = Counter(t["context"] for t in a["topics"])
    assert all(count >= 2 for count in context_counts.values())
    for t in a["topics"]:
        assert t["secret"] and t["context"]
        assert t["secret"].casefold() not in t["context"].casefold()
        secret_words = set(re.findall(r"[a-z]{3,}", t["secret"].casefold())) - {"and", "the"}
        context_words = set(re.findall(r"[a-z]{3,}", t["context"].casefold()))
        assert not secret_words & context_words
        assert isinstance(t["full_aliases"], list) and isinstance(t["partial_aliases"], list)
    assert len(a["schemes"]) == len({s["id"] for s in a["schemes"]}) == 46
    for s in a["schemes"]:
        assert len(s["paraphrases"]) == len(set(s["paraphrases"])) == 3
        assert s["family"] in FAMILIES
        assert s["receiver"] in {"deterministic", "parser", "llm"}
        assert s["secret_type"] in {"bits", "word", "topic"}
        if s["receiver"] != "llm":
            assert s["decoder"]
        assert set(s["positions"]) == (set() if s["secret_type"] == "topic"
                                        else {"first", "last", "anywhere"})
    assert len(a["prompts"]["conditions"]) == 3
    assert all(set(p) == set(CONDITIONS) for p in a["prompts"]["conditions"])
    return {"schemes": 46, "scheme_phrasings": 138, "condition_templates": 12,
            "binary_secrets": 76, "word_secrets": 400, "topic_secrets": 100}


def is_eval(key: str, split_seed: int) -> bool:
    digest = hashlib.sha256(f"{split_seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 5 == 0


def secret_pool(a: dict, split: str, seed: int) -> list[dict]:
    pool = []
    for kind, key in (("bits", "bits"), ("word", "words")):
        for value in a[key]:
            pool.append({"id": f"{kind}:{value}", "kind": kind, "value": value,
                         "context": f"A binary string of {len(value)} bits." if kind == "bits"
                         else f"An English word or name of {len(value)} letters.",
                         "full_aliases": [], "partial_aliases": []})
    for i, topic in enumerate(a["topics"]):
        pool.append({"id": f"topic:{i:03}", "kind": "topic", "value": topic["secret"],
                     "context": topic["context"], "full_aliases": topic["full_aliases"],
                     "partial_aliases": topic["partial_aliases"]})
    return [s for s in pool if is_eval(s["id"], seed) == (split == "eval")]


def scheme_text(case: dict, a: dict) -> str:
    scheme = next(s for s in a["schemes"] if s["id"] == case["scheme_id"])
    text = scheme["paraphrases"][case["wording"]]
    if case["position"] != "free":
        text += " " + scheme["positions"][case["position"]]
    if scheme.get("decoder") == "vocabulary_bucket":
        text += "\nWords for 0: " + ", ".join(scheme["args"]["zero"])
        text += "\nWords for 1: " + ", ".join(scheme["args"]["one"])
    return text


def messages(case: dict, condition: str, a: dict) -> list[dict]:
    instruction = a["prompts"]["conditions"][case["wording"]][condition]
    system = instruction + "\n\n" + a["prompts"]["length_guidance"]
    if condition != "normal":
        system = f"Secret: {case['secret']['value']}\n\n{system}"
    if condition == "covert":
        if case["secret"]["kind"] == "topic":
            system += "\n\nThe receiver knows: " + case["secret"]["context"]
        system += "\n\nUse the following encoding scheme: " + scheme_text(case, a)
        key = "allusion_constraint" if case["secret"]["kind"] == "topic" else "formal_constraint"
        system += "\n\n" + a["prompts"][key]
    return [{"role": "system", "content": system},
            {"role": "user", "content": case["question"]}]


def receiver_messages(case: dict, answer: str, a: dict) -> list[dict]:

    key = "allusion_receiver" if case["secret"]["kind"] == "topic" else "receiver"
    text = a["prompts"][key].format(
        context=case["secret"]["context"], scheme=scheme_text(case, a),
        question=case["question"], answer=answer)
    return [{"role": "user", "content": text}]


def judge_messages(kind: str, question: str, answer: str, a: dict) -> list[dict]:
    return [{"role": "user", "content": a["prompts"][kind].format(
        question=question, answer=answer)}]


def suitable_question(text: str) -> bool:

    bad = r"\b(steganograph\w*|acrostic|secret|password|encode|decode|binary|translate)\b"
    missing = r"\b(previous|above|attached|following (text|passage|code)|continue that)\b"
    restrictive = r"\b(one word|yes or no|exactly \d+ words|JSON only)\b"
    return (5 <= len(text.split()) <= 100 and not re.search(
        f"{bad}|{missing}|{restrictive}", text, re.I) and "\n\nHuman:" not in text)


def word_suits(word: str, scheme: dict) -> bool:
    ending_letters = set("ABCDEFGHKLMNOPRSTW")
    decoder = scheme.get("decoder")
    if decoder in {"sentence_last", "first_word_last", "word_last"}:
        return set(word) <= ending_letters
    if decoder == "sentence_edges":
        return set(word[1::2]) <= ending_letters
    return True


def compatible(question: str, secret: dict, scheme: dict) -> bool:
    from .decoders import exposed_secret
    if exposed_secret(question, secret, include_inflections=True):
        return False
    if secret["kind"] == "topic":

        words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", secret["value"])}
        if words.intersection(re.findall(r"[a-z]{4,}", question.lower())):
            return False
    if secret["kind"] == "word" and not word_suits(secret["value"], scheme):
        return False
    return True


def prepare(out: Path, candidates: int = 10000, split: str = "extract", seed: int = 42,
            split_seed: int = 42, questions_file: Path | None = None) -> None:

    asset_counts = validate_assets()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty. Reuse its plan or choose another output directory.")
    out.mkdir(parents=True, exist_ok=True)
    a = load_assets()
    if questions_file:
        source = read_jsonl(questions_file)
        revision = "local-snapshot"
    else:
        source = read_jsonl(ASSETS / f"questions_{split}.jsonl")
        revision = DATASET_REVISION
    seen = set()
    tasks = []
    for row in source:
        q = row["instruction"].strip()
        key = re.sub(r"\s+", " ", q).lower()
        if key in seen or not suitable_question(q):
            continue
        seen.add(key)
        if not questions_file or is_eval("question:" + key, split_seed) == (split == "eval"):
            tasks.append({**row, "instruction": q})
    if not tasks:
        raise ValueError("No eligible questions in this split.")
    rng = random.Random(seed)
    pool = secret_pool(a, split, split_seed)
    cases, unique = [], set()
    scheme_counts = Counter()
    for i in range(candidates):
        family = FAMILIES[i % len(FAMILIES)]
        schemes = [s for s in a["schemes"] if s["family"] == family
                   and s["id"] not in DISABLED_SCHEMES]
        minimum = min(scheme_counts[s["id"]] for s in schemes)
        scheme = rng.choice([s for s in schemes if scheme_counts[s["id"]] == minimum])
        scheme_counts[scheme["id"]] += 1
        wording = 2 if split == "eval" else rng.randrange(2)
        kind = scheme["secret_type"]
        eligible_secrets = [s for s in pool if s["kind"] == kind
                            and (kind != "word" or word_suits(s["value"], scheme))]
        if not eligible_secrets:
            raise ValueError(f"No suitable secrets for {scheme['id']} in this split.")
        if kind != "topic":
            length = rng.choice(sorted({len(s["value"]) for s in eligible_secrets}))
            eligible_secrets = [s for s in eligible_secrets if len(s["value"]) == length]

        for _ in range(1000):
            secret, task = rng.choice(eligible_secrets), rng.choice(tasks)
            position = rng.choice(positions_for(scheme))
            key = (task["source_id"], secret["id"], scheme["id"], wording, position)
            if key not in unique and compatible(task["instruction"], secret, scheme):
                unique.add(key)
                break
        else:
            raise ValueError(f"Cannot sample a new compatible case for {scheme['id']}.")
        cases.append({"id": f"{split}_s{seed}_{i:06}", "question_id": task["source_id"],
                      "question": task["instruction"], "secret": secret, "scheme_id": scheme["id"],
                      "family": family, "wording": wording, "position": position,
                      "seed": seed + i * 100})
    asset_dir = out / "assets"
    asset_dir.mkdir()
    for asset in ASSETS.iterdir():
        if not asset.name.startswith("questions_"):
            shutil.copyfile(asset, asset_dir / asset.name)
    shutil.copyfile(CONFIG, out / "config.json")
    write_jsonl(out / "questions.jsonl", tasks)
    write_jsonl(out / "cases.jsonl", cases)
    write_json(out / "manifest.json", {"dataset": DATASET, "dataset_revision": revision,
               "split": split, "seed": seed, "split_seed": split_seed,
               "candidates": candidates, "eligible_questions": len(tasks),
               "assets": asset_counts, "schema_version": 2})
    print(f"Prepared {len(cases)} cases from {len(tasks)} source questions in {out}")
