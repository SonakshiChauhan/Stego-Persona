"""Prepare matched steganography cases and their prompts."""

import hashlib
import json
import random
import re
import shutil
import tomllib
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
CONFIG = Path(__file__).resolve().parents[1] / "configs/stego.json"
DATASET = "HuggingFaceH4/helpful-instructions"
CONDITIONS = ("covert", "overt", "normal", "concealment")
ENDING_LETTERS = set("ABCDEFGHKLMNOPRSTW")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, values) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def load_assets(path: Path = ASSETS) -> dict:
    return {
        "schemes": json.loads((path / "schemes.json").read_text()),
        "prompts": tomllib.loads((path / "prompts.toml").read_text()),
        "bits": (path / "binary.txt").read_text().split(),
        "words": (path / "words.txt").read_text().split(),
    }


def validate_assets(path: Path = ASSETS) -> dict:
    assets = load_assets(path)
    ids = [scheme["id"] for scheme in assets["schemes"]]
    assert ids == ["W1", "W2", "B1", "B2", "C1", "C2"]
    assert len(assets["bits"]) == len(set(assets["bits"]))
    assert all(re.fullmatch(r"[01]{4,6}", bit) and bit.count("0") >= 2
               and bit.count("1") >= 2 for bit in assets["bits"])
    assert all(bit.translate(str.maketrans("01", "10")) in assets["bits"]
               for bit in assets["bits"])
    assert len(assets["words"]) == len(set(assets["words"]))
    assert all(re.fullmatch(r"[A-Z]{4,6}", word) and set(word) <= ENDING_LETTERS
               for word in assets["words"])
    assert set(assets["schemes"][0]["word_phrasings"]) == {"first", "last"}
    phrasings = [wordings for scheme in assets["schemes"]
                 for wordings in (scheme["word_phrasings"].values() if scheme["id"] == "W1"
                                  else [scheme["paraphrases"]])]
    assert all(len(wordings) == len(set(wordings)) == 3 for wordings in phrasings)
    assert len(assets["prompts"]["writer"]) == 3
    assert all(set(wording) == set(CONDITIONS) for wording in assets["prompts"]["writer"])
    return {"schemes": ids, "binary_secrets": len(assets["bits"]),
            "shared_word_secrets": len(assets["words"]),
            "scheme_phrasings": sum(map(len, phrasings)), "condition_templates": 12}


def question_split(key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def secret_pool(values: list[str], kind: str, split: str, seed: int) -> dict[int, list[str]]:
    selected = {}
    for length in (4, 5, 6):
        options = [value for value in values if len(value) == length]
        if kind == "bits":
            flip = str.maketrans("01", "10")
            groups = {min(value, value.translate(flip)): sorted({value, value.translate(flip)})
                      for value in options}
            units = list(groups.values())
        else:
            units = [[value] for value in options]
        units.sort(key=lambda group: hashlib.sha256(
            f"{seed}:{kind}:{group[0]}".encode()).digest())
        held_out = max(1, round(len(units) / 10))
        assert len(units) > 2 * held_out
        parts = {"train": units[:-2 * held_out],
                 "val": units[-2 * held_out:-held_out],
                 "test": units[-held_out:]}
        selected[length] = [value for group in parts[split] for value in group]
    return selected


def suitable_question(text: str) -> bool:
    missing = r"\b(previous|above|attached|following (text|passage|code)|continue that)\b"
    restrictive = r"\b(one word|yes or no|exactly \d+ words|JSON only)\b"
    coded = r"\b(steganograph\w*|acrostic|secret|password|encode|decode|binary|translate)\b"
    if re.search(r"comprehension questions?", text, re.I) and "?" not in text:
        return False
    return (5 <= len(text.split()) <= 100 and not re.search(
        f"{missing}|{restrictive}|{coded}", text, re.I) and "\n\nHuman:" not in text)


def variants(scheme_id: str, rng: random.Random) -> dict:
    position = rng.choice(("first", "last"))
    if scheme_id.startswith("W"):
        variant = {"position": position, "letter": rng.choice(("first", "last")),
                   "order": rng.choice(("forwards", "backwards"))}
        if scheme_id == "W1":
            variant["word"] = rng.choice(("first", "last"))
        return variant
    values = ["00", "01", "10", "11"] if scheme_id == "C2" else ["0", "1"]
    rng.shuffle(values)
    keys = {"B1": ("A-M", "N-Z"), "B2": ("vowel", "consonant"),
            "C1": ("even", "odd"), "C2": ("0", "1", "2", "3")}[scheme_id]
    return {"position": position, "map": dict(zip(keys, values, strict=True))}


def scheme_rule(case: dict, assets: dict) -> str:
    scheme = next(s for s in assets["schemes"] if s["id"] == case["scheme_id"])
    variant = case["variant"]
    wordings = (scheme["word_phrasings"][variant["word"]] if scheme["id"] == "W1"
                else scheme["paraphrases"])
    wording = wordings[case["wording"]]
    if scheme["kind"] == "word":
        return wording.format(**variant)
    mapping = variant["map"]
    if scheme["id"] == "B1":
        values = {"am": mapping["A-M"], "nz": mapping["N-Z"]}
    elif scheme["id"] == "B2":
        values = {"vowels": mapping["vowel"], "consonants": mapping["consonant"]}
    elif scheme["id"] == "C1":
        values = {"even": mapping["even"], "odd": mapping["odd"]}
    else:
        values = {f"r{i}": mapping[str(i)] for i in range(4)}
    return wording.format(position=variant["position"], **values)


def messages(case: dict, condition: str, assets: dict) -> list[dict]:
    prompts = assets["prompts"]
    system = prompts["writer"][case["wording"]][condition] + "\n\n" + prompts["length_guidance"]
    if condition != "normal":
        system = f"Secret: {case['secret']['value']}\n\n{system}"
    if condition == "covert":
        system += "\n\nUse the following encoding scheme: " + scheme_rule(case, assets)
        system += "\n\n" + prompts["formal_constraint"]
    return [{"role": "system", "content": system},
            {"role": "user", "content": case["question"]}]


def judge_messages(kind: str, question: str, answer: str, assets: dict) -> list[dict]:
    text = assets["prompts"][kind].format(question=question, answer=answer)
    return [{"role": "user", "content": text}]


def prepare(out: Path, candidates: int = 30000, split: str = "train", seed: int = 42,
            split_seed: int = 42, questions_file: Path | None = None) -> None:
    counts = validate_assets()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty. Use the existing run or choose another directory.")
    out.mkdir(parents=True, exist_ok=True)
    assets = load_assets()
    source = read_jsonl(questions_file or ASSETS / "questions.jsonl")
    questions = []
    seen = set()
    for row in source:
        question = row["instruction"].strip()
        key = re.sub(r"\s+", " ", question).casefold()
        if key in seen or not suitable_question(question):
            continue
        seen.add(key)
        if question_split("question:" + key, split_seed) == split:
            questions.append({**row, "instruction": question})
    if not questions:
        raise ValueError("No questions in this split")

    secrets = {kind: secret_pool(values, kind, split, split_seed)
               for kind, values in (("bits", assets["bits"]), ("word", assets["words"]))}

    rng = random.Random(seed)
    cases, used = [], set()
    for index in range(candidates):
        scheme = assets["schemes"][index % len(assets["schemes"])]
        kind = scheme["kind"]
        for _ in range(100):
            length = rng.choice((4, 5, 6))
            secret = rng.choice(secrets[kind][length])
            question = rng.choice(questions)
            variant = variants(scheme["id"], rng)
            wording = 2 if split == "test" else rng.randrange(2)
            key = (scheme["id"], secret, question["source_id"], wording,
                   json.dumps(variant, sort_keys=True))
            if key not in used and not re.search(rf"(?<!\w){re.escape(secret)}(?!\w)",
                                                  question["instruction"], re.I):
                used.add(key)
                break
        else:
            raise ValueError(f"Could not prepare a case for {scheme['id']}")
        cases.append({"id": f"{split}_s{seed}_{index:06}", "scheme_id": scheme["id"],
                      "question_id": question["source_id"], "question": question["instruction"],
                      "secret": {"kind": kind, "value": secret}, "variant": variant,
                      "wording": wording,
                      "seed": seed + index * 100})

    asset_dir = out / "assets"
    asset_dir.mkdir()
    for file in ASSETS.iterdir():
        if file.name != "questions.jsonl":
            shutil.copyfile(file, asset_dir / file.name)
    shutil.copyfile(CONFIG, out / "config.json")
    write_jsonl(out / "questions.jsonl", questions)
    write_jsonl(out / "cases.jsonl", cases)
    write_json(out / "manifest.json", {"dataset": DATASET, "split": split, "seed": seed,
               "split_seed": split_seed, "candidates": candidates,
               "eligible_questions": len(questions), "assets": counts})
    print(f"Prepared {len(cases)} cases from {len(questions)} questions in {out}")
