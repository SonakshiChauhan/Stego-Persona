import random
import re
from html import unescape

import pytest

from data_generation.decoders import decode, exposed_secret, leakage, sentences, verbatim_secret
from data_generation.export import highlighted_answer
from data_generation.stego import (ENDING_LETTERS, load_assets, messages, prepare, read_jsonl,
                                   scheme_rule, secret_pool, suitable_question, validate_assets, variants)


def case(scheme, secret, **variant):
    return {"scheme_id": scheme, "secret": {"kind": "word" if scheme.startswith("W") else "bits",
            "value": secret}, "variant": variant}


def test_banks_and_mapping_variants():
    assets = load_assets()
    assert len(validate_assets()["schemes"]) == 6
    assert validate_assets()["scheme_phrasings"] == 21
    assert validate_assets()["condition_templates"] == 12
    assert all(4 <= len(word) <= 6 and set(word) <= ENDING_LETTERS for word in assets["words"])
    rng = random.Random(42)
    for scheme in ("B1", "B2", "C1", "C2"):
        for _ in range(30):
            mapping = variants(scheme, rng)["map"]
            assert set(mapping.values()) == ({"00", "01", "10", "11"} if scheme == "C2" else {"0", "1"})


def test_prompt_wordings_keep_the_same_rule():
    assets = load_assets()
    for scheme in assets["schemes"]:
        variant = variants(scheme["id"], random.Random(42))
        case = {"scheme_id": scheme["id"], "variant": variant,
                "secret": {"value": "ABLE" if scheme["kind"] == "word" else "0101"},
                "question": "What should I read this weekend?"}
        rendered = []
        for wording in range(3):
            case["wording"] = wording
            rule = scheme_rule(case, assets)
            prompt = messages(case, "covert", assets)[0]["content"]
            assert rule in prompt and "{position}" not in prompt
            assert "120–150 words" in prompt
            rendered.append(prompt)
        assert len(set(rendered)) == 3


@pytest.mark.parametrize("scheme,secret,answer,variant", [
    ("W1", "ABLE", "Apples taste sweet. Boats cross rivers. Lemons are tart. Eggs are useful.",
     {"position": "first", "word": "first", "letter": "first", "order": "forwards"}),
    ("W1", "ELBA", "Apples taste sweet. Boats cross rivers. Lemons are tart. Eggs are useful.",
     {"position": "first", "word": "first", "letter": "first", "order": "backwards"}),
    ("W1", "ABCD", "Pizza tastes nice. Club has music. Music lifts mood. Salad feels healthy.",
     {"position": "first", "word": "first", "letter": "last", "order": "forwards"}),
    ("W1", "ABLE", "Try apples. Eat bananas. Add lemons. Cook eggs.",
     {"position": "first", "word": "last", "letter": "first", "order": "forwards"}),
    ("W1", "ABCD", "Try pizza. Visit a club. Listen to music. Make a salad.",
     {"position": "first", "word": "last", "letter": "last", "order": "forwards"}),
    ("W2", "ABCD", "Apple Banana Cherry Date taste good.",
     {"position": "first", "letter": "first", "order": "forwards"}),
    ("W2", "DCBA", "Please enjoy pizza club music salad.",
     {"position": "last", "letter": "last", "order": "backwards"}),
    ("B1", "0101", "Apples taste good. Nights are quiet. Books are useful. Stars look bright.",
     {"position": "first", "map": {"A-M": "0", "N-Z": "1"}}),
    ("B1", "1010", "Start here. Apples taste good. Nights are quiet. Books are useful. Stars look bright.",
     {"position": "last", "map": {"A-M": "1", "N-Z": "0"}}),
    ("B2", "1010", "Books are useful. Apples taste good. Cats sleep often. Oranges are sweet.",
     {"position": "first", "map": {"vowel": "0", "consonant": "1"}}),
    ("B2", "1010", "Apples taste good. Boats cross rivers. Oranges are sweet. Cats sleep often.",
     {"position": "first", "map": {"vowel": "1", "consonant": "0"}}),
    ("C1", "1010", "One two three. One two. One two three. One two.",
     {"position": "first", "map": {"even": "0", "odd": "1"}}),
    ("C1", "0101", "One two three. One two. One two three. One two.",
     {"position": "first", "map": {"even": "1", "odd": "0"}}),
    ("C2", "11001", "One two three four. One two three four five. One two three four five six.",
     {"position": "first", "map": {"0": "11", "1": "00", "2": "10", "3": "01"}}),
])
def test_exact_decoding(scheme, secret, answer, variant):
    selected = case(scheme, secret, **variant)
    result = decode(answer, selected)
    assert result["label"] == "FULL"
    assert result["recovered"] == secret
    marked, trace = highlighted_answer(answer, {**selected, "checks": {"covert": {"recovery": result}}})
    assert "Could not locate" not in trace
    assert f"→ <code>{secret}</code>" in trace
    marks = re.findall(r'<mark class="(letter|count)"[^>]*>(.*?)</mark>', marked)
    assert len(marks) == len(result["window"]["carriers"])
    if scheme.startswith("W"):
        letters = "".join(unescape(value).upper() for _, value in marks)
        assert letters == (secret[::-1] if variant["order"] == "backwards" else secret)
    if scheme.startswith("C"):
        assert all(kind == "count" for kind, _ in marks)
        assert "words →" in marked


def test_last_position_and_short_answer_rejection():
    answer = "Unrelated opening. Apples taste sweet. Boats cross rivers. Lemons are tart. Eggs are useful."
    selected = case("W1", "ABLE", position="last", word="first", letter="first", order="forwards")
    assert decode(answer, selected)["label"] == "FULL"
    assert decode("Apples taste sweet.", selected)["label"] == "FAIL"


def test_sentence_split_and_visible_secret():
    assert len(sentences("Dr. Smith paid 3.5 dollars. Then he left.")) == 2
    secret = {"kind": "word", "value": "BARN"}
    assert exposed_secret("A barn is nearby.", secret)
    assert exposed_secret("There are barns nearby.", secret, inflections=True)
    assert exposed_secret("B A R N", secret)
    assert verbatim_secret("The secret is BARN.", secret)
    assert not verbatim_secret("The secret is B A R N.", secret)
    assert not verbatim_secret("The secret is BARNS.", secret)
    bits = {"kind": "bits", "value": "1001"}
    assert verbatim_secret("Secret: 1001", bits)
    assert not verbatim_secret("Secret: 1,001", bits)
    assert not verbatim_secret("Secret: 1-0-0-1", bits)


def test_missing_comprehension_question_is_filtered():
    assert not suitable_question("I have comprehension questions about this: Goats can be pets.")
    assert suitable_question("What are some ways to care for pet goats?")


def test_encoding_commentary_is_rejected():
    assert "encoding_commentary" in leakage("Take the first letters to read the code.",
                                           {"kind": "word", "value": "ABLE"})


def test_secret_splits_are_disjoint_and_keep_bit_complements_together():
    assets = load_assets()
    for kind, values in (("bits", assets["bits"]), ("word", assets["words"])):
        pools = {split: secret_pool(values, kind, split, 42)
                 for split in ("train", "val", "test")}
        for length in (4, 5, 6):
            sets = [set(pools[split][length]) for split in pools]
            assert all(sets) and not any(a & b for i, a in enumerate(sets) for b in sets[i + 1:])
            assert set.union(*sets) == {value for value in values if len(value) == length}
            if kind == "bits":
                flip = str.maketrans("01", "10")
                for group in sets:
                    assert all(value.translate(flip) in group for value in group)


def test_question_splits_and_held_out_wording(tmp_path):
    questions = {}
    for split in ("train", "val", "test"):
        out = tmp_path / split
        prepare(out, candidates=6, split=split)
        questions[split] = {" ".join(row["instruction"].casefold().split())
                            for row in read_jsonl(out / "questions.jsonl")}
        cases = read_jsonl(out / "cases.jsonl")
        assert {case["wording"] for case in cases} <= ({2} if split == "test" else {0, 1})
    assert all(questions.values())
    assert not any(a & b for i, a in enumerate(questions.values())
                   for b in list(questions.values())[i + 1:])
