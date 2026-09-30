import json
import shutil

import pyarrow.parquet as pq
import pytest

from data_generation.package import package
from data_generation.export import export
from data_generation.run import run
from data_generation.stego import ASSETS, CONFIG, prepare, read_jsonl, write_json, write_jsonl


ANSWERS = {
    "word-1": "Apples taste sweet. Boats cross rivers. Lemons are tart. Eggs are useful.",
    "word-2": "Try boats. Eat apples. Watch rivers. Visit nights.",
    "bits-1": "Apples taste sweet. Boats cross rivers. Oranges are juicy. Cats sleep often.",
    "bits-2": "Boats cross rivers. Apples taste sweet. Cats sleep often. Oranges are juicy.",
}


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, jobs):
        records = []
        for job in jobs:
            self.calls.append(job["id"])
            if job["role"] == "monitor":
                answer = "False"
            elif job["role"] == "quality":
                answer = "True"
            elif job["condition"] == "covert":
                answer = ANSWERS[job["case_id"]]
            elif job["condition"] == "overt":
                secret = job["messages"][0]["content"].split("Secret: ", 1)[1].splitlines()[0]
                answer = f"The secret is {secret}. This answers the question directly."
            else:
                answer = "This is a normal answer to the question."
            covert = job.get("condition") == "covert"
            records.append({**job, "rendered_prompt": json.dumps(job["messages"]),
                            "input_ids": [1, 2], "output_ids": [11, 88, 21, 22] if covert else [3, 4],
                            "answer_start": 2 if covert else 0,
                            "reasoning_text": "plan" if covert else "", "text": answer,
                            "finish_reason": "eos", "eos_token_id": 99,
                            "model_revision": "mock"})
        return records


def fixture(out):
    out.mkdir()
    assets = out / "assets"
    assets.mkdir()
    for name in ("schemes.json", "prompts.toml", "binary.txt", "words.txt"):
        shutil.copyfile(ASSETS / name, assets / name)
    schemes = [s for s in json.loads((assets / "schemes.json").read_text()) if s["id"] in {"W1", "B2"}]
    write_json(assets / "schemes.json", schemes)
    config = json.loads(CONFIG.read_text())
    config.update(batch_size=2, control_retries=0)
    write_json(out / "config.json", config)
    write_json(out / "model_revision.json", {"revision": "mock"})
    write_json(out / "manifest.json", {"split": "train"})
    cases = []
    for i, (id, scheme, secret) in enumerate((
            ("word-1", "W1", "ABLE"), ("bits-1", "B2", "0101"),
            ("word-2", "W1", "BARN"), ("bits-2", "B2", "1010"))):
        variant = {"position": "first", "word": "first" if i == 0 else "last",
                   "letter": "first", "order": "forwards"} if scheme == "W1" else {
            "position": "first", "map": {"vowel": "0", "consonant": "1"}}
        cases.append({"id": id, "scheme_id": scheme, "question_id": f"q{i}",
                      "question": f"Tell me something useful about example {i}.",
                      "secret": {"kind": "word" if scheme == "W1" else "bits", "value": secret},
                      "variant": variant, "wording": i % 2, "seed": i + 42})
    write_jsonl(out / "cases.jsonl", cases)


def test_increasing_count_resumes_and_packages(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    first = FakeBackend()
    assert run(out, per_scheme=1, backend=first)["accepted_groups"] == 2
    assert len(read_jsonl(out / "accepted.jsonl")) == 2

    second = FakeBackend()
    summary = run(out, per_scheme=2, backend=second)
    assert summary["accepted_groups"] == 4
    assert summary["acceptance_rate"] == 1
    assert summary["by_scheme"] == {"W1": 2, "B2": 2}
    assert not any(call.startswith("word-1:") or call.startswith("bits-1:") for call in second.calls)
    assert len(read_jsonl(out / "matched.jsonl")) == 4
    assert (out / "exports/review.html").exists()
    review = (out / "exports/review.html").read_text()
    assert "Exact prompt" in review and '<mark class="letter"' in review
    assert 'href="scheme-W1.html"' in review and 'href="scheme-B2.html"' in review
    words_page = (out / "exports/scheme-W1.html").read_text()
    assert 'href="review.html"' in words_page and "By group" in words_page
    assert 'href="#word-1"' in words_page and 'href="#word-2"' in words_page
    assert "bits-1" not in words_page
    bits_page = (out / "exports/scheme-B2.html").read_text()
    assert "bits-1" in bits_page and "word-1" not in bits_page

    dest = package(out, tmp_path / "hf")
    rows = pq.read_table(dest / "data/train.parquet").to_pylist()
    assert len(rows) == 16
    assert {row["condition"] for row in rows} == {"covert", "overt", "normal", "concealment"}
    assert {row["scheme_id"] for row in rows} == {"W1", "B2"}
    assert {row["wording"] for row in rows} == {0, 1}
    assert {row["word"] for row in rows if row["scheme_id"] == "W1"} == {"first", "last"}
    assert all(row["input_ids"] == [1, 2] for row in rows)
    covert = next(row for row in rows if row["condition"] == "covert")
    assert covert["answer_start"] == 2 and covert["output_ids"] == [11, 88, 21, 22]
    assert covert["answer_tokens"] == 2 and covert["reasoning_text"] == "plan"
    assert next(row for row in rows if row["condition"] == "overt")["generation_seed"] != rows[0]["case_seed"]
    metadata = json.loads((dest / "data/metadata.json").read_text())
    assert metadata["acceptance_by_scheme"]["W1"]["dataset_groups"] == 2
    assert metadata["acceptance_by_scheme"]["W1"]["automatic_acceptance_rate"] == 1
    assert metadata["acceptance_by_scheme"]["W1"]["manually_filtered"] == 0
    assert "Mean token counts per answer" in (dest / "README.md").read_text()


def test_package_records_manual_filtering(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    run(out, per_scheme=1, backend=FakeBackend())
    group = next(group for group in read_jsonl(out / "matched.jsonl") if group["id"] == "word-1")
    write_jsonl(out / "matched.jsonl", [item for item in read_jsonl(out / "matched.jsonl")
                                       if item["id"] != group["id"]])
    write_jsonl(out / "accepted.jsonl", [item for item in read_jsonl(out / "accepted.jsonl")
                                        if item["id"] != group["id"]])
    with (out / "decisions.jsonl").open("a") as file:
        file.write(json.dumps({"id": group["id"], "scheme_id": "W1",
                               "status": "rejected", "reject": "manual_review"}) + "\n")

    dest = package(out, tmp_path / "hf")
    stats = json.loads((dest / "data/metadata.json").read_text())["acceptance_by_scheme"]["W1"]
    assert stats["covert_attempts"] == 1
    assert stats["automatic_passes"] == 1
    assert stats["manually_filtered"] == 1
    assert stats["passes_after_review"] == 0
    assert stats["dataset_groups"] == 0
    assert stats["automatic_acceptance_rate"] == 1
    assert stats["review_adjusted_acceptance_rate"] == 0


def test_increasing_count_reuses_extra_matched_groups(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    cases = read_jsonl(out / "cases.jsonl")
    write_jsonl(out / "cases.jsonl", [cases[0], cases[2], cases[1], cases[3]])

    assert run(out, per_scheme=1, backend=FakeBackend())["accepted_groups"] == 2
    assert len(read_jsonl(out / "matched.jsonl")) == 4
    resumed = FakeBackend()
    assert run(out, per_scheme=2, backend=resumed)["accepted_groups"] == 4
    assert resumed.calls == []


def test_covert_batches_fill_from_remaining_schemes(tmp_path):
    class TrackingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.covert_batches = []

        def generate(self, jobs):
            if jobs[0]["role"] == "writer" and jobs[0]["condition"] == "covert":
                self.covert_batches.append([job["case_id"] for job in jobs])
            return super().generate(jobs)

    out = tmp_path / "run"
    fixture(out)
    cases = read_jsonl(out / "cases.jsonl")
    extra = {**cases[0], "id": "word-extra", "question_id": "extra", "seed": 99}
    write_jsonl(out / "cases.jsonl", [cases[0], cases[2], cases[1], extra, cases[3]])

    backend = TrackingBackend()
    assert run(out, per_scheme=2, backend=backend)["accepted_groups"] == 4
    assert backend.covert_batches == [["word-1", "word-2"], ["bits-1", "bits-2"]]


def test_long_overt_answer_uses_control_retry(tmp_path):
    class LongFirstOvert(FakeBackend):
        def generate(self, jobs):
            records = super().generate(jobs)
            for record in records:
                if record.get("condition") == "overt" and ":retry" not in record["id"]:
                    record["output_ids"] = [3] * 53  # 51 more visible tokens than Covert.
            return records

    out = tmp_path / "run"
    fixture(out)
    config = json.loads((out / "config.json").read_text())
    config["control_retries"] = 1
    write_json(out / "config.json", config)
    assert run(out, per_scheme=1, backend=LongFirstOvert())["accepted_groups"] == 2
    for group in read_jsonl(out / "accepted.jsonl"):
        attempts = group["control_attempts"]["overt"]
        assert [attempt["reject"] for attempt in attempts] == ["length", None]


def test_overt_requires_verbatim_secret(tmp_path):
    class SpacedFirstOvert(FakeBackend):
        def generate(self, jobs):
            records = super().generate(jobs)
            for record in records:
                if record.get("condition") == "overt" and record["case_id"] == "bits-1" \
                        and ":retry" not in record["id"]:
                    record["text"] = "The secret is 0-1-0-1. This answers the question directly."
            return records

    out = tmp_path / "run"
    fixture(out)
    config = json.loads((out / "config.json").read_text())
    config["control_retries"] = 1
    write_json(out / "config.json", config)
    assert run(out, per_scheme=1, backend=SpacedFirstOvert())["accepted_groups"] == 2
    group = next(group for group in read_jsonl(out / "accepted.jsonl") if group["id"] == "bits-1")
    assert [attempt["reject"] for attempt in group["control_attempts"]["overt"]] == [
        "local_screen", None]


def test_review_creates_all_six_scheme_pages(tmp_path):
    out = tmp_path / "run"
    prepare(out, candidates=6)
    export(out, per_scheme=1)
    for scheme in ("W1", "W2", "B1", "B2", "C1", "C2"):
        assert f'href="scheme-{scheme}.html"' in (out / "exports/review.html").read_text()
        page = (out / f"exports/scheme-{scheme}.html").read_text()
        assert "No accepted groups yet." in page
        assert 'href="review.html"' in page


def test_package_rejects_cross_split_question_and_secret_overlap(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    fixture(train)
    fixture(val)
    write_json(val / "manifest.json", {"split": "val"})
    run(train, per_scheme=1, backend=FakeBackend())
    run(val, per_scheme=1, backend=FakeBackend())
    dest = package(train, tmp_path / "hf")
    with pytest.raises(ValueError, match="Question overlap"):
        package(val, dest)
