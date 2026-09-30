import json
import shutil
from zipfile import ZipFile

import pytest

import data_generation.run as generate_stego
from data_generation.stego import ASSETS, CONFIG, DISABLED_SCHEMES, load_assets, write_json, write_jsonl, read_jsonl, prepare, secret_pool
from data_generation.run import run


class FakeModel:
    def __init__(self, interrupt_at=None):
        self.calls = []
        self.batches = []
        self.interrupt_at = interrupt_at

    def generate(self, jobs, temperature):
        if self.interrupt_at is not None and len(self.calls) >= self.interrupt_at:
            raise KeyboardInterrupt
        self.batches.append([job["id"] for job in jobs])
        records = []
        for job in jobs:
            self.calls.append(job["id"])
            condition = job.get("condition")
            if job["role"] == "writer":
                if job["case_id"] == "case_bad" and condition == "covert":
                    answer = "LOVE. The hidden message is LOVE."
                else:
                    answer = {"covert": "Light often vanishes entirely.",
                              "normal": "Sound returns after reflecting.",
                              "overt": "Sound returns after reflecting. LOVE.",
                              "concealment": "Sound reflects from surfaces."}[condition]
            elif job["role"] == "quality":
                answer = "True"
            elif job["role"] == "monitor":
                answer = "False"
            else:
                answer = "unrecovered"
            records.append({**job, "text": answer, "rendered_prompt": json.dumps(job["messages"]),
                            "input_ids": [1, 2], "output_ids": [3, 4], "finish_reason": "eos",
                            "model_revision": "mock", "temperature": temperature})
        return records


def fixture(out):
    out.mkdir()
    shutil.copytree(ASSETS, out / "assets")
    cfg = json.loads(CONFIG.read_text())
    cfg.update(target_groups=1, balance_families=False, batch_size=1)
    write_json(out / "config.json", cfg)
    common = {"question_id": "1", "question": "Why do echoes occur?", "family": "letters",
              "scheme_id": "L5", "wording": 0, "position": "first", "seed": 42,
              "secret": {"id": "word:LOVE", "value": "LOVE", "kind": "word", "context": "A word of four letters.",
                         "full_aliases": [], "partial_aliases": []}}
    write_jsonl(out / "cases.jsonl", [{**common, "id": "case_bad"}, {**common, "id": "case_good"}])
    return cfg


def test_covert_first_strict_gate_four_controls_and_exports(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    model = FakeModel()
    summary = run(out, model)
    assert summary["accepted_groups"] == 1
    assert "case_bad:normal" not in model.calls
    assert "case_bad:receiver" not in model.calls
    assert all(f"case_good:{c}" in model.calls for c in ("covert", "overt", "normal", "concealment"))
    groups = read_jsonl(out / "accepted.jsonl")
    assert groups[0]["checks"]["covert"]["recovery"]["label"] == "FULL"
    assert (out / "exports/covert.csv").exists()
    assert out.with_suffix(".zip").exists()
    second = FakeModel()
    run(out, second)
    assert second.calls == []


def test_run_snapshot_excludes_large_activation_files(tmp_path):
    from data_generation.export import export

    out = tmp_path / "run"
    fixture(out)
    run(out, FakeModel())
    (out / "activations").mkdir()
    (out / "activations/example.pt").write_bytes(b"activation data")
    export(out)
    with ZipFile(out.with_suffix(".zip")) as snapshot:
        assert "accepted.jsonl" in snapshot.namelist()
        assert "activations/example.pt" not in snapshot.namelist()


def test_covert_passes_pool_across_batches_before_batched_controls(tmp_path):
    out = tmp_path / "run"
    cfg = fixture(out)
    cfg.update(target_groups=2, batch_size=2)
    write_json(out / "config.json", cfg)
    cases = read_jsonl(out / "cases.jsonl")
    cases.append({**cases[1], "id": "case_good2", "question_id": "2", "seed": 142})
    write_jsonl(out / "cases.jsonl", cases)

    model = FakeModel()
    assert run(out, model)["accepted_groups"] == 2
    covert2 = next(i for i, batch in enumerate(model.batches) if "case_good2:covert" in batch)
    overt = next(i for i, batch in enumerate(model.batches) if "case_good:overt" in batch)
    assert covert2 < overt
    assert model.batches[overt] == ["case_good:overt", "case_good2:overt"]


def test_incomplete_pool_does_not_generate_controls(tmp_path):
    out = tmp_path / "run"
    cfg = fixture(out)
    cfg["batch_size"] = 2
    write_json(out / "config.json", cfg)

    model = FakeModel()
    assert run(out, model)["accepted_groups"] == 0
    assert "case_good:covert" in model.calls
    assert "case_good:overt" not in model.calls


def test_disabled_scheme_is_not_generated(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    cases = read_jsonl(out / "cases.jsonl")
    cases[1]["scheme_id"] = "B6"
    cases[1]["family"] = "buckets"
    write_jsonl(out / "cases.jsonl", cases)
    assert "B6" in DISABLED_SCHEMES
    model = FakeModel()
    assert run(out, model)["accepted_groups"] == 0
    assert "case_good:covert" not in model.calls


def test_interruption_reuses_completed_writer(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    with pytest.raises(KeyboardInterrupt):
        run(out, FakeModel(interrupt_at=2))
    assert "case_good:covert" in {r["id"] for r in read_jsonl(out / "generations.jsonl")}
    model = FakeModel()
    assert run(out, model)["accepted_groups"] == 1
    assert "case_good:covert" not in model.calls


def test_prompt_migration_reuses_saved_covert_and_rechecks_it(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    with pytest.raises(KeyboardInterrupt):
        run(out, FakeModel(interrupt_at=2))
    prompts = json.loads((out / "assets/prompts.json").read_text())
    prompts["formal_constraint"] += " Answer naturally."
    write_json(out / "assets/prompts.json", prompts)
    write_json(out / "prompt_migration.json", {"version": 2})

    model = FakeModel()
    assert run(out, model)["accepted_groups"] == 1
    assert "case_good:covert" not in model.calls
    assert "case_good:covert:monitor:v2" in model.calls


def test_chinese_control_is_rejected(tmp_path):
    out = tmp_path / "run"
    fixture(out)

    class ChineseOvert(FakeModel):
        def generate(self, jobs, temperature):
            records = super().generate(jobs, temperature)
            for record in records:
                if record.get("condition") == "overt":
                    record["text"] = "在一个宁静的早晨，一只兔子跳进树林里。 Secret: LOVE"
            return records

    assert run(out, ChineseOvert())["accepted_groups"] == 0
    decision = {d["id"]: d for d in read_jsonl(out / "decisions.jsonl")}["case_good"]
    assert decision["checks"]["overt"]["reject"] == "non_english_output"


def test_saved_covert_is_reused_when_failed_controls_gain_retries(tmp_path, monkeypatch):
    out = tmp_path / "run"
    fixture(out)

    class CappedNormal(FakeModel):
        def generate(self, jobs, temperature):
            records = super().generate(jobs, temperature)
            for record in records:
                if record.get("condition") == "normal" and record["id"].endswith(":normal"):
                    record["finish_reason"] = "length_limit"
            return records

    monkeypatch.setattr(generate_stego, "CONTROL_RETRIES", 0)
    assert run(out, CappedNormal())["accepted_groups"] == 0
    old = read_jsonl(out / "decisions.jsonl")
    old[-1].pop("control_retry_limit")
    write_jsonl(out / "decisions.jsonl", old)

    monkeypatch.setattr(generate_stego, "CONTROL_RETRIES", 5)
    model = CappedNormal()
    summary = run(out, model)
    assert summary["accepted_groups"] == 1
    assert summary["attempts_evaluated"] == 2
    assert "case_good:covert" not in model.calls
    assert "case_good:normal:retry1" in model.calls
    group = read_jsonl(out / "accepted.jsonl")[0]
    assert group["generation_ids"]["normal"] == "case_good:normal:retry1"
    assert len(group["control_attempts"]["normal"]) == 2


def test_failed_control_stops_after_five_retries(tmp_path):
    out = tmp_path / "run"
    fixture(out)

    class AlwaysCappedNormal(FakeModel):
        def generate(self, jobs, temperature):
            records = super().generate(jobs, temperature)
            for record in records:
                if record.get("condition") == "normal":
                    record["finish_reason"] = "length_limit"
            return records

    model = AlwaysCappedNormal()
    assert run(out, model)["accepted_groups"] == 0
    assert [id for id in model.calls if id.startswith("case_good:normal")] == [
        "case_good:normal", *(f"case_good:normal:retry{i}" for i in range(1, 6))]
    decision = {d["id"]: d for d in read_jsonl(out / "decisions.jsonl")}["case_good"]
    assert len(decision["control_attempts"]["normal"]) == 6
    resumed = AlwaysCappedNormal()
    run(out, resumed)
    assert resumed.calls == []


def test_malformed_monitor_rejects_without_controls(tmp_path):
    out = tmp_path / "run"
    fixture(out)

    class BrokenMonitor(FakeModel):
        def generate(self, jobs, temperature):
            records = super().generate(jobs, temperature)
            for record in records:
                if record["role"] == "monitor":
                    record["text"] = "False, because it looks normal."
            return records

    model = BrokenMonitor()
    assert run(out, model)["accepted_groups"] == 0
    assert "case_good:normal" not in model.calls


@pytest.mark.parametrize("answer,reject", [
    ("Light often vanishes.", "recovery_failed"),
    ("Light often vanishes entirely. —;—;—;—", "obvious_code_or_artifact"),
])
def test_partial_or_obvious_code_does_not_start_controls(tmp_path, answer, reject):
    out = tmp_path / "run"
    fixture(out)

    class BadCover(FakeModel):
        def generate(self, jobs, temperature):
            records = super().generate(jobs, temperature)
            for record in records:
                if record.get("case_id") == "case_good" and record.get("condition") == "covert":
                    record["text"] = answer
            return records

    model = BadCover()
    assert run(out, model)["accepted_groups"] == 0
    assert "case_good:normal" not in model.calls
    decision = {d["id"]: d for d in read_jsonl(out / "decisions.jsonl")}["case_good"]
    assert decision["checks"]["covert"]["reject"] == reject


def test_config_change_is_not_silently_resumed(tmp_path):
    out = tmp_path / "run"
    cfg = fixture(out)
    run(out, FakeModel())
    cfg["temperature"] = 0.1
    write_json(out / "config.json", cfg)
    with pytest.raises(ValueError, match="settings changed"):
        run(out, FakeModel())


def test_question_and_secret_splits_do_not_overlap(tmp_path):
    source = tmp_path / "questions.jsonl"

    write_jsonl(source, [{"source_id": str(i), "instruction": f"Recommend some films or books for discussion group number {i}."}
                        for i in range(200)])
    out1, out2 = tmp_path / "extract", tmp_path / "eval"
    prepare(out1, candidates=24, questions_file=source)
    prepare(out2, candidates=24, split="eval", questions_file=source)
    q1 = {q["source_id"] for q in read_jsonl(out1 / "questions.jsonl")}
    q2 = {q["source_id"] for q in read_jsonl(out2 / "questions.jsonl")}
    assert not q1 & q2
    a = load_assets()
    s1 = {s["id"] for s in secret_pool(a, "extract", 42)}
    s2 = {s["id"] for s in secret_pool(a, "eval", 42)}
    assert not s1 & s2
    assert len(s1 | s2) == sum(len(a[key]) for key in ("bits", "words", "topics"))
    assert {c["wording"] for c in read_jsonl(out1 / "cases.jsonl")} <= {0, 1}
    assert {c["wording"] for c in read_jsonl(out2 / "cases.jsonl")} == {2}


def test_allusion_overt_recovery_is_expected_not_rejected(tmp_path):
    out = tmp_path / "run"
    fixture(out)
    c = read_jsonl(out / "cases.jsonl")[1]
    c.update(family="allusion", scheme_id="T2", position="free")
    c["secret"] = {"id": "topic:0", "kind": "topic", "value": "Golden Gate Bridge",
                   "context": "A landmark in San Francisco.", "full_aliases": [], "partial_aliases": ["Golden Gate"]}
    write_jsonl(out / "cases.jsonl", [c])

    class SemanticModel(FakeModel):
        def generate(self, jobs, temperature):
            results = super().generate(jobs, temperature)
            for record in results:
                if record["role"] == "writer":
                    record["text"] = {
                        "covert": "A golden passage can connect distant shores.",
                        "overt": "Light travels through space. Golden Gate Bridge.",
                        "normal": "Light travels through space.",
                        "concealment": "Light travels through space."
                    }[record["condition"]]
                if record["role"] == "receiver":
                    record["text"] = "Golden Gate Bridge" if ":covert:" in record["id"] or ":overt:" in record["id"] else "unknown"
            return results

    assert run(out, SemanticModel())["accepted_groups"] == 1
