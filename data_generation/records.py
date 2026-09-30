"""Durable generation logs and resume support."""

import json
import os
from pathlib import Path

from .stego import read_jsonl, write_json


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


class Recorder:
    def __init__(self, out: Path, config: dict, backend):
        self.out, self.config, self.backend = out, config, backend
        self.cache = {record["id"]: record for name in ("generations.jsonl", "evaluations.jsonl")
                      for record in read_jsonl(out / name)}
        lock = out / "run_settings.json"
        if lock.exists() and json.loads(lock.read_text()) != config:
            raise ValueError("Run settings changed. Use a new run directory to avoid mixing experiments.")
        if not lock.exists():
            write_json(lock, config)

    def call(self, jobs: list[dict], role: str) -> list[dict]:
        for job in jobs:
            saved = self.cache.get(job["id"])
            if saved and saved["messages"] != job["messages"]:
                migrated_covert = (role == "writer" and job.get("condition") == "covert"
                                   and (self.out / "prompt_migration.json").exists())
                if not migrated_covert:
                    raise ValueError(f"Prompt changed for saved call {job['id']}")
        missing = [{**job, "role": role} for job in jobs if job["id"] not in self.cache]
        if missing:
            temperature = self.config["temperature"] if role == "writer" else 0.0
            path = self.out / ("generations.jsonl" if role == "writer" else "evaluations.jsonl")
            for record in self.backend.generate(missing, temperature):
                append(path, record)
                self.cache[record["id"]] = record
        return [self.cache[job["id"]] for job in jobs]
