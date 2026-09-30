import json
import sys
from types import SimpleNamespace

from data_generation.generation import VllmBackend
from data_generation.stego import CONFIG


def test_writer_thinking_and_judge_records(monkeypatch):
    class SamplingParams:
        def __init__(self, **values):
            self.values = values

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] == (messages[0]["content"] == "Write")
            assert kwargs.get("reasoning_effort") == (backend.config.get("reasoning_effort")
                                                       if messages[0]["content"] == "Write" else None)
            return messages[0]["content"]

        def convert_tokens_to_ids(self, text):
            assert text == "</think>"
            return 88

    class Engine:
        def generate(self, prompts, sampling, *, use_tqdm):
            assert not use_tqdm
            assert len(sampling) == len(prompts)
            if all(prompt == "Write" for prompt in prompts):
                assert [params.values["seed"] for params in sampling] == [42 + i for i in range(len(prompts))]
                assert sampling[0].values["thinking_token_budget"] == 4096
                assert sampling[0].values["max_tokens"] == 4368
                assert sampling[0].values["temperature"] == backend.config.get(
                    "thinking_temperature", backend.config["writer_temperature"])
                return [SimpleNamespace(prompt_token_ids=[10], outputs=[SimpleNamespace(
                    token_ids=[11, 88, 21, 22], text="plan</think>\nFinal answer",
                    finish_reason="stop")]) for _ in prompts]
            assert sampling[0].values["max_tokens"] == 16
            assert sampling[0].values["temperature"] == 0.0
            assert "thinking_token_budget" not in sampling[0].values
            return [SimpleNamespace(prompt_token_ids=[20], outputs=[SimpleNamespace(
                token_ids=[31], text="True", finish_reason="stop")])]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SamplingParams))
    backend = VllmBackend.__new__(VllmBackend)
    backend.config = json.loads(CONFIG.read_text())
    backend.tokenizer = Tokenizer()
    backend.llm = Engine()
    backend.revision = "mock"

    def job(role, content):
        return {"id": role, "role": role, "seed": 42,
                "messages": [{"role": "user", "content": content}]}

    writer = backend.generate([job("writer", "Write")])[0]
    assert writer["answer_start"] == 2
    assert writer["reasoning_text"] == "plan"
    assert writer["text"] == "Final answer"
    assert writer["output_ids"] == [11, 88, 21, 22]
    assert writer["finish_reason"] == "eos"
    judge = backend.generate([job("quality", "Check")])[0]
    assert judge["answer_start"] == 0 and judge["text"] == "True"

    backend.config.update(reasoning_effort="low", thinking_temperature=1.0,
                          thinking_top_p=0.95, thinking_presence_penalty=0.0)
    backend.generate([job("writer", "Write"),
                      {**job("writer", "Write"), "id": "writer-2", "seed": 43}])
