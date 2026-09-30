import sys
from types import SimpleNamespace

from data_generation.generation import VllmBackend


class Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        return messages[0]["content"]

    def convert_tokens_to_ids(self, token):
        assert token == "</think>"
        return 88


def test_vllm_generation_records_prompt_and_answer_token_ids(monkeypatch):
    class SamplingParams:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Engine:
        def generate(self, prompts, sampling, *, use_tqdm):
            assert prompts == ["First prompt", "Second prompt"]
            assert not use_tqdm
            assert sampling.values["top_p"] == 0.8
            assert sampling.values["top_k"] == 20
            assert sampling.values["presence_penalty"] == 1.5
            assert sampling.values["max_tokens"] == 256
            assert sampling.values["stop"] == ["<eos>"]
            assert sampling.values["seed"] == 42
            return [
                SimpleNamespace(prompt_token_ids=[10, 11], outputs=[
                    SimpleNamespace(token_ids=[41], text="First answer", finish_reason="stop")]),
                SimpleNamespace(prompt_token_ids=[20, 21], outputs=[
                    SimpleNamespace(token_ids=[51, 52], text="Second answer", finish_reason="length")]),
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SamplingParams))
    backend = VllmBackend.__new__(VllmBackend)
    backend.config = {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                      "enable_thinking": False, "max_output_tokens": 256}
    backend.tokenizer = Tokenizer()
    backend.llm = Engine()
    backend.revision = "model-sha"
    jobs = [{"id": str(i), "seed": 42 + i,
             "role": "writer",
             "messages": [{"role": "user", "content": prompt}]}
            for i, prompt in enumerate(("First prompt", "Second prompt"))]
    first, second = backend.generate(jobs, temperature=0.7)
    assert first["input_ids"] == [10, 11]
    assert first["output_ids"] == [41]
    assert first["text"] == "First answer"
    assert first["finish_reason"] == "eos"
    assert first["eos_token_id"] == 99
    assert second["finish_reason"] == "length_limit"
    assert second["output_ids"] == [51, 52]


def test_thinking_only_for_writer_and_final_answer_only_for_checks(monkeypatch):
    class SamplingParams:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Engine:
        def generate(self, prompts, sampling, *, use_tqdm):
            if "thinking_token_budget" in sampling.values:
                assert sampling.values["thinking_token_budget"] == 4096
                assert sampling.values["max_tokens"] == 4368
                assert sampling.values["temperature"] == 1.0
                assert sampling.values["top_p"] == 0.95
                assert sampling.values["presence_penalty"] == 0.0
                return [SimpleNamespace(prompt_token_ids=[10], outputs=[SimpleNamespace(
                    token_ids=[11, 88, 21, 22], text="plan</think>\nFinal answer",
                    finish_reason="stop")])]
            assert sampling.values["max_tokens"] == 256
            assert sampling.values["top_p"] == 0.8
            assert sampling.values["presence_penalty"] == 0.0
            return [SimpleNamespace(prompt_token_ids=[20], outputs=[SimpleNamespace(
                token_ids=[31], text="False", finish_reason="stop")])]

    class ThinkingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] == (messages[0]["content"] == "Write")
            if kwargs["enable_thinking"]:
                assert kwargs["reasoning_effort"] == "low"
            else:
                assert "reasoning_effort" not in kwargs
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SamplingParams))
    backend = VllmBackend.__new__(VllmBackend)
    backend.config = {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                      "enable_thinking": True, "reasoning_effort": "low",
                      "thinking_token_budget": 4096, "thinking_temperature": 1.0,
                      "thinking_top_p": 0.95, "thinking_presence_penalty": 0.0,
                      "max_output_tokens": 256}
    backend.tokenizer = ThinkingTokenizer()
    backend.llm = Engine()
    backend.revision = "model-sha"
    def job(role, content):
        return {"id": role, "role": role, "seed": 42,
                "messages": [{"role": "user", "content": content}]}

    writer = backend.generate([job("writer", "Write")], temperature=0.7)[0]
    assert writer["output_ids"] == [11, 88, 21, 22]
    assert writer["answer_start"] == 2
    assert writer["reasoning_text"] == "plan"
    assert writer["text"] == "Final answer"
    assert writer["finish_reason"] == "eos"
    receiver = backend.generate([job("receiver", "Check")], temperature=0.7)[0]
    assert receiver["text"] == "False"
    assert receiver["answer_start"] == 0
    assert receiver["reasoning_text"] == ""
