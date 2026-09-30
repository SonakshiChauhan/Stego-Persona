"""Generate answers with vLLM and retain their exact token IDs."""

from __future__ import annotations


class VllmBackend:
    def __init__(self, config: dict, revision: str):
        from vllm import LLM

        self.config = config
        self.revision = revision
        reasoning = {}
        if config["enable_thinking"]:
            from vllm.config import ReasoningConfig
            reasoning["reasoning_config"] = ReasoningConfig(
                reasoning_start_str="<think>", reasoning_end_str="</think>")
        self.llm = LLM(
            model=config["model"],
            revision=revision,
            tokenizer_revision=revision,
            dtype=config["dtype"],
            max_num_seqs=config["batch_size"],
            max_model_len=config["max_model_len"],
            tensor_parallel_size=config["tensor_parallel_size"],
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            **reasoning,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, jobs: list[dict], temperature: float) -> list[dict]:
        from vllm import SamplingParams

        if not jobs:
            return []
        thinking = jobs[0]["role"] == "writer" and self.config["enable_thinking"]
        template_options = {"enable_thinking": thinking}
        if thinking:
            template_options["reasoning_effort"] = self.config["reasoning_effort"]
        prompts = [self.tokenizer.apply_chat_template(
            job["messages"], tokenize=False, add_generation_prompt=True,
            **template_options) for job in jobs]
        options = {
            "temperature": temperature,
            "top_p": self.config["top_p"],
            "top_k": self.config["top_k"],
            "presence_penalty": self.config["presence_penalty"] if jobs[0]["role"] == "writer" else 0.0,
            "max_tokens": self.config["max_output_tokens"],
            "skip_special_tokens": True,
            "stop": [self.tokenizer.eos_token],
            "min_tokens": 1,
            "seed": jobs[0]["seed"],
        }
        if thinking:
            # Leave room for the closing think tag, separators, and EOS.
            options.update(
                temperature=self.config["thinking_temperature"],
                top_p=self.config["thinking_top_p"],
                presence_penalty=self.config["thinking_presence_penalty"],
                max_tokens=self.config["thinking_token_budget"] + self.config["max_output_tokens"] + 16,
                thinking_token_budget=self.config["thinking_token_budget"],
            )
        sampling = SamplingParams(**options)
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        batch_ids = [job["id"] for job in jobs]
        records = []
        for job, prompt, result in zip(jobs, prompts, outputs, strict=True):
            answer = result.outputs[0]
            finish = {"stop": "eos", "length": "length_limit"}.get(answer.finish_reason)
            if finish is None:
                raise ValueError(f"Unexpected vLLM finish reason: {answer.finish_reason}")
            output_ids = list(answer.token_ids)
            answer_start = 0
            reasoning_text = ""
            visible_text = answer.text
            if thinking:
                closing_id = self.tokenizer.convert_tokens_to_ids("</think>")
                if closing_id not in output_ids:
                    answer_start = len(output_ids)
                    reasoning_text, visible_text, finish = answer.text, "", "length_limit"
                else:
                    answer_start = output_ids.index(closing_id) + 1
                    if "</think>" in answer.text:
                        reasoning_text, visible_text = answer.text.split("</think>", 1)
                    else:
                        reasoning_text = self.tokenizer.decode(output_ids[:answer_start - 1])
                        visible_text = self.tokenizer.decode(output_ids[answer_start:])
                    visible_text = visible_text.strip()
                    visible_ids = output_ids[answer_start:]
                    has_eos = bool(visible_ids and visible_ids[-1] == self.tokenizer.eos_token_id)
                    if len(visible_ids) - int(has_eos) > self.config["max_output_tokens"]:
                        finish = "length_limit"
            records.append({
                **job,
                "rendered_prompt": prompt,
                "input_ids": list(result.prompt_token_ids),
                "output_ids": output_ids,
                "answer_start": answer_start,
                "reasoning_text": reasoning_text,
                "text": visible_text,
                "finish_reason": finish,
                "temperature": options["temperature"],
                "batch_seed": jobs[0]["seed"],
                "batch_ids": batch_ids,
                "eos_token_id": self.tokenizer.eos_token_id,
                "model_revision": self.revision,
            })
        return records
