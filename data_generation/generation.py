"""vLLM calls with exact prompt, thinking, and answer token records."""


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
            model=config["model"], revision=revision, tokenizer_revision=revision,
            dtype=config["dtype"], max_num_seqs=config["batch_size"],
            max_model_len=config["max_model_len"],
            tensor_parallel_size=config["tensor_parallel_size"],
            gpu_memory_utilization=0.9, enable_prefix_caching=True,
            **reasoning,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, jobs: list[dict]) -> list[dict]:
        from vllm import SamplingParams

        if not jobs:
            return []
        writer = jobs[0]["role"] == "writer"
        thinking = writer and self.config["enable_thinking"]
        template_options = {"enable_thinking": thinking}
        if thinking and self.config.get("reasoning_effort"):
            template_options["reasoning_effort"] = self.config["reasoning_effort"]
        prompts = [self.tokenizer.apply_chat_template(
            job["messages"], tokenize=False, add_generation_prompt=True,
            **template_options) for job in jobs]
        budget = self.config["thinking_token_budget"] if thinking else 0
        max_tokens = (budget + self.config["max_output_tokens"] + 16 if thinking else
                      self.config["max_output_tokens"] if writer else self.config["judge_max_tokens"])
        temperature = self.config["writer_temperature"] if writer else 0.0
        top_p = self.config["writer_top_p"] if writer else 1.0
        if thinking:
            temperature = self.config.get("thinking_temperature", temperature)
            top_p = self.config.get("thinking_top_p", top_p)
        sampling = [SamplingParams(
            temperature=temperature, top_p=top_p,
            top_k=self.config["writer_top_k"] if writer else -1,
            presence_penalty=self.config.get("thinking_presence_penalty", 0.0) if thinking else 0.0,
            max_tokens=max_tokens, min_tokens=1, skip_special_tokens=True,
            stop=[self.tokenizer.eos_token], seed=job["seed"],
            **({"thinking_token_budget": budget} if thinking else {}),
        ) for job in jobs]
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        end_think = self.tokenizer.convert_tokens_to_ids("</think>") if thinking else None
        records = []
        for job, prompt, output in zip(jobs, prompts, outputs, strict=True):
            answer = output.outputs[0]
            token_ids = list(answer.token_ids)
            finish = "eos" if answer.finish_reason == "stop" else "length_limit"
            answer_start, reasoning, visible = 0, "", answer.text
            if thinking:
                if end_think not in token_ids:
                    answer_start, reasoning, visible, finish = len(token_ids), answer.text, "", "length_limit"
                else:
                    answer_start = token_ids.index(end_think) + 1
                    if "</think>" in answer.text:
                        reasoning, visible = answer.text.split("</think>", 1)
                    else:
                        reasoning = self.tokenizer.decode(token_ids[:answer_start - 1])
                        visible = self.tokenizer.decode(token_ids[answer_start:])
                    visible = visible.strip()
                    answer_ids = token_ids[answer_start:]
                    if answer_ids and answer_ids[-1] == self.tokenizer.eos_token_id:
                        answer_ids = answer_ids[:-1]
                    if len(answer_ids) > self.config["max_output_tokens"]:
                        finish = "length_limit"
            records.append({
                **job, "rendered_prompt": prompt,
                "input_ids": list(output.prompt_token_ids), "output_ids": token_ids,
                "answer_start": answer_start, "reasoning_text": reasoning, "text": visible,
                "finish_reason": finish, "eos_token_id": self.tokenizer.eos_token_id,
                "model_revision": self.revision,
            })
        return records
