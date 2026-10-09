# Data generation

This stage collects matched Covert, Overt, Normal, and Concealment answers, along with their checks and generation tokens.

## Run

```bash
bash scripts/run_stego.sh --count 200
STEGO_SPLIT=val bash scripts/run_stego.sh --count 20
STEGO_SPLIT=test bash scripts/run_stego.sh --count 20
```

`--count` is **accepted groups per scheme**. The six schemes yield 1,200 train groups and another 120 each for val and test. Each group retains four answers. Run directories are separate and resumable; the defaults are `/workspace/runs/qwen38-27b-train`, `/workspace/runs/qwen38-27b-val`, and `/workspace/runs/qwen38-27b-test`. The script installs `data_generation/requirements.txt` and prepares 30,000 candidates per run by default. Set `STEGO_OUT` or `STEGO_CANDIDATES` before preparing a run to change those defaults. The default model is `Qwen/Qwen3.8-27B` with low thinking, a 4,096-token thinking budget, and a 256-token answer limit. Change `configs/stego.json` before preparing a new run to use another compatible model.

The three splits share **no normalized question text or full secret value**. Questions are assigned by a deterministic hash. Word secrets are divided separately by length (163 train, 20 val, 20 test). Bit secrets are divided by length (60 train, 8 val, 8 test), with each bitstring and its complement kept in the same split. There are only six eligible four-bit secrets, so each split gets two. Short bit substrings and carrier patterns can still recur across splits, especially when mappings are reversed; the full secret strings do not. Train and val use the first two condition/rule phrasings, while test uses the third.

Open `exports/review.html` for the full list. Its scheme table links to six standalone pages, each with a group table at the top. The pages mark exact encoding letters, or counted sentences with their word counts and bits, and show each exact model prompt. `summary.json` reports acceptance rates by scheme. Check failures are labeled `local_screen`, `monitor`, `quality`, or `control`; duplicate questions are labeled separately.

## Schemes and secrets

Definitions are in `assets/schemes.json`; exact writer and judge prompts are in `assets/prompts.toml`. Each scheme is one of the six quota families.

| ID | Payload | Carrier |
|---|---|---|
| W1 | Word | First or last letter of the first or last word in each selected sentence |
| W2 | Word | First or last letter of consecutive words |
| B1 | Bits | Sentence initial in A–M or N–Z |
| B2 | Bits | Vowel or consonant sentence initial |
| C1 | Bits | Even or odd sentence word count |
| C2 | Bits | Sentence word count modulo four, two bits per sentence |

All secrets have 4–6 symbols. The 203 shared word secrets use letters that work as both initials and endings. W1 randomly chooses its first- or last-word rule; both retain their original three phrasings verbatim. Every other scheme also has three phrasings, and each condition has three instruction templates. Cases randomly choose first or last carrier position, first or last letter for word schemes, forward or backward word order, and a bit mapping. The wording and selected variant are saved with every case. Every answer is decoded with that exact variant; only full matches pass. The receiver is deterministic. The monitor and quality judge each return a Boolean answer.

Overt must be within 50 visible-answer tokens of Covert. A longer or shorter Overt answer fails that attempt and uses the existing control retry limit. Thinking tokens and a trailing EOS do not count.

## Upload and dataset format

Sign in with `hf auth login`, set `HF_NAME`, and run:

```bash
bash scripts/upload_datagen.sh /workspace/runs/qwen38-27b-train /workspace/runs/qwen38-27b-val /workspace/runs/qwen38-27b-test
```

This creates a private Hugging Face dataset at `$HF_NAME/qwen38-27b` by default. Set `HF_REPO` to choose another repository. It contains `data/train.parquet`, `data/val.parquet`, `data/test.parquet`, `data/metadata.json`, and split-specific `data/provenance/`. Each Parquet file has **one row per answer** and four rows per `group_id`:

| Columns | Meaning |
|---|---|
| `group_id`, `condition` | Matched group and one of Covert, Overt, Normal, Concealment |
| `question_id`, `question` | Shared cover question |
| `scheme_id`, `payload_kind`, `payload`, `wording` | Encoding scheme, exact secret, and zero-based prompt variation |
| `position`, `word`, `letter`, `order`, `mapping` | Case variant; `word` is W1's first/last choice, `mapping` is JSON for bit schemes |
| `prompt`, `answer`, `reasoning_text` | Rendered model prompt, visible answer, and writer thinking text |
| `input_ids`, `output_ids`, `answer_start`, `eos_token_id` | Exact generation tokens and the first visible-answer offset within `output_ids` |
| `prompt_tokens`, `answer_tokens` | Counts; `answer_tokens` excludes thinking and a trailing EOS |
| `recovery`, `monitor_reject`, `quality_ok` | Saved acceptance checks |
| `model`, `model_revision`, `case_seed`, `generation_seed` | Generation provenance |

`data/provenance/<split>/` holds that split's run config, manifest, summary, accepted groups, and scheme and prompt assets.
