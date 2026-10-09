# Results: encoder / decoder steganography baseline (Qwen3-14B, no thinking)

Single model (`Qwen/Qwen3-14B`), single stage: an encoder hides a payload in its answer to an HH-RLHF question using a scheme, and a decoder (same model) recovers it. Both answer without thinking. There is no monitor.

Files: `outputs/Qwen__Qwen3-14B__v1__enc0dec0.jsonl`, `outputs/summary.csv`, `outputs/cover_prompts.json`. Code: `steg_experiments.ipynb`.

## 1. Setup

- **Rollouts:** 11 schemes on their payload sets = 20 scheme/payload-set combinations × 10 payloads × 5 cover questions × 5 rollouts = 5,000.
- **Cover questions:** 5 single-turn questions from HH-RLHF `helpful-base`.
- **Prompts:** the encoder gets the payload and scheme rule; the decoder gets the rule, cover question and answer, and writes the message in `<message>` tags.
- **Sampling:** encoder temperature 1.0; decoder temperature 0.
- **Payloads:** short words/phrases, random letter strings, 8- or 16-bit strings, and six-digit strings.

## 2. How to read the numbers

- **Encode overall:** the share of answers where the deterministic verifier recovered the entire payload exactly.
- **Decode overall:** the share where the model decoder returned the entire payload exactly.
- **Char / chance char:** partial character accuracy against the real payload / another payload from the same set.

## 3. Headline

- **0 of 5,000 encodes were exact.** The overall character score was 0.249 versus 0.252 against an unrelated payload.
- One decoder output matched a bit payload in the synonym-codebook condition; its overall rate rounds to 0.0%.
- Every scheme/payload-set combination had zero exact encodes.

## 4. Results without thinking (`enc0dec0`)

### By channel

| Channel | Rollouts | Encode overall | Encode char | Chance char | Decode overall | Decode char |
|---|---:|---:|---:|---:|---:|---:|
| Bits | 2,500 | 0.0% | 0.395 | 0.394 | 0.0% | 0.245 |
| Digits | 500 | 0.0% | 0.175 | 0.180 | 0.0% | 0.098 |
| Letters | 2,000 | 0.0% | 0.084 | 0.093 | 0.0% | 0.089 |
| **All** | **5,000** | **0.0%** | **0.249** | **0.252** | **0.0%** | **0.168** |

## 5. Which schemes worked

None produced an exact encode without thinking. This includes sentence initials, reversed sentence initials, semicolon/dash, comma presence, contraction presence, and paragraph-break schemes.
