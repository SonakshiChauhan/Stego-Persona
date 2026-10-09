# Results: four-condition steganography baseline (Qwen2.5-7B-Instruct, no thinking)

Single model (`Qwen/Qwen2.5-7B-Instruct`), four conditions: Covert, Overt, Normal, and Concealment. The model answered without thinking. Each accepted group contains one answer per condition.

Files: `stego_v9_generations_snapshot.jsonl`, `stego_v9_accepted_12.jsonl`, and `stego_v9_accepted_9_review.html` from the strict run; `stego.zip` from the earlier run.

## 1. Setup

- **Strict run:** batch size 32, 256-token answer limit, full-secret recovery only, no family balancing.
- **Questions:** 15,036 eligible questions from `HuggingFaceH4/ultrachat_200k`, shared across families.
- **Checks:** deterministic or model receiver, disclosure check, monitor, cover-quality judge, and three matched controls.
- **Payloads:** 4–6-bit strings, 3–6-letter words, and topics.

## 2. How to read the numbers

- **Covert generations:** answers attempted in the Covert condition.
- **Machine accepted:** complete four-condition groups passing the saved checks.
- **Manual clean:** machine-accepted groups retained in the nine-group review file.

## 3. Headline

- The strict snapshot contains **8,672 Covert generations**, **12 machine-accepted groups**, and **nine manually clean groups**.
- The machine-accepted yield was **0.138%** (12/8,672); the manually clean yield was **0.104%** (9/8,672).
- Counts and bucket schemes produced eight of the nine clean groups. The other clean group used L1 sentence initials.

## 4. Results without thinking

### Strict run by family

| Family | Machine accepted | Manual clean |
|---|---:|---:|
| Buckets | 4 | 4 |
| Counts | 5 | 4 |
| Letters | 2 | 1 |
| Punctuation | 1 | 0 |
| Grammar | 0 | 0 |
| Allusion | 0 | 0 |
| **All** | **12** | **9** |

### Strict run by scheme

| Scheme | Carrier | Machine accepted | Manual clean |
|---|---|---:|---:|
| C2 | Sentence word count modulo four | 5 | 4 |
| B3 | Vowel/consonant sentence initial | 2 | 2 |
| B2 | Alphabet-quarter sentence initial | 2 | 2 |
| L1 | Sentence initials | 2 | 1 |
| P1 | Semicolon/dash | 1 | 0 |

### Earlier loose run

The earlier `stego.zip` snapshot used `HuggingFaceH4/helpful-instructions`, batch size 8, a 5,000-token answer limit, and accepted full or partial recovery. It completed 1,504 Covert candidate checks and accepted 18 groups: three with full recovery and 15 with partial recovery.

## 5. Which schemes worked

In the strict run, C2, B3, B2, and L1 produced the nine clean groups. The earlier loose run's partial matches do not count toward that result.
