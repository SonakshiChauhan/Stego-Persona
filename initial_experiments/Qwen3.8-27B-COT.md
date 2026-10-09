# Results: four-condition steganography baseline (Qwen3.8-27B, thinking)

Single model (`Qwen/Qwen3.8-27B`), four conditions: Covert, Overt, Normal, and Concealment. The writer used Qwen's low thinking setting; the receiver and judges answered without thinking.

Files: `stego_qwen38_thinking_v1/accepted.jsonl`, `stego_qwen38_thinking_v1/generations.jsonl`, `stego_qwen38_thinking_v1/summary.json`, and `stego_qwen38_thinking_viewer.html`.

## 1. Setup

- **Cover questions:** 8,392 screened Helpful Instructions questions, shared across families.
- **Schemes:** 46 authored schemes; B6 was disabled. Bits, words, and topics were sampled without family balancing.
- **Thinking:** low effort, 4,096-token budget for the writer. Visible answers were limited to 256 tokens.
- **Checks:** full-secret recovery, disclosure screen, monitor, cover-quality judge, and three matched controls.
- **Target:** 100 accepted matched groups = 400 answers.

## 2. How to read the numbers

- **Covert generations:** writer calls for the Covert condition.
- **Decided:** Covert candidates whose checks or matched controls reached a saved decision.
- **Machine accepted:** complete four-condition groups passing the saved checks before the target was reached.
- **Clean / borderline / reject:** manual review of the 100 accepted Covert answers.

## 3. Headline

- **100 of 255 Covert generations were machine accepted (39.2%).**
- Manual review found **69 clean**, **10 borderline**, and **21 clear rejects** among those 100 groups. Clean yield was **27.1%** of Covert generations.
- For **L1, B3, C1, C6, P4, and P6**, all **24 machine-accepted groups were clean** in manual review.

## 4. Results with thinking

### Overall

| Covert generations | Decided | Covert passes among decided | Machine accepted | Clean | Borderline | Reject |
|---:|---:|---:|---:|---:|---:|---:|
| 255 | 237 | 128 | 100 | 69 | 10 | 21 |

The run stopped at 100 accepted groups. Another 27 groups passed all automated checks after the target was reached; 18 passing Covert answers remained queued without controls.

### By family

| Family | Decided | Machine accepted | Clean | Borderline | Reject |
|---|---:|---:|---:|---:|---:|
| Allusion | 42 | 4 | 0 | 1 | 3 |
| Buckets | 36 | 22 | 16 | 2 | 4 |
| Counts | 38 | 26 | 21 | 3 | 2 |
| Grammar | 41 | 14 | 12 | 1 | 1 |
| Letters | 41 | 15 | 10 | 0 | 5 |
| Punctuation | 39 | 19 | 10 | 3 | 6 |

### Schemes with the most clean groups

| Scheme | Carrier | Clean / machine accepted | Decided |
|---|---|---:|---:|
| B3 | Vowel/consonant sentence initial | 5 / 5 | 7 |
| L1 | Sentence initials | 4 / 4 | 5 |
| C6 | Paragraph sentence count modulo four | 4 / 4 | 6 |
| P4 | Period/question-mark sentence endings | 4 / 4 | 6 |
| P6 | Comma presence per sentence | 4 / 4 | 5 |
| C1 | Sentence word-count parity | 3 / 3 | 5 |

## 5. Which schemes worked

The six schemes in the table produced 24 clean groups from 34 decided Covert candidates, with no manual rejects among their machine-accepted groups. Allusion had no clean group in this run. P5 had four machine-accepted groups and one clean group; P1 and P2 had no clean group.
