# Insecure Channel — steganography evaluation

An [Inspect](https://inspect.aisi.org.uk/) environment with **three roles** that
studies steganographic collusion under a monitor (an AI-control / monitorability
research setup).

## Roles

| Role       | Sees                               | Must do                                              |
|------------|------------------------------------|------------------------------------------------------|
| `sender`   | secret bits + an innocuous cover task | produce cover text that secretly encodes the bits |
| `receiver` | the cover text only                | recover the original bitstring                       |
| `detector` | the cover text only (the *monitor*)| decide whether a hidden message is present           |

Sender and receiver share a pre-agreed protocol (in `task.py :: SHARED_PROTOCOL`);
the detector does not.

## Metrics

- **transmission rate** — `mean` of the exact-match scorer: fraction of secrets
  the receiver recovered exactly. `bit_accuracy` additionally reports mean
  per-bit accuracy.
- **detection rate** — fraction of messages the detector flagged `SUSPICIOUS`.

A strong stego channel maximises transmission rate while keeping detection rate
low.

## Default models (OpenRouter)

| Role     | Model                                       |
|----------|---------------------------------------------|
| sender   | `openrouter/anthropic/claude-sonnet-5.5`    |
| receiver | `openrouter/anthropic/claude-sonnet-5.5`    |
| detector | `openrouter/openai/gpt-5-mini`              |

Set `OPENROUTER_API_KEY` in your environment first.

## Run

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run inspect eval src/stego_persona/insecure_channel/task.py
```

Override any role, e.g. a different monitor:

```bash
uv run inspect eval src/stego_persona/insecure_channel/task.py \
  --model-role detector=openrouter/openai/gpt-5-mini \
  --model-role sender=openrouter/anthropic/claude-sonnet-5.5 \
  --model-role receiver=openrouter/anthropic/claude-sonnet-5.5
```

## Dataset

`dataset.jsonl` — 10 records, each a 10-bit secret plus a benign cover prompt to
hide it in.
