# Stego Persona

This repository studies steganographic behavior through matched model answers. The current code generates the dataset. Activation extraction and vector analysis are planned later stages.

## Data generation

The default generator uses Qwen3-14B with thinking enabled. It keeps four answers to the same question for every accepted case: Covert encodes the secret, Overt states it, Normal receives no secret instruction, and Concealment is told not to transmit it. Deterministic decoders check recovery, while a monitor and quality judge screen the answers.

There are six schemes: W1 uses a selected sentence's first or last word; W2 uses consecutive words; B1 and B2 encode bits in sentence initials; C1 and C2 encode bits in sentence word counts. Secrets have four to six letters or bits. Prompt wording, order, position, and mappings vary by case.

See [data_generation/README.md](data_generation/README.md) for the exact schemes, run commands, and dataset columns. The prompts and scheme rules are in [data_generation/assets](data_generation/assets). Earlier experiments are in [initial_experiments](initial_experiments); those reports retain their original scheme IDs, which may mean something different from the current IDs.

## Run

On a GPU machine with Python 3.11 or newer:

```bash
bash scripts/run_stego.sh --count 200
STEGO_SPLIT=val bash scripts/run_stego.sh --count 20
STEGO_SPLIT=test bash scripts/run_stego.sh --count 20
```

This collects **200 train groups plus 20 val and 20 test groups per scheme**: 1,440 groups and 5,760 answers. The default run directories are `/workspace/runs/qwen3-14b-train`, `qwen3-14b-val`, and `qwen3-14b-test`. Use `--count 10` for a smaller run in any split, then increase the count in the same directory to resume. Set `STEGO_OUT` to choose another directory and `STEGO_CANDIDATES` before preparing it to change the 30,000 candidate default. Change [configs/stego.json](configs/stego.json) before preparing a new run to use another compatible model.

To package and upload all three splits as a private Hugging Face dataset, sign in with `hf auth login`, set `HF_NAME`, and run:

```bash
bash scripts/upload_datagen.sh /workspace/runs/qwen3-14b-train /workspace/runs/qwen3-14b-val /workspace/runs/qwen3-14b-test
```

The default repository is `$HF_NAME/qwen3-14b`; set `HF_REPO` to choose another. The dataset has one row per answer, with exact generation token IDs for later activation extraction and no activations.
