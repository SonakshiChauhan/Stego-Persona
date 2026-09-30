import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Steganography data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=10000)
    p.add_argument("--split", choices=["extract", "eval"], default="extract")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--questions-file", type=Path, help="Optional existing HF instruction snapshot JSONL")
    for command in ("run", "export"):
        p = sub.add_parser(command)
        p.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "validate":
        from data_generation.stego import validate_assets
        print(json.dumps(validate_assets(), indent=2))
    elif command == "prepare":
        from data_generation.stego import prepare
        prepare(**args)
    elif command == "run":
        from data_generation.run import run
        run(**args)
    else:
        from data_generation.export import export
        print(json.dumps(export(**args), indent=2))


if __name__ == "__main__":
    main()
