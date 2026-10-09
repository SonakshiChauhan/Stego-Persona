"""Command-line entry point for steganography data generation."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Steganography data generation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--candidates", type=int, default=30000)
    prepare.add_argument("--split", choices=("train", "val", "test"), default="train")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--split-seed", type=int, default=42)
    prepare.add_argument("--questions-file", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--per-scheme", type=int, default=200)
    package = commands.add_parser("package")
    package.add_argument("--out", type=Path, required=True)
    package.add_argument("--dest", type=Path, required=True)
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
        from data_generation.package import package
        package(**args)


if __name__ == "__main__":
    main()
