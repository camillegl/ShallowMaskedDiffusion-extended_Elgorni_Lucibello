"""maskeddiffusion-validate-artifact: validate a run directory against the
artifact schema (ADR 003)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="run directory to validate")
    args = parser.parse_args(argv)

    from ..artifacts import validate_artifact

    problems = validate_artifact(args.path)
    if problems:
        for p in problems:
            print(f"INVALID: {p}")
        return 1
    print(f"valid artifact: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
