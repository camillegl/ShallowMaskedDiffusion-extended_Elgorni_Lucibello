"""Command-line entry points (ADR 002): --config TOML, --output dir,
--device, --dry-run."""

from __future__ import annotations

import argparse


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="TOML run configuration")
    p.add_argument("--output", required=True, help="run/output directory")
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="compute device (mps is best-effort: determinism/resume guaranteed on cpu only)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config, validate dimensions, print derived fields and planned "
        "paths; perform no training and no writes",
    )
    return p
