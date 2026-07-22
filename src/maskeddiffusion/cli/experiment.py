"""maskeddiffusion-experiment: Phase 4C paired-experiment engine.

Expands one experiment TOML (one intervention over a shared base
configuration) into paired comparison groups, writes the deterministic
pre-run manifest, and executes every condition resumably — skipping
completed valid artifacts, refusing incomplete or provenance-inconsistent
ones, and never overwriting existing run output
(docs/PHASE4C_EXPERIMENT_PROTOCOL.md).

--dry-run prints the exact run count, dimensions, every seed stream,
model/optimizer/sampler identities, the projected generated-sample count,
and all output paths — and writes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..experiments import execute_plan, load_experiment_config
from ..training import resolve_device
from . import base_parser


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "experiment")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the pre-run experiment manifest and resolved specs without executing",
    )
    args = parser.parse_args(argv)
    plan = load_experiment_config(args.config)
    out = Path(args.output)

    if args.dry_run:
        print(json.dumps(plan.dry_run_dict(out), indent=2))
        return 0

    device = resolve_device(args.device)
    if args.plan_only:
        path = plan.write_plan_manifest(out, device=device, config_path=args.config)
        print(f"plan manifest written (no runs executed): {path}")
        return 0

    command = " ".join(["maskeddiffusion-experiment", *sys.argv[1:]])
    summary = execute_plan(plan, out, device=device, command=command, config_path=args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
