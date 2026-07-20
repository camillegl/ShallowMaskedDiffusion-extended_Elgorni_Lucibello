"""maskeddiffusion-train: train a linear masked score on a hidden-manifold teacher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..artifacts import RunArtifact
from ..config import load_config
from ..training import resolve_device, train
from . import base_parser


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "train")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = Path(args.output)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "resolved_dimensions": config.dimensions.to_dict(),
                    "seeds": config.seeds.to_dict(),
                    "model": config.model.identity(),
                    "sampler": config.sampler.identity(),
                    "training": config.training.__dict__,
                    "planned_paths": {
                        "manifest": str(out / "manifest.json"),
                        "resolved_config": str(out / "resolved_config.json"),
                        "metrics": str(out / "metrics.jsonl"),
                        "summary": str(out / "summary.json"),
                        "checkpoints": str(out / "checkpoints"),
                    },
                },
                indent=2,
            )
        )
        return 0

    device = resolve_device(args.device)
    artifact = RunArtifact(out)
    config.to_json(out / "resolved_config.json")

    state, teacher, summary = train(
        config,
        device=device,
        on_log=artifact.log_metrics,
        checkpoint_dir=out / "checkpoints",
    )
    final_ckpt = out / "checkpoints" / "final.pt"
    if final_ckpt.exists():
        artifact.register_file(final_ckpt, "final checkpoint")

    teacher_path = out / "teacher.pt"
    teacher.save(teacher_path)
    artifact.register_file(teacher_path, "quenched teacher state")

    artifact.write_summary(summary)
    artifact.write_manifest(
        command=" ".join(["maskeddiffusion-train", *sys.argv[1:]]),
        device=device,
        teacher_id=teacher.teacher_id,
        seeds=config.seeds.to_dict(),
        sampler=config.sampler.identity(),
        objective={
            "name": "continuous_time_masked_bce",
            "l2reg": config.training.l2reg,
            "min_time": config.training.min_time,
        },
        model=config.model.identity(),
    )
    print(f"run complete: step {summary['final_step']}, artifact at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
