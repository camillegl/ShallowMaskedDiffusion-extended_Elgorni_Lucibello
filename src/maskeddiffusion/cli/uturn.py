"""maskeddiffusion-uturn: U-turn / reconstruction experiment.

Mask each coordinate of a clean example — a training row (source "train")
or a fresh draw from the same finite-F teacher (source "fresh") —
independently with probability t, then reconstruct the masked coordinates
with the configured sampler starting from that partially observed state
(observed coordinates held fixed). Reports the retrieval overlap
q_U(t) = mean_i[x_hat_i * x_clean_i] against the no-recovery baseline 1 - t,
the excess recovery, reconstruction Hamming error, and nearest-training
diagnostics, per example and aggregated per (source, t).

The sampler identity is taken from --config (as in maskeddiffusion-sample);
train-source and fresh-source cells at the same (example_index, t) share
paired mask/order/token seeds (src/maskeddiffusion/uturn.py). No
train-source result is labelled memorization without the paired
fresh-source comparison (docs/RESEARCH_SPEC.md claim discipline).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ..artifacts import RunArtifact, sha256_file
from ..checkpoints import load_checkpoint
from ..config import load_config
from ..dimensions import Dimensions
from ..models import LinearMaskedScore, LinearScoreConfig
from ..randomness import SeedHierarchy
from ..teacher import HiddenManifoldTeacher
from ..training import resolve_device
from ..uturn import UTURN_SOURCES, UTurnConfig, run_uturn, summarize_uturn
from . import base_parser


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "uturn")
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt from a train run")
    parser.add_argument("--teacher", required=True, help="teacher.pt from the train run")
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        required=True,
        help="mask probabilities t (each in [0, 1])",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=8,
        help="examples per source (train source uses the first rows of the "
        "reconstructed training set; must not exceed train_size)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(UTURN_SOURCES),
        default=list(UTURN_SOURCES),
        help="clean-example sources to run (default: both)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = Path(args.output)
    uturn_config = UTurnConfig(
        t_values=tuple(args.t_values),
        n_examples=args.n_examples,
        sources=tuple(args.sources),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment": "uturn_reconstruction",
                    "checkpoint": args.checkpoint,
                    "teacher": args.teacher,
                    "sampler": config.sampler.identity(),
                    "t_values": list(uturn_config.t_values),
                    "n_examples": uturn_config.n_examples,
                    "sources": list(uturn_config.sources),
                    "planned_paths": {
                        "summary": str(out / "summary.json"),
                        "per_example": str(out / "results" / "per_example.jsonl"),
                        "tensors": str(out / "results" / "uturn_tensors.pt"),
                    },
                },
                indent=2,
            )
        )
        return 0

    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint)
    teacher = HiddenManifoldTeacher.load(args.teacher)
    if teacher.teacher_id != payload["teacher_id"]:
        raise ValueError(
            f"--teacher {args.teacher} (teacher_id={teacher.teacher_id}) does not match "
            f"--checkpoint {args.checkpoint} (teacher_id={payload['teacher_id']})"
        )
    checkpoint_visible_dim = payload["model_config"].get("visible_dim")
    if checkpoint_visible_dim != config.dimensions.visible_dim:
        raise ValueError(
            f"--config resolves visible_dim={config.dimensions.visible_dim}, but "
            f"--checkpoint was trained at visible_dim={checkpoint_visible_dim} — "
            "pass the same dimensions used for training"
        )

    # The training set actually used to fit --checkpoint is reconstructed
    # from the checkpoint's own stored config (train_size, train_data_seed),
    # not from --config — same provenance discipline as
    # maskeddiffusion-evaluate (a --config with a different train_size or
    # seed would silently reconstruct the wrong train-source examples and
    # corrupt nearest-training diagnostics).
    checkpoint_config = payload.get("config") or {}
    if "dimensions" not in checkpoint_config or "seeds" not in checkpoint_config:
        raise ValueError(
            f"--checkpoint {args.checkpoint} has no config recorded — cannot "
            "reconstruct the exact training set it was trained on (regenerate "
            "the checkpoint with the current package version)"
        )
    checkpoint_dims = Dimensions.from_dict(checkpoint_config["dimensions"])
    checkpoint_seeds = SeedHierarchy.from_dict(checkpoint_config["seeds"])
    train_set = teacher.sample_batch(
        checkpoint_dims.train_size, checkpoint_seeds.generator("train_data_seed")
    )

    model_cfg_stored = dict(payload["model_config"])
    model_cfg_stored.pop("model", None)
    model = LinearMaskedScore(
        LinearScoreConfig(**model_cfg_stored), torch.Generator().manual_seed(0)
    )
    model.load_state_dict(payload["model_state"])
    model = model.to(device).eval()

    result = run_uturn(
        model,
        config.sampler,
        teacher,
        train_set,
        uturn_config,
        config.seeds,
    )
    summary = summarize_uturn(result)

    artifact = RunArtifact(out)
    config.to_json(out / "resolved_config.json")

    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = results_dir / "per_example.jsonl"
    with open(per_example_path, "w") as f:
        for cell in result.cells:
            f.write(json.dumps(cell.to_dict()) + "\n")
    artifact.register_file(
        per_example_path,
        "per-example U-turn cell results, one JSON object per (source, example_index, t) cell",
    )
    tensors_path = results_dir / "uturn_tensors.pt"
    torch.save(
        {
            "masks": result.masks,
            "clean": result.clean,
            "reconstructions": result.reconstructions,
        },
        tensors_path,
    )
    artifact.register_file(
        tensors_path,
        "clean examples, paired masks (t, example, coordinate), and reconstructions per source",
    )

    for point in summary["points"]:
        artifact.log_metrics({"event": "uturn_point", **point})
    artifact.log_metrics({"event": "uturn_completed", "n_cells": len(result.cells)})
    artifact.write_summary(summary)
    artifact.write_manifest(
        command=" ".join(["maskeddiffusion-uturn", *sys.argv[1:]]),
        device=device,
        teacher_id=teacher.teacher_id,
        seeds=config.seeds.to_dict(),
        sampler=config.sampler.identity(),
        objective={
            "name": "uturn_reconstruction",
            "t_values": list(uturn_config.t_values),
            "n_examples": uturn_config.n_examples,
            "sources": list(uturn_config.sources),
        },
        model=payload["model_config"],
        input_paths=[args.checkpoint, args.teacher],
        extra={
            "checkpoint_id": payload.get("checkpoint_id"),
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_file_sha256": sha256_file(args.checkpoint),
            # Train-source examples and nearest-training references are
            # reconstructed from the checkpoint's own recorded config, not
            # from this invocation's --config seeds (see above).
            "checkpoint_train_size": checkpoint_dims.train_size,
            "checkpoint_train_data_seed": checkpoint_seeds.train_data_seed,
            "cell_seed_derivation": (
                "uturn_cell_seed(purpose, stream_seed, example_index[, t_value]) — "
                "SHA-256, domain 'maskeddiffusion.uturn.v1'; mask/order/token "
                "cell seeds are source-independent (paired across train/fresh); "
                "fresh clean draws derive per-example from evaluation_data_seed"
            ),
        },
    )
    print(
        f"uturn complete: {len(result.cells)} cells "
        f"({len(uturn_config.sources)} sources x {uturn_config.n_examples} examples x "
        f"{len(uturn_config.t_values)} t values), artifact at {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
