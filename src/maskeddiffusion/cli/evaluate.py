"""maskeddiffusion-evaluate: MMD, correlation, and overlap diagnostics for
generated samples against the finite-F teacher law.

Outputs are labeled by comparison (model_vs_true, true_vs_true,
train_vs_true, model_vs_train) and by the sampler-indexed terminal law that
produced the samples. An MMD decrease is evidence only under this
diagnostic.

`--samples` takes a **sample-run artifact directory** (as written by
`maskeddiffusion-sample`), not a bare tensor path — this lets evaluate
verify the samples' own manifest (teacher_id, checkpoint_id, sampler
identity, resolved dimensions) rather than trusting an arbitrary tensor to
actually correspond to `--checkpoint`/`--teacher`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from ..artifacts import RunArtifact, sha256_file, validate_artifact
from ..checkpoints import load_checkpoint
from ..config import load_config
from ..dimensions import Dimensions
from ..metrics.correlations import correlation_error, empirical_pair_correlation
from ..metrics.mmd import (
    model_vs_train,
    model_vs_true,
    sqrt_clipped_mmd,
    train_vs_true,
    true_vs_true,
)
from ..metrics.overlaps import nearest_training_excess
from ..randomness import SeedHierarchy
from ..teacher import HiddenManifoldTeacher
from ..training import resolve_device
from . import base_parser


def _load_sample_artifact(samples_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    manifest_path = samples_dir / "manifest.json"
    tensor_path = samples_dir / "samples" / "samples.pt"
    if not manifest_path.exists():
        raise ValueError(
            f"--samples {samples_dir} has no manifest.json — pass the sample-run "
            "artifact directory written by maskeddiffusion-sample, not a bare tensor path"
        )
    if not tensor_path.exists():
        raise ValueError(f"--samples {samples_dir} has no samples/samples.pt")
    manifest = json.loads(manifest_path.read_text())
    tensor = torch.load(tensor_path, map_location="cpu", weights_only=False)
    return tensor, manifest


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "evaluate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", required=True, help="teacher.pt from the train run")
    parser.add_argument(
        "--samples", required=True, help="sample-run artifact directory (not a bare .pt path)"
    )
    parser.add_argument("--n-true", type=int, default=1000, help="fresh P_F evaluation samples")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[4.0, 8.0])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = Path(args.output)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "comparisons": [
                        "model_vs_true",
                        "true_vs_true",
                        "train_vs_true",
                        "model_vs_train",
                    ],
                    "lambdas": args.lambdas,
                    "n_true": args.n_true,
                    "planned_paths": {"summary": str(out / "summary.json")},
                },
                indent=2,
            )
        )
        return 0

    requested_device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint)
    teacher = HiddenManifoldTeacher.load(args.teacher)
    if teacher.teacher_id != payload["teacher_id"]:
        raise ValueError(
            f"--teacher {args.teacher} (teacher_id={teacher.teacher_id}) does not match "
            f"--checkpoint {args.checkpoint} (teacher_id={payload['teacher_id']})"
        )

    samples_dir = Path(args.samples)
    artifact_problems = validate_artifact(samples_dir)
    if artifact_problems:
        raise ValueError(
            f"--samples {samples_dir} failed artifact validation: {'; '.join(artifact_problems)}"
        )
    model_samples, sample_manifest = _load_sample_artifact(samples_dir)

    sample_teacher_id = sample_manifest.get("teacher_id")
    if sample_teacher_id != teacher.teacher_id:
        raise ValueError(
            f"--samples {samples_dir} was generated under teacher_id "
            f"{sample_teacher_id!r}, but --teacher/--checkpoint resolve to "
            f"{teacher.teacher_id!r}"
        )

    checkpoint_id = payload.get("checkpoint_id")
    sample_checkpoint_id = sample_manifest.get("checkpoint_id")
    if checkpoint_id is None or sample_checkpoint_id is None:
        raise ValueError(
            "checkpoint_id missing from --checkpoint or from the --samples manifest — "
            "cannot verify the samples were generated from this exact checkpoint content "
            "(regenerate both with the current package version)"
        )
    if checkpoint_id != sample_checkpoint_id:
        raise ValueError(
            f"--samples {samples_dir} was generated from checkpoint_id "
            f"{sample_checkpoint_id!r}, but --checkpoint {args.checkpoint} has "
            f"checkpoint_id {checkpoint_id!r} — the checkpoint file has changed since "
            "the samples were produced"
        )

    # checkpoint_id is a semantic hash over model_state + model_config +
    # teacher/step/examples_seen (checkpoints.py); it does not cover every
    # byte of the checkpoint file (e.g. optimizer_state, generator_states).
    # Comparing the whole file's sha256, recorded by maskeddiffusion-sample
    # at the moment it read --checkpoint, catches any change to the file
    # between sampling and this evaluate invocation, not just the subset of
    # fields checkpoint_id happens to hash.
    sample_checkpoint_sha256 = sample_manifest.get("checkpoint_file_sha256")
    if sample_checkpoint_sha256 is None:
        raise ValueError(
            f"--samples {samples_dir} manifest has no checkpoint_file_sha256 — "
            "regenerate the samples with the current package version"
        )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != sample_checkpoint_sha256:
        raise ValueError(
            f"--checkpoint {args.checkpoint} has sha256 {checkpoint_sha256!r}, but "
            f"--samples {samples_dir} was generated from a checkpoint file with sha256 "
            f"{sample_checkpoint_sha256!r} — the checkpoint file has changed since the "
            "samples were produced, even though its checkpoint_id still matches"
        )

    checkpoint_visible_dim = payload["model_config"].get("visible_dim")
    if checkpoint_visible_dim != config.dimensions.visible_dim:
        raise ValueError(
            f"--config resolves visible_dim={config.dimensions.visible_dim}, but "
            f"--checkpoint was trained at visible_dim={checkpoint_visible_dim} — "
            "pass the same config used for training/sampling"
        )
    if model_samples.shape[-1] != config.dimensions.visible_dim:
        raise ValueError(
            f"--samples tensor has last dim {model_samples.shape[-1]}, expected "
            f"visible_dim={config.dimensions.visible_dim}"
        )

    # Sampler identity is a property of how the samples were actually
    # generated (recorded in the sample artifact's own manifest), not of
    # whatever --config happens to be passed to this evaluate invocation —
    # using config.sampler.identity() here would silently mislabel samples
    # if a different config were passed to sample vs evaluate.
    sampler_identity = sample_manifest.get("sampler") or {}
    if not sampler_identity:
        raise ValueError(f"--samples {samples_dir} manifest has no sampler identity recorded")

    # The training set actually used to fit --checkpoint is determined by the
    # checkpoint's own stored config (train_size, train_data_seed), not by
    # whatever --config is passed to this evaluate invocation. A --config
    # with the same visible_dim but a different train_size or seed would
    # otherwise silently reconstruct the wrong training set and corrupt
    # Train-True, Model-Train, and nearest-training diagnostics.
    checkpoint_config = payload.get("config") or {}
    if "dimensions" not in checkpoint_config or "seeds" not in checkpoint_config:
        raise ValueError(
            f"--checkpoint {args.checkpoint} has no config recorded — cannot "
            "reconstruct the exact training set it was trained on (regenerate "
            "the checkpoint with the current package version)"
        )
    checkpoint_dims = Dimensions.from_dict(checkpoint_config["dimensions"])
    checkpoint_seeds = SeedHierarchy.from_dict(checkpoint_config["seeds"])

    seeds = config.seeds
    true_a = teacher.sample_batch(args.n_true, seeds.generator("evaluation_data_seed"))
    true_b = teacher.sample_batch(args.n_true, seeds.generator("metric_seed"))
    train_set = teacher.sample_batch(
        checkpoint_dims.train_size, checkpoint_seeds.generator("train_data_seed")
    )

    def summarize(res) -> dict:
        return {
            "biased_mmd2": {str(k): v for k, v in res.biased_mmd2.items()},
            "unbiased_mmd2_raw": {str(k): v for k, v in res.unbiased_mmd2_raw.items()},
            "mixture_biased_mmd2": res.mixture_biased_mmd2,
            "mixture_unbiased_mmd2_raw": res.mixture_unbiased_mmd2_raw,
            "display_sqrt_clipped_mixture": sqrt_clipped_mmd(res.mixture_biased_mmd2),
        }

    lams = args.lambdas
    comparisons = ("model_vs_true", "true_vs_true", "train_vs_true", "model_vs_train")
    results = {
        "model_vs_true": summarize(model_vs_true(model_samples, true_a, lams)),
        "true_vs_true": summarize(true_vs_true(true_a, true_b, lams)),
        "train_vs_true": summarize(train_vs_true(train_set, true_b, lams)),
        "model_vs_train": summarize(model_vs_train(model_samples, train_set, lams)),
        "nearest_training": nearest_training_excess(model_samples, true_a, train_set),
        "pair_correlation_error": correlation_error(
            empirical_pair_correlation(model_samples), teacher.correlation_matrix()
        ),
    }

    artifact = RunArtifact(out)
    config.to_json(out / "resolved_config.json")
    artifact.log_metrics(
        {
            "event": "evaluated",
            **{
                f"{cmp}_mixture_biased_mmd2": results[cmp]["mixture_biased_mmd2"]
                for cmp in comparisons
            },
        }
    )
    artifact.write_summary(results)
    artifact.write_manifest(
        command=" ".join(["maskeddiffusion-evaluate", *sys.argv[1:]]),
        device=requested_device,
        teacher_id=teacher.teacher_id,
        seeds=seeds.to_dict(),
        sampler=sampler_identity,
        objective={"name": "n/a (evaluation run)", "mmd_lambdas": lams},
        model=payload["model_config"],
        input_paths=[args.checkpoint, args.teacher, str(samples_dir)],
        extra={
            "checkpoint_id": checkpoint_id,
            "checkpoint_file_sha256": checkpoint_sha256,
            # This CLI never moves the teacher/sample/model tensors onto a
            # non-cpu device — every op here is plain-tensor MMD/correlation
            # math on whatever device the inputs were saved on (always cpu
            # for teacher.sample_batch outputs and saved sample tensors).
            # Recording both makes clear that --device is not (yet) honored
            # for computation in this CLI, rather than letting the manifest
            # imply GPU computation that did not happen.
            "requested_device": requested_device,
            "actual_device": "cpu",
        },
    )
    print(json.dumps({k: results[k] for k in comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
