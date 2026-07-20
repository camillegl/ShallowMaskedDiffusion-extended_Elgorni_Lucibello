"""maskeddiffusion-evaluate: MMD, correlation, and overlap diagnostics for
generated samples against the finite-F teacher law.

Outputs are labeled by comparison (model_vs_true, true_vs_true,
train_vs_true) and by the sampler-indexed terminal law that produced the
samples. An MMD decrease is evidence only under this diagnostic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ..artifacts import RunArtifact
from ..checkpoints import load_checkpoint
from ..config import load_config
from ..metrics.correlations import correlation_error, empirical_pair_correlation
from ..metrics.mmd import model_vs_true, sqrt_clipped_mmd, train_vs_true, true_vs_true
from ..metrics.overlaps import nearest_training_excess
from ..teacher import HiddenManifoldTeacher
from ..training import resolve_device
from . import base_parser


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "evaluate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", required=True, help="teacher.pt from the train run")
    parser.add_argument("--samples", required=True, help="samples.pt from a sample run")
    parser.add_argument("--n-true", type=int, default=1000, help="fresh P_F evaluation samples")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[4.0, 8.0])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = Path(args.output)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "comparisons": ["model_vs_true", "true_vs_true", "train_vs_true"],
                    "lambdas": args.lambdas,
                    "n_true": args.n_true,
                    "planned_paths": {"summary": str(out / "summary.json")},
                },
                indent=2,
            )
        )
        return 0

    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint)
    teacher = HiddenManifoldTeacher.load(args.teacher)
    if teacher.teacher_id != payload["teacher_id"]:
        raise ValueError("teacher file does not match checkpoint teacher_id")

    model_samples = torch.load(args.samples, map_location="cpu", weights_only=False)
    seeds = config.seeds
    true_a = teacher.sample_batch(args.n_true, seeds.generator("evaluation_data_seed"))
    true_b = teacher.sample_batch(args.n_true, seeds.generator("metric_seed"))
    train_set = teacher.sample_batch(
        config.dimensions.train_size, seeds.generator("train_data_seed")
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
    results = {
        "model_vs_true": summarize(model_vs_true(model_samples, true_a, lams)),
        "true_vs_true": summarize(true_vs_true(true_a, true_b, lams)),
        "train_vs_true": summarize(train_vs_true(train_set, true_b, lams)),
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
                for cmp in ("model_vs_true", "true_vs_true", "train_vs_true")
            },
        }
    )
    artifact.write_summary(results)
    artifact.write_manifest(
        command=" ".join(["maskeddiffusion-evaluate", *sys.argv[1:]]),
        device=device,
        teacher_id=teacher.teacher_id,
        seeds=seeds.to_dict(),
        sampler=config.sampler.identity(),
        objective={"name": "n/a (evaluation run)", "mmd_lambdas": lams},
        model=payload["model_config"],
        input_paths=[args.checkpoint, args.teacher, args.samples],
    )
    print(json.dumps({k: results[k] for k in ("model_vs_true", "true_vs_true")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
