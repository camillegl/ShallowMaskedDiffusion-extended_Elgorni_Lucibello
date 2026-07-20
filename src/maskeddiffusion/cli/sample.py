"""maskeddiffusion-sample: generate samples from a trained checkpoint.

Samples follow the sampler-indexed terminal law P_{θ,A,k} of the configured
sampler; the sampler identity is recorded in the output manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ..artifacts import RunArtifact
from ..checkpoints import load_checkpoint
from ..config import load_config
from ..models import LinearMaskedScore, LinearScoreConfig
from ..samplers import sample as run_sampler
from ..training import resolve_device
from . import base_parser


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "sample")
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt from a train run")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = Path(args.output)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "sampler": config.sampler.identity(),
                    "n_samples": args.n_samples,
                    "planned_paths": {"samples": str(out / "samples" / "samples.pt")},
                },
                indent=2,
            )
        )
        return 0

    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint)
    model_cfg_stored = dict(payload["model_config"])
    model_cfg_stored.pop("model", None)
    model = LinearMaskedScore(
        LinearScoreConfig(**model_cfg_stored), torch.Generator().manual_seed(0)
    )
    model.load_state_dict(payload["model_state"])
    model = model.to(device).eval()

    with torch.no_grad():
        result = run_sampler(
            model,
            config.sampler,
            args.n_samples,
            order_generator=config.seeds.generator("sampler_order_seed"),
            token_generator=config.seeds.generator("sampler_token_seed"),
        )

    artifact = RunArtifact(out)
    config.to_json(out / "resolved_config.json")
    artifact.save_tensor(
        "samples/samples.pt",
        result.values.cpu(),
        f"terminal samples of {config.sampler.sampler_name}",
    )
    artifact.log_metrics({"event": "sampled", "n_samples": args.n_samples})
    artifact.write_summary({"n_samples": args.n_samples, "sampler": config.sampler.identity()})
    artifact.write_manifest(
        command=" ".join(["maskeddiffusion-sample", *sys.argv[1:]]),
        device=device,
        teacher_id=payload["teacher_id"],
        seeds=config.seeds.to_dict(),
        sampler=config.sampler.identity(),
        objective={"name": "n/a (sampling run)"},
        model=payload["model_config"],
        input_paths=[str(args.checkpoint)],
    )
    print(f"wrote {args.n_samples} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
