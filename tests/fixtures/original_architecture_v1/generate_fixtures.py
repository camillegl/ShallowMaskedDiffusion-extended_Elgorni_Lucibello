"""Generate deterministic fixtures pinning the LEGACY implementation's behavior.

Run from the repository root (legacy flat modules must be importable):

    uv run python tests/fixtures/original_architecture_v1/generate_fixtures.py

This script only READS legacy code and writes small fixture files into this
directory. It is kept for provenance; the fixtures it produced are committed
and must not be regenerated casually (a regeneration invalidates the pinned
behavior — record any regeneration in docs/MIGRATION_REPORT.md).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

FIXTURE_DIR = Path(__file__).parent
REPO_ROOT = FIXTURE_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from diffusion import MaskedDiffusion  # noqa: E402  (legacy module)


def make_config(L: int, l2reg: float = 0.1, alpha: float = 0.5, bias: bool = False):
    return argparse.Namespace(
        model="linear",
        L=L,
        l2reg=l2reg,
        alpha=alpha,
        lr=1e-3,
        bias=bias,
        freeze_mask_weights=False,
        pbar=False,
        epochs=10,
    )


def manifest(entry_name: str, **kwargs) -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {
        "fixture": entry_name,
        "source_commit": sha,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        **kwargs,
    }


def fx_linear_model() -> dict:
    """Legacy LinearBackbone forward: out = xt@W.T + mask@V.T (+ b); mask = (xt==0).
    NO runtime 1/sqrt(L) (discrepancy D3 - legacy quirk pinned intentionally)."""
    torch.manual_seed(101)
    L = 8
    model = MaskedDiffusion(make_config(L, bias=True))
    bb = model.backbone
    with torch.no_grad():
        bb.W.copy_(torch.randn(L, L))
        bb.V.copy_(torch.randn(L, L) * 0.3)
        bb.b.copy_(torch.randn(L) * 0.1)
    x0 = torch.where(torch.rand(3, L) < 0.5, 1.0, -1.0)
    mask = torch.rand(3, L) < 0.4
    xt = x0.clone()
    xt[mask] = 0.0
    with torch.no_grad():
        logits = bb(xt)
    torch.save(
        {
            "W": bb.W.detach(),
            "V": bb.V.detach(),
            "b": bb.b.detach(),
            "x0": x0,
            "mask": mask,
            "xt": xt,
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
        },
        FIXTURE_DIR / "linear_model.pt",
    )
    return manifest(
        "linear_model",
        source_module="models.py",
        source_symbol="LinearBackbone.forward",
        dimensions={"L": L, "batch": 3},
        seed=101,
        tolerance=1e-6,
        discrepancies=["D3 (no runtime 1/sqrt(L))", "D4 (in-band mask token 0)"],
        preserves="legacy quirk: normalization only in init, in-band mask encoding",
        normalization="none at runtime; W init scale 1/sqrt(L) overwritten by fixed W here",
    )


def fx_masking() -> dict:
    """Legacy corruption: t=rand(B) shared per row; mask = rand_like(x0) < t; xt[mask]=0."""
    torch.manual_seed(202)
    L, B = 8, 4
    x0 = torch.where(torch.rand(B, L) < 0.5, 1.0, -1.0)
    t = torch.rand(B)
    uniforms = torch.rand(B, L)
    mask = uniforms < t.unsqueeze(1)
    xt = x0.clone()
    xt[mask] = 0.0
    torch.save(
        {"x0": x0, "t": t, "uniforms": uniforms, "mask": mask, "xt": xt},
        FIXTURE_DIR / "masking.pt",
    )
    return manifest(
        "masking",
        source_module="diffusion.py",
        source_symbol="MaskedDiffusion._compute_loss (masking block, lines 71-81)",
        dimensions={"L": L, "batch": B},
        seed=202,
        tolerance=0.0,
        discrepancies=["D4"],
        preserves="intended behavior (Bernoulli(t) masks)",
        randomness="explicit uniforms stored; mask = uniforms < t",
    )


def fx_objective() -> dict:
    """Legacy loss: sum BCE*(1/t) over masked / (L*B); l2 = 0.5*l2reg/(L*alpha)*sqnorm
    over ALL params (including any frozen - D6 legacy quirk)."""
    torch.manual_seed(303)
    L, B = 8, 4
    l2reg, alpha = 0.1, 0.5
    model = MaskedDiffusion(make_config(L, l2reg=l2reg, alpha=alpha, bias=False))
    with torch.no_grad():
        model.backbone.W.copy_(torch.randn(L, L) * 0.5)
        model.backbone.V.copy_(torch.randn(L, L) * 0.2)
    x0 = torch.where(torch.rand(B, L) < 0.5, 1.0, -1.0)
    # replay the exact RNG the legacy loss will consume
    rng_state = torch.get_rng_state()
    t = torch.rand(B)
    uniforms = torch.rand(B, L)
    mask = uniforms < t.unsqueeze(1)
    torch.set_rng_state(rng_state)
    loss, acc = model._compute_loss(x0)
    l2loss = model.l2coeff * model.sqnorm()
    torch.save(
        {
            "W": model.backbone.W.detach(),
            "V": model.backbone.V.detach(),
            "x0": x0,
            "t": t,
            "uniforms": uniforms,
            "mask": mask,
            "data_loss": loss.detach(),
            "accuracy": torch.tensor(acc),
            "l2coeff": torch.tensor(model.l2coeff),
            "sqnorm": model.sqnorm().detach(),
            "l2loss": l2loss.detach(),
            "total_loss": (loss + l2loss).detach(),
        },
        FIXTURE_DIR / "objective.pt",
    )
    return manifest(
        "objective",
        source_module="diffusion.py",
        source_symbol="MaskedDiffusion._compute_loss + sqnorm + l2coeff",
        dimensions={"L": L, "batch": B},
        seed=303,
        tolerance=1e-6,
        parameters={"l2reg": l2reg, "alpha_legacy_M_over_N": alpha},
        discrepancies=["D1 (alpha=M/N)", "D5 (1/(L*B) normalization)", "D6 (all-params sqnorm)"],
        preserves="legacy behavior incl. D6 quirk (V penalized; frozen params would be too)",
        weighting="1/t per masked position; t shared across positions per row",
    )


def fx_sampler_stochastic() -> dict:
    """Legacy sample() k=1 fair path, full trajectory via stepwise _sample_k_update."""
    torch.manual_seed(404)
    L, B = 6, 2
    model = MaskedDiffusion(make_config(L))
    with torch.no_grad():
        model.backbone.W.copy_(torch.randn(L, L))
        model.backbone.V.copy_(torch.randn(L, L) * 0.5)
    xt = torch.zeros(B, L)
    states = [xt.clone()]
    with torch.no_grad():
        while bool((xt == 0).any()):
            xt = model._sample_k_update(xt, 1)
            states.append(xt.clone())
    # cross-check: same seed through the public sample() gives the same final state
    torch.manual_seed(404)
    model2 = MaskedDiffusion(make_config(L))
    with torch.no_grad():
        model2.backbone.W.copy_(torch.randn(L, L))
        model2.backbone.V.copy_(torch.randn(L, L) * 0.5)
        final2 = model2.sample(nsamples=B, k=1)
    assert torch.equal(states[-1], final2), "stepwise replay != public sample()"
    torch.save(
        {
            "W": model.backbone.W.detach(),
            "V": model.backbone.V.detach(),
            "states": torch.stack(states),
            "final": states[-1],
        },
        FIXTURE_DIR / "sampler_stochastic.pt",
    )
    return manifest(
        "sampler_stochastic",
        source_module="diffusion.py",
        source_symbol="MaskedDiffusion.sample/_sample_k_update (k=1 fair path)",
        dimensions={"L": L, "batch": B},
        seed=404,
        tolerance=0.0,
        discrepancies=[],
        preserves="intended behavior: Algorithm-1 sequential stochastic (fair) sampling; "
        "no revision of committed tokens",
        randomness="torch.manual_seed(404); multinomial for coordinates, bernoulli for tokens",
        note="intermediate states recorded after every single-token update",
    )


def fx_greedy_reconstruction() -> dict:
    """Legacy mask_and_sample greedy: pre-scheduled random order, threshold decoding."""
    torch.manual_seed(505)
    L, B, T0 = 6, 2, 4
    model = MaskedDiffusion(make_config(L))
    with torch.no_grad():
        model.backbone.W.copy_(torch.randn(L, L))
        model.backbone.V.copy_(torch.randn(L, L) * 0.5)
    x0 = torch.where(torch.rand(B, L) < 0.5, 1.0, -1.0)
    rng_state = torch.get_rng_state()
    # replay the order draw mask_and_sample will make internally
    indx_seq = torch.cat([torch.randperm(L) for _ in range(B)]).view(B, L)
    torch.set_rng_state(rng_state)
    with torch.no_grad():
        xt_final, history = model.mask_and_sample(x0, T0=T0, decoding_strategy="greedy")
    # replay the loop to capture intermediates and per-step logits
    torch.set_rng_state(rng_state)
    _ = torch.cat([torch.randperm(L) for _ in range(B)]).view(B, L)  # consume order draw
    xt = x0.clone()
    for i in range(B):
        xt[i, indx_seq[i, :T0]] = 0.0
    corrupted = xt.clone()
    intermediates, step_logits = [xt.clone()], []
    with torch.no_grad():
        for T in range(T0, 0, -1):
            logits = model(xt)
            to_unmask = indx_seq[:, T - 1]
            lg = logits[torch.arange(B), to_unmask]
            step_logits.append(lg.clone())
            xt[torch.arange(B), to_unmask] = (torch.sigmoid(lg) >= 0.5).float() * 2 - 1
            intermediates.append(xt.clone())
    assert torch.equal(xt, xt_final), "replayed greedy loop != mask_and_sample output"
    overlap = (xt_final * x0).mean()
    torch.save(
        {
            "W": model.backbone.W.detach(),
            "V": model.backbone.V.detach(),
            "x0": x0,
            "corrupted": corrupted,
            "decoding_order": indx_seq,
            "T0": torch.tensor(T0),
            "step_logits": torch.stack(step_logits),
            "intermediates": torch.stack(intermediates),
            "final": xt_final,
            "history": torch.tensor(history),
            "final_overlap": overlap,
        },
        FIXTURE_DIR / "greedy_reconstruction.pt",
    )
    return manifest(
        "greedy_reconstruction",
        source_module="diffusion.py",
        source_symbol="MaskedDiffusion.mask_and_sample (decoding_strategy='greedy')",
        dimensions={"L": L, "batch": B, "T0": T0},
        seed=505,
        tolerance=1e-6,
        discrepancies=[],
        preserves="intended behavior: pre-scheduled random order, threshold-at-0.5 "
        "decoding, no revision",
        randomness="torch.manual_seed(505); randperm order draw replayed explicitly",
    )


def main() -> None:
    entries = [
        fx_linear_model(),
        fx_masking(),
        fx_objective(),
        fx_sampler_stochastic(),
        fx_greedy_reconstruction(),
    ]
    (FIXTURE_DIR / "manifest.json").write_text(
        json.dumps({"format": "original_architecture_fixtures.v1", "fixtures": entries}, indent=2)
        + "\n"
    )
    print(f"wrote {len(entries)} fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
