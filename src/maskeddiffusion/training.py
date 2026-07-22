"""Direct typed PyTorch training loop (ADR 001).

Deterministic initialization, explicit generators for data and masking,
checkpoint/restore with optimizer and RNG state, progress in optimizer steps
and examples seen (not only epochs).

Device support: cpu is the reference device — exact resume and determinism
are guaranteed and tested on cpu only. cuda is implemented (generators are
placed on the training device) but untested in this environment; mps is
accepted best-effort with known nondeterminism and incomplete generator
support. Verify determinism yourself before trusting non-cpu runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint, restore_into, save_checkpoint
from .config import RunConfig, TrainingConfig
from .models import LinearMaskedScore
from .objectives import continuous_time_masked_bce
from .randomness import SeedHierarchy
from .teacher import HiddenManifoldTeacher


@dataclass
class TrainState:
    model: LinearMaskedScore
    optimizer: torch.optim.Optimizer
    step: int
    examples_seen: int
    generators: dict[str, torch.Generator]


def optimizer_identity(training: TrainingConfig) -> dict[str, Any]:
    """Recorded identity of the optimizer `build_state` constructs.

    AdamW with `weight_decay=0.0` — regularization lives in the objective
    (ADR 001/legacy parity) — and torch-default betas/eps. Experiment specs
    and manifests record this dict; it must stay in sync with the AdamW
    construction in `build_state` below.
    """
    return {
        "name": "adamw",
        "learning_rate": training.learning_rate,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    }


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps requested but not available")
    return requested


def build_state(
    config: RunConfig, device: str = "cpu"
) -> tuple[TrainState, HiddenManifoldTeacher, torch.Tensor]:
    """Construct teacher, training data, model, and optimizer deterministically."""
    seeds: SeedHierarchy = config.seeds
    teacher = HiddenManifoldTeacher.sample(config.dimensions, seeds.generator("teacher_seed"))
    train_data = teacher.sample_batch(
        config.dimensions.train_size, seeds.generator("train_data_seed")
    ).to(device)

    model = LinearMaskedScore(config.model, seeds.generator("model_seed")).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=0.0,  # regularization lives in the objective (ADR 001/legacy parity)
    )
    # Generators must live on the same device as the tensors they fill
    # (torch requires generator.device == target device for rand/randint).
    generators = {
        "mask_seed": seeds.generator("mask_seed", device=device),
        "dataloader_seed": seeds.generator("dataloader_seed", device=device),
    }
    state = TrainState(
        model=model, optimizer=optimizer, step=0, examples_seen=0, generators=generators
    )
    return state, teacher, train_data


def _draw_batch(
    train_data: torch.Tensor, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    m = train_data.shape[0]
    idx = torch.randint(0, m, (min(batch_size, m),), generator=generator, device=train_data.device)
    return train_data[idx]


def train(
    config: RunConfig,
    *,
    device: str = "cpu",
    state: TrainState | None = None,
    teacher: HiddenManifoldTeacher | None = None,
    train_data: torch.Tensor | None = None,
    validation_data: torch.Tensor | None = None,
    on_log: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> tuple[TrainState, HiddenManifoldTeacher, dict[str, Any]]:
    """Run the training loop to config.training.max_steps. Returns final state,
    teacher, and a summary dict."""
    if state is None or teacher is None or train_data is None:
        state, teacher, train_data = build_state(config, device)
    if validation_data is None and config.training.validation_size > 0:
        validation_data = teacher.sample_batch(
            config.training.validation_size,
            config.seeds.generator("validation_data_seed"),
        ).to(device)

    if resume_from is not None:
        payload = load_checkpoint(resume_from)
        if payload["teacher_id"] != teacher.teacher_id:
            raise ValueError(
                f"checkpoint teacher_id {payload['teacher_id']} != current {teacher.teacher_id}"
            )
        state.step, state.examples_seen = restore_into(
            payload,
            model=state.model,
            optimizer=state.optimizer,
            generators=state.generators,
        )

    tcfg = config.training
    start_time = time.monotonic()
    last: dict[str, Any] = {}

    while state.step < tcfg.max_steps:
        batch = _draw_batch(train_data, tcfg.batch_size, state.generators["dataloader_seed"])
        result = continuous_time_masked_bce(
            state.model,
            batch,
            state.generators["mask_seed"],
            l2reg=tcfg.l2reg,
            train_size=config.dimensions.train_size,
            min_time=tcfg.min_time,
        )
        state.optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        state.optimizer.step()
        state.step += 1
        state.examples_seen += batch.shape[0]

        should_log = state.step % tcfg.log_every == 0 or state.step == tcfg.max_steps
        if should_log:
            last = {
                "step": state.step,
                "examples_seen": state.examples_seen,
                "train_loss": float(result.data_loss.item()),
                "regularization": float(result.regularization.item()),
                "total_loss": float(result.total.item()),
                "masked_token_count": result.masked_token_count,
                "wall_clock_s": time.monotonic() - start_time,
                **{f"order_{k}": v for k, v in state.model.order_parameters().items()},
            }
            if validation_data is not None and state.step % tcfg.validation_every == 0:
                with torch.no_grad():
                    val = continuous_time_masked_bce(
                        state.model,
                        validation_data,
                        # metric_seed stream so validation never perturbs training RNG;
                        # must share validation_data's device (torch requires
                        # generator.device == tensor.device for torch.rand).
                        config.seeds.generator("metric_seed", device=validation_data.device),
                        l2reg=0.0,
                        train_size=config.dimensions.train_size,
                        min_time=tcfg.min_time,
                    )
                last["validation_loss"] = float(val.data_loss.item())
            if on_log is not None:
                on_log(last)

        if (
            checkpoint_dir is not None
            and tcfg.checkpoint_every > 0
            and state.step % tcfg.checkpoint_every == 0
        ):
            _save(checkpoint_dir, state, config, teacher)

    if checkpoint_dir is not None:
        _save(checkpoint_dir, state, config, teacher, final=True)

    summary = {
        "final_step": state.step,
        "examples_seen": state.examples_seen,
        "wall_clock_s": time.monotonic() - start_time,
        **{k: v for k, v in last.items() if k not in ("step", "examples_seen")},
    }
    return state, teacher, summary


def _save(
    directory: Path,
    state: TrainState,
    config: RunConfig,
    teacher: HiddenManifoldTeacher,
    final: bool = False,
) -> Path:
    name = "final.pt" if final else f"step_{state.step:08d}.pt"
    path = directory / name
    save_checkpoint(
        path,
        model=state.model,
        optimizer=state.optimizer,
        step=state.step,
        examples_seen=state.examples_seen,
        config_dict=config.to_dict(),
        teacher_id=teacher.teacher_id,
        generator_states={k: g.get_state() for k, g in state.generators.items()},
    )
    return path
