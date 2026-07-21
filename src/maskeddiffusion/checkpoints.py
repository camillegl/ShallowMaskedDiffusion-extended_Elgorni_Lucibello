"""Checkpoint save/restore with exact CPU resume.

A checkpoint contains model state, optimizer state, step counters, resolved
configuration, teacher identifier, RNG generator states, package version, and
git metadata where available.

Generator-state coverage is partial by design: only the streams a caller
actually passes to `restore_into` (in practice `mask_seed` and
`dataloader_seed` — the two consumed continuously throughout the training
loop) are persisted and restored. The one-shot streams drawn once at run
start (`teacher_seed`, `model_seed`, `validation_data_seed`,
`evaluation_data_seed`) are never replayed on resume; teacher identity across
a resume is instead checked via the content hash `teacher_id`
(`teacher.py`), not RNG replay. This means a resumed run's teacher, initial
model weights, and validation set are verified to match by content hash and
config, not reproduced from their original RNG draws.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .models import LinearMaskedScore


def git_metadata(repo_root: Path | None = None) -> dict[str, Any]:
    try:
        root = repo_root or Path.cwd()
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if sha.returncode != 0:
            return {"git_sha": None, "git_dirty": None}
        return {
            "git_sha": sha.stdout.strip(),
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def checkpoint_identity(
    model_state: dict[str, Any], teacher_id: str, step: int, examples_seen: int
) -> str:
    """Stable content hash of a checkpoint: model parameter bytes + the
    teacher/step/examples_seen it was trained under. Two checkpoints with
    this same identity have identical weights under an identical teacher at
    an identical point in training (mirrors `HiddenManifoldTeacher.teacher_id`'s
    design in teacher.py). Used to let downstream artifacts (e.g. a sample
    run's manifest) verify they were produced from *this exact* checkpoint,
    not merely one with the same file path."""
    h = hashlib.sha256()
    for key in sorted(model_state):
        tensor = model_state[key]
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    h.update(teacher_id.encode())
    h.update(f"{step}:{examples_seen}".encode())
    return "ckpt-" + h.hexdigest()[:16]


def save_checkpoint(
    path: str | Path,
    *,
    model: LinearMaskedScore,
    optimizer: torch.optim.Optimizer,
    step: int,
    examples_seen: int,
    config_dict: dict[str, Any],
    teacher_id: str,
    generator_states: dict[str, torch.Tensor],
) -> None:
    model_state = model.state_dict()
    payload = {
        "format": "maskeddiffusion.checkpoint.v1",
        "model_state": model_state,
        "model_config": model.config.identity(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "examples_seen": examples_seen,
        "config": config_dict,
        "teacher_id": teacher_id,
        "checkpoint_id": checkpoint_identity(model_state, teacher_id, step, examples_seen),
        "generator_states": generator_states,
        "package_version": __version__,
        **git_metadata(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "maskeddiffusion.checkpoint.v1":
        raise ValueError(f"unknown checkpoint format {payload.get('format')!r}")
    return payload


def restore_into(
    payload: dict[str, Any],
    *,
    model: LinearMaskedScore,
    optimizer: torch.optim.Optimizer,
    generators: dict[str, torch.Generator],
) -> tuple[int, int]:
    """Restore model, optimizer, and generator states. Returns (step, examples_seen)."""
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    for name, gen in generators.items():
        state = payload["generator_states"].get(name)
        if state is None:
            raise KeyError(f"checkpoint has no generator state for stream {name!r}")
        gen.set_state(state)
    return int(payload["step"]), int(payload["examples_seen"])
