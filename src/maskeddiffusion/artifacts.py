"""Run-artifact schema (ADR 003) and validation.

Every run directory contains manifest.json, resolved_config.json,
metrics.jsonl, summary.json, and optionally checkpoints/, samples/, figures/.
Every binary file has a manifest entry with path, sha256, and size.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .checkpoints import git_metadata

SCHEMA_VERSION = "maskeddiffusion.run.v1"
REQUIRED_FILES = ("manifest.json", "resolved_config.json", "metrics.jsonl", "summary.json")
REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "command",
    "timestamp",
    "git_sha",
    "git_dirty",
    "python_version",
    "torch_version",
    "platform",
    "device",
    "uv_lock_sha256",
    "teacher_id",
    "seeds",
    "sampler",
    "objective",
    "model",
    "files",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment_metadata(device: str, repo_root: Path | None = None) -> dict[str, Any]:
    uv_lock = (repo_root or Path.cwd()) / "uv.lock"
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "package_version": __version__,
        "platform": platform.platform(),
        "device": device,
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock.exists() else None,
        **git_metadata(repo_root),
    }


class RunArtifact:
    """Writer for one run directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.root / "metrics.jsonl"
        self._files: list[dict[str, Any]] = []

    def log_metrics(self, record: dict[str, Any]) -> None:
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def register_file(self, path: Path, description: str, **extra: Any) -> None:
        self._files.append(
            {
                "path": str(path.relative_to(self.root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "description": description,
                **extra,
            }
        )

    def save_tensor(self, relative: str, tensor: torch.Tensor, description: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
        self.register_file(path, description, dtype=str(tensor.dtype), shape=list(tensor.shape))
        return path

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def write_manifest(
        self,
        *,
        command: str,
        device: str,
        teacher_id: str,
        seeds: dict[str, Any],
        sampler: dict[str, Any],
        objective: dict[str, Any],
        model: dict[str, Any],
        input_paths: list[str] | None = None,
        repo_root: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "command": command,
            "timestamp": datetime.now(UTC).isoformat(),
            **environment_metadata(device, repo_root),
            "teacher_id": teacher_id,
            "seeds": seeds,
            "sampler": sampler,
            "objective": objective,
            "model": model,
            "input_paths": input_paths or [],
            "files": self._files,
            **(extra or {}),
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate_artifact(root: str | Path) -> list[str]:
    """Return a list of problems; empty list means the artifact is valid."""
    root = Path(root)
    problems: list[str] = []

    for name in REQUIRED_FILES:
        if not (root / name).exists():
            problems.append(f"missing required file: {name}")
    if problems:
        return problems

    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except json.JSONDecodeError as e:
        return [f"malformed manifest.json: {e}"]
    try:
        config = json.loads((root / "resolved_config.json").read_text())
    except json.JSONDecodeError as e:
        return [f"malformed resolved_config.json: {e}"]
    try:
        json.loads((root / "summary.json").read_text())
    except json.JSONDecodeError as e:
        problems.append(f"malformed summary.json: {e}")
    with open(root / "metrics.jsonl") as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"malformed metrics.jsonl line {i + 1}")
                    break

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            problems.append(f"manifest missing key: {key}")
    if manifest.get("uv_lock_sha256", "missing") is None:
        problems.append(
            "manifest uv_lock_sha256 is null: dependency set not pinned "
            "(uv.lock absent at run time)"
        )

    seeds = manifest.get("seeds", {})
    from .randomness import STREAMS

    for stream in STREAMS:
        if stream not in seeds:
            problems.append(f"manifest seeds missing stream: {stream}")

    sampler = manifest.get("sampler", {})
    if sampler and "sampler_name" not in sampler:
        problems.append("manifest sampler missing sampler_name")
    if not sampler:
        problems.append("manifest has empty sampler specification")

    cfg_dims = config.get("dimensions", {})
    if cfg_dims:
        expected_n = round(cfg_dims.get("aspect_ratio", 0) * cfg_dims.get("latent_dim", 0))
        if cfg_dims.get("visible_dim") != expected_n:
            problems.append(
                f"resolved visible_dim {cfg_dims.get('visible_dim')} inconsistent with "
                f"round(aspect_ratio*latent_dim)={expected_n}"
            )
        model_cfg = manifest.get("model", {})
        if model_cfg.get("visible_dim") not in (None, cfg_dims.get("visible_dim")):
            problems.append("manifest model visible_dim inconsistent with resolved config")

    # teacher-id consistency across manifest and any checkpoints
    teacher_id = manifest.get("teacher_id")
    for entry in manifest.get("files", []):
        p = root / entry["path"]
        if not p.exists():
            problems.append(f"registered file missing: {entry['path']}")
            continue
        if "sha256" in entry and sha256_file(p) != entry["sha256"]:
            problems.append(f"hash mismatch: {entry['path']}")
        if entry["path"].startswith("checkpoints/") and p.suffix == ".pt":
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
                if payload.get("teacher_id") not in (None, teacher_id):
                    problems.append(
                        f"checkpoint teacher_id {payload.get('teacher_id')} != "
                        f"manifest teacher_id {teacher_id}"
                    )
            except Exception as e:  # noqa: BLE001 - report, don't crash validation
                problems.append(f"unreadable checkpoint {entry['path']}: {e}")

    return problems
