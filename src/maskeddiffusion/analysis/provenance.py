"""Phase 4C analysis provenance manifest (docs/PHASE4C_ANALYSIS_SPEC.md §8).

Same spirit as the ADR 003 run manifest: environment metadata (git sha + dirty
flag, package/python/torch versions, platform, uv.lock sha256) plus explicit
input and output file lists with sha256 — filenames alone are not provenance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..artifacts import environment_metadata, sha256_file

MANIFEST_SCHEMA_VERSION = "maskeddiffusion.p4c_analysis_manifest.v1"
MANIFEST_JSON = "p4c_analysis_manifest.json"


def file_entry(path: str | Path, description: str, role: str) -> dict[str, Any]:
    """sha256 + size entry for one consumed or produced file."""
    p = Path(path)
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "size_bytes": p.stat().st_size,
        "role": role,
        "description": description,
    }


def build_analysis_manifest(
    *,
    command: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    validation: dict[str, Any],
    statistics_policy: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict (spec §8)."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "command": command,
        "timestamp": datetime.now(UTC).isoformat(),
        **environment_metadata("cpu", repo_root),
        "statistics_policy": statistics_policy,
        "validation": validation,
        "inputs": inputs,
        "outputs": outputs,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
