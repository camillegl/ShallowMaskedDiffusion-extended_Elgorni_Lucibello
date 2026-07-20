"""Preservation: protected notebooks and result dependencies exist and match
the recorded SHA-256 hashes (docs/REFERENCE_RESULTS_MANIFEST.md)."""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
MANIFEST = REPO / "artifacts" / "reference" / "mmd_final_run" / "manifest.json"


def entries():
    manifest = json.loads(MANIFEST.read_text())
    return [(e["path"], e) for e in manifest["protected_files"]]


def test_manifest_exists_and_wellformed():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema"] == "maskeddiffusion.reference_manifest.v1"
    assert manifest["notebooks_rerun_during_migration"] is False
    assert len(manifest["protected_files"]) >= 4


@pytest.mark.parametrize("path,entry", entries())
def test_protected_file_exists_and_hash_matches(path, entry):
    p = REPO / path
    assert p.exists(), f"protected file missing: {path}"
    assert p.stat().st_size == entry["size_bytes"], f"size changed: {path}"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == entry["sha256"], f"hash mismatch for protected file: {path}"
