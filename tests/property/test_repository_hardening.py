"""Static repository-structure guarantees (Phase 3 CI hardening,
docs/PHASE3_BRANCH_REPORT.md). Bare `alpha`/`gamma` enforcement lives in
test_notation_enforcement.py; this file covers the remaining invariants."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src" / "maskeddiffusion"
FROZEN_ROOT_MODULES = ["datasets", "diffusion", "models"]


def test_active_package_never_imports_frozen_root_modules():
    """Nothing under src/maskeddiffusion/ may import the root-level frozen
    compatibility modules (docs/FROZEN_LEGACY_RUNTIME.md: one-way
    legacy-to-notebook dependency only)."""
    offenders = []
    pattern = re.compile(
        r"^\s*(from\s+("
        + "|".join(FROZEN_ROOT_MODULES)
        + r")\s+import|import\s+("
        + "|".join(FROZEN_ROOT_MODULES)
        + r")\b)"
    )
    for path in SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert offenders == [], f"active package imports frozen root modules: {offenders}"


def test_frozen_root_modules_stay_outside_active_package():
    """The retained root compatibility modules must remain importable from
    the repository root (docs/FROZEN_LEGACY_RUNTIME.md), not be relocated
    into src/maskeddiffusion/. src/maskeddiffusion/models.py is a distinct,
    independent module and is not a copy of the root one (see
    test_active_package_never_imports_frozen_root_modules for the actual
    import-boundary guarantee)."""
    for name in FROZEN_ROOT_MODULES:
        assert (ROOT / f"{name}.py").exists(), f"{name}.py missing from repository root"


def test_protected_reference_files_exist_and_are_hash_pinned():
    """Every file pinned in the protected-artifact manifest exists and
    matches its recorded SHA-256 (scripts/validate_reference_artifacts.sh
    performs the same check as a standalone script; this test keeps it
    under `pytest -q`)."""
    manifest_path = ROOT / "artifacts" / "reference" / "mmd_final_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for entry in manifest["protected_files"]:
        p = ROOT / entry["path"]
        if not p.exists():
            failures.append(f"missing: {entry['path']}")
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            failures.append(f"hash mismatch: {entry['path']}")
    assert failures == [], failures
    assert len(manifest["protected_files"]) == 8
