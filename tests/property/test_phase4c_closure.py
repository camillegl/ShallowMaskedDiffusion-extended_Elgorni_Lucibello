"""Phase 4C closure checks: package unification, immutability, and
cross-layer contract enforcement that don't belong to any single module's
unit-test file.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "maskeddiffusion"
TESTS = REPO_ROOT / "tests"

_SINGULAR_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?:\.{1,2})?experiment(?:\s+import\b)|"
    r"import\s+(?:maskeddiffusion\.)?experiment\b)"
)


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py")) + sorted(TESTS.rglob("*.py"))


def test_old_singular_experiment_package_does_not_exist():
    assert not (SRC / "experiment").exists(), (
        "src/maskeddiffusion/experiment/ (singular) must not exist — the "
        "engine and schema were unified into experiments/ (plural)"
    )
    assert (SRC / "experiments").is_dir()
    assert (SRC / "experiments" / "runner.py").exists()
    assert (SRC / "experiments" / "schema.py").exists()


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_source_or_test_file_imports_the_singular_package(path):
    text = path.read_text()
    for lineno, line in enumerate(text.splitlines(), start=1):
        # `from ..experiments...` / `maskeddiffusion.experiments...` must not
        # false-positive: the regex requires the import target to end
        # exactly at "experiment" (a word boundary), not "experiments".
        if _SINGULAR_IMPORT_RE.match(line):
            pytest.fail(
                f"{path.relative_to(REPO_ROOT)}:{lineno}: imports the old singular "
                f"'experiment' package: {line.strip()!r}"
            )


def test_no_duplicate_comparisons_constant_within_experiments_or_analysis():
    """Within the Phase 4C engine (experiments/) and analysis layer
    (analysis/) — the three layers named by the closure requirement
    (runner, uturn stage, analysis ingestion) plus the schema itself —
    COMPARISONS must be imported from experiments.schema, never
    independently redefined. `cli/evaluate.py`'s own COMPARISONS predates
    the schema and is a separate, foundational CLI-level concern (used by
    the plain `maskeddiffusion-evaluate` CLI, not only the engine); it is
    intentionally out of scope for this check to avoid a fragile
    `cli -> experiments` import-direction dependency."""
    offenders = []
    scope = list((SRC / "experiments").rglob("*.py")) + list((SRC / "analysis").rglob("*.py"))
    for path in sorted(scope):
        if path == SRC / "experiments" / "schema.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "COMPARISONS":
                        offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_uturn_stage_and_schema_share_the_uturn_block_validator():
    """experiments.uturn_stage must call experiments.schema's
    check_uturn_block rather than re-implementing shape validation."""
    uturn_stage_src = (SRC / "experiments" / "uturn_stage.py").read_text()
    assert "check_uturn_block" in uturn_stage_src
    assert "from .schema import" in uturn_stage_src


def _imports_experiments_schema(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "experiments.schema" in (node.module)
        ):
            return True
        if isinstance(node, ast.Import):
            if any("experiments.schema" in alias.name for alias in node.names):
                return True
    return False


def test_analysis_ingestion_only_reads_via_phase4c_run_record():
    """analysis.ingest is the only analysis module allowed to import
    experiments.schema (docs/PHASE4C_ANALYSIS_SPEC.md §0); analysis.rows
    additionally imports it for the shared COMPARISONS constant."""
    ingest_path = SRC / "analysis" / "ingest.py"
    assert _imports_experiments_schema(ingest_path)
    assert "Phase4CRunRecord" in ingest_path.read_text()
    for path in sorted((SRC / "analysis").glob("*.py")):
        if path.name in ("ingest.py", "rows.py"):
            continue
        assert not _imports_experiments_schema(path), (
            f"{path.relative_to(REPO_ROOT)} imports experiments.schema directly; "
            "only analysis.ingest (and analysis.rows, for the shared COMPARISONS "
            "constant) may"
        )
