"""Provenance-manifest tests (docs/PHASE4C_ANALYSIS_SPEC.md §8): environment
metadata, explicit hashed input/output lists, and the validation report."""

import json

from maskeddiffusion.analysis.provenance import (
    MANIFEST_SCHEMA_VERSION,
    build_analysis_manifest,
    file_entry,
    write_manifest,
)
from maskeddiffusion.analysis.report import STATISTICS_POLICY, write_tables
from maskeddiffusion.analysis.rows import validate_rows
from maskeddiffusion.analysis.synthetic import synthetic_rows
from maskeddiffusion.artifacts import sha256_file


def test_file_entry_hashes_real_files(tmp_path):
    rows = validate_rows(synthetic_rows()).accepted
    paths = write_tables(tmp_path, rows)
    entry = file_entry(paths["per_run"], "tidy per-run table", role="output")
    assert entry["sha256"] == sha256_file(paths["per_run"])
    assert entry["size_bytes"] == paths["per_run"].stat().st_size
    assert entry["role"] == "output"


def test_manifest_structure_and_environment(tmp_path):
    consumed = tmp_path / "run_record_0.json"
    consumed.write_text(json.dumps({"synthetic": True}))
    inputs = [file_entry(consumed, "synthetic run record", role="input")]
    produced = tmp_path / "out.csv"
    produced.write_text("a,b\n1,2\n")
    outputs = [file_entry(produced, "placeholder output", role="output")]
    validation = {"accepted_row_count": 24, "rejection_count": 0, "rejections": []}
    manifest = build_analysis_manifest(
        command="maskeddiffusion-p4c-analyze --synthetic-demo",
        inputs=inputs,
        outputs=outputs,
        validation=validation,
        statistics_policy=STATISTICS_POLICY,
    )
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    for key in (
        "git_sha",
        "git_dirty",
        "python_version",
        "torch_version",
        "package_version",
        "platform",
        "uv_lock_sha256",
    ):
        assert key in manifest
    assert manifest["statistics_policy"]["bootstrap_ci"] is False
    assert manifest["inputs"][0]["sha256"] == sha256_file(consumed)
    assert manifest["outputs"][0]["sha256"] == sha256_file(produced)

    out = write_manifest(manifest, tmp_path / "p4c_analysis_manifest.json")
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert loaded["validation"] == validation
