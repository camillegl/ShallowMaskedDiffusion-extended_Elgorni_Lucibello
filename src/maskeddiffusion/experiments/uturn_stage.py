"""Optional U-turn stage for the Phase 4C engine (docs/PHASE4C_EXPERIMENT_PROTOCOL.md).

When an `ExperimentSpec` carries a `uturn` config, the runner executes the
U-turn / reconstruction protocol (`maskeddiffusion.uturn`) against the arm's
own trained checkpoint and teacher — mirroring `cli/uturn.py`'s wiring — and
writes a fourth ADR-003 stage artifact, `<run_dir>/uturn/`.

Reduced-form design decision. `Phase4CRunRecord.uturn` and
`analysis.rows.UTurnSummary` hold exactly one curve
(`mask_densities`/`overlap`) and one scalar pair
(`baseline_recovery`/`excess_recovery`) — there is no room for a per-source
breakdown or the train-vs-fresh delta. `uturn_summary_to_record_block` picks
the **fresh-source** curve when available, falling back to **train** only
when `sources = ("train",)` (fresh unavailable). This choice is deliberate:
a fresh-source recovery curve is interpretable on its own (retrieval against
an unseen draw of the same finite-`F` law), whereas a train-source curve
alone must never be read as a memorization signal
(`maskeddiffusion.uturn` module docstring, `docs/RESEARCH_SPEC.md` claim
discipline) — using it as the record's headline curve when a
memorization-neutral alternative exists would invite exactly that
misreading. `baseline_recovery`/`excess_recovery` are the mean over
`t_values` of that source's `no_recovery_baseline` / `excess_recovery_mean`
points (`uturn.summarize_uturn`) — a single-number summary of the sweep, not
a claim at any particular `t`. The full per-source and train-vs-fresh
comparison data is never discarded: it stays in the stage artifact's own
`summary.json`, which `artifact_path`-style provenance can always reach.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..artifacts import RunArtifact, sha256_file
from ..config import RunConfig
from ..models import LinearMaskedScore
from ..teacher import HiddenManifoldTeacher
from ..uturn import run_uturn, summarize_uturn
from .schema import check_uturn_block
from .spec import ExperimentSpec, spec_fingerprint

_PREFERRED_SOURCE_ORDER: tuple[str, ...] = ("fresh", "train")


def _headline_source(sources: tuple[str, ...]) -> str:
    for candidate in _PREFERRED_SOURCE_ORDER:
        if candidate in sources:
            return candidate
    raise ValueError(f"no known U-turn source in {sources!r}")


def uturn_summary_to_record_block(summary: dict[str, Any]) -> dict[str, Any]:
    """`summarize_uturn` output -> the schema's reduced `uturn` block.

    See module docstring for the source-selection and scalar-reduction
    rationale. The output is validated against
    `experiments.schema.check_uturn_block` — the same function
    `Phase4CRunRecord.__post_init__` calls — before being returned, so a
    broken adapter fails here, at the point of construction, rather than
    later and more confusingly inside record validation.
    """
    sources = tuple(summary["sources"])
    source = _headline_source(sources)
    points = [p for p in summary["points"] if p["source"] == source]
    points.sort(key=lambda p: p["t_value"])
    if not points:
        raise ValueError(f"summary has no points for headline source {source!r}")
    block = {
        "mask_densities": [p["t_value"] for p in points],
        "overlap": [p["q_u_mean"] for p in points],
        "baseline_recovery": sum(p["no_recovery_baseline"] for p in points) / len(points),
        "excess_recovery": sum(p["excess_recovery_mean"] for p in points) / len(points),
    }
    check_uturn_block(block)
    return block


def run_uturn_stage(
    *,
    model: LinearMaskedScore,
    spec: ExperimentSpec,
    config: RunConfig,
    teacher: HiddenManifoldTeacher,
    train_set: torch.Tensor,
    checkpoint_path: Path,
    checkpoint_id: str | None,
    run_dir: Path,
    device: str,
    command: str,
) -> dict[str, Any]:
    """Run the U-turn protocol for one arm and write its ADR-003 artifact.

    `spec.uturn` must not be None. Returns the `summarize_uturn` output (the
    full per-source summary — callers reduce it via
    `uturn_summary_to_record_block` for the canonical run record).
    """
    uturn_config = spec.uturn
    assert uturn_config is not None  # narrowed by caller
    result = run_uturn(model, spec.sampler, teacher, train_set, uturn_config, spec.seeds)
    summary = summarize_uturn(result)

    artifact = RunArtifact(run_dir)
    config.to_json(run_dir / "resolved_config.json")

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = results_dir / "per_example.jsonl"
    with open(per_example_path, "w") as f:
        for cell in result.cells:
            f.write(json.dumps(cell.to_dict()) + "\n")
    artifact.register_file(
        per_example_path,
        "per-example U-turn cell results, one JSON object per (source, example_index, t) cell",
    )
    tensors_path = results_dir / "uturn_tensors.pt"
    torch.save(
        {"masks": result.masks, "clean": result.clean, "reconstructions": result.reconstructions},
        tensors_path,
    )
    artifact.register_file(
        tensors_path,
        "clean examples, paired masks (t, example, coordinate), and reconstructions per source",
    )

    for point in summary["points"]:
        artifact.log_metrics({"event": "uturn_point", **point})
    artifact.log_metrics({"event": "uturn_completed", "n_cells": len(result.cells)})
    artifact.write_summary(summary)
    artifact.write_manifest(
        command=command,
        device=device,
        teacher_id=teacher.teacher_id,
        seeds=spec.seeds.to_dict(),
        sampler=spec.sampler.identity(),
        objective={
            "name": "uturn_reconstruction",
            "t_values": list(uturn_config.t_values),
            "n_examples": uturn_config.n_examples,
            "sources": list(uturn_config.sources),
        },
        model=config.model.identity(),
        input_paths=[str(checkpoint_path)],
        extra={
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            "experiment_id": spec.experiment_id,
            "pair_id": spec.pair_id,
            "repeat_id": spec.repeat_id,
            "intervention": spec.intervention,
            "condition": spec.condition,
            "spec_fingerprint": spec_fingerprint(spec),
        },
    )
    return summary


def load_uturn_block(run_dir: Path) -> dict[str, Any] | None:
    """Reduced record block from an already-written U-turn stage, or None.

    Reads the stage's own `summary.json` rather than re-running the (much
    more expensive) reconstruction sweep — used on resume/backfill.
    """
    summary_path = run_dir / "uturn" / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    return uturn_summary_to_record_block(summary)
