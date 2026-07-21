"""Generate fixture.json by executing the exact, hash-verified MMD function
block extracted from the protected notebook cell, in isolation.

Run from the repository root:

    uv run python tests/fixtures/mmd_notebook_reference_v1/generate_fixture.py

What this script does, in order:
1. Reads the protected notebook (read-only; never modified or executed as a
   notebook) and extracts the exact source text of the MMD function block
   from code cell 11 (lambda_key, kernel_sums_exponential_hamming,
   normalized_mmd_weights, compute_mmd_biased_unbiased).
2. Verifies the notebook's own SHA-256 against docs/REFERENCE_RESULTS_MANIFEST.md
   and the extracted block's SHA-256 against the pinned value below, so a
   silent notebook edit or a bad extraction boundary fails loudly instead of
   quietly producing a different fixture.
3. `exec()`s that exact block in an isolated namespace containing only torch
   and numpy — `maskeddiffusion.metrics.mmd` is never imported here, so the
   expected values in fixture.json do not depend on the production module
   being correct.
4. Evaluates the reference implementation on a small fixed pair of explicit
   ±1 arrays at chunk_size=1024 and chunk_size=3, checks the two agree (the
   reference is chunking-invariant on its own), and writes fixture.json.

This script is kept for provenance/audit; the fixture it produces is
committed and must not be regenerated casually (a regeneration invalidates
the pinned reference — record any regeneration in
docs/MMD_NOTEBOOK_PROVENANCE.md, not by silently overwriting the file).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

FIXTURE_DIR = Path(__file__).parent
REPO_ROOT = FIXTURE_DIR.parents[2]

NOTEBOOK_PATH = (
    REPO_ROOT / "experiments-analysis" / "analysis_mmd_distribution_distance_corrected.ipynb"
)
EXPECTED_NOTEBOOK_SHA256 = "3bc29d1904fe1444db0e5815bc4da28c4ec40b895c4889c2513d87a715b5966c"
CELL_INDEX = 11
BLOCK_START_MARKER = "def lambda_key(lam: float) -> str:"
BLOCK_END_MARKER = (
    "# ===========================================================================\n"
    "# Nearest-neighbour overlap helper (prefix-parameterised)"
)
EXPECTED_BLOCK_SHA256 = "4fe18103d5628f89b7d69b465eee603d29377e0ac91dc6353b6a70698ba689f1"


def extract_block() -> str:
    notebook_bytes = NOTEBOOK_PATH.read_bytes()
    notebook_sha256 = hashlib.sha256(notebook_bytes).hexdigest()
    if notebook_sha256 != EXPECTED_NOTEBOOK_SHA256:
        raise SystemExit(
            f"protected notebook hash mismatch: got {notebook_sha256}, "
            f"expected {EXPECTED_NOTEBOOK_SHA256} (docs/REFERENCE_RESULTS_MANIFEST.md); "
            "refusing to regenerate the fixture from an unverified notebook"
        )

    nb = json.loads(notebook_bytes)
    cell_src = "".join(nb["cells"][CELL_INDEX]["source"])
    i0 = cell_src.index(BLOCK_START_MARKER)
    i1 = cell_src.index(BLOCK_END_MARKER)
    block = cell_src[i0:i1].rstrip() + "\n"

    block_sha256 = hashlib.sha256(block.encode()).hexdigest()
    if block_sha256 != EXPECTED_BLOCK_SHA256:
        raise SystemExit(
            f"extracted block hash mismatch: got {block_sha256}, "
            f"expected {EXPECTED_BLOCK_SHA256}; the notebook cell or the "
            "extraction markers above have changed - update "
            "docs/MMD_NOTEBOOK_PROVENANCE.md deliberately, don't silently accept this"
        )
    return block


def main() -> None:
    block = extract_block()

    # Isolated namespace: torch/numpy only, production module never imported.
    namespace = {"torch": torch, "np": np}
    exec(compile(block, "<notebook-cell-11-mmd-block>", "exec"), namespace)
    compute_mmd_biased_unbiased = namespace["compute_mmd_biased_unbiased"]

    X = torch.tensor(
        [
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1, 1, 1, -1, -1],
        ],
        dtype=torch.float32,
    )
    Y = torch.tensor(
        [
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, -1, 1, -1, 1, -1, 1, -1, -1],
        ],
        dtype=torch.float32,
    )
    lambdas = [4.0, 8.0]

    result_unchunked = compute_mmd_biased_unbiased(X, Y, lambdas, chunk_size=1024)
    result_chunked = compute_mmd_biased_unbiased(X, Y, lambdas, chunk_size=3)
    for key, unchunked_val in result_unchunked.items():
        chunked_val = result_chunked[key]
        if abs(unchunked_val - chunked_val) >= 1e-5:
            raise SystemExit(f"reference block is not chunking-invariant on {key}")

    fixture = {
        "provenance": {
            "notebook_path": str(NOTEBOOK_PATH.relative_to(REPO_ROOT)),
            "notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
            "cell_index": CELL_INDEX,
            "block_sha256": EXPECTED_BLOCK_SHA256,
            "block_functions": [
                "lambda_key",
                "kernel_sums_exponential_hamming",
                "normalized_mmd_weights",
                "compute_mmd_biased_unbiased",
            ],
            "generation_method": "exec of the exact byte-hashed block above in an "
            "isolated namespace (torch, numpy only); production "
            "maskeddiffusion.metrics.mmd was NOT imported while generating "
            "these expected values.",
        },
        "inputs": {
            "dtype": "float32",
            "X": X.tolist(),
            "Y": Y.tolist(),
            "lambdas": lambdas,
            "chunk_sizes_checked": [1024, 3],
        },
        "expected": {
            "mmd2_biased_lambda_4": result_unchunked["mmd2_biased_lambda_4"],
            "mmd2_unbiased_lambda_4_raw": result_unchunked["mmd2_unbiased_lambda_4_raw"],
            "mmd2_biased_lambda_8": result_unchunked["mmd2_biased_lambda_8"],
            "mmd2_unbiased_lambda_8_raw": result_unchunked["mmd2_unbiased_lambda_8_raw"],
            "mmd2_biased_mixture": result_unchunked["mmd2_biased_mixture"],
            "mmd2_unbiased_mixture_raw": result_unchunked["mmd2_unbiased_mixture_raw"],
            "mmd_biased_mixture": result_unchunked["mmd_biased_mixture"],
        },
        # rel=1e-6 was observed to be right at the boundary of, and abs=1e-9
        # tighter than, the actual float32 CPU reduction-order divergence
        # between macOS (arm64) and Linux (x86_64) CI runners — both legal
        # IEEE-754 float32 roundings of the same chunked sum, not a
        # correctness bug. Widened once CI (added after this fixture was
        # first generated) exposed the cross-platform gap; see
        # docs/MMD_NOTEBOOK_PROVENANCE.md and
        # docs/UPSTREAM_DISCREPANCIES.md D15. Still 5-6 orders of magnitude
        # below any scientifically meaningful MMD difference discussed in
        # this repo (~1e-2-1e-1).
        "tolerance": {"rel": 5e-6, "abs": 1e-7},
    }

    out_path = FIXTURE_DIR / "fixture.json"
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(json.dumps(fixture["expected"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
