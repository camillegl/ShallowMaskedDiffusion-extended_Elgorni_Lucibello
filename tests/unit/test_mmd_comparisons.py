"""Unit coverage for the four named MMD comparison wrappers
(true_vs_true, train_vs_true, model_vs_true, model_vs_train). The
notebook-equivalence regression suite (tests/regression/test_mmd_notebook_equivalence.py)
covers compute_mmd's numerical correctness in depth; this file only checks
each wrapper delegates to compute_mmd with the documented argument order —
added because model_vs_train had zero test coverage before being wired into
maskeddiffusion-evaluate (docs/PHASE4A_DESIGN_AUDIT.md)."""

import pytest
import torch

from maskeddiffusion.metrics.mmd import (
    compute_mmd,
    model_vs_train,
    model_vs_true,
    train_vs_true,
    true_vs_true,
)

LAMBDAS = (4.0, 8.0)


def _spins(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand((n, d), generator=g) < 0.5, 1.0, -1.0)


def test_true_vs_true_matches_compute_mmd_argument_order():
    a, b = _spins(5, 6, 1), _spins(4, 6, 2)
    expected = compute_mmd(a, b, LAMBDAS)
    got = true_vs_true(a, b, LAMBDAS)
    assert got.biased_mmd2 == expected.biased_mmd2
    assert got.unbiased_mmd2_raw == expected.unbiased_mmd2_raw


def test_train_vs_true_matches_compute_mmd_argument_order():
    train_set, true_fresh = _spins(5, 6, 3), _spins(4, 6, 4)
    expected = compute_mmd(train_set, true_fresh, LAMBDAS)
    got = train_vs_true(train_set, true_fresh, LAMBDAS)
    assert got.biased_mmd2 == expected.biased_mmd2
    assert got.unbiased_mmd2_raw == expected.unbiased_mmd2_raw


def test_model_vs_true_matches_compute_mmd_argument_order():
    model_samples, true_fresh = _spins(5, 6, 5), _spins(4, 6, 6)
    expected = compute_mmd(model_samples, true_fresh, LAMBDAS)
    got = model_vs_true(model_samples, true_fresh, LAMBDAS)
    assert got.biased_mmd2 == expected.biased_mmd2
    assert got.unbiased_mmd2_raw == expected.unbiased_mmd2_raw


def test_model_vs_train_matches_compute_mmd_argument_order():
    model_samples, train_set = _spins(5, 6, 7), _spins(4, 6, 8)
    expected = compute_mmd(model_samples, train_set, LAMBDAS)
    got = model_vs_train(model_samples, train_set, LAMBDAS)
    assert got.biased_mmd2 == expected.biased_mmd2
    assert got.unbiased_mmd2_raw == expected.unbiased_mmd2_raw


def test_model_vs_train_is_symmetric_under_argument_swap():
    """MMD² is symmetric in its two sample sets — compute_mmd(a, b) ==
    compute_mmd(b, a) even when the two sets have different sizes (the
    biased/unbiased formulas swap the (m, n) roles consistently). This is a
    property of compute_mmd itself, verified here because model_vs_train
    now feeds it (model_samples, train_set) directly from the CLI and any
    accidental swap must not be silently masked by symmetry. Compared with
    pytest.approx, not ==: swapping (x, y) changes the chunked summation
    order of the cross term, giving the same float32 cross-platform
    rounding sensitivity as docs/UPSTREAM_DISCREPANCIES.md D15 — the
    mathematical identity holds to ~1e-7 relative, not bit-for-bit."""
    model_samples, train_set = _spins(5, 6, 9), _spins(9, 6, 10)  # different sizes
    forward = model_vs_train(model_samples, train_set, LAMBDAS)
    swapped = compute_mmd(train_set, model_samples, LAMBDAS)
    for lam in LAMBDAS:
        assert forward.biased_mmd2[lam] == pytest.approx(swapped.biased_mmd2[lam], rel=1e-5)
        assert forward.unbiased_mmd2_raw[lam] == pytest.approx(
            swapped.unbiased_mmd2_raw[lam], rel=1e-5
        )
