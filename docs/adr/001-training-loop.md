# ADR 001 — Training loop: direct typed PyTorch, no Lightning

Status: accepted (Phase 2, 2026-07-20).

## Decision

The active package (`src/maskeddiffusion`) uses a direct, typed PyTorch training loop.
PyTorch Lightning is not used in, and must not be required by, the active implementation.

## Reasons

- The model and training procedure are small (a linear score, AdamW, one loss); a framework
  adds indirection without proportional benefit.
- Exact control over random generators matters scientifically (hierarchical seed streams,
  reproducible masking); Lightning's implicit seeding and loop hooks obscure it.
- Loss normalization and regularization scaling are contract-level quantities (see
  `docs/UPSTREAM_DISCREPANCIES.md` D5/D6); they must be visible in one function, not spread
  across callbacks.
- Deterministic checkpoint/resume (model + optimizer + RNG states) must be transparent and
  testable on CPU.

## Consequences

- Progress is recorded in optimizer steps and examples seen, not only epochs.
- The legacy Lightning-based behavior (`diffusion.py`, `train.py`) is preserved as
  regression-tested knowledge via `tests/fixtures/original_architecture_v1/`; Lightning
  remains a dependency only as long as legacy modules remain importable, and is removed once
  they are retired.
