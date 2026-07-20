# ADR 004 — Legacy compatibility: fixtures + explicit adapters, one implementation

Status: accepted (Phase 2, 2026-07-20).

## Decision

The active implementation follows the hidden-manifold specification
(`docs/RESEARCH_SPEC.md`). Legacy behavior is preserved through:

- deterministic fixtures in `tests/fixtures/original_architecture_v1/` capturing the actual
  behavior of the legacy modules at their source commit;
- regression tests that replay those fixtures against explicitly named compatibility
  switches (e.g., the model's `normalization="legacy_init_only"` mode reproducing the
  init-scaled, no-runtime-`1/√L` legacy score — discrepancy D3);
- nothing else. There are not two full implementations.

Legacy defects are not reproduced as defaults:

- `sign(0)` is repaired to `+1` in the teacher (fixes D8 in the active path);
- regularization applies only to trainable parameters (fixes D6 in the active path);
- runtime `1/√N` normalization is explicit in the forward pass (resolves D3's ambiguity for
  new work; the legacy convention survives only as the named compat mode);
- the CLI does not accept bare `--alpha` (D1); a `--legacy-visible-load` escape hatch, if
  ever needed, must warn loudly and never map silently.

## Consequences

- Legacy flat modules remain in place until fixtures and replacement tests pass, then are
  retired per `docs/MIGRATION_REPORT.md`; modules imported by the protected notebooks are
  kept or shimmed, never silently removed.
- Fixture manifests record whether each pinned behavior is intended behavior or a legacy
  quirk, with discrepancy IDs.
