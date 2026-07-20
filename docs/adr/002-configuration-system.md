# ADR 002 — Configuration: typed dataclasses + TOML + argparse

Status: accepted (Phase 2, 2026-07-20).

## Decision

Configuration uses:

- typed dataclasses (`maskeddiffusion.config`) as the single in-memory representation;
- TOML files read with standard-library `tomllib` for run configuration;
- argparse only for CLI selection (`--config`, `--output`, `--device`, `--dry-run`) and
  explicit overrides;
- JSON for resolved-configuration snapshots written into run artifacts
  (`resolved_config.json`).

Hydra is not introduced. The unused `hydra-core` dependency is removed (no active code ever
imported it — verified by grep before removal; see `docs/REPO_AUDIT.md`).

## Reasons

- The configuration space is small and flat; dataclasses give type checking and defaults
  without a framework.
- `tomllib` is stdlib — zero dependency cost, no plugin magic, no composition-order
  surprises.
- A resolved JSON snapshot makes every run self-describing independent of the config file's
  later evolution.

## Consequences

- There is exactly one active configuration system. The legacy argparse interface of
  `train.py` is legacy-only and is not extended.
- Contract naming is enforced at the config boundary (`latent_dim`, `aspect_ratio`,
  `sample_ratio`, …); bare `alpha` is rejected (see `docs/NOTATION.md`).
