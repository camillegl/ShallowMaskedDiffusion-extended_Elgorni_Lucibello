# ADR 003 — Artifact format: one run directory with JSON metadata

Status: accepted (Phase 2, 2026-07-20).

## Decision

Every run produces one directory:

```
<run>/
  manifest.json          # schema version, command, git SHA + dirty flag, environment,
                         # teacher id, seed hierarchy, model/objective/sampler identity,
                         # uv.lock sha256, input/output paths, per-binary-file metadata
  resolved_config.json   # full resolved configuration snapshot
  metrics.jsonl          # one JSON object per logging event (step, examples_seen, losses)
  summary.json           # final scalars
  checkpoints/           # optional; torch.save payloads (model+optimizer+RNG+config)
  samples/               # optional; .pt / .npz tensors
  figures/               # optional
```

Metadata is portable JSON. Tensors use PyTorch `.pt` (or `.npz` where NumPy interop is
wanted). Every binary file must have an entry in `manifest.json` (path, sha256, size,
dtype/shape description).

`maskeddiffusion-validate-artifact` validates the schema (missing files, malformed metadata,
inconsistent teacher ids or dimensions, missing seed streams or sampler spec, hash
mismatches).

## Reasons

- JSON+JSONL is diff-able, greppable, and language-neutral; no database or experiment
  tracker dependency.
- Filename-encoded provenance (the legacy pattern) is lossy; the manifest replaces it.

## Consequences

- Reference results from the protected notebooks are catalogued (not rewritten) under
  `artifacts/reference/mmd_final_run/manifest.json` referencing files at their existing
  paths.
