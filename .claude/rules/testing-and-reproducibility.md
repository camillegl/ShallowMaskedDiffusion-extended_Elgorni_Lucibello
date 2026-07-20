# Testing and reproducibility rules

- Use `uv` exclusively (`uv run python …`). Never use pip, Poetry, or Conda in this repo.
- Core scientific functions must take explicit random generators or seeds as arguments; no
  implicit global randomness (`torch.randn` without a seeded context) in new library code.
  Notebook-level convenience seeding is acceptable only in notebooks.
- Every generated artifact (CSV, NPZ, figure data) produced by new code should carry metadata:
  git commit, arguments/config, seeds, and date — as a sidecar manifest or embedded fields.
  Filenames alone are not provenance.
- Prefer deterministic CPU regression tests for scientific behavior where feasible; when a
  result is platform-dependent (MPS/CUDA nondeterminism), document that in the test or
  artifact rather than loosening tolerances silently.
- Do not delete or regenerate committed numerical results (`data/*.npz`,
  `experiments-analysis/*.csv`) without explicit user instruction.
- Note: `uv.lock` is currently gitignored (see `docs/REPO_AUDIT.md`); do not assume
  collaborators share your resolved dependency versions.
