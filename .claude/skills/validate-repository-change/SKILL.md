---
name: validate-repository-change
description: Inspect the current uncommitted diff, run only verified cheap checks, and report consistency between docs and implementation. User-invoked only — never runs automatically and never commits.
disable-model-invocation: true
---

# Validate repository change

Validate the current working-tree changes. Never commit, stage, or revert anything.

1. **Inspect the diff.** `git status --short`, `git diff` (and `git diff --check` for
   whitespace/conflict markers). Summarize what changed and classify each file: scientific
   code, docs, notebook, artifact, config.
2. **Run only checks verified to exist.** Currently verified in this repo:
   - `uv run python -c "import diffusion, models, datasets"` (cheap import check, run from
     repo root);
   - `uv run python train.py --help` (CLI intact) — **on `main` only**; `train.py` was
     retired on the `guthlac` branch (see `docs/MIGRATION_REPORT.md`'s Phase 3F section),
     skip this check there.
   There is no test suite or CI as of the 2026-07-20 audit — do not invent test commands;
   if tests have since been added, run only ones that finish in seconds on CPU.
3. **Docs/implementation consistency.** If scientific code changed, check the affected rows
   of `docs/EQUATION_TO_CODE_MAP.md` and `docs/ORIGINAL_ARCHITECTURE.md` still describe the
   behavior; if docs changed, check they still match the code. Run the
   `validate-scientific-contract` skill on new configs or notebook changes.
4. **Report.** List: checks run with exact outcomes (including failures — never report a
   skipped check as passed), doc/code inconsistencies found, and behavior that remains
   untested or unresolved. Stop there; leave commit decisions to the user.
