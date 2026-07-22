# AGENTS.md (deprecated pointer)

This file previously carried full project instructions and has been retired
because it drifted from reality (it claimed Python 3.11, a `.python-version`
pin that did not exist, and an outdated layout).

**Authoritative instructions live in [CLAUDE.md](CLAUDE.md)**, backed by the
source-of-truth documents in [docs/](docs/) — start with
`docs/RESEARCH_SPEC.md`, `docs/NOTATION.md`, and `docs/CLAUDE_CODE.md`.
Where anything here or in older history conflicts with those files, they win.

Quick facts (mirrored from CLAUDE.md, not independently authoritative):

- Package manager: `uv` only; Python 3.12 (`.python-version`, `uv.lock`
  un-gitignored and kept in version control).
- Active implementation: `src/maskeddiffusion/` with `maskeddiffusion-train`
  / `-sample` / `-evaluate` / `-uturn` / `-validate-artifact` CLIs; tests via
  `uv run pytest -q`.
- Legacy flat modules (`train.py`, `diffusion.py`, `models.py`,
  `datasets.py`) are superseded but retained — the protected MMD notebooks
  import them (`docs/REFERENCE_RESULTS_MANIFEST.md`).
