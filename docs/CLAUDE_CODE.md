# Claude Code setup for this repository

How agent tooling is organized here, and how to use it.

## Roles

- **`CLAUDE.md` (root)** — the always-loaded project instruction file: purpose,
  source-of-truth hierarchy, notation, invariants, verified commands, forbidden actions.
  Stable facts only; procedures live in skills, long reference material in `docs/`.
- **`.claude/rules/`** — focused rule files loaded as project guidance:
  `scientific-contract.md` (notation, finite-F target, sampler-indexed laws, claim
  discipline), `testing-and-reproducibility.md`, `notebooks.md`.
- **`.claude/skills/`** — reusable procedures. `validate-scientific-contract` and
  `review-scientific-claim` may be invoked by the model when relevant;
  `migrate-legacy-component` and `validate-repository-change` are user-invoked only
  (`disable-model-invocation: true`) because they can drive broad changes or run commands.
- **`.claude/agents/`** — specialist subagents: `scientific-auditor`,
  `architecture-auditor`, `reproducibility-reviewer`, `claim-reviewer`. All are read-only
  inspectors by design.

## When to use which agent

- Use the **built-in Explore** agent for generic "where is X / how does Y work" searches
  with no scientific-judgment component.
- Use the **project agents** when the question involves the scientific contract: verifying
  equations against code (`scientific-auditor`), reconstructing runtime behavior
  (`architecture-auditor`), auditing seeds/artifacts/locking (`reproducibility-reviewer`),
  or vetting prose for overclaims (`claim-reviewer`).
- **Parallel agents** are appropriate for independent read-only audits (the four roles
  above cover disjoint concerns). Keep all file edits in the lead session — parallel
  writers on one working tree cause conflicts and unreviewable merges.
- **Why inspectors are read-only:** audit findings must be evidence, not self-fulfilling
  edits; a reviewer that can modify what it reviews can silently make its own report true,
  and concurrent editors corrupt each other's context.

## Session notes

- Subagents defined in `.claude/agents/` are discovered at session start. If a
  newly created agent is not offered in the current session, **restart Claude Code** (or
  start a new session) from the repository root; do not try to force-load it.
- Memory: keep durable, repo-relevant facts in the docs hierarchy, not in ad-hoc memory —
  future sessions must be able to reconstruct the contract from the repository alone.

## Recommended future hooks (not installed)

No hooks are installed by design (none existed before, and adding automation was out of
scope). Candidates worth adding deliberately later, all non-destructive and non-LLM:

1. Block deletion of regression-test fixtures and committed numerical results
   (`data/*.npz`, `experiments-analysis/results/`).
2. Warn when a diff introduces a new bare `alpha` or `gamma` identifier in Python.
3. Run the cheap import check (`uv run python -c "import diffusion, models, datasets"`)
   before task completion.
4. Warn when a new generated artifact lacks a manifest (commit, args, seed).

Do not add LLM-based hooks, and do not enable bypass permissions.
