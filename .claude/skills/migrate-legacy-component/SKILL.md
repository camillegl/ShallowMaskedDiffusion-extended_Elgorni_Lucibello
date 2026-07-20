---
name: migrate-legacy-component
description: Migrate exactly one legacy component (module function, notebook-trapped logic) into contract-compliant package code with regression protection. User-invoked only — do not trigger automatically.
disable-model-invocation: true
---

# Migrate legacy component

Migrate **exactly one** named legacy component per invocation. Refuse multi-component
requests; ask the user to split them.

Procedure:

1. **Map first.** The component must have an entry in `docs/EQUATION_TO_CODE_MAP.md`. If it
   doesn't, add it (with the architecture-auditor agent's help) before writing any new code.
2. **Pin behavior.** Write a deterministic CPU regression test capturing the current
   behavior of the original (fixed seeds/generators, small sizes). It must pass against the
   legacy implementation before migration starts.
3. **Migrate.** Implement the replacement using contract naming (`docs/NOTATION.md`) and
   explicit generators (`.claude/rules/testing-and-reproducibility.md`). The regression test
   must pass against the replacement bit-for-bit, or the tolerance and its reason must be
   documented in the test.
4. **Do not delete the original.** Leave legacy code in place, marked as superseded, until
   the user has validated the replacement in a later session.
5. **Log.** Update `docs/EQUATION_TO_CODE_MAP.md` (new module column),
   `docs/UPSTREAM_DISCREPANCIES.md` (if the migration resolves or exposes a discrepancy),
   and `docs/PROVENANCE.md` is not touched; add a short entry to `docs/REPO_AUDIT.md`'s
   action column if status changed.
6. **Never commit automatically.** Report the diff and stop.
