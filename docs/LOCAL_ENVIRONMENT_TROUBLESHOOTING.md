# Local environment troubleshooting

Machine-specific issues observed while validating this repository. These are
**not** repository defects; nothing here is fixed by changing package code,
imports, or run scripts. Keep this file limited to issues that were actually
reproduced and diagnosed, with the diagnostic trail, not general FAQ content.

## `ModuleNotFoundError: No module named 'maskeddiffusion'` despite a clean editable install

### Observed symptom

On at least one macOS development machine, `uv run pytest`,
`uv run maskeddiffusion-train --help`, and `scripts/reproduce_smoke.sh`
intermittently fail with `ModuleNotFoundError: No module named
'maskeddiffusion'`, even though:

- `uv sync` reports success with no errors;
- `uv pip show maskeddiffusion` reports a valid editable install pointing at
  `src/`;
- the failure is not consistent — the same command can pass on one invocation
  and fail on the next, with no source or config change in between.

### Root cause (diagnosed, local to this machine)

`uv`'s editable install writes a `.pth` file,
`.venv/lib/python3.12/site-packages/_editable_impl_maskeddiffusion.pth`,
containing the absolute path to `src/`. CPython's `site.addpackage()`
silently **skips** any `.pth` file that has the macOS `UF_HIDDEN` filesystem
flag set (see `Lib/site.py`, the `st_flags & stat.UF_HIDDEN` check) — the
`src/` directory never reaches `sys.path`, so `import maskeddiffusion` fails
even though the venv metadata is entirely correct.

Something on this machine (not identified — candidates include Spotlight
indexing exclusions, a cloud-sync client, or a security/EDR tool; not
anything in this repository, `uv`, or `pyproject.toml`) re-applies
`UF_HIDDEN` to this specific underscore-prefixed `.pth` file. The file's own
`mtime` does not change when the flag reappears, confirming it's an
attribute being toggled externally, not the file being rewritten by `uv` or
by anything in this repo's tooling.

Timing, as observed during this review: clearing the flag
(`chflags nohidden`) and then running an unrelated no-op command (`uv run
true`) left it cleared, but running an actual `maskeddiffusion-*` entry
point (e.g. `uv run maskeddiffusion-train --help`) left the flag re-set
immediately afterward, on the same invocation. This is consistent with
something reacting to the file being *accessed* or to the venv script being
*executed*, not a fixed-interval timer — so a single `chflags nohidden`
before a multi-step script is **not reliably sufficient**: a script that
invokes the CLI more than once (as `scripts/reproduce_smoke.sh` does) can
still fail partway through even when cleared immediately beforehand. This
was observed directly: `reproduce_smoke.sh` completed its train step and
then failed on the following sample step with the same
`ModuleNotFoundError`, with the flag re-set in between despite having been
cleared at the start of the same shell invocation.

### Diagnostic commands

```bash
# Confirm the flag is set (look for "hidden" in the output):
ls -lO .venv/lib/python3.12/site-packages/*maskeddiffusion*.pth

# Confirm this is the cause: clear it and immediately retry the import.
chflags nohidden .venv/lib/python3.12/site-packages/_editable_impl_maskeddiffusion.pth
uv run python -c "import maskeddiffusion; print(maskeddiffusion.__file__)"
```

If the import succeeds immediately after `chflags nohidden` and fails again
on a later, unrelated invocation with the flag back on and the file's mtime
unchanged, that is this issue.

### Confirmed local workaround (partial — not fully reliable)

Run `chflags nohidden` on the `.pth` file immediately before the command that
needs the import, in the same shell invocation (chaining with `&&`):

```bash
chflags nohidden .venv/lib/python3.12/site-packages/_editable_impl_maskeddiffusion.pth && uv run pytest -q
```

This is sufficient for a single-command invocation (e.g. `pytest -q`,
`maskeddiffusion-train --help` in isolation), since the flag reappears only
after the import-triggering process runs. It is **not sufficient for a
multi-step script** that invokes the CLI more than once —
`scripts/reproduce_smoke.sh` calls `maskeddiffusion-train` and then
`maskeddiffusion-sample` in sequence, and the flag was observed to reappear
between those two calls even when cleared once at the very start of the
script's shell invocation. On this machine, the reliable workaround for a
multi-step script was to re-run `chflags nohidden` immediately before *each*
CLI invocation, not once per script:

```bash
chflags nohidden .venv/lib/python3.12/site-packages/_editable_impl_maskeddiffusion.pth && uv run maskeddiffusion-train ...
chflags nohidden .venv/lib/python3.12/site-packages/_editable_impl_maskeddiffusion.pth && uv run maskeddiffusion-sample ...
```

This is manual, ad hoc, and deliberately not baked into
`scripts/reproduce_smoke.sh` or any other shared script (see "Scope and
status" below).

### Scope and status

- **Not classified as a repository defect.** It is not caused by anything in
  `pyproject.toml`, `src/maskeddiffusion/`, or the `uv` lockfile, and no
  change to any of those is proposed to work around it.
- **Not reproduced in CI or on another clean machine.** This repository does
  not currently have CI configured (no `.github/workflows/`), so there is no
  clean-environment run to cross-check against yet. If CI is added, a plain
  `uv sync && uv run python -c "import maskeddiffusion"` step is sufficient
  to confirm this does not occur there — no macOS-specific workaround should
  be added to CI, since the cause is local-machine tooling, not the package.
- **This workaround is intentionally not automated.** It is not added to
  `scripts/reproduce_smoke.sh`, `scripts/validate_reference_artifacts.sh`, or
  any other project command, and `PYTHONPATH=src` is not added anywhere as a
  substitute. Baking a one-machine `chflags` call or a `PYTHONPATH` override
  into shared scripts would mask a real failure on a machine where the
  editable install is broken for a different reason (e.g. a genuinely stale
  `.pth` file, or a real packaging bug) by always working around the symptom
  rather than surfacing it.
- If this reappears on a different machine, or under CI, that would upgrade
  it from "observed once, local, environmental" to a real reproducibility
  concern worth revisiting here.
