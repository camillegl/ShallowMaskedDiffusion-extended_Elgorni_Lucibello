#!/usr/bin/env bash
# Verify the protected MMD notebooks and their result dependencies still exist
# and match the recorded SHA-256 hashes (docs/REFERENCE_RESULTS_MANIFEST.md /
# artifacts/reference/mmd_final_run/manifest.json).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python - <<'EOF'
import hashlib, json, sys
from pathlib import Path

manifest = json.loads(Path("artifacts/reference/mmd_final_run/manifest.json").read_text())
failures = []
for entry in manifest["protected_files"]:
    p = Path(entry["path"])
    if not p.exists():
        failures.append(f"missing: {p}")
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != entry["sha256"]:
        failures.append(f"hash mismatch: {p}")
for f in failures:
    print("FAIL:", f)
if failures:
    sys.exit(1)
print(f"all {len(manifest['protected_files'])} protected reference files verified")
EOF
