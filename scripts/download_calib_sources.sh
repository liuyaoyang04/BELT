#!/usr/bin/env bash
# Download the pinned public source files listed in data/calib_sources/manifest.json
# for the 5-category calibration corpus (scripts/calib.py) and verify their
# file sizes and SHA-256 checksums.
#
# Usage:
#   bash scripts/download_calib_sources.sh [OUT_DIR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${REPO_ROOT}/data/calib_sources}"
MANIFEST="${REPO_ROOT}/data/calib_sources/manifest.json"
HF="https://huggingface.co"
mkdir -p "${OUT}"

CURL_OPTS=(-fL --retry 3 --retry-delay 2)

while IFS=$'\t' read -r repo revision rel_file expected_size expected_sha; do
    dst="${OUT}/${repo/\//__}/${rel_file}"
    if [[ -s "${dst}" ]]; then
        echo "exists: ${dst}"
        continue
    fi
    mkdir -p "$(dirname "${dst}")"
    url="${HF}/datasets/${repo}/resolve/${revision}/${rel_file}"
    echo "fetching ${dst} ..."
    curl "${CURL_OPTS[@]}" -o "${dst}.part" "${url}"
    mv "${dst}.part" "${dst}"
done < <(
    python3 - "${MANIFEST}" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
for entry in manifest:
    print(f"{entry['repo']}\t{entry['revision']}\t{entry['file']}\t{entry['size']}\t{entry['sha256']}")
PY
)

python3 - "${OUT}" "${MANIFEST}" <<'PY'
import hashlib, json, pathlib, sys

out = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
for entry in manifest:
    rel = pathlib.Path(entry["repo"].replace("/", "__")) / entry["file"]
    p = out / rel
    if not p.is_file():
        raise SystemExit(f"missing downloaded file: {p}")
    size = p.stat().st_size
    if size != entry["size"]:
        raise SystemExit(f"{rel}: size {size} != {entry['size']}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"{rel}: sha256 {digest} != {entry['sha256']}")
    print(f"verified {rel}: {size:,} bytes, sha256={digest[:16]}...")
PY
