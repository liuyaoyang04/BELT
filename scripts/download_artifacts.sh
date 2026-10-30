#!/usr/bin/env bash
# Download pre-computed BELT release artifacts from GitHub Releases and verify
# their exact byte sizes and SHA-256 checksums.
#
# Usage:
#   bash scripts/download_artifacts.sh            # download c50 low-rank factors (.pt)
#   bash scripts/download_artifacts.sh --factors  # download c50 low-rank factors (.pt)
#   bash scripts/download_artifacts.sh --calib    # download pre-generated docs.jsonl
#   bash scripts/download_artifacts.sh --all      # download both artifacts
#
# Optional environment variables:
#   RELEASE_TAG      : GitHub release tag (default: v0.1.0)
#   RELEASE_BASE_URL : Override base download URL (default: https://github.com/liuyaoyang04/BELT/releases/download/${RELEASE_TAG})
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_TAG="${RELEASE_TAG:-v0.1.0}"
RELEASE_BASE_URL="${RELEASE_BASE_URL:-https://github.com/liuyaoyang04/BELT/releases/download/${RELEASE_TAG}}"

MODE="${1:---factors}"
case "${MODE}" in
    --factors)
        WANT_FACTORS=1
        WANT_CALIB=0
        ;;
    --calib)
        WANT_FACTORS=0
        WANT_CALIB=1
        ;;
    --all|--with-calib)
        WANT_FACTORS=1
        WANT_CALIB=1
        ;;
    -h|--help)
        sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "Unknown option: ${MODE} (expected --factors, --calib, or --all)" >&2
        exit 1
        ;;
esac

FACTORS_NAME="qwen3_8_27b_docs_L8192_c50_ours.pt"
FACTORS_DST="${REPO_ROOT}/checkpoints/${FACTORS_NAME}"
FACTORS_SIZE=201354438
FACTORS_SHA256="3dcc4f674dafce985522c2f50979a7c1cc5e865bc913a0ff9946320b499dffa6"

CALIB_NAME="docs.jsonl"
CALIB_DST="${REPO_ROOT}/data/calib/${CALIB_NAME}"
CALIB_SIZE=194406906
CALIB_SHA256="e1480f65860257aed70eb046d248e4a4b778a46af68152d6916d71f99c9d9afd"

CURL_OPTS=(-fL --retry 3 --retry-delay 2)

verify_file() {
    local path="$1"
    local expected_size="$2"
    local expected_sha="$3"
    python3 - "${path}" "${expected_size}" "${expected_sha}" <<'PY'
import hashlib, pathlib, sys

p = pathlib.Path(sys.argv[1])
expected_size = int(sys.argv[2])
expected_sha = sys.argv[3]

if not p.is_file():
    raise SystemExit(f"missing file: {p}")
size = p.stat().st_size
if size != expected_size:
    raise SystemExit(f"{p}: size {size} != {expected_size}")
h = hashlib.sha256()
with p.open("rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
digest = h.hexdigest()
if digest != expected_sha:
    raise SystemExit(f"{p}: sha256 {digest} != {expected_sha}")
print(f"verified {p}: {size:,} bytes, sha256={digest[:16]}...")
PY
}

fetch_artifact() {
    local asset_name="$1"
    local dst="$2"
    local expected_size="$3"
    local expected_sha="$4"

    mkdir -p "$(dirname "${dst}")"
    if [[ -f "${dst}" ]]; then
        echo "exists: ${dst} (verifying checksum...)"
        verify_file "${dst}" "${expected_size}" "${expected_sha}"
        return
    fi
    local url="${RELEASE_BASE_URL%/}/${asset_name}"
    echo "fetching ${url} -> ${dst} ..."
    curl "${CURL_OPTS[@]}" -o "${dst}.part" "${url}"
    mv "${dst}.part" "${dst}"
    verify_file "${dst}" "${expected_size}" "${expected_sha}"
}

if [[ "${WANT_FACTORS}" == "1" ]]; then
    fetch_artifact "${FACTORS_NAME}" "${FACTORS_DST}" "${FACTORS_SIZE}" "${FACTORS_SHA256}"
fi

if [[ "${WANT_CALIB}" == "1" ]]; then
    fetch_artifact "${CALIB_NAME}" "${CALIB_DST}" "${CALIB_SIZE}" "${CALIB_SHA256}"
fi
