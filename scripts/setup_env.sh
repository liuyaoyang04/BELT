#!/usr/bin/env bash
# Set up the main LRKV compression & calibration Python environment (.venv).
#
# Usage:
#   bash scripts/setup_env.sh
#
# Optional environment variables:
#   PYTHON_BIN : Python 3.10+ interpreter (default: python3)
#   VENV_DIR   : Target virtualenv path (default: <repo>/.venv)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

echo "[setup_env] Creating virtual environment at ${VENV_DIR} using ${PYTHON_BIN}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install -r "${REPO_ROOT}/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu121
"${VENV_DIR}/bin/pip" install --no-deps --no-build-isolation -e "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/data" "${REPO_ROOT}/checkpoints" "${REPO_ROOT}/models" "${REPO_ROOT}/results"

echo "[setup_env] Environment ready at ${VENV_DIR}"
"${VENV_DIR}/bin/python" -c "import torch, transformers, lrkv; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'transformers:', transformers.__version__)"

