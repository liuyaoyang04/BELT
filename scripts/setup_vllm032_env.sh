#!/usr/bin/env bash
# Set up the vLLM 0.32.0 + CuTe DSL environment (.venv-vllm032) and install the
# out-of-tree vllm_lrkv plugin in editable mode.
#
# Usage:
#   bash scripts/setup_vllm032_env.sh
#
# Optional environment variables:
#   PYTHON_BIN    : Python 3.10+ interpreter (default: python3)
#   VLLM_VENV_DIR : Target virtualenv path (default: <repo>/.venv-vllm032)
#   VLLM_WHL_URL  : vLLM 0.32.0 wheel URL (default: official vLLM 0.32.0+cu129 release wheel)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-${REPO_ROOT}/.venv-vllm032}"
VLLM_WHL_URL="${VLLM_WHL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl}"

echo "[setup_vllm032_env] Creating virtual environment at ${VLLM_VENV_DIR} using ${PYTHON_BIN}..."
"${PYTHON_BIN}" -m venv "${VLLM_VENV_DIR}"

"${VLLM_VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

echo "[setup_vllm032_env] Installing vLLM 0.32.0 and Hopper CuTe DSL dependencies..."
"${VLLM_VENV_DIR}/bin/pip" install "${VLLM_WHL_URL}" \
    --extra-index-url https://download.pytorch.org/whl/cu129
"${VLLM_VENV_DIR}/bin/pip" install "nvidia-cutlass-dsl==4.8.0" "quack-kernels==0.6.5"

echo "[setup_vllm032_env] Installing repo and vllm_lrkv plugin in editable mode..."
"${VLLM_VENV_DIR}/bin/pip" install --no-deps --no-build-isolation -e "${REPO_ROOT}"
"${VLLM_VENV_DIR}/bin/pip" install --no-deps --no-build-isolation -e "${REPO_ROOT}/plugins/vllm_lrkv"

echo "[setup_vllm032_env] Verifying installed packages..."
"${VLLM_VENV_DIR}/bin/python" -c "import torch, vllm, vllm_lrkv; print('torch:', torch.__version__, 'vllm:', vllm.__version__)"
echo "[setup_vllm032_env] Done. Activate with: source scripts/vllm032_env.sh"

