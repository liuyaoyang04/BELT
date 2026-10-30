#!/usr/bin/env bash
# Launch an OpenAI-compatible vLLM 0.32.0 server for either a dense Qwen3.8-27B model or a
# latent-KV exported model (with the vllm_lrkv plugin automatically loaded).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh models/latent_exports/<model>_latent [PORT]
#
# Optional environment variables:
#   VLLM_VENV_DIR : Path to vLLM 0.32 virtualenv (default: <repo>/.venv-vllm032)
#   HOST          : Bind address (default: 127.0.0.1)
#   PORT          : Bind port (default: 8000, or second positional arg)
#   MAX_MODEL_LEN : Maximum context length (default: 32768)
#   MAX_NUM_SEQS  : Maximum concurrent sequences (default: 256; fits BF16 Qwen3.8-27B GDN blocks on 80GB Hopper)
#   GPU_MEM_UTIL  : vLLM --gpu-memory-utilization (default: 0.90)
#   EXTRA_ARGS    : Additional flags passed to `vllm serve`
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/vllm032_env.sh"

MODEL_DIR="${1:?Usage: bash scripts/serve_vllm.sh <MODEL_DIR> [PORT]}"
PORT="${2:-${PORT:-8000}}"
HOST="${HOST:-127.0.0.1}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-${REPO_ROOT}/.venv-vllm032}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

EXTRA=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
fi

if [[ ! -x "${VLLM_VENV_DIR}/bin/vllm" ]]; then
    echo "Error: ${VLLM_VENV_DIR}/bin/vllm not found. Run 'bash scripts/setup_vllm032_env.sh' first (or set VLLM_VENV_DIR)." >&2
    exit 1
fi

exec "${VLLM_VENV_DIR}/bin/vllm" serve "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    "${EXTRA[@]}"

