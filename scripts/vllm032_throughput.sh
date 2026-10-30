#!/usr/bin/env bash
# Measure online serving throughput of a model directory on vLLM 0.32 with `vllm bench serve`
# (random token IDs, ignore_eos=True, all requests sent at once).
#
# Usage:
#   GPU=0 MODEL_DIR=models/latent_exports/<model>_latent bash scripts/vllm032_throughput.sh [TAG]
#
# Environment variables:
#   MODEL_DIR    : HF model directory (dense BF16 or latent export from lrkv.runtime.export_latent)
#   GPU          : Single GPU index (default: 0)
#   PORT         : Local server port (default: 8910 + GPU)
#   PROMPTS      : Number of requests per cell (default: 64)
#   CONCURRENCY  : Comma-separated max-concurrency list (default: 1,4,16,32,64)
#   INPUT_LEN    : Prompt tokens (default: 2048)
#   OUTPUT_LEN   : Generated tokens (default: 256)
#   MAX_LEN      : --max-model-len (default: 8192)
#   GPU_MEM      : --gpu-memory-utilization (default: 0.90)
#   SERVE_EXTRA  : Extra `vllm serve` flags (e.g. "--enforce-eager")
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/vllm032_env.sh"

VLLM_VENV_DIR="${VLLM_VENV_DIR:-${REPO_ROOT}/.venv-vllm032}"
VLLM="${VLLM_VENV_DIR}/bin/vllm"
if [[ ! -x "${VLLM}" ]]; then
    echo "Error: ${VLLM} not found. Run 'bash scripts/setup_vllm032_env.sh' first (or set VLLM_VENV_DIR)." >&2
    exit 1
fi
MODEL_DIR="${MODEL_DIR:?Please set MODEL_DIR=/path/to/model}"
GPU="${GPU:-0}"
PORT="${PORT:-$((8910 + GPU))}"
PROMPTS="${PROMPTS:-64}"
CONCURRENCY="${CONCURRENCY:-1,4,16,32,64}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM="${GPU_MEM:-0.90}"
SERVE_EXTRA="${SERVE_EXTRA:-}"
TAG="${1:-$(basename "${MODEL_DIR}")}"
OUT_DIR="results/throughput/${TAG}"
mkdir -p "${OUT_DIR}"

SERVER_LOG="${OUT_DIR}/server_in${INPUT_LEN}_out${OUTPUT_LEN}.log"
echo "[throughput] starting vllm serve ${MODEL_DIR} on GPU ${GPU} port ${PORT} -> ${SERVER_LOG}"
CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM}" serve "${MODEL_DIR}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --no-enable-prefix-caching \
    --disable-uvicorn-access-log \
    ${SERVE_EXTRA} >"${SERVER_LOG}" 2>&1 &
SPID=$!
trap 'kill ${SPID} 2>/dev/null || true; wait ${SPID} 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
        break
    fi
    if ! kill -0 "${SPID}" 2>/dev/null; then
        echo "[throughput] server died; tail of ${SERVER_LOG}:" >&2
        tail -n 50 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null

grep -E "Available KV cache memory|KV cache size|Maximum concurrency|vllm_lrkv" "${SERVER_LOG}" \
    | tee "${OUT_DIR}/kv_cache_in${INPUT_LEN}_out${OUTPUT_LEN}.txt" || true

IFS=',' read -ra CONCS <<< "${CONCURRENCY}"
for C in "${CONCS[@]}"; do
    N=$(( PROMPTS < C ? C : PROMPTS ))
    RES="${OUT_DIR}/in${INPUT_LEN}_out${OUTPUT_LEN}_c${C}.json"
    echo "[throughput] ${TAG}  in=${INPUT_LEN} out=${OUTPUT_LEN}  concurrency=${C}  prompts=${N}"
    "${VLLM}" bench serve \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "${MODEL_DIR}" \
        --dataset-name random \
        --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --num-prompts "${N}" \
        --max-concurrency "${C}" \
        --request-rate inf \
        --ignore-eos \
        --num-warmups 2 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-filename "${RES}"
done

