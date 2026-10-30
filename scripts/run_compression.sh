#!/usr/bin/env bash
# End-to-end Low-Rank KV Cache Compression & Latent Export Pipeline for Qwen3.8-27B.
#
# Steps:
#   1. Build shared token artifact (lrkv.experiment_data)
#   2. Fit full-rank curvature-aware K/V bases (lrkv.allocate basis)
#   3. Train exact-budget rank allocator via output KD + WSD (lrkv.allocate train)
#   4. Export canonical low-rank factor checkpoint (lrkv.allocate export)
#   5. Export true-latent HF model directory for vllm_lrkv 0.32.0 (lrkv.runtime.export_latent)
#
# Usage:
#   MODEL=/path/to/Qwen3.8-27B bash scripts/run_compression.sh
#
# Configurable environment variables:
#   MODEL          : Path to local Qwen3.8-27B HuggingFace model directory (required)
#   TAG            : Experiment name tag (default: basename of MODEL, lowercase)
#   PYTHON         : Python interpreter for compression env (default: .venv/bin/python)
#   DOCS_SOURCE    : Trace document JSONL path (default: data/calib/docs.jsonl)
#   DOC_BUDGET     : Carry-over token cap per document (default: 16384)
#   SEQ_LEN        : Rank-training sequence length (default: 8192)
#   CALIB_SAMPLES  : Decomposition calibration row count (default: 512)
#   CALIB_LEN      : Decomposition calibration sequence length (default: 8192)
#   BUDGET_RATIOS  : Space-separated KV cache retain ratios (default: "0.50"; e.g. 0.70 -> c30, 0.50 -> c50)
#   LR             : Peak WSD AdamW learning rate in rank units (default: 2.0)
#   TEACHER_GPU    : CUDA device for dense teacher during basis fit (default: 0)
#   WORKER_GPUS    : Space-separated CUDA devices for parallel layer fitting (default: "0")
#   TRAIN_GPUS     : Comma-separated CUDA_VISIBLE_DEVICES for rank training (default: "0,1,2,3")
#   STUDENT_GPUS   : Model-parallel student GPU count per rank, 0 to disable (default: 3)
#   TEACHER_GPUS   : Model-parallel teacher GPU count when STUDENT_GPUS > 0 (default: 1)
#   ROPE_ALIGN     : Set to 1 to enable --rope-align during export_latent (default: 0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:?Please set MODEL=/path/to/Qwen3.8-27B}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL_NAME="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
MODEL_NAME="${MODEL_NAME%_}"
TAG="${TAG:-${MODEL_NAME}}"
DOCS_SOURCE="${DOCS_SOURCE:-data/calib/docs.jsonl}"
DOC_BUDGET="${DOC_BUDGET:-16384}"
SEQ_LEN="${SEQ_LEN:-8192}"
CALIB_SAMPLES="${CALIB_SAMPLES:-512}"
CALIB_LEN="${CALIB_LEN:-8192}"
BUDGET_RATIOS="${BUDGET_RATIOS:-0.50}"
LR="${LR:-2.0}"
TEACHER_GPU="${TEACHER_GPU:-0}"
WORKER_GPUS="${WORKER_GPUS:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
STUDENT_GPUS="${STUDENT_GPUS-3}"
TEACHER_GPUS="${TEACHER_GPUS:-1}"
ROPE_ALIGN="${ROPE_ALIGN:-0}"

EXP_DATA="data/experiment/${TAG}_docs_c${CALIB_SAMPLES}x${CALIB_LEN}_t${SEQ_LEN}.pt"
BASIS="checkpoints/bases/${TAG}_docs_L${SEQ_LEN}_ours.pt"

mkdir -p data/experiment checkpoints/bases results/allocation checkpoints models/latent_exports

# Step 1: Prepare shared calibration & rank-training token artifact
if [[ ! -f "${EXP_DATA}" ]]; then
    echo "=== [1/5] Building experiment token artifact: ${EXP_DATA} ==="
    "${PYTHON}" -m lrkv.experiment_data \
        --model "${MODEL}" \
        --source "${DOCS_SOURCE}" \
        --document-budget "${DOC_BUDGET}" \
        --train-samples 4000 \
        --validation-per-category 2 \
        --sequence-length "${SEQ_LEN}" \
        --calibration-samples "${CALIB_SAMPLES}" \
        --calibration-length "${CALIB_LEN}" \
        --output "${EXP_DATA}"
else
    echo "=== [1/5] Reusing existing token artifact: ${EXP_DATA} ==="
fi

# Step 2: Fit full-rank behavior-aligned bases (kronecker_surrogate K + joint gated behavior_svd V)
if [[ ! -f "${BASIS}" ]]; then
    echo "=== [2/5] Fitting full-rank bases: ${BASIS} ==="
    LAYER_DEV_ARGS=()
    if [[ -n "${WORKER_GPUS}" ]]; then
        read -r -a W_ARR <<< "${WORKER_GPUS}"
        if [[ "${#W_ARR[@]}" -gt 1 || "${W_ARR[0]}" != "${TEACHER_GPU}" ]]; then
            LAYER_DEV_ARGS=(--layer-devices)
            for g in "${W_ARR[@]}"; do
                LAYER_DEV_ARGS+=("cuda:${g}")
            done
        fi
    fi
    "${PYTHON}" -m lrkv.allocate basis \
        --model "${MODEL}" \
        --experiment-data "${EXP_DATA}" \
        --device "cuda:${TEACHER_GPU}" \
        "${LAYER_DEV_ARGS[@]}" \
        --output "${BASIS}"
else
    echo "=== [2/5] Reusing existing basis: ${BASIS} ==="
fi

# Steps 3-5: Train rank allocation, export factor checkpoint, and export latent HF model directory
for RATIO in ${BUDGET_RATIOS}; do
    PCT="$(awk -v r="${RATIO}" 'BEGIN { printf "%d", (1.0 - r) * 100 + 0.5 }')"
    ALLOC="results/allocation/${TAG}_docs_L${SEQ_LEN}_c${PCT}_ours.json"
    CKPT="checkpoints/${TAG}_docs_L${SEQ_LEN}_c${PCT}_ours.pt"
    LATENT_DIR="models/latent_exports/${TAG}_docs_L${SEQ_LEN}_c${PCT}_ours_latent"

    if [[ ! -f "${ALLOC}" ]]; then
        echo "=== [3/5] Training rank allocation at retain-ratio=${RATIO}: ${ALLOC} ==="
        MP_ARGS=()
        if [[ -n "${STUDENT_GPUS}" && "${STUDENT_GPUS}" != "0" ]]; then
            MP_ARGS=(--student-gpus "${STUDENT_GPUS}" --teacher-gpus "${TEACHER_GPUS}")
        fi
        CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" "${PYTHON}" -m lrkv.allocate train \
            --model "${MODEL}" \
            --basis "${BASIS}" \
            --experiment-data "${EXP_DATA}" \
            --retain-ratio "${RATIO}" \
            --learning-rate "${LR}" \
            "${MP_ARGS[@]}" \
            --output "${ALLOC}"
    else
        echo "=== [3/5] Reusing existing rank allocation: ${ALLOC} ==="
    fi

    if [[ ! -f "${CKPT}" ]]; then
        echo "=== [4/5] Exporting factor checkpoint at retain-ratio=${RATIO}: ${CKPT} ==="
        "${PYTHON}" -m lrkv.allocate export \
            --model "${MODEL}" \
            --basis "${BASIS}" \
            --allocation "${ALLOC}" \
            --output "${CKPT}"
    else
        echo "=== [4/5] Reusing existing factor checkpoint: ${CKPT} ==="
    fi

    if [[ ! -f "${LATENT_DIR}/config.json" ]]; then
        echo "=== [5/5] Exporting latent vLLM model directory: ${LATENT_DIR} ==="
        ROPE_ARGS=()
        if [[ "${ROPE_ALIGN}" == "1" ]]; then
            ROPE_ARGS=(--rope-align)
        fi
        "${PYTHON}" -m lrkv.runtime.export_latent \
            --model "${MODEL}" \
            --checkpoint "${CKPT}" \
            --output "${LATENT_DIR}" \
            "${ROPE_ARGS[@]}"
    else
        echo "=== [5/5] Reusing existing latent vLLM model directory: ${LATENT_DIR} ==="
    fi
done

echo "=== Pipeline complete! ==="
