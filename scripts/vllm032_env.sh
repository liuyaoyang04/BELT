#!/usr/bin/env bash
# Source before running vLLM 0.32.0 with the vllm_lrkv plugin:
#   source scripts/vllm032_env.sh
#
# Configures repo-local cache directories (to avoid filling /tmp or ~/.cache), ensures
# CUDA_HOME points to a CUDA 12.9+ toolkit when /usr/local/cuda-12.9 is present (needed
# by DeepGEMM / FlashInfer JIT), and prepends CUDA 12.9 forward-compatibility libraries
# if CUDA_COMPAT_DIR is set (or on hosts with an older driver < 575).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${CUDA_HOME:-}" && -x "/usr/local/cuda-12.9/bin/nvcc" ]]; then
    export CUDA_HOME="/usr/local/cuda-12.9"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/bin" ]]; then
    case ":${PATH:-}:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) export PATH="${CUDA_HOME}/bin:${PATH:-}" ;;
    esac
fi

if [[ -n "${CUDA_COMPAT_DIR:-}" && -f "${CUDA_COMPAT_DIR}/libcuda.so.1" ]]; then
    export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
elif [[ -f "/usr/local/cuda-12.9/compat/libcuda.so.1" ]]; then
    # Automatically enable CUDA 12.9 forward-compat if host driver is older than 575
    HOST_DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true)"
    if [[ -n "${HOST_DRIVER_MAJOR}" && "${HOST_DRIVER_MAJOR}" -lt 575 ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}"
    fi
fi

CACHE_BASE="${LRKV_CACHE_DIR:-${REPO_ROOT}/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_BASE}/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_BASE}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_BASE}/torchinductor}"
export LRKV_CUTE_CACHE_DIR="${LRKV_CUTE_CACHE_DIR:-${VLLM_CACHE_ROOT}/lrkv_cute}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${LRKV_CUTE_CACHE_DIR}"

