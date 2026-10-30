# BELT: Behavior-Aligned Low-Rank KV Cache Compression for Qwen3.8-27B

📄 [**Paper (arXiv:2610.17966)**](https://arxiv.org/abs/2610.17966) | 📦 [**Release Artifacts (`v0.1.0`)**](https://github.com/liuyaoyang04/BELT/releases/tag/v0.1.0) | 📜 [**License: Apache-2.0**](LICENSE)

Official **Qwen3.8-27B** implementation of **BELT** (*BEhavior-Aligned Low-Rank KV Cache Compression with Learned Top-k Rank Allocation*), paired with **`vllm_lrkv`**, an out-of-tree **vLLM 0.32.0** plugin featuring a BF16 paged latent KV cache and a single-pass **NVIDIA Hopper (SM90) CuTe DSL** decode kernel.

- **Lossless 50% KV Cache Compression**: Preserves **89.39%** (`+0.00 pp`) on GPQA Diamond while expanding single-GPU KV cache capacity by **1.95×** and accelerating end-to-end reasoning throughput by **1.63×**.
- **Lightweight Factor Checkpoints**: Released/exported low-rank factors are only **~192 MB** (K/V factors of the 16 full-attention layers only) and merge with local `Qwen3.8-27B` in under a minute.
- **Drop-In Hopper `vLLM 0.32.0` Plugin**: Supports `vllm serve`, `vllm.LLM`, FULL CUDA graphs, and chunked FlashAttention prefill out of the box.

---

## Benchmark Results (`Qwen3.8-27B`)

Evaluated on **GPQA Diamond** (`198` questions, `avg@1`, thinking mode enabled with official `Qwen3.8-27B` sampling: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `max_model_len=65536`) using `vLLM 0.32.0` (`BF16`, `gpu_memory_utilization=0.90`, `max_num_seqs=128`) on **2×NVIDIA H800 80GB** (`DP=2`):

| Method | KV Cache Budget | GPQA Diamond (`avg@1`) | KV Cache Capacity / GPU | Mean Output Throughput (`2×H800`) | Total Eval Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full KV** (Stock `vLLM 0.32.0`, Dense BF16) | 100% | **89.39%** (`177 / 198`) | 271,183 tokens (`1.00×`) | ~720 tok/s (`1.00×`) | 65.9 min |
| **BELT `c50`** (50% Compressed, BF16 Latent + Hopper CuTe) | **50%** | **89.39%** (`177 / 198`, `+0.00 pp`) | **528,756 tokens (`1.95×`)** | **~1,171 tok/s (`1.63×`, peak `1,422 tok/s`)** | **43.7 min (`1.51×` faster)** |

---

## Requirements

- **OS & Python**: Linux (`x86_64`), Python 3.10 – 3.12
- **GPU**:
  - **Latent Inference (`vllm_lrkv`)**: **NVIDIA Hopper (SM90)** GPU (H100 / H800 / H20)
  - **Offline Compression (`lrkv`)**: NVIDIA Hopper or Ampere 80 GB GPUs (1–8 GPUs for basis fitting; ≥4×80 GB GPUs for 27B 8k-context rank allocation training)
- **CUDA & Driver**:
  - **Latent Inference (`vLLM 0.32.0+cu129`)**: **CUDA Toolkit `>= 12.9`** (`nvcc >= 12.9` via `CUDA_HOME` or `PATH`, required by DeepGEMM / FlashInfer JIT) and **NVIDIA Driver `>= 575`** (or Driver `>= 535` with CUDA 12.9 forward-compatibility libraries in `/usr/local/cuda-12.9/compat` or `CUDA_COMPAT_DIR`, auto-enabled by `scripts/vllm032_env.sh`)
  - **Offline Compression (`lrkv`)**: CUDA `>= 12.1`
- **Base Weights**: Official unquantized [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) HuggingFace checkpoint

---

## Released Artifacts

Pre-computed low-rank factors and the 5-category `Qwen3.8-27B` calibration corpus are hosted on [**GitHub Releases (`v0.1.0`)**](https://github.com/liuyaoyang04/BELT/releases/tag/v0.1.0) and can be fetched with SHA-256 verification via `scripts/download_artifacts.sh`:

| Artifact | Size | Local Path | Direct Download |
| :--- | :---: | :--- | :--- |
| **`c50` Low-Rank Factors** (50% KV budget, 16 full-attention layers) | 192 MB | `checkpoints/qwen3_8_27b_docs_L8192_c50_ours.pt` | [`qwen3_8_27b_docs_L8192_c50_ours.pt`](https://github.com/liuyaoyang04/BELT/releases/download/v0.1.0/qwen3_8_27b_docs_L8192_c50_ours.pt) |
| **Calibration Corpus** (5,450 `Qwen3.8-27B` reasoning traces across 5 domains) | 185 MB | `data/calib/docs.jsonl` | [`docs.jsonl`](https://github.com/liuyaoyang04/BELT/releases/download/v0.1.0/docs.jsonl) |

```bash
# Download c50 low-rank factors only (for Part 1 inference)
bash scripts/download_artifacts.sh

# Or download both c50 factors and calibration docs.jsonl
bash scripts/download_artifacts.sh --all
```

---

## Part 1: Direct Inference (Using Low-Rank Factors)

If you already have a low-rank factor checkpoint (`.pt`, e.g., pre-computed `c50` factors or one exported from [Part 2](#part-2-generating-low-rank-factors-from-scratch)), you only need the **`vLLM 0.32.0` serving environment** to export and run inference.

### 1. Install Serving Environment

```bash
bash scripts/setup_vllm032_env.sh
source scripts/vllm032_env.sh
```

### 2. Build the Latent Model Directory

Download the pre-computed `c50` factors (if not already present) and combine them with your local `Qwen3.8-27B` directory into a `vllm_lrkv` latent model directory (~45 seconds on CPU):

```bash
bash scripts/download_artifacts.sh

.venv-vllm032/bin/python -m lrkv.runtime.export_latent \
  --model /path/to/Qwen3.8-27B \
  --checkpoint checkpoints/qwen3_8_27b_docs_L8192_c50_ours.pt \
  --output models/qwen3_8_27b_c50_latent
```

### 3. Run Inference

#### Option A: OpenAI-Compatible Server (`vllm serve`)

```bash
source scripts/vllm032_env.sh
CUDA_VISIBLE_DEVICES=0 MAX_MODEL_LEN=32768 \
  bash scripts/serve_vllm.sh models/qwen3_8_27b_c50_latent 8000
```

> Latent models run at `TP=1` with `--language-model-only` (configured automatically by `scripts/serve_vllm.sh`). Across multiple GPUs, launch one replica per GPU for data-parallel serving.

#### Option B: Offline Batched Inference in Python

```python
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        model="models/qwen3_8_27b_c50_latent",
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=32768,
        max_num_seqs=256,
        gpu_memory_utilization=0.90,
    )
    outputs = llm.generate(
        ["Explain how low-rank latent KV caches reduce memory bandwidth during decoding:"],
        SamplingParams(temperature=0.0, max_tokens=256),
    )
    print(outputs[0].outputs[0].text)
```

#### Option C: Throughput & KV Cache Capacity Benchmark

```bash
# 50% compressed latent model
GPU=0 MODEL_DIR=models/qwen3_8_27b_c50_latent \
  INPUT_LEN=2048 OUTPUT_LEN=256 MAX_LEN=8192 CONCURRENCY="1,4,16,32,64" \
  bash scripts/vllm032_throughput.sh qwen3_8_27b_c50_latent

# Uncompressed Full-KV baseline
GPU=0 MODEL_DIR=/path/to/Qwen3.8-27B \
  INPUT_LEN=2048 OUTPUT_LEN=256 MAX_LEN=8192 CONCURRENCY="1,4,16,32,64" \
  bash scripts/vllm032_throughput.sh qwen3_8_27b_fullkv
```

---

## Part 2: Generating Low-Rank Factors from Scratch

To reproduce the low-rank factor checkpoint (`.pt`) from an uncompressed `Qwen3.8-27B` model, set up the compression environment and run the BELT pipeline.

### 1. Install Compression Environment

```bash
bash scripts/setup_env.sh
```

### 2. Prepare Calibration Data

To use the released calibration corpus directly, download `data/calib/docs.jsonl` and skip to Step 3:

```bash
bash scripts/download_artifacts.sh --calib
```

Otherwise, to regenerate `docs.jsonl` from scratch, download the pinned 5-category prompt sources and generate `Qwen3.8-27B` calibration traces:

```bash
# 1) Download & SHA-256 verify prompt shards
bash scripts/download_calib_sources.sh

# 2) Build deterministic 5-category prompt set
.venv/bin/python scripts/calib.py requests \
  --out data/calib/requests.jsonl \
  --heldout data/calib/heldout.jsonl

# 3) Generate reasoning traces against one or more dense Qwen3.8-27B vLLM endpoints
#    (scripts/serve_vllm.sh uses .venv-vllm032 from Part 1 — run 'bash scripts/setup_vllm032_env.sh' first,
#     then launch in another terminal: CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh /path/to/Qwen3.8-27B 8000;
#     or point --servers to any existing OpenAI-compatible Qwen3.8-27B endpoint)
.venv/bin/python scripts/calib.py generate \
  --requests data/calib/requests.jsonl \
  --servers http://127.0.0.1:8000 \
  --out data/calib/responses.jsonl

# 4) Format full chat documents
.venv/bin/python scripts/calib.py texts \
  --responses data/calib/responses.jsonl \
  --requests data/calib/requests.jsonl \
  --tokenizer /path/to/Qwen3.8-27B \
  --out data/calib/docs.jsonl
```

### 3. Run BELT Compression (One Command)

`scripts/run_compression.sh` packs the calibration tokens, fits full-rank behavior-aligned K/V bases, learns the global rank allocation under `--retain-ratio 0.50` (`c50`), exports the `.pt` factor checkpoint under `checkpoints/`, and exports the latent vLLM directory under `models/latent_exports/`:

```bash
MODEL=/path/to/Qwen3.8-27B \
BUDGET_RATIOS="0.50" \
TEACHER_GPU=0 WORKER_GPUS="0 1 2 3" \
TRAIN_GPUS="0,1,2,3" STUDENT_GPUS=3 TEACHER_GPUS=1 \
bash scripts/run_compression.sh
```

---

## Citation

```bibtex
@article{liu2026belt,
  title   = {BELT: BEhavior-Aligned Low-Rank KV Cache Compression with Learned Top-k Rank Allocation},
  author  = {Liu, Yaoyang and Shen, Shuwei and Tang, Jincheng and Liu, Jinguang and Xue, Jinbao and Yang, Hailong and Zhang, Ke and Wu, Yinjun},
  journal = {arXiv preprint arXiv:2610.17966},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.17966}
}
```

---

## License

This project is released under the [Apache License 2.0](LICENSE).
