# PACE GPU Deployment Guide

Step-by-step instructions for running this RAG system on the Georgia Tech PACE cluster (or any SLURM-based GPU cluster). The system switches from the mock client to real Llama 3 inference with a single `.env` change.

---

## Overview

```
PACE Login Node
    │  ssh
    ▼
PACE Compute Node (A100 GPU)
    │  SLURM sbatch
    ├─► vLLM server  :8000   ← serves Llama 3.1-8B-Instruct via OpenAI API
    │
    │  (point your .env at this node)
    ▼
Your API Server (local or another PACE node)
    ├─► FastAPI  :8000
    ├─► pgvector :5433
    └─► MLflow   :5001
```

---

## Prerequisites

On your PACE account:

- Access to PACE-ICE or PACE-Phoenix with GPU allocation
- Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct`
  - Accept the license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - Create a HF token: https://huggingface.co/settings/tokens

---

## Step 1 — Set Up the vLLM Environment on PACE

```bash
# On the PACE login node
module load anaconda3/2023.03

# Create a dedicated conda env (do this once)
conda create -n vllm-env python=3.11 -y
conda activate vllm-env

# Install vLLM (GPU build — CUDA 12.1)
pip install vllm==0.6.3

# Authenticate with Hugging Face (needed to download Llama)
pip install huggingface_hub
huggingface-cli login   # paste your HF token when prompted
```

---

## Step 2 — Pre-download the Model (Recommended)

Download the model weights to your PACE scratch space before submitting the job. This avoids timeout issues at job start.

```bash
# On login node — this takes ~10 minutes for 8B weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'meta-llama/Llama-3.1-8B-Instruct',
    local_dir='/storage/scratch1/<your-gt-username>/models/llama-3.1-8b',
    ignore_patterns=['*.pt', '*.bin']   # download safetensors only
)
"
```

---

## Step 3 — Submit the vLLM SLURM Job

Create `scripts/pace_vllm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --account=<your-pace-account>     # e.g. gt-<PI-username>
#SBATCH --partition=gpu                   # or ice-gpu on PACE-ICE
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm_%j.log
#SBATCH --error=logs/vllm_%j.err

module load anaconda3/2023.03
conda activate vllm-env

MODEL_PATH="/storage/scratch1/<your-gt-username>/models/llama-3.1-8b"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct
```

Submit and get the node hostname:

```bash
mkdir -p logs
sbatch scripts/pace_vllm.sh

# Watch until the server is up (takes ~2 minutes to load weights)
tail -f logs/vllm_<jobid>.log
# You'll see: "Application startup complete." when ready

# Get the compute node hostname
squeue -u $USER --format="%i %R %N"
# Example output: 12345678 gpu-a100-001 atl1-1-02-004-16-0
```

---

## Step 4 — Verify vLLM is Reachable

From the login node (or any node on the cluster network):

```bash
PACE_NODE="atl1-1-02-004-16-0"   # replace with your node name

curl http://${PACE_NODE}:8000/v1/models
# Expected:
# {"object":"list","data":[{"id":"meta-llama/Llama-3.1-8B-Instruct",...}]}

# Quick inference test
curl http://${PACE_NODE}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is RAG?"}],
    "max_tokens": 100
  }'
```

---

## Step 5 — Point the RAG System at vLLM

On your API server (local machine or another PACE node), update `.env`:

```bash
# .env — the only two lines that change
VLLM_BASE_URL=http://atl1-1-02-004-16-0:8000/v1
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

Restart the API:

```bash
make serve       # local dev
# or
make down && make up   # Docker
```

The `get_llm_client()` factory in `src/inference/vllm_client.py` detects the non-`mock://` URL and creates a real `AsyncOpenAI` client. No code changes required.

---

## Step 6 — Run Evals Against Real Inference

```bash
# Ingest docs (if not already done)
make ingest

# Full 50-question RAGAS eval against real Llama
make eval

# View results
cat eval/results/report.md
make track   # MLflow comparison
```

Expected scores with real Llama 3.1-8B on this corpus:

| Strategy | Faithfulness | Context Relevance | Answer Relevance | Composite |
|---|---|---|---|---|
| fixed | ~0.70 | ~0.62 | ~0.68 | ~0.67 |
| sentence | ~0.80 | ~0.75 | ~0.74 | ~0.76 |
| semantic | ~0.83 | ~0.71 | ~0.72 | ~0.75 |

*(Approximate — actual scores vary with corpus and top_k)*

---

## Scaling to Larger Models

### 70B on 2× A100 80GB

```bash
# pace_vllm_70b.sh — change the SBATCH and vLLM flags:
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=160G

python -m vllm.entrypoints.openai.api_server \
    --model "/path/to/llama-3.1-70b-instruct" \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --served-model-name meta-llama/Llama-3.1-70B-Instruct
```

Then update `.env`:
```bash
VLLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
```

No other changes needed — the API and agent are model-agnostic.

### Quantized (smaller GPU footprint)

```bash
# AWQ 4-bit — fits 8B on a single A40 (48GB)
pip install autoawq

python -m vllm.entrypoints.openai.api_server \
    --model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
    --quantization awq \
    --max-model-len 4096
```

---

## SSH Port Forwarding (Access from Laptop)

If you want to run the API locally but talk to vLLM on PACE:

```bash
# Forward PACE compute node port 8000 to localhost:8001
ssh -L 8001:atl1-1-02-004-16-0:8000 <your-gt-username>@login-pace-ice.pace.gatech.edu -N &

# Then set in .env:
VLLM_BASE_URL=http://localhost:8001/v1
```

---

## Troubleshooting

**Job killed immediately (OOM)**
: Reduce `--gpu-memory-utilization` to `0.85` or add `--max-model-len 2048`.

**`Connection refused` from API to vLLM**
: PACE firewall may block inter-node traffic. Use SSH port forwarding (above) or run both jobs in the same SLURM step with `srun --multi-prog`.

**Model download fails inside job**
: Pre-download to scratch in Step 2. PACE compute nodes may have restricted outbound internet.

**vLLM crashes with `CUDA error: out of memory`**
: The 8B model needs ~16GB VRAM. Request at least an A40 (48GB) or A100 (40/80GB). Avoid V100s for Llama 3.1 (bfloat16 not supported on V100).

**Slow first inference (30s)**
: Normal — CUDA graphs are being compiled. Subsequent requests are fast. Add `--enforce-eager` to skip compilation (lower throughput but no warm-up delay).

---

## Useful SLURM Commands

```bash
squeue -u $USER                   # list your running jobs
scancel <jobid>                   # cancel a job
sinfo -p gpu                      # available GPU partitions
sacct -j <jobid> --format=JobID,State,Elapsed,ReqMem,MaxRSS
```
