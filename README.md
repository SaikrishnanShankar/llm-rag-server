# Production LLM RAG Server with Evals

A production-grade Retrieval-Augmented Generation (RAG) system demonstrating
end-to-end ML engineering: vector retrieval, agent orchestration, LLM serving,
automated evaluation, and full observability.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│  FastAPI      │  async endpoints: /query, /ingest, /eval
│  (port 8000) │  Prometheus metrics on :8001
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  LangGraph Agent                                      │
│                                                       │
│  classify_query ──(needs retrieval?)──► retrieve     │
│       │                                    │          │
│       │ (no)                               ▼          │
│       │                            rerank_chunks      │
│       │                                    │          │
│       └────────────────────────────────────┘          │
│                                            │          │
│                                    generate_answer    │
│                                            │          │
│                                    format_response    │
└──────────────────────────────────────────┬───────────┘
                                           │
              ┌──────────────┬─────────────┤
              │              │             │
              ▼              ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ pgvector  │  │  vLLM    │  │  MLflow  │
        │ (pg:5432) │  │ (mock or │  │ (port    │
        │           │  │  PACE)   │  │  5000)   │
        └──────────┘  └──────────┘  └──────────┘
              │
        ┌─────────────────────┐
        │  Prometheus :9090   │
        │  Grafana :3000      │
        └─────────────────────┘
```

### Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Vector DB | pgvector on PostgreSQL | Single-service deployment; IVFFlat ANN; SQL joins for metadata filtering |
| Chunking | 3 strategies (fixed/sentence/semantic) | Enables RAGAS comparison to find optimal strategy |
| Agent | LangGraph | Stateful graph supports conditional routing + future multi-hop RAG |
| LLM Client | OpenAI-compatible → vLLM | Single env-var swap between mock (local) and GPU (PACE) |
| Evals | RAGAS (faithfulness, context_relevance, answer_relevance) | Industry standard; computable without human annotation |
| Tracking | MLflow | Experiment comparison across chunking strategies; model registry |

---

## Quick Start (Local Dev — No GPU Required)

### Prerequisites
- Docker + docker-compose
- Python 3.11+ (for local dev without Docker)

### 1. Clone and configure

```bash
git clone <repo>
cd llm-serving-evals
cp .env.example .env          # default uses mock LLM — no GPU needed
```

### 2. Start all services

```bash
make up
# Starts: PostgreSQL/pgvector, FastAPI, MLflow, Prometheus, Grafana
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000  (admin/admin)
```

### 3. Ingest sample documents

```bash
make ingest
# Ingests 3 sample docs with all 3 chunking strategies into pgvector
```

### 4. Query the system

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Retrieval-Augmented Generation?", "top_k": 5}'
```

**Response:**
```json
{
  "query": "What is Retrieval-Augmented Generation?",
  "answer": "Based on the available information: RAG combines LLMs with...",
  "sources": [
    {
      "doc_id": "llm_and_rag_a3f2b1c4",
      "filename": "llm_and_rag.txt",
      "similarity_score": 0.92,
      "content_preview": "RAG combines LLMs with information retrieval..."
    }
  ],
  "metadata": {
    "needs_retrieval": true,
    "chunks_retrieved": 5,
    "retrieval_latency_seconds": 0.012,
    "usage": { "total_tokens": 487, "tokens_per_second": 210 }
  }
}
```

### 5. Run RAGAS evaluations

```bash
make eval
# Evaluates all 3 chunking strategies on 50 questions
# Output: eval/results/eval_results.json + eval/results/report.md
```

### 6. View experiment comparison

```bash
make track    # opens MLflow at localhost:5000
make monitor  # opens Grafana at localhost:3000
```

---

## Project Structure

```
llm-serving-evals/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app + lifespan (schema init, metrics server)
│   │   └── routes.py        # /query, /query/stream, /ingest, /ingest/file, /eval
│   ├── agent/
│   │   ├── graph.py         # LangGraph state machine (compile + run)
│   │   ├── nodes.py         # 5 agent nodes: classify→retrieve→rerank→generate→format
│   │   └── state.py         # AgentState TypedDict
│   ├── retrieval/
│   │   ├── chunking.py      # fixed, sentence, semantic chunking strategies
│   │   ├── embeddings.py    # sentence-transformers (all-MiniLM-L6-v2, singleton)
│   │   ├── vectorstore.py   # pgvector CRUD + similarity search
│   │   └── ingest.py        # CLI ingestion pipeline (txt + PDF)
│   ├── inference/
│   │   ├── vllm_client.py   # OpenAI-compatible client + mock switcher
│   │   └── mock_client.py   # context-grounded mock (enables local RAGAS evals)
│   ├── evals/
│   │   ├── run_evals.py     # RAGAS evaluation runner (CLI + async API)
│   │   ├── dataset.py       # Q&A dataset builder (runs RAG pipeline per question)
│   │   └── report.py        # Markdown report generator
│   ├── tracking/
│   │   └── mlflow_logger.py # per-request + eval run logging
│   └── metrics/
│       └── prometheus.py    # metric definitions + metrics server
├── eval/
│   ├── qa_dataset.json      # 50 ML/RAG/MLOps question-answer pairs
│   └── results/             # generated: eval_results.json + report.md
├── data/
│   └── sample_docs/         # 3 sample documents (ML, LLM/RAG, MLOps)
├── monitoring/
│   ├── prometheus.yml       # scrape config (api:8001)
│   ├── grafana_dashboard.json
│   └── grafana/provisioning/
│       ├── datasources/     # auto-provisions Prometheus datasource
│       └── dashboards/      # auto-loads dashboard JSON
├── scripts/
│   └── init_db.sql          # pgvector schema + indexes
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── Makefile
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | RAG query (JSON response) |
| `POST` | `/api/v1/query/stream` | Streaming query (SSE) |
| `POST` | `/api/v1/ingest` | Ingest text directly |
| `POST` | `/api/v1/ingest/file` | Upload .txt or .pdf |
| `POST` | `/api/v1/eval` | Trigger background RAGAS eval |
| `GET`  | `/api/v1/health` | Health check |
| `GET`  | `/docs` | Swagger UI |
| `GET`  | `:<PROMETHEUS_PORT>/metrics` | Prometheus metrics |

### Query Request Options

```json
{
  "query": "string",
  "top_k": 5,
  "chunk_strategy": "sentence",  // fixed | sentence | semantic | null (all)
  "stream": false
}
```

---

## Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `rag_request_latency_seconds` | Histogram | `chunk_strategy` | End-to-end latency |
| `rag_retrieval_latency_seconds` | Histogram | `chunk_strategy` | pgvector search time |
| `rag_tokens_per_second` | Gauge | `model` | LLM throughput |
| `rag_active_requests` | Gauge | — | Concurrent requests |
| `rag_requests_total` | Counter | `chunk_strategy`, `status` | Request count |
| `rag_eval_faithfulness_score` | Gauge | `chunk_strategy` | RAGAS faithfulness |
| `rag_eval_context_relevance_score` | Gauge | `chunk_strategy` | RAGAS context relevance |
| `rag_eval_answer_relevance_score` | Gauge | `chunk_strategy` | RAGAS answer relevance |
| `rag_ingestion_chunks_total` | Counter | `chunk_strategy` | Chunks inserted |

---

## MLflow Experiment Tracking

Each request and each eval run is logged as an MLflow run:

**Per-request params:** `chunk_strategy`, `top_k`, `model`, `needs_retrieval`  
**Per-request metrics:** `latency_seconds`, `retrieval_latency_seconds`, `tokens_used`, `tokens_per_second`, `chunks_retrieved`

**Eval run params:** `chunk_strategy`, `num_questions`, `eval_type`  
**Eval run metrics:** `faithfulness`, `context_relevance`, `answer_relevance`, `composite_score`

Compare strategies in the MLflow UI (Experiments → select runs → Compare):
```bash
make track  # opens http://localhost:5000
```

---

## RAGAS Evaluation

### Running Evals

```bash
# All 3 strategies (50 questions each)
make eval

# Quick test (10 questions)
python -m src.evals.run_evals --limit 10

# Single strategy
python -m src.evals.run_evals --strategies sentence --limit 20
```

### Output Files

```
eval/results/
├── eval_results.json        # summary scores per strategy
├── eval_detailed_<ts>.json  # per-question answers + contexts
└── report.md                # markdown comparison report
```

### Sample Report

```
| Strategy | Faithfulness | Context Relevance | Answer Relevance | Composite |
|----------|-------------|-------------------|------------------|-----------|
| fixed    | 0.712       | 0.681             | 0.643            | 0.679     |
| sentence | 0.798       | 0.743             | 0.721            | 0.754     |
| semantic | 0.831       | 0.779             | 0.748            | 0.786     |
```

---

## Deployment on PACE (Georgia Tech GPU Cluster)

### Step 1: Start vLLM on a PACE GPU node

```bash
# SLURM job (pace_vllm.sh)
#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00

module load anaconda3
conda activate vllm-env

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

### Step 2: Update .env — single line change

```bash
# .env on your API server
VLLM_BASE_URL=http://<pace-node-hostname>:8000/v1
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

That's it. The `get_llm_client()` factory detects the non-mock URL and
creates a real `AsyncOpenAI` client pointing at vLLM's OpenAI-compatible endpoint.

### Step 3: Verify

```bash
curl http://<pace-node>:8000/v1/models
# Should return the loaded model name

make serve  # or docker-compose up api
curl -X POST http://localhost:8000/api/v1/query \
  -d '{"query": "Explain transformer attention"}'
```

### PACE Architecture Notes

- vLLM's **PagedAttention** eliminates KV cache fragmentation → ~3× throughput vs naive serving
- Use `--tensor-parallel-size 2` for 70B models across 2 × A100 80GB GPUs
- For long-running evals on PACE, run `make eval` from a login node pointing to PACE vLLM
- The mock client and real vLLM client share the identical async interface — no code changes

---

## Local Dev Without Docker

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start PostgreSQL with pgvector (Docker just for DB)
docker run -d -p 5432:5432 \
  -e POSTGRES_DB=ragdb -e POSTGRES_USER=raguser -e POSTGRES_PASSWORD=ragpassword \
  pgvector/pgvector:pg16

# 3. Configure
cp .env.example .env  # uses mock LLM by default

# 4. Run API
make serve

# 5. Ingest + eval
make ingest
make eval
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Serving | vLLM (OpenAI-compatible) / Mock for local dev |
| Vector Store | PostgreSQL + pgvector (IVFFlat ANN) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim, CPU) |
| Agent | LangGraph (stateful graph, conditional routing) |
| API | FastAPI (async, SSE streaming) |
| Evals | RAGAS (faithfulness, context_relevance, answer_relevance) |
| Experiment Tracking | MLflow |
| Metrics | Prometheus + Grafana |
| Containerization | Docker + docker-compose |
