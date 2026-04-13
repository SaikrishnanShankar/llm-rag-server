# Production LLM RAG Server

[![CI](https://github.com/SaikrishnanShankar/llm-rag-server/actions/workflows/ci.yml/badge.svg)](https://github.com/SaikrishnanShankar/llm-rag-server/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade Retrieval-Augmented Generation (RAG) system built as an end-to-end ML engineering portfolio project. Covers the full stack: vector retrieval, stateful agent orchestration, LLM serving, automated evaluation, and observability — runnable locally with a mock LLM and deployable to a GPU cluster (PACE / any SLURM cluster) with a single `.env` change.

---

## What This Demonstrates

| Skill Area | Implementation |
|---|---|
| **LLM Serving** | vLLM OpenAI-compatible endpoint; mock client for local dev with no GPU |
| **RAG pipeline** | LangGraph stateful agent: classify → retrieve → rerank → generate → format |
| **Vector search** | pgvector with HNSW index; 3 chunking strategies benchmarked via RAGAS |
| **Evaluation** | RAGAS (faithfulness, context relevance, answer relevance) over 50 Q&A pairs |
| **Experiment tracking** | MLflow: per-request metrics + eval run comparison across chunking strategies |
| **Observability** | Prometheus custom metrics + Grafana dashboard; LLM-tuned histogram buckets |
| **API design** | FastAPI async endpoints, SSE streaming, file upload, background tasks |
| **CI/CD** | GitHub Actions: lint → unit tests → pgvector integration → eval smoke test |

---

## Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────┐
│  FastAPI  :8000                                     │
│  /query  /query/stream  /ingest  /eval  /health     │
│  Prometheus metrics on :8001                        │
└────────────────────┬───────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────┐
│  LangGraph Agent                                    │
│                                                     │
│  classify_query ──(needs retrieval?)──► retrieve   │
│       │                                    │        │
│       │ (no: direct answer)                ▼        │
│       │                            rerank_chunks    │
│       │                         (cross-encoder or   │
│       │                          cosine fallback)   │
│       └────────────────────────────────────┘        │
│                                            │        │
│                                    generate_answer  │
│                                            │        │
│                                    format_response  │
│                                  (dedup by doc_id)  │
└────────────────────────────────────────────────────┘
          │                  │                │
          ▼                  ▼                ▼
   pgvector :5433       vLLM / Mock      MLflow :5001
   (HNSW index)     (OpenAI-compat.)   (experiment UI)
          │
   Prometheus :9090
   Grafana    :3000
```

### Key Design Decisions

| Component | Choice | Why |
|---|---|---|
| Vector DB | pgvector + HNSW | Single-service deployment; HNSW works at any dataset size (unlike IVFFlat which needs ≥10k rows) |
| Chunking | 3 strategies (fixed / sentence / semantic) | Benchmarked via RAGAS to find the optimal strategy for your corpus |
| Agent | LangGraph | Stateful graph; conditional routing avoids retrieval for direct queries; extensible to multi-hop RAG |
| LLM Client | Factory pattern → AsyncOpenAI or MockLLMClient | Single `.env` line swap between mock and real vLLM — zero code changes |
| Mock client | Context-grounded synthesis | Extracts sentences from retrieved chunks by keyword overlap, so RAGAS faithfulness scores are meaningful locally |
| Evals | RAGAS heuristic mode | Runs without an LLM judge — usable offline and in CI |
| MLflow | Per-request + per-eval runs | Strategy comparison in the MLflow UI; composite score picks the winner |

---

## Quick Start (No GPU Required)

### Prerequisites

- Docker + Docker Compose
- Python 3.11+

### 1. Clone and configure

```bash
git clone https://github.com/SaikrishnanShankar/llm-rag-server.git
cd llm-rag-server
cp .env.example .env          # defaults to mock LLM — no GPU needed
```

### 2. Start services

```bash
make up
```

Starts: PostgreSQL/pgvector (`:5433`), FastAPI API (`:8000`), MLflow (`:5001`), Prometheus (`:9090`), Grafana (`:3000`).

### 3. Ingest sample documents

```bash
make ingest
# Indexes 3 sample docs (ML basics, LLM/RAG, MLOps) with all 3 chunking strategies
```

### 4. Query the system

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Retrieval-Augmented Generation?", "top_k": 5}'
```

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
    "usage": {"total_tokens": 487, "tokens_per_second": 210}
  }
}
```

### 5. Run RAGAS evaluations

```bash
make eval
# Evaluates all 3 strategies across 50 questions
# Output: eval/results/report.md + eval/results/eval_results.json
```

### 6. Explore results

```bash
make track    # MLflow UI at http://localhost:5001
make monitor  # Grafana at http://localhost:3000  (admin / admin)
```

---

## Eval Results (Mock LLM, 50 Questions)

Scores from a real local run using the context-grounded mock client:

| Strategy | Faithfulness | Context Relevance | Answer Relevance | **Composite** |
|---|---|---|---|---|
| fixed | 0.309 | 0.180 | 0.063 | 0.184 |
| semantic | 0.476 | 0.104 | 0.050 | 0.210 |
| **sentence** | 0.376 | **0.285** | 0.067 | **0.243** |

**Winner: `sentence`** — highest composite score, best context relevance (retriever returns tightly scoped passages). Semantic chunking wins on faithfulness (answers stay grounded) but suffers on context relevance at small corpus sizes.

> Scores reflect the mock LLM's heuristic synthesis. On a real vLLM + Llama-3.1-8B-Instruct run you can expect faithfulness ~0.75–0.85 and answer relevance ~0.70–0.80.

---

## Project Structure

```
llm-rag-server/
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI app + lifespan (schema init, metrics)
│   │   └── routes.py          # /query, /query/stream, /ingest, /ingest/file, /eval
│   ├── agent/
│   │   ├── graph.py           # LangGraph compile + run_agent() entry point
│   │   ├── nodes.py           # 5 async nodes: classify → retrieve → rerank → generate → format
│   │   └── state.py           # AgentState TypedDict
│   ├── retrieval/
│   │   ├── chunking.py        # fixed, sentence, semantic strategies
│   │   ├── embeddings.py      # sentence-transformers singleton (all-MiniLM-L6-v2)
│   │   ├── vectorstore.py     # pgvector CRUD + HNSW similarity search
│   │   └── ingest.py          # CLI pipeline: txt + PDF → chunks → pgvector
│   ├── inference/
│   │   ├── vllm_client.py     # factory: mock:// → MockLLMClient, else AsyncOpenAI
│   │   └── mock_client.py     # context-grounded mock for local RAGAS evals
│   ├── evals/
│   │   ├── run_evals.py       # RAGAS runner with heuristic fallback (no LLM judge)
│   │   ├── dataset.py         # builds HF-compatible dataset by running RAG pipeline
│   │   └── report.py          # markdown strategy comparison report
│   ├── tracking/
│   │   └── mlflow_logger.py   # log_request() + log_eval_run() (fire-and-forget)
│   └── metrics/
│       └── prometheus.py      # metric definitions + start_metrics_server()
├── eval/
│   ├── qa_dataset.json        # 50 Q&A pairs (ML basics, LLM/RAG, MLOps)
│   └── results/               # generated: eval_results.json + report.md
├── data/
│   └── sample_docs/           # 3 sample documents for ingestion demo
├── tests/
│   ├── test_chunking.py       # 16 unit tests (all strategies + edge cases)
│   ├── test_vectorstore.py    # 7 integration tests (skipped without DB)
│   └── test_agent.py          # 13 unit tests (routing, reranking, format)
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana_dashboard.json
│   └── grafana/provisioning/  # auto-provisions datasource + dashboard
├── scripts/
│   └── init_db.sql            # pgvector schema + HNSW index
├── .github/workflows/ci.yml   # lint → unit → integration → eval smoke
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
└── .env.example
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/query` | RAG query — JSON response with answer + sources |
| `POST` | `/api/v1/query/stream` | Streaming query — Server-Sent Events |
| `POST` | `/api/v1/ingest` | Ingest raw text directly |
| `POST` | `/api/v1/ingest/file` | Upload `.txt` or `.pdf` file |
| `POST` | `/api/v1/eval` | Trigger background RAGAS eval (returns immediately) |
| `GET`  | `/api/v1/health` | Health check |
| `GET`  | `/docs` | Swagger UI |
| `GET`  | `:8001/metrics` | Prometheus scrape endpoint |

**Query options:**
```json
{
  "query": "string (required)",
  "top_k": 5,
  "chunk_strategy": "sentence",
  "stream": false
}
```

`chunk_strategy` filters retrieval to chunks ingested with that strategy. Omit to search across all strategies.

---

## Prometheus Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `rag_request_latency_seconds` | Histogram | `chunk_strategy` | End-to-end latency |
| `rag_retrieval_latency_seconds` | Histogram | `chunk_strategy` | pgvector search time |
| `rag_tokens_per_second` | Gauge | `model` | LLM throughput |
| `rag_active_requests` | Gauge | — | In-flight requests |
| `rag_requests_total` | Counter | `chunk_strategy`, `status` | Request count by outcome |
| `rag_eval_faithfulness_score` | Gauge | `chunk_strategy` | Latest RAGAS faithfulness |
| `rag_eval_context_relevance_score` | Gauge | `chunk_strategy` | Latest RAGAS context relevance |
| `rag_eval_answer_relevance_score` | Gauge | `chunk_strategy` | Latest RAGAS answer relevance |
| `rag_ingestion_chunks_total` | Counter | `chunk_strategy` | Chunks inserted |

Buckets are LLM-tuned: `[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]` seconds.

---

## MLflow Experiment Tracking

Every request and every eval run is logged as an MLflow run under the `rag-chunking-comparison` experiment.

**Per-request:** `chunk_strategy`, `top_k`, `model`, `needs_retrieval`, `latency_seconds`, `retrieval_latency_seconds`, `tokens_used`, `tokens_per_second`, `chunks_retrieved`

**Per-eval:** `chunk_strategy`, `num_questions`, `faithfulness`, `context_relevance`, `answer_relevance`, `composite_score`

Compare strategies in the MLflow UI — select multiple runs and click **Compare** to see a side-by-side metric table and parallel coordinates plot.

```bash
make track   # opens http://localhost:5001
```

---

## PACE / GPU Deployment

See [docs/pace_deployment.md](docs/pace_deployment.md) for the complete step-by-step guide.

**TL;DR:** One `.env` change and the system switches from mock to real Llama inference:

```bash
# .env
VLLM_BASE_URL=http://<pace-node-hostname>:8000/v1
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

The `get_llm_client()` factory detects the non-mock URL and creates a real `AsyncOpenAI` client pointing at vLLM's OpenAI-compatible endpoint — no code changes.

---

## Running Tests

```bash
# Unit tests only (no DB required)
pytest tests/test_chunking.py tests/test_agent.py -v

# Integration tests (requires pgvector running)
pytest tests/test_vectorstore.py -v

# Full suite
pytest -v

# Lint
ruff check src/ tests/
```

CI runs all three on every push — see `.github/workflows/ci.yml`.

---

## Local Dev Without Docker

```bash
# Start only the DB (Docker for just postgres)
docker run -d -p 5433:5432 \
  -e POSTGRES_DB=ragdb -e POSTGRES_USER=raguser -e POSTGRES_PASSWORD=ragpassword \
  pgvector/pgvector:pg16

# Install deps and configure
pip install -r requirements.txt
cp .env.example .env

# Start API (port 8000, hot-reload)
make serve

# Ingest + eval
make ingest
make eval
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Serving | vLLM (OpenAI-compatible) · Mock client for local dev |
| Vector Store | PostgreSQL 16 + pgvector · HNSW index (cosine distance) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim, CPU/MPS) |
| Agent | LangGraph (stateful graph, conditional routing) |
| Reranker | sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2` (graceful fallback) |
| API | FastAPI · async · SSE streaming · BackgroundTasks |
| Evals | RAGAS (heuristic mode — no LLM judge required) |
| Experiment Tracking | MLflow 2.18 |
| Metrics | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| CI | GitHub Actions (3-job pipeline) |
