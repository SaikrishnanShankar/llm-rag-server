.PHONY: help up down ingest serve eval track monitor logs clean

COMPOSE = docker-compose
API_URL  = http://localhost:8000
ENV_FILE = .env

help:
	@echo ""
	@echo "  Production LLM RAG Server — make targets"
	@echo "  ─────────────────────────────────────────"
	@echo "  make up       → start all containers (postgres, mlflow, prometheus, grafana, api)"
	@echo "  make down     → stop and remove containers"
	@echo "  make ingest   → load sample docs into pgvector"
	@echo "  make serve    → start FastAPI in dev mode (hot-reload)"
	@echo "  make eval     → run RAGAS evals and generate report"
	@echo "  make track    → open MLflow UI in browser"
	@echo "  make monitor  → open Grafana dashboard in browser"
	@echo "  make logs     → tail API container logs"
	@echo "  make clean    → remove all Docker volumes (⚠ deletes data)"
	@echo ""

# ─── Docker ──────────────────────────────────────────────────────────────────
up:
	@cp -n $(ENV_FILE).example $(ENV_FILE) 2>/dev/null || true
	$(COMPOSE) up -d --build
	@echo "Services started. API: $(API_URL)  MLflow: http://localhost:5000  Grafana: http://localhost:3000"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

clean:
	$(COMPOSE) down -v
	@echo "All volumes removed."

# ─── Dev (local, no Docker for API) ─────────────────────────────────────────
serve:
	@cp -n $(ENV_FILE).example $(ENV_FILE) 2>/dev/null || true
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

# ─── Ingestion ───────────────────────────────────────────────────────────────
ingest:
	python -m src.retrieval.ingest --docs-dir data/sample_docs --strategy sentence
	python -m src.retrieval.ingest --docs-dir data/sample_docs --strategy fixed
	python -m src.retrieval.ingest --docs-dir data/sample_docs --strategy semantic
	@echo "Ingestion complete. Docs indexed with all 3 chunking strategies."

# ─── Evals ───────────────────────────────────────────────────────────────────
eval:
	python -m src.evals.run_evals
	@echo "Eval complete. See eval/results/eval_results.json and eval/results/report.md"

# ─── MLflow ──────────────────────────────────────────────────────────────────
track:
	@open http://localhost:5001 2>/dev/null || xdg-open http://localhost:5001

# ─── Grafana ─────────────────────────────────────────────────────────────────
monitor:
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000
	@echo "Grafana credentials: admin / admin"
