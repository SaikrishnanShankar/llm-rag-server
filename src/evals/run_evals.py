"""
RAGAS evaluation runner.

Runs the full evaluation pipeline for one or more chunking strategies,
computes real RAGAS scores, logs to MLflow, pushes to Prometheus, and
writes eval/results/eval_results.json + eval/results/report.md.

Usage:
    python -m src.evals.run_evals
    python -m src.evals.run_evals --strategies fixed sentence
    python -m src.evals.run_evals --limit 10  # quick test with 10 questions

Design decision: RAGAS uses the LLM to compute metrics (faithfulness is
measured by asking the LLM to verify each claim). For local dev, we use
the mock client which produces context-grounded answers so faithfulness
scores are meaningful (~0.7–0.9) rather than degenerate zeros.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("eval/results")


async def _evaluate_strategy(
    chunk_strategy: str,
    top_k: int = 5,
    question_limit: Optional[int] = None,
) -> Dict:
    """
    Run RAGAS evaluation for one chunking strategy.

    Returns a dict with faithfulness, context_relevance, answer_relevance,
    num_questions, and per-question details.
    """
    from src.evals.dataset import build_eval_dataset

    # Build dataset by running questions through the RAG pipeline
    raw_data = await build_eval_dataset(
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        question_limit=question_limit,
    )

    dataset = Dataset.from_dict(raw_data)
    logger.info("Dataset built: %d samples for strategy=%s", len(dataset), chunk_strategy)

    # Run RAGAS evaluation
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI
        from src.config import settings

        # Configure RAGAS to use the same vLLM endpoint (or mock)
        if settings.vllm_base_url.startswith("mock://"):
            # Use a lightweight evaluation approach for mock mode
            scores = _mock_ragas_scores(raw_data)
        else:
            llm = ChatOpenAI(
                base_url=settings.vllm_base_url,
                api_key="EMPTY",
                model=settings.vllm_model,
                temperature=0.0,
            )
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=llm,
            )
            scores = {
                "faithfulness": float(result["faithfulness"]),
                "context_relevance": float(result["context_precision"]),
                "answer_relevance": float(result["answer_relevancy"]),
            }

    except ImportError as e:
        logger.warning("RAGAS import issue: %s — using heuristic scoring", e)
        scores = _mock_ragas_scores(raw_data)

    scores["num_questions"] = len(raw_data["question"])
    scores["chunk_strategy"] = chunk_strategy
    scores["per_question"] = [
        {
            "question": q,
            "answer": a,
            "ground_truth": gt,
            "num_contexts": len(ctx),
        }
        for q, a, gt, ctx in zip(
            raw_data["question"],
            raw_data["answer"],
            raw_data["ground_truth"],
            raw_data["contexts"],
        )
    ]

    return scores


def _mock_ragas_scores(raw_data: Dict) -> Dict:
    """
    Heuristic RAGAS scores for local mock mode.

    Computes real scores by measuring:
    - faithfulness: word overlap between answer and context
    - context_relevance: word overlap between context and question
    - answer_relevance: word overlap between answer and question
    """
    import re

    def tokenize(text: str):
        return set(re.findall(r"\b\w{3,}\b", text.lower()))

    faithfulness_scores = []
    context_relevance_scores = []
    answer_relevance_scores = []

    stop_words = {"the", "and", "for", "are", "was", "were", "that", "this", "with", "from", "have", "has"}

    for question, answer, contexts, ground_truth in zip(
        raw_data["question"],
        raw_data["answer"],
        raw_data["contexts"],
        raw_data["ground_truth"],
    ):
        q_words = tokenize(question) - stop_words
        a_words = tokenize(answer) - stop_words
        ctx_words = tokenize(" ".join(contexts)) - stop_words

        # Faithfulness: fraction of answer words grounded in context
        if a_words:
            faithfulness_scores.append(len(a_words & ctx_words) / len(a_words))
        else:
            faithfulness_scores.append(0.0)

        # Context relevance: fraction of context words relevant to question
        if ctx_words:
            context_relevance_scores.append(
                min(1.0, len(q_words & ctx_words) / max(len(q_words), 1))
            )
        else:
            context_relevance_scores.append(0.0)

        # Answer relevance: cosine-like overlap between answer and question
        union = a_words | q_words
        if union:
            answer_relevance_scores.append(len(a_words & q_words) / len(union))
        else:
            answer_relevance_scores.append(0.0)

    return {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "context_relevance": sum(context_relevance_scores) / len(context_relevance_scores),
        "answer_relevance": sum(answer_relevance_scores) / len(answer_relevance_scores),
    }


async def run_full_eval(
    strategies: Optional[List[str]] = None,
    top_k: int = 5,
    question_limit: Optional[int] = None,
) -> Dict:
    """
    Run evaluation for all strategies, save results, generate report.

    Returns the full results dict.
    """
    from src.evals.report import generate_report
    from src.tracking.mlflow_logger import log_eval_run, log_eval_artifact
    from src.metrics.prometheus import record_eval_scores

    if strategies is None:
        strategies = ["fixed", "sentence", "semantic"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for strategy in strategies:
        logger.info("=== Evaluating strategy: %s ===", strategy)
        try:
            scores = await _evaluate_strategy(strategy, top_k, question_limit)
        except Exception as e:
            logger.error("Eval failed for strategy %s: %s", strategy, e)
            scores = {
                "faithfulness": 0.0,
                "context_relevance": 0.0,
                "answer_relevance": 0.0,
                "num_questions": 0,
                "error": str(e),
            }

        # Log to MLflow
        run_id = log_eval_run(
            chunk_strategy=strategy,
            faithfulness=scores.get("faithfulness", 0.0),
            context_relevance=scores.get("context_relevance", 0.0),
            answer_relevance=scores.get("answer_relevance", 0.0),
            num_questions=scores.get("num_questions", 0),
            run_name=f"eval_{strategy}_{timestamp}",
        )
        scores["mlflow_run_id"] = run_id

        # Push to Prometheus
        record_eval_scores(strategy, scores)

        all_results[strategy] = scores
        logger.info(
            "Strategy %s: F=%.3f CR=%.3f AR=%.3f",
            strategy,
            scores.get("faithfulness", 0),
            scores.get("context_relevance", 0),
            scores.get("answer_relevance", 0),
        )

    # Save JSON results
    results_path = RESULTS_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        # Don't serialise per_question in top-level summary
        summary = {
            k: {kk: vv for kk, vv in v.items() if kk != "per_question"}
            for k, v in all_results.items()
        }
        json.dump({"timestamp": timestamp, "results": summary}, f, indent=2)

    # Save detailed results
    detailed_path = RESULTS_DIR / f"eval_detailed_{timestamp}.json"
    with open(detailed_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate markdown report
    report_path = RESULTS_DIR / "report.md"
    generate_report(
        {k: {kk: vv for kk, vv in v.items() if kk != "per_question"} for k, v in all_results.items()},
        report_path,
    )

    # Attach results to MLflow
    log_eval_artifact(str(results_path), "eval_results")
    log_eval_artifact(str(report_path), "report")

    logger.info("Eval complete. Results: %s  Report: %s", results_path, report_path)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluations")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["fixed", "sentence", "semantic"],
        default=["fixed", "sentence", "semantic"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for quick tests)")
    args = parser.parse_args()

    asyncio.run(run_full_eval(strategies=args.strategies, top_k=args.top_k, question_limit=args.limit))
