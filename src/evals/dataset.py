"""
Build a RAGAS-compatible evaluation dataset from the Q&A JSON file.

RAGAS expects a HuggingFace Dataset with columns:
  question, answer, contexts (list of strings), ground_truth

We generate 'answer' and 'contexts' by running each question through
the full RAG pipeline, then evaluate those results against ground_truth.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

QA_DATASET_PATH = Path("eval/qa_dataset.json")


def load_qa_pairs(path: Path = QA_DATASET_PATH, limit: Optional[int] = None):
    """Load question-answer pairs from JSON file."""
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


async def _run_single_query(question: str, chunk_strategy: str, top_k: int = 5):
    """Run one question through the RAG pipeline; return (answer, contexts)."""
    from src.agent.graph import run_agent

    result = await run_agent(
        query=question,
        top_k=top_k,
        chunk_strategy=chunk_strategy,
    )
    answer = result.get("answer", "")
    contexts = [s.get("content_preview", "") for s in result.get("sources", [])]
    # Use full content from retrieved chunks if available via sources
    return answer, contexts


async def build_eval_dataset(
    chunk_strategy: str,
    top_k: int = 5,
    question_limit: Optional[int] = None,
) -> dict:
    """
    Run all Q&A pairs through the RAG pipeline for a given chunking strategy.

    Returns a dict with lists: questions, answers, contexts, ground_truths.
    """
    pairs = load_qa_pairs(limit=question_limit)
    questions, answers, contexts_list, ground_truths = [], [], [], []

    logger.info(
        "Building eval dataset: %d questions, strategy=%s", len(pairs), chunk_strategy
    )

    # Run sequentially to avoid overwhelming the DB / LLM
    for pair in tqdm(pairs, desc=f"Evaluating [{chunk_strategy}]"):
        try:
            answer, contexts = await _run_single_query(
                pair["question"], chunk_strategy, top_k
            )
        except Exception as e:
            logger.warning("Query failed for %r: %s", pair["question"][:50], e)
            answer = ""
            contexts = []

        questions.append(pair["question"])
        answers.append(answer)
        contexts_list.append(contexts if contexts else ["No context retrieved"])
        ground_truths.append(pair["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
