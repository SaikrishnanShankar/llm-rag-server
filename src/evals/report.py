"""
Generate a markdown report comparing chunking strategies across RAGAS metrics.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def generate_report(results: Dict[str, Dict], output_path: Path) -> str:
    """
    Generate a markdown report from eval results.

    Args:
        results: {strategy: {faithfulness, context_relevance, answer_relevance, ...}}
        output_path: where to write report.md

    Returns:
        Markdown string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# RAG Evaluation Report",
        f"\n**Generated:** {now}\n",
        "## Summary: Chunking Strategy Comparison",
        "",
        "| Strategy | Faithfulness | Context Relevance | Answer Relevance | Composite |",
        "|----------|-------------|-------------------|------------------|-----------|",
    ]

    best_strategy = None
    best_composite = -1.0

    for strategy, scores in sorted(results.items()):
        f = scores.get("faithfulness", 0.0)
        cr = scores.get("context_relevance", 0.0)
        ar = scores.get("answer_relevance", 0.0)
        composite = (f + cr + ar) / 3

        lines.append(
            f"| {strategy} | {f:.3f} | {cr:.3f} | {ar:.3f} | {composite:.3f} |"
        )

        if composite > best_composite:
            best_composite = composite
            best_strategy = strategy

    lines += [
        "",
        f"**Winner:** `{best_strategy}` with composite score {best_composite:.3f}",
        "",
        "## Metric Definitions",
        "",
        "- **Faithfulness**: Are all claims in the answer supported by the retrieved context?",
        "  A score of 1.0 means every statement is grounded in the provided passages.",
        "",
        "- **Context Relevance**: Do the retrieved chunks actually contain information",
        "  needed to answer the question? Penalises noisy retrieval.",
        "",
        "- **Answer Relevance**: Does the generated answer directly address the question?",
        "  Penalises verbose or off-topic responses.",
        "",
        "## Per-Strategy Details",
        "",
    ]

    for strategy, scores in sorted(results.items()):
        lines += [
            f"### {strategy.capitalize()} Chunking",
            "",
            f"- Faithfulness: **{scores.get('faithfulness', 0):.4f}**",
            f"- Context Relevance: **{scores.get('context_relevance', 0):.4f}**",
            f"- Answer Relevance: **{scores.get('answer_relevance', 0):.4f}**",
            f"- Questions evaluated: {scores.get('num_questions', 'N/A')}",
            f"- MLflow run ID: `{scores.get('mlflow_run_id', 'N/A')}`",
            "",
        ]

    lines += [
        "## Interpretation",
        "",
        "1. **If faithfulness is low**: The LLM is hallucinating facts not present in",
        "   retrieved context. Consider reducing temperature or improving chunk quality.",
        "",
        "2. **If context relevance is low**: The retriever is returning irrelevant chunks.",
        "   Consider increasing `top_k`, switching chunking strategy, or tuning the",
        "   similarity threshold.",
        "",
        "3. **If answer relevance is low**: The answer is off-topic. Review the system",
        "   prompt and ensure the LLM is instructed to stay on-topic.",
        "",
        "## Next Steps",
        "",
        "- View full experiment comparison in MLflow: `make track`",
        "- View metrics dashboard: `make monitor`",
        "- Deploy best strategy: set `DEFAULT_CHUNK_STRATEGY` in `.env`",
        "- For GPU inference on PACE: set `VLLM_BASE_URL=http://<pace-node>:8000/v1`",
    ]

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report
