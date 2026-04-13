"""
Three chunking strategies for document ingestion.

Design decision: each strategy returns List[str] of raw text chunks.
Metadata (doc_id, index, strategy) is attached upstream in vectorstore.py
so chunking stays a pure text-transformation concern.

- fixed:    simple character-window splits; fast, baseline
- sentence: NLTK sentence boundaries; preserves semantic units cheaply
- semantic: embed → cosine-similarity breakpoints; highest quality,
            most compute. Threshold=0.8 balances chunk count vs coherence.
"""
from __future__ import annotations

import re
from typing import List


# ─── Fixed-size chunking ──────────────────────────────────────────────────────

def fixed_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Sliding window over characters with overlap."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ─── Sentence-boundary chunking ───────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no NLTK dependency required)."""
    # Split on period/exclamation/question followed by whitespace + capital
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in raw if s.strip()]


def sentence_chunk(
    text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1
) -> List[str]:
    """Group sentences into chunks with 1-sentence overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: List[str] = []
    step = max(1, sentences_per_chunk - overlap_sentences)
    for i in range(0, len(sentences), step):
        group = sentences[i : i + sentences_per_chunk]
        chunk = " ".join(group).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ─── Semantic chunking ────────────────────────────────────────────────────────

def semantic_chunk(
    text: str,
    embed_fn,                   # callable: List[str] → np.ndarray
    similarity_threshold: float = 0.80,
    max_chunk_size: int = 1024,
) -> List[str]:
    """
    Split at sentence boundaries where consecutive sentence embeddings
    drop below similarity_threshold (cosine distance breakpoints).

    Falls back to sentence_chunk when text is too short to embed meaningfully.
    """
    import numpy as np

    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return sentence_chunk(text)

    embeddings = embed_fn(sentences)  # shape: (N, D)
    # Normalise for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-8)
    # Cosine similarity between consecutive sentences
    similarities = (normed[:-1] * normed[1:]).sum(axis=1)  # shape: (N-1,)

    # Build chunks by splitting at low-similarity boundaries
    chunks: List[str] = []
    current: List[str] = [sentences[0]]
    current_len = len(sentences[0])

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        next_len = len(next_sentence)
        would_exceed = current_len + next_len + 1 > max_chunk_size

        if sim < similarity_threshold or would_exceed:
            chunks.append(" ".join(current))
            current = [next_sentence]
            current_len = next_len
        else:
            current.append(next_sentence)
            current_len += next_len + 1

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    strategy: str,
    embed_fn=None,
    **kwargs,
) -> List[str]:
    """
    Unified entry point used by the ingestion pipeline.

    Args:
        text:       Raw document text.
        strategy:   "fixed" | "sentence" | "semantic"
        embed_fn:   Required only for strategy="semantic".
        **kwargs:   Forwarded to the strategy function.
    """
    strategy = strategy.lower()
    if strategy == "fixed":
        return fixed_chunk(text, **kwargs)
    if strategy == "sentence":
        return sentence_chunk(text, **kwargs)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("embed_fn is required for semantic chunking")
        return semantic_chunk(text, embed_fn=embed_fn, **kwargs)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose fixed | sentence | semantic")
