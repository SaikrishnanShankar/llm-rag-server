"""
Embedding generation via sentence-transformers (all-MiniLM-L6-v2).

Design decision: singleton pattern for the model object — loading
sentence-transformers takes ~2 s and allocates ~90 MB; we load once
at module import time and reuse across requests.

The public surface is just two functions:
  embed(texts)        → np.ndarray  (N, 384)
  embed_query(query)  → np.ndarray  (384,)

Both accept strings or lists; embed_query is a convenience wrapper
that always returns a 1-D array suitable for pgvector queries.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded (dim=%d)", settings.embedding_dimension)
    return model


def embed(texts: Union[str, List[str]], batch_size: int = 64) -> np.ndarray:
    """
    Encode one or more texts into dense vectors.

    Returns:
        np.ndarray of shape (N, embedding_dimension) with float32 values.
    """
    if isinstance(texts, str):
        texts = [texts]
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # pre-normalised → cosine = dot product
        show_progress_bar=len(texts) > 100,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string; returns 1-D array of shape (384,)."""
    return embed([query])[0]


def embed_for_chunking(texts: List[str]) -> np.ndarray:
    """
    Thin wrapper used by semantic_chunk so the chunking module
    doesn't need to import embeddings directly (avoids circular imports).
    """
    return embed(texts)
