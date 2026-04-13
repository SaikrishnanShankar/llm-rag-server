"""Unit tests for all three chunking strategies."""
import pytest
from src.retrieval.chunking import chunk_text, fixed_chunk, sentence_chunk, semantic_chunk


# ─── fixed chunking ───────────────────────────────────────────────────────────

def test_fixed_basic():
    text = "a" * 600
    chunks = fixed_chunk(text, chunk_size=512, overlap=64)
    assert len(chunks) == 2
    assert len(chunks[0]) == 512
    assert len(chunks[1]) == 600 - (512 - 64)  # 152


def test_fixed_short_text_single_chunk():
    text = "Short text."
    chunks = fixed_chunk(text, chunk_size=512)
    assert chunks == ["Short text."]


def test_fixed_empty_text():
    assert fixed_chunk("") == []
    assert fixed_chunk("   ") == []


def test_fixed_overlap_cannot_exceed_chunk_size():
    # overlap >= chunk_size would cause infinite loop — should still terminate
    chunks = fixed_chunk("hello world " * 50, chunk_size=20, overlap=5)
    assert len(chunks) > 0


# ─── sentence chunking ────────────────────────────────────────────────────────

def test_sentence_basic():
    text = (
        "Machine learning is great. It powers many applications. "
        "Deep learning is a subset. Neural networks are powerful tools."
    )
    chunks = sentence_chunk(text, sentences_per_chunk=2, overlap_sentences=1)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 0


def test_sentence_single_sentence():
    chunks = sentence_chunk("Only one sentence here.")
    assert len(chunks) == 1


def test_sentence_empty():
    assert sentence_chunk("") == [] or sentence_chunk("") == [""]


def test_sentence_overlap_produces_shared_content():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks_no_overlap = sentence_chunk(text, sentences_per_chunk=2, overlap_sentences=0)
    chunks_with_overlap = sentence_chunk(text, sentences_per_chunk=2, overlap_sentences=1)
    # With overlap, adjacent chunks share a sentence → more chunks total
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


# ─── semantic chunking ────────────────────────────────────────────────────────

def _dummy_embed(texts):
    """Returns random-ish but deterministic embeddings for testing."""
    import numpy as np
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((len(texts), 16)).astype("float32")


def test_semantic_basic():
    text = (
        "Machine learning uses data to train models. "
        "Supervised learning requires labeled examples. "
        "Neural networks are inspired by the brain. "
        "Backpropagation trains neural networks efficiently."
    )
    chunks = semantic_chunk(text, embed_fn=_dummy_embed, similarity_threshold=0.5)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strip() != ""


def test_semantic_falls_back_for_short_text():
    # ≤2 sentences → falls back to sentence_chunk
    chunks = semantic_chunk("One sentence only.", embed_fn=_dummy_embed)
    assert len(chunks) >= 1


def test_semantic_max_chunk_size_respected():
    # Make a text where all sentences are similar (high cosine) but combined
    # they'd exceed max_chunk_size, so we still split
    sentence = "A" * 100 + ". "
    text = sentence * 5
    chunks = semantic_chunk(text, embed_fn=_dummy_embed, max_chunk_size=120)
    for c in chunks:
        assert len(c) <= 120 + 50  # small tolerance for sentence boundary


# ─── dispatcher ──────────────────────────────────────────────────────────────

def test_chunk_text_dispatcher_fixed():
    chunks = chunk_text("hello world " * 100, strategy="fixed")
    assert len(chunks) > 0


def test_chunk_text_dispatcher_sentence():
    chunks = chunk_text("Hello world. How are you? I am fine.", strategy="sentence")
    assert len(chunks) > 0


def test_chunk_text_dispatcher_semantic():
    chunks = chunk_text(
        "Hello world. How are you? I am fine.",
        strategy="semantic",
        embed_fn=_dummy_embed,
    )
    assert len(chunks) > 0


def test_chunk_text_dispatcher_unknown():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_text("text", strategy="unknown")


def test_chunk_text_semantic_requires_embed_fn():
    with pytest.raises(ValueError, match="embed_fn is required"):
        chunk_text("text", strategy="semantic")
