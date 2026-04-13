"""
Document ingestion pipeline: load → chunk → embed → store.

Usage (CLI):
    python -m src.retrieval.ingest --docs-dir data/sample_docs --strategy sentence
    python -m src.retrieval.ingest --docs-dir data/sample_docs --strategy fixed
    python -m src.retrieval.ingest --docs-dir data/sample_docs --strategy semantic
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from tqdm import tqdm

from src.retrieval.chunking import chunk_text
from src.retrieval.embeddings import embed, embed_for_chunking
from src.retrieval.vectorstore import ensure_schema, insert_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _doc_id(path: Path) -> str:
    """Stable doc_id: stem + first 8 chars of content hash."""
    content = path.read_bytes()
    h = hashlib.sha256(content).hexdigest()[:8]
    return f"{path.stem}_{h}"


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def ingest_file(path: Path, strategy: str) -> int:
    text = load_document(path)
    if not text.strip():
        logger.warning("Empty document: %s", path)
        return 0

    doc_id = _doc_id(path)
    logger.info("Ingesting %s → doc_id=%s strategy=%s", path.name, doc_id, strategy)

    embed_fn = embed_for_chunking if strategy == "semantic" else None
    chunks = chunk_text(text, strategy=strategy, embed_fn=embed_fn)

    if not chunks:
        logger.warning("No chunks produced for %s", path)
        return 0

    logger.info("  %d chunks produced", len(chunks))
    embeddings = embed(chunks)
    metadata = {"source": str(path), "filename": path.name}
    inserted = insert_chunks(chunks, embeddings, doc_id, strategy, metadata)
    return inserted


def ingest_directory(docs_dir: str, strategy: str) -> None:
    ensure_schema()
    docs_path = Path(docs_dir)
    files = list(docs_path.glob("**/*.txt")) + list(docs_path.glob("**/*.pdf"))

    if not files:
        logger.warning("No .txt or .pdf files found in %s", docs_dir)
        return

    total = 0
    for f in tqdm(files, desc=f"Ingesting ({strategy})"):
        total += ingest_file(f, strategy)

    logger.info("Ingestion complete: %d total chunks inserted (strategy=%s)", total, strategy)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--docs-dir", default="data/sample_docs", help="Directory of .txt/.pdf files")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "semantic"], default="sentence")
    args = parser.parse_args()
    ingest_directory(args.docs_dir, args.strategy)
