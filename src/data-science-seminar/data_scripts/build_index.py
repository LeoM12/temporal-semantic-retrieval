"""
build_index.py
==============

Dense-retrieval indexing pipeline for the seminar project
"Temporal Relevance and Bias in Semantic RAG Systems".

This script builds the semantic search foundation used by all downstream
retrieval experiments. Given a single per-topic JSON corpus file, it:

    1. Loads the news articles (stanford-oval/ccnews subset, 2017-2024).
    2. Concatenates each article's title and body into one passage.
    3. Encodes every passage into a dense vector with a QA-tuned
       sentence-transformer (multi-qa-mpnet-base-dot-v1).
    4. L2-normalizes the vectors so that inner product == cosine similarity.
    5. Indexes them with an exact FAISS IndexFlatIP index.
    6. Writes the FAISS index and a parallel, position-aligned metadata file.
    7. Reloads the index and validates that the vector count is correct.

The design decisions below are fixed by the experiment protocol: exact
search, one embedding per document, 512-token truncation, batch size 64,
CPU only. They are intentionally not configurable.

Required third-party packages:
    sentence-transformers, faiss-cpu, tqdm, numpy
    (torch is installed transitively by sentence-transformers)

Usage:
    python build_index.py path/to/financial_markets.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import faiss  # provided by the `faiss-cpu` package
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Configuration constants.                 #
# --------------------------------------------------------------------------- #

OUTPUT_DIR = Path(r"C:\Programming\rag_seminar\data-science-seminar\data\processed\embeddings")

# QA-retrieval-tuned bi-encoder, appropriate for query-to-document matching.
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

# Maximum number of input tokens fed to the encoder. Longer passages are
# truncated by the model's tokenizer; documents are never chunked.
MAX_SEQ_LENGTH = 512

# Number of passages encoded per forward pass.
BATCH_SIZE = 64
DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #

def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def log_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Corpus loading.                                                             #
# --------------------------------------------------------------------------- #


def load_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        log_error(f"Input file does not exist: {corpus_path}")
        sys.exit(1)
    if not corpus_path.is_file():
        log_error(f"Input path is not a file: {corpus_path}")
        sys.exit(1)

    documents: list[dict] = []
    try:
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:          # skip blank lines (e.g. trailing newline)
                    continue
                documents.append(json.loads(line))
    except json.JSONDecodeError as exc:
        log_error(f"Could not parse JSON on line {line_no} of {corpus_path}: {exc}")
        sys.exit(1)
    except OSError as exc:
        log_error(f"Could not read {corpus_path}: {exc}")
        sys.exit(1)

    if not documents:
        log_error(f"Corpus file {corpus_path} contains no documents.")
        sys.exit(1)

    return documents


# --------------------------------------------------------------------------- #
# Passage and metadata construction.                                          #
# --------------------------------------------------------------------------- #

def build_passage(document: dict) -> str:
    title = document.get("article_title", "") or ""
    body = document.get("plain_text", "") or ""
    return f"{title}. {body}"


def build_metadata(documents: list[dict]) -> list[dict]:
    """Build the lightweight, position-aligned metadata list.

    metadata[i] corresponds 1:1 with FAISS index position i, because both the
    metadata list and the embeddings are produced in original document order.
    Only id, published_date and topic are retained.
    """
    return [
        {
            "id": document.get("id"),
            "published_date": document.get("published_date"),
            "topic": document.get("topic"),
        }
        for document in documents
    ]


# --------------------------------------------------------------------------- #
# Main pipeline.                                                              #
# --------------------------------------------------------------------------- #

def main() -> None:
    start_time = time.perf_counter()

    if len(sys.argv) < 2:
        print("Error: corpus_path argument is required.", file=sys.stderr)
        sys.exit(1)
    corpus_path = Path(sys.argv[1])

    # The output filename is derived from the input file name.
    topic = corpus_path.stem
    index_path = OUTPUT_DIR / f"{topic}_index.faiss"
    metadata_path = OUTPUT_DIR / f"{topic}_metadata.json"

    # Startup banner
    log_info(f"Input corpus file : {corpus_path}")
    log_info(f"Output directory  : {OUTPUT_DIR}")
    log_info(f"Derived topic     : {topic}")

    # Load corpus
    documents = load_corpus(corpus_path)
    num_docs = len(documents)
    log_info(f"Loaded {num_docs} documents.")

    try:
        # Load embedding model
        log_info(f"Loading embedding model '{MODEL_NAME}' on device '{DEVICE}' ...")
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        # Enforce the 512-token truncation limit; documents are never chunked.
        model.max_seq_length = MAX_SEQ_LENGTH
        embedding_dim = model.get_sentence_embedding_dimension()
        log_ok(f"Model loaded (embedding dimension = {embedding_dim}).")

        # Build the position-aligned metadata
        metadata = build_metadata(documents)

        # --- Create the FAISS index ---------------------------------------- #
        # IndexFlatIP performs exact, brute-force inner-product search. It
        # requires no training and uses no randomness, so the resulting index
        # is fully deterministic and reproducible for a fixed corpus + model.
        index = faiss.IndexFlatIP(embedding_dim)

        # --- Embed in batches and add to the index ------------------------- #
        # Passages are built per batch (not as one big list) so that only the
        # raw documents plus a single batch of text live in memory at a time.
        num_batches = (num_docs + BATCH_SIZE - 1) // BATCH_SIZE
        for start in tqdm(
            range(0, num_docs, BATCH_SIZE),
            total=num_batches,
            desc="Embedding",
            unit="batch",
        ):
            batch_docs = documents[start:start + BATCH_SIZE]
            batch_passages = [build_passage(doc) for doc in batch_docs]

            # normalize_embeddings=True yields unit-length vectors, so the
            # IndexFlatIP inner product is exactly the cosine similarity.
            embeddings = model.encode(
                batch_passages,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            # FAISS requires C-contiguous float32 arrays.
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            index.add(embeddings)

        # The raw articles are no longer needed once everything is indexed.
        del documents

        log_ok(f"Indexing complete. index.ntotal = {index.ntotal}")

        # --- Persist outputs ----------------------------------------------- #
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        log_ok(f"Wrote FAISS index    -> {index_path}")
        log_ok(f"Wrote metadata file  -> {metadata_path}")

        # --- Validation ---------------------------------------------------- #
        # Reload the index from disk and verify the vector count matches the
        # number of documents. Implemented as an explicit check (rather than a
        # bare `assert`) so a mismatch produces a clean [ERROR] and non-zero
        # exit instead of an AssertionError traceback.
        reloaded = faiss.read_index(str(index_path))
        if reloaded.ntotal != num_docs:
            log_error(
                "Validation failed: reloaded index has "
                f"{reloaded.ntotal} vectors but {num_docs} documents were indexed."
            )
            sys.exit(1)
        log_ok(
            f"Validation passed: reloaded index contains {reloaded.ntotal} "
            f"vectors (matches {num_docs} documents)."
        )

    except SystemExit:
        # Let intentional sys.exit() calls (e.g. failed validation) pass through.
        raise
    except Exception as exc: 
        # Catch-all so the pipeline never terminates with an unhandled trace.
        log_error(f"Unexpected failure: {type(exc).__name__}: {exc}")
        sys.exit(1)

    elapsed = time.perf_counter() - start_time
    log_info(f"Done. Total runtime: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()