import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration constants.
# --------------------------------------------------------------------------- #

OUTPUT_DIR = Path(r"C:\Programming\rag_seminar\data-science-seminar\experiments\testing")

# QA-retrieval-tuned bi-encoder, appropriate for query-to-document matching.
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

# Maximum number of input tokens fed to the encoder. Longer passages are
# truncated by the model's tokenizer; documents are never chunked.
MAX_SEQ_LENGTH = 512

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
# Loading and saving helpers.
# --------------------------------------------------------------------------- #

def save_results(retrieval_results: list[dict], output_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / output_name
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(retrieval_results, handle, ensure_ascii=False, indent=2)

    log_ok(f"Wrote {len(retrieval_results)} query results to {results_path}.")

def load_index(index_path: Path) -> faiss.Index:
    log_info("Loading Index...")
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        log_error("Index file path doesn't exist.")
        sys.exit(1)

    return index

def load_metadata(metadata_path: Path) -> list[dict]:
    log_info("Loading Metadata...")
    if metadata_path.exists():
        with open(metadata_path) as md:
            metadata = json.load(md)
    else:
        log_error("Metadata file path doesn't exist.")
        sys.exit(1)

    return metadata

def load_queries(queries_path: Path) -> list[dict]:
    log_info("Loading Queries...")
    if queries_path.exists():
        with open(queries_path) as q:
            queries = json.load(q)
            if len(queries) == 0:
                log_error("Query set is empty.")
                sys.exit(1)
    else:
        log_error("Query-set file path doesn't exist.")
        sys.exit(1)
    
    return queries

def load_indices(index_dir: Path) -> dict:
    indices = {}

    for faiss_path in index_dir.glob("*.faiss"):
        topic = faiss_path.stem.removesuffix("_index")
        meta_path = index_dir / f"{topic}_metadata.json"
        if not meta_path.exists():
            log_error(f"Metadata missing for topic '{topic}': {meta_path}")
            sys.exit(1)
        indices[topic] = {
            "index": faiss.read_index(str(faiss_path)),
            "metadata": json.loads(meta_path.read_text()),
        }
    log_ok(f"Loaded {len(indices)} indices: {list(indices.keys())}")
    return indices

def check_index_meta_align(indices: dict):
    for topic, bundle in indices.items():
        if bundle["index"].ntotal != len(bundle["metadata"]):
            log_error(f"[{topic}] Misalignment: index has {bundle['index'].ntotal} vectors "
                    f"but metadata has {len(bundle['metadata'])} entries.")
            sys.exit(1)

    

# --------------------------------------------------------------------------- #
# Query embedding.
# --------------------------------------------------------------------------- #

def embed_queries(queries_by_topic: defaultdict) -> dict:
    log_info(f"Loading embedding model '{MODEL_NAME}'.")
    model = SentenceTransformer(MODEL_NAME, device = DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH
    embedding_dim = model.get_embedding_dimension()
    log_ok(f"Model loaded (embedding dimension = {embedding_dim}).")

    embeddings_by_topic = {}
    for topic, queries in queries_by_topic.items():
        plain_queries = [q["query"] for q in queries]
        embeddings = model.encode(
            plain_queries,
            normalize_embeddings = True,
            convert_to_numpy = True,
        )
        embeddings_by_topic[topic] = embeddings
        log_ok(f"Embedded {len(embeddings)} queries for topic '{topic}'.")

    log_ok(f"Embedding complete. Embedded {len(embeddings_by_topic)} topics.")
    return embeddings_by_topic

# --------------------------------------------------------------------------- #
# Retrieval.
# --------------------------------------------------------------------------- #

def retrieve(
    document_index: faiss.Index,
    query_embeddings: np.ndarray,
    queries: list[dict],
    metadata: list[dict],
    k: int,
) -> list[dict]:
    scores, indices = document_index.search(query_embeddings, k)
    results = []

    for i, query in enumerate(tqdm(queries)):
        query_results = []

        for j, (score, pos) in enumerate(zip(scores[i], indices[i])):
            if pos == -1:
                continue

            doc_meta = metadata[pos]
            query_results.append({
                "rank": j + 1,
                "faiss_position": int(pos),
                "id": doc_meta["id"],
                "published_date": doc_meta["published_date"],
                "score": float(score)
            })
        results.append({
            "id": query["id"],
            "query_type": query["query_type"],
            "topic": query["topic"],
            "results": query_results
        })
    log_ok(f"Retrieved {np.size(scores)} documents for {len(query_embeddings)} queries.")
    return results

# --------------------------------------------------------------------------- #
# Main pipeline.
# --------------------------------------------------------------------------- #

#TODO: Refactoring
#TODO: Adding automatic configurations log
def main() -> None:
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Retrieve top-k documents for a query set."
    )

    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--queries_path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-name", type=str)

    args = parser.parse_args()
    index_dir: Path = args.index_dir
    queries_path: Path = args.queries_path
    k: int = args.top_k
    output_name: Path = args.output_name
    if(output_name == None):
        output_name = queries_path.stem + "_results.json"

    log_ok("Done parsing arguments.")

    indices = load_indices(index_dir)
    check_index_meta_align(indices)

    queries = load_queries(queries_path)
    log_info(f"Loaded {len(queries)} queries from {queries_path}.")

    queries_by_topic = defaultdict(list)
    for q in queries:
        queries_by_topic[q["topic"]].append(q)

    embeddings_by_topic = embed_queries(queries_by_topic)

    retrieval_results = []
    for topic, embeddings in embeddings_by_topic.items():
        index = indices[topic]["index"]
        metadata = indices[topic]["metadata"]
        queries = queries_by_topic[topic]
        results = retrieve(index, embeddings, queries, metadata, k)
        retrieval_results.extend(results)

    save_results(retrieval_results, output_name)

    elapsed = time.perf_counter() - start_time
    log_info(f"Done. Total runtime: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()