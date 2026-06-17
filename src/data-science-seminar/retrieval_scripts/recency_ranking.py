import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time

"""
Usage:
    uv run recency_ranking.py
    Required Arguments:
        --input_path {Path to file with retrieval results}
    Optional arguments:
        --output-name {custom name of output file; Default: reranked_ + input_filename}
"""

# --------------------------------------------------------------------------- #
# Shell & Logging helpers
# --------------------------------------------------------------------------- #

def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def log_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)

def ask_overwrite_confirm(output_path: str) -> None:
    log_info(f"Output target at {output_path} already exists.")
    answer = input("Overwrite file? ('yes' or 'no'): ")
    if answer.upper() in ["Y", "YES"]:
        print("Continuing by overwriting existing file.")
    elif answer.upper() in ["N", "NO"]:
        log_info(f"Terminating current run. Overwriting file at {output_path} was denied by user")
        sys.exit(1)
    else:
        log_error("Invalid input. Please type 'yes' or 'no'.")
        ask_overwrite_confirm(output_path)

# --------------------------------------------------------------------------- #
# Loading & Saving.
# --------------------------------------------------------------------------- #

def load_input(input_path: Path) -> list[dict]:
    log_info("Loading retrieved documents as input...")
    if input_path.exists():
        with open(input_path) as d:
            query_results = json.load(d)
            if len(query_results) == 0:
                log_error("No retrieved documents found in file.")
                sys.exit(1)
    else:
        log_error("File path for retrieved documents doesn't exits.")
        sys.exit(1)
    return query_results

def save_results(ranked_results: list[dict], output_path: Path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(ranked_results, handle, ensure_ascii=False, indent=2)

    log_ok(f"Wrote {len(ranked_results)} recency-ranked results to {output_path}.")

# --------------------------------------------------------------------------- #
# Assign recency-based rank to each retrieved document.
# --------------------------------------------------------------------------- #

def assign_recency_rank(query_results: list[dict]) -> list[dict]:
    for query_result in query_results:
        retrieved = query_result["results"]
        by_recency = sorted(
            retrieved,
            key= lambda r: date.fromisoformat(r["published_date"]),
            reverse=True
        )
        for recency_rank, result in enumerate(by_recency, start=1):
            result["recency_rank"] = recency_rank
        query_result["results"] = by_recency
    return query_results

# --------------------------------------------------------------------------- #
# Main function.
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank based on recency of retrieved documents."
    )

    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output-name", type=str)

    args = parser.parse_args()
    input_path: Path = args.input_path
    output_name: Path = args.output_name
    if(output_name == None):
        output_name = "reranked_" + input_path.name
    output_path = input_path.parent / output_name

    if output_path.exists():
        ask_overwrite_confirm(output_path)

    query_results = load_input(input_path)
    ranked_results = assign_recency_rank(query_results)
    save_results(ranked_results, output_path)

if __name__ == "__main__":
    main()