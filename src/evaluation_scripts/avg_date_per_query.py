import argparse
from datetime import date
import json
from pathlib import Path
import sys

"""
    Computes, for every query, the unweighted and the score-weighted average
    publication date of its top-k retrieval candidates, and the delta between
    them

    Required Arguments:
        --results_path {Path to retrieval-results file of one category}
    Optional arguments:
        --output-name {custom name of output JSONL file; Default: avg_date_per_query_<query_type>.jsonl}
"""

REFERENCE_DATE = date(2025, 1, 1)

FIELDNAMES = [
    "query_id",
    "topic",
    "query_type",
    "n_candidates",
    "unweighted_avg_age_days",
    "unweighted_avg_date",
    "weighted_avg_age_days",
    "weighted_avg_date",
    "delta_days",
]

# --------------------------------------------------------------------------- #
# Shell & Logging helpers
# --------------------------------------------------------------------------- #

def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def log_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)


def ask_overwrite_confirm(output_path: Path) -> None:
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
# Loading.
# --------------------------------------------------------------------------- #

def load_json_list(input_path: Path, description: str) -> list[dict]:
    log_info(f"Loading {description} from {input_path}...")
    if not input_path.exists():
        log_error(f"File path for {description} doesn't exist: {input_path}")
        sys.exit(1)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if len(data) == 0:
        log_error(f"No entries found in {description} file.")
        sys.exit(1)
    return data


def check_single_query_type(query_results: list[dict]) -> None:
    query_types = {entry["query_type"] for entry in query_results}
    if len(query_types) > 1:
        log_error(
            "Results file contains more than one query_type "
            f"({sorted(query_types)}) - a single run must cover exactly one category."
        )
        sys.exit(1)

# --------------------------------------------------------------------------- #
# Core metric: unweighted vs. score-weighted average age/date per query.
# --------------------------------------------------------------------------- #

def compute_query_row(query_result: dict) -> dict:
    query_id = query_result["id"]
    candidates = query_result["results"]
    n_candidates = len(candidates)

    # relative age in days vs. REFERENCE_DATE for every candidate.
    # Larger age = older document, smaller age = newer document.
    ages = [
        (REFERENCE_DATE - date.fromisoformat(c["published_date"])).days
        for c in candidates
    ]
    scores = [c["score"] for c in candidates]

    # unweighted average age.
    unweighted_avg_age_days = sum(ages) / n_candidates

    # score-weighted average age (weights normalized to sum to 1).
    score_sum = sum(scores)
    weights = [s / score_sum for s in scores]
    weighted_avg_age_days = sum(w * a for w, a in zip(weights, ages))

    # Step 4: delta = unweighted - weighted (ages, not calendar dates:
    # smaller age = newer, so a positive delta means score-weighted
    # candidates are on average NEWER than the unweighted candidate pool).
    delta_days = unweighted_avg_age_days - weighted_avg_age_days

    return {
        "query_id": query_id,
        "topic": query_result["topic"],
        "query_type": query_result["query_type"],
        "n_candidates": n_candidates,
        "unweighted_avg_age_days": unweighted_avg_age_days,
        "unweighted_avg_date": age_days_to_date(unweighted_avg_age_days).isoformat(),
        "weighted_avg_age_days": weighted_avg_age_days,
        "weighted_avg_date": age_days_to_date(weighted_avg_age_days).isoformat(),
        "delta_days": delta_days,
    }


def age_days_to_date(avg_age_days: float) -> date:
    # round to the nearest whole day only for this human-readable calendar-date
    # representation. The age_days columns keep full precision.
    return date.fromordinal(REFERENCE_DATE.toordinal() - round(avg_age_days))


def compute_all_rows(query_results: list[dict]) -> list[dict]:
    return [compute_query_row(q) for q in query_results]

# --------------------------------------------------------------------------- #
# Saving.
# --------------------------------------------------------------------------- #

def save_results(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            ordered_row = {field: row[field] for field in FIELDNAMES}
            handle.write(json.dumps(ordered_row, ensure_ascii=False))
            handle.write("\n")

    log_ok(f"Wrote {len(rows)} query rows to {output_path}.")

# --------------------------------------------------------------------------- #
# Main function.
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute unweighted vs. score-weighted average publication date "
            "per query (RQ1 recency-bias metric) for a single category."
        )
    )

    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--output-name", type=str)

    args = parser.parse_args()
    results_path: Path = args.results_path
    output_name: str = args.output_name

    query_results = load_json_list(results_path, "retrieval results")
    check_single_query_type(query_results)

    if output_name is None:
        category = query_results[0]["query_type"]
        output_name = f"avg_date_per_query_{category}.jsonl"
    output_path = results_path.parent / output_name

    if output_path.exists():
        ask_overwrite_confirm(output_path)

    rows = compute_all_rows(query_results)
    save_results(rows, output_path)


if __name__ == "__main__":
    main()
