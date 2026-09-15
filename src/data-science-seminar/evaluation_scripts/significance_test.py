import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import scipy
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

"""
    Runs a two-sided significance test comparing the per-query metrics
    produced by avg_date_per_query.py between two categories (e.g. evergreen
    vs. time_sensitive), both pooled across all queries and split per topic.

    Two tests are available:
        mwu      - Mann-Whitney-U test for independent (unpaired) samples.
                   Effect size: rank-biserial correlation r_rb.
        wilcoxon - Wilcoxon signed-rank test for paired samples. Queries are
                   paired by topic and query number, i.e. "fm_eg_01" is paired
                   with "fm_tsn_01". Effect size: matched-pairs rank-biserial
                   correlation r_rb.
    Both effect sizes are written as "rank_biserial".

    "delta_days" and "weighted_avg_age_days" are tested seperately.
    delta_days isolates the score-internal recency preference within the
    candidate pool that the retriever already selected, while
    weighted_avg_age_days additionally mixes in the retriever's selection
    effect (which candidates end up in the pool at all).

    Required Arguments:
        --baseline_path {Path to baseline-category avg_date_per_query JSONL file}
        --comparison_path {Path to comparison-category avg_date_per_query JSONL file}
    Optional arguments:
        --test {'mwu' or 'wilcoxon'; Default: mwu}
        --alpha {significance threshold; Default: 0.05}
"""

VALUE_FIELDS = {
    "delta": "delta_days",
    "weighted_avg": "weighted_avg_age_days",
}

TEST_NAMES = {
    "mwu": "Mann-Whitney-U",
    "wilcoxon": "Wilcoxon signed-rank",
}

# --------------------------------------------------------------------------- #
# Shell & Logging helpers
# --------------------------------------------------------------------------- #

def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def log_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)

# --------------------------------------------------------------------------- #
# Loading.
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path, label: str) -> tuple[list[dict], int]:
    log_info(f"Loading '{label}' queries from {path}...")
    if not path.exists():
        log_error(f"File path for '{label}' doesn't exist: {path}")
        sys.exit(1)

    rows = []
    n_skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                _ = row["query_id"]
                _ = row["topic"]
                _ = row["n_candidates"]
                _ = row[VALUE_FIELDS["delta"]]
                _ = row[VALUE_FIELDS["weighted_avg"]]
            except (json.JSONDecodeError, KeyError):
                n_skipped += 1
                continue
            rows.append(row)

    if len(rows) == 0:
        log_error(f"No usable rows found for '{label}' in {path}.")
        sys.exit(1)

    return rows, n_skipped

# --------------------------------------------------------------------------- #
# Pairing (Wilcoxon only).
# --------------------------------------------------------------------------- #

def pair_key(row: dict) -> tuple[str, str]:
    # query ids follow "<topic-prefix>_<query-type>_<number>" (e.g. "fm_tsn_01"),
    # so topic + number identifies the same query across categories.
    return row["topic"], row["query_id"].rsplit("_", 1)[-1]


def pair_rows(baseline_rows: list[dict], comparison_rows: list[dict]) -> list[tuple[dict, dict]]:
    def index(rows: list[dict], label: str) -> dict[tuple[str, str], dict]:
        indexed = {}
        for row in rows:
            key = pair_key(row)
            if key in indexed:
                log_error(f"Duplicate pair key {key} in '{label}' ({indexed[key]['query_id']}, {row['query_id']}).")
                sys.exit(1)
            indexed[key] = row
        return indexed

    baseline_index = index(baseline_rows, "baseline")
    comparison_index = index(comparison_rows, "comparison")

    unmatched_baseline = sorted(set(baseline_index) - set(comparison_index))
    unmatched_comparison = sorted(set(comparison_index) - set(baseline_index))
    if unmatched_baseline or unmatched_comparison:
        log_error(
            "Paired test requires a one-to-one match of queries. "
            f"Unmatched baseline keys: {unmatched_baseline}; "
            f"unmatched comparison keys: {unmatched_comparison}."
        )
        sys.exit(1)

    return [(baseline_index[key], comparison_index[key]) for key in sorted(baseline_index)]

# --------------------------------------------------------------------------- #
# Core statistics.
# --------------------------------------------------------------------------- #

def run_mwu(baseline_values: list[float], comparison_values: list[float], alpha: float) -> dict:
    n_baseline = len(baseline_values)
    n_comparison = len(comparison_values)

    result = mannwhitneyu(baseline_values, comparison_values, alternative="two-sided", method="auto")
    # how often was baseline higher than comparison in cross-comparison
    # the smaller u_stat is, the higher are the values in comparison group
    u_stat = float(result.statistic)
    p_value = float(result.pvalue)

    # Rank-biserial correlation effect size, with baseline as the focal group.
    # r_rb = 2U / (n_baseline * n_comparison) - 1
    #      = P(baseline > comparison) - P(baseline < comparison).
    # Negative r_rb = baseline group tends to have smaller (i.e. newer,
    # for age-based fields) values than the comparison group.
    rank_biserial = (2 * u_stat) / (n_baseline * n_comparison) - 1

    return {
        "n_baseline": n_baseline,
        "n_comparison": n_comparison,
        "median_baseline": median(baseline_values),
        "median_comparison": median(comparison_values),
        "mean_baseline": mean(baseline_values),
        "mean_comparison": mean(comparison_values),
        "u_statistic": u_stat,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "significant": p_value < alpha,
    }


def run_wilcoxon(baseline_values: list[float], comparison_values: list[float], alpha: float) -> dict:
    # values are expected in pair order: baseline_values[i] belongs to comparison_values[i].
    differences = [b - c for b, c in zip(baseline_values, comparison_values)]
    nonzero = [d for d in differences if d != 0]
    n_pairs = len(differences)
    n_zero_differences = n_pairs - len(nonzero)

    if len(nonzero) == 0:
        # e.g. delta_days at top_k=1 is always 0. No evidence for any difference.
        log_info("All paired differences are zero - reporting p = 1 and effect size 0.")
        w_stat, p_value, rank_biserial = 0.0, 1.0, 0.0
    else:
        # zero_method="wilcox" drops zero differences before ranking.
        result = wilcoxon(
            baseline_values, comparison_values, zero_method="wilcox", alternative="two-sided", method="auto"
        )
        # for two-sided tests scipy reports min(R+, R-).
        w_stat = float(result.statistic)
        p_value = float(result.pvalue)

        # Matched-pairs rank-biserial correlation, with baseline as the focal group
        # (same sign convention as the rank-biserial correlation in run_mwu).
        # r_rb = (R+ - R-) / (R+ + R-), with differences = baseline - comparison.
        # Negative r_rb = baseline values tend to be smaller (i.e. newer, for
        # age-based fields) than their paired comparison values.
        ranks = rankdata([abs(d) for d in nonzero])
        r_plus = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)
        r_minus = sum(rank for rank, d in zip(ranks, nonzero) if d < 0)
        rank_biserial = float((r_plus - r_minus) / (r_plus + r_minus))

    return {
        "n_pairs": n_pairs,
        "n_zero_differences": n_zero_differences,
        "median_baseline": median(baseline_values),
        "median_comparison": median(comparison_values),
        "mean_baseline": mean(baseline_values),
        "mean_comparison": mean(comparison_values),
        "median_difference": median(differences),
        "w_statistic": w_stat,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "significant": p_value < alpha,
    }


TEST_FUNCTIONS = {
    "mwu": run_mwu,
    "wilcoxon": run_wilcoxon,
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def build_value_type_results(
    baseline_rows: list[dict], comparison_rows: list[dict], field: str, alpha: float, test: str
) -> dict:
    # for wilcoxon, baseline_rows[i] and comparison_rows[i] are already paired,
    # and grouping by topic keeps that order intact.
    run_test = TEST_FUNCTIONS[test]
    baseline_values = [row[field] for row in baseline_rows]
    comparison_values = [row[field] for row in comparison_rows]
    pooled = run_test(baseline_values, comparison_values, alpha)

    baseline_by_topic = group_by_topic(baseline_rows, field)
    comparison_by_topic = group_by_topic(comparison_rows, field)
    topics = sorted(set(baseline_by_topic) | set(comparison_by_topic))

    per_topic = {}
    for topic in topics:
        b_values = baseline_by_topic.get(topic)
        c_values = comparison_by_topic.get(topic)
        if not b_values or not c_values:
            log_warn(
                f"Topic '{topic}' is missing from one of the two files for "
                f"field '{field}' - skipping per-topic test for this topic."
            )
            continue
        per_topic[topic] = run_test(b_values, c_values, alpha)

    return {"pooled": pooled, "per_topic": per_topic}


def group_by_topic(rows: list[dict], field: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["topic"]].append(row[field])
    return dict(grouped)

# --------------------------------------------------------------------------- #
# Main function.
# --------------------------------------------------------------------------- #

def guess_label(rows: list[dict], fallback: str) -> str:
    query_types = {row.get("query_type") for row in rows if row.get("query_type")}
    if len(query_types) == 1:
        return next(iter(query_types))
    return fallback


def get_top_k(rows: list[dict]) -> int:
    # top-k (n_candidates) is assumed constant within a single input file.
    return rows[0]["n_candidates"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two-sided Mann-Whitney-U (unpaired) or Wilcoxon signed-rank (paired) "
            "significance test comparing delta_days and weighted_avg_age_days "
            "between two categories (RQ1/RQ2/RQ3)."
        )
    )
    parser.add_argument("--baseline_path", type=Path, required=True, help="Path to baseline-category avg_date_per_query JSONL file")
    parser.add_argument("--comparison_path", type=Path, required=True, help="Path to comparison-category avg_date_per_query JSONL file")
    parser.add_argument(
        "--test", choices=sorted(TEST_FUNCTIONS), default="mwu",
        help="'mwu' for unpaired samples, 'wilcoxon' for paired samples (default: mwu)",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default: 0.05)")
    parser.add_argument("--description", type=str, default="", help="Description of evaluation run for metadata.")

    args = parser.parse_args()
    baseline_path: Path = args.baseline_path
    comparison_path: Path = args.comparison_path
    test: str = args.test
    alpha: float = args.alpha
    description = args.description
    test_name = TEST_NAMES[test]

    baseline_rows, n_skipped_baseline = load_jsonl(baseline_path, "baseline")
    comparison_rows, n_skipped_comparison = load_jsonl(comparison_path, "comparison")

    baseline_label = guess_label(baseline_rows, "baseline")
    comparison_label = guess_label(comparison_rows, "comparison")

    baseline_top_k = get_top_k(baseline_rows)
    comparison_top_k = get_top_k(comparison_rows)
    if baseline_top_k != comparison_top_k:
        log_warn(
            f"top-k (n_candidates) differs between baseline ({baseline_top_k}) "
            f"and comparison ({comparison_top_k})."
        )

    baseline_topics = {row["topic"] for row in baseline_rows}
    comparison_topics = {row["topic"] for row in comparison_rows}
    if baseline_topics != comparison_topics:
        log_warn(
            f"Topic sets differ between baseline ({sorted(baseline_topics)}) "
            f"and comparison ({sorted(comparison_topics)})."
        )
    all_topics = sorted(baseline_topics | comparison_topics)

    if test == "wilcoxon":
        pairs = pair_rows(baseline_rows, comparison_rows)
        baseline_rows = [b for b, _ in pairs]
        comparison_rows = [c for _, c in pairs]
        log_ok(f"Matched {len(pairs)} query pairs.")

    results = {}
    for value_type, field in VALUE_FIELDS.items():
        log_info(f"Running {test_name} test for '{value_type}' ({field})...")
        results[value_type] = build_value_type_results(baseline_rows, comparison_rows, field, alpha, test)

    now = datetime.now(timezone.utc)
    metadata = {
        "script": Path(__file__).name,
        "description": description,
        "run_timestamp_utc": now.isoformat(),
        "inputs": {
            "baseline": {
                "path": str(baseline_path.resolve()),
                "label": baseline_label,
                "n_queries_loaded": len(baseline_rows),
                "n_rows_skipped": n_skipped_baseline,
                "top_k": baseline_top_k,
            },
            "comparison": {
                "path": str(comparison_path.resolve()),
                "label": comparison_label,
                "n_queries_loaded": len(comparison_rows),
                "n_rows_skipped": n_skipped_comparison,
                "top_k": comparison_top_k,
            },
        },
        "topics": all_topics,
        "value_fields_used": VALUE_FIELDS,
        "alpha": alpha,
        "test": test,
        "samples": "paired" if test == "wilcoxon" else "unpaired",
        "effect_size": "matched_pairs_rank_biserial" if test == "wilcoxon" else "rank_biserial",
        "test_type": "two-sided",
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
    }
    if test == "wilcoxon":
        metadata["pairing"] = "topic + trailing query number of query_id"
        metadata["zero_method"] = "wilcox"

    output = {"metadata": metadata, "results": results}

    output_dir = comparison_path.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")
    output_name = f"{test}_{baseline_label}_vs_{comparison_label}_{timestamp_str}.json"
    output_path = output_dir / output_name

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    log_ok(f"Wrote {test_name} test results to {output_path}.")


if __name__ == "__main__":
    main()
