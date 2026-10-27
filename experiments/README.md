# Experiments

Raw retrieval runs and statistical tests for the three research questions (see the top-level [README](../README.md)). Every run retrieves against the FAISS indices built by `build_index.py` using queries from `data/processed/queries/`, and is repeated at `top_k` ∈ {1, 10, 20, 50, 100} to check whether effects depend on cutoff depth.

## Folders

- **`evergreen_results/`** — baseline condition: evergreen (non-time-sensitive) queries against the un-injected index. Serves as the baseline for RQ1 and for the evergreen arm of RQ3.
- **`rq1_results/`** — time-sensitive queries *without* explicit temporal cues, against the un-injected index. Tested against `evergreen_results` (unpaired, Mann–Whitney U) to see whether retrievers favor recent documents for time-sensitive queries by default.
- **`rq2_results/`** — time-sensitive queries *with* explicit temporal cues, against the un-injected index, split into three cue types: `absolute_cues/` (e.g. "in 2021"), `relative_cues/` (e.g. "last year"), `present_oriented_cues/` (e.g. "currently"). Each is tested against `rq1_results` (same queries, no cue) to see whether adding a cue shifts recency.
- **`rq3_results/`** — same three query sets (`absolute_queries/`, `evergreen_queries/`, `time_sensitive_queries/`) but retrieved against an index built with publication-date metadata injected into the document text before embedding. Each is tested against its non-injected counterpart (`rq2_results`/`evergreen_results`/`rq1_results`, same queries, paired Wilcoxon signed-rank) to see whether date injection shifts retrieved-document age.

## Per-`top_k_N/` contents

- `*_results.json` — raw ranked retrieval output per query (doc id, FAISS position, score, `published_date`).
- `avg_date_per_query_*.jsonl` — per-query recency metrics derived from the above (`delta_days`, `weighted_avg_age_days`), the actual input to the significance tests.
- `*_results_config.yaml` — run provenance: model, index/queries paths, timestamp.
- `mwu_*.json` / `wilcoxon_*.json` — significance test output (pooled across topics and per-topic, for both metrics), produced by `significance_test.py`.

`values.tex` (one per condition folder) collects the pooled/per-topic effect sizes and p-values across all `top_k` values as LaTeX macros for the report tables in `evaluation/`.

## Pipeline

See `src/data-science-seminar/`: `retrieval_scripts/retrieval.py` runs the queries, `evaluation_scripts/avg_date_per_query.py` computes the recency metrics, `evaluation_scripts/significance_test.py` runs MWU/Wilcoxon tests, `helpers/significance_results_to_latex.py` builds `values.tex`.
