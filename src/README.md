# Source Code

Python package (`data-science-seminar/`) with the pipeline code for the seminar project
"Temporal Relevance and Bias in Semantic RAG Systems" (see the [project README](../README.md)
for the research questions and overall status).

The pipeline runs roughly in this order: `data_scripts` build the corpus and search index,
`retrieval_scripts` run queries against it, and `evaluation_scripts` turn the raw
results into statistical data for evaluation.

## data_scripts/

- `build_dataset.py` — streams the `stanford-oval/ccnews` dataset year by year and assembles a
  balanced, timestamped article corpus per topic (keyword-routed, quota per topic/year cell).
- `build_index.py` — embeds a topic's corpus with `multi-qa-mpnet-base-dot-v1`, L2-normalizes the
  vectors, and builds an exact FAISS `IndexFlatIP` index plus aligned metadata file.

## retrieval_scripts/

- `retrieval.py` — runs a query set against a topic's FAISS index and writes the top-k results
  (with publication dates) as structured JSON, per research question.

## evaluation_scripts/

- `avg_date_per_query.py` — computes unweighted and score-weighted average publication date of the
  retrieved candidates per query, and the delta between them.
- `significance_test.py` — compares per-query date metrics between two conditions (e.g. evergreen
  vs. time-sensitive) with Mann-Whitney-U (unpaired) or Wilcoxon signed-rank (paired) tests.
- `run_evaluation.py` — orchestrates the full evaluation for one experiment: runs the two scripts
  above across all `top_k_X` subfolders, then renders the comparison as LaTeX.

## Usage

Scripts are invoked directly, e.g.:

```
uv run src/data-science-seminar/data_scripts/build_index.py path/to/financial_markets.json
uv run src/data-science-seminar/retrieval_scripts/retrieval.py --index_dir ... --queries_path ... --rq 1
uv run src/data-science-seminar/evaluation_scripts/run_evaluation.py --baseline_dir ... --comparison_dir ... --samples paired
```

Each script documents its own CLI arguments in its module docstring.
