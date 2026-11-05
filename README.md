# Temporal Relevance in Semantic Retrieval

**Investigating Implicit Temporal Behaviour in a Standard Dense Retriever**

Data Science Seminar — University of Passau · Summer Semester 2026
Prof. Dr. Michael Granitzer, University of Passau
Author: Leonard Motte, BSc Informatik

---

Retrieval-augmented generation (RAG) systems typically rely on plain semantic
(dense) retrievers that match *meaning*, not *time*. This repository contains the
code, query sets, indices and raw results of a controlled study on whether such a
retriever is implicitly sensitive to temporal relevance, and whether that
sensitivity can be induced with simple temporal signals in the query or in the
document. The full write-up is the
[final report](docs/leonard_motte_temporal_relevance_in_semantic_retrieval.pdf).

> **Headline result.** A standard dense retriever shows no implicit temporal
> sensitivity. It shifts toward newer documents only where query and document
> share a *literal* year expression — never for temporal meaning that is conveyed
> implicitly through topic or wording.


## Motivation

Facts age at different rates. Evergreen knowledge (the speed of light) stays valid
for years, while time-sensitive information (a company's CEO, market prices) can 
become outdated within months. A retriever that ignores
this distinction fails in both directions: forcing recency onto a stable query
surfaces newer but possibly worse documents, while ignoring document age for a
dynamic query can hand the generator a confidently wrong context.


## Research Questions

The three questions form one experiment in which the location of the temporal
signal moves each time: nowhere → query → document.

- **RQ1 — Implicit behaviour.** Does a semantic retriever systematically favour
  more recent documents for time-sensitive queries than for evergreen queries,
  without any explicit temporal cue?
- **RQ2 — Temporal cues in the query.** To what extent do explicit cues in a
  time-sensitive query (present-oriented, relative, absolute) shift the recency
  of the retrieved documents?
- **RQ3 — Temporal cues in the document.** Does injecting a document's
  publication date into its text before embedding shift the age of the retrieved
  documents?


## Experimental Design

### Corpus

[`stanford-oval/ccnews`](https://huggingface.co/datasets/stanford-oval/ccnews),
chosen because every article carries an exact **publication timestamp** (not just
a crawl date) and the data spans multiple years, allowing facts to drift.

- Three topic domains: **Financial Markets**, **Internet & Online Platforms**,
  **UK Football**, selected via domain-specific keyword lists matched against
  article titles.
- Capped at **3,700 articles per topic and year**, so retrieval cannot look
  recency-biased merely because more articles were crawled in recent years.
- Years 2017–2024 (2016 dropped for insufficient coverage):
  3,700 × 3 topics × 8 years ≈ **90k articles**.

### Query sets

Five manually written and reviewed sets with 20 questions per topic (60 each,
**300 queries** in total), all derived from the same keyword lists.

| Set | ID prefix | Example |
|---|---|---|
| Evergreen (control) | `eg` | *What does FOMC stand for?* |
| Time-sensitive, unmarked (foundation) | `tsn` | *Who chairs the Federal Reserve?* |
| + present-oriented cue | `tsp` | *Who **currently** chairs the Federal Reserve?* |
| + relative cue | `tsr` | *Who chairs the Federal Reserve **since last year**?* |
| + absolute cue | `tsa` | *Who chairs the Federal Reserve **in 2024**?* |

The three cue sets are built by extending the unmarked time-sensitive questions,
so each marked query has a 1:1 unmarked counterpart. The absolute cue always
names 2024, the last year of the corpus.

### Retrieval setup

- **Model:** `multi-qa-mpnet-base-dot-v1` (768-dim, 512-token input limit;
  documents are truncated, never chunked)
- **Index:** FAISS `IndexFlatIP` over L2-normalised embeddings (exact cosine
  similarity), with a **separate index per topic** to avoid cross-topic noise
- **Depths:** top-*k* with *k* ∈ {1, 10, 20, 50, 100}, run independently to check
  whether effects depend on the cutoff

### Evaluation

For every query, the **score-weighted average publication age** of the top-*k*
documents is computed:

$$
\bar{a}_{\text{weighted}} = \sum_{i=1}^{k} w_i \, a_i, \qquad w_i = \frac{s_i}{\sum_{j=1}^{k} s_j}
$$

where *a*<sub>i</sub> is the age in days relative to a fixed reference date
(2025-01-01, outside the corpus) and *s*<sub>i</sub> the retrieval score. The
age distributions of two conditions are then compared:

| Research question | Comparison | Test |
|---|---|---|
| RQ1 | TS unmarked vs. evergreen (independent samples) | Mann–Whitney U, two-sided |
| RQ2 | each cue type vs. TS unmarked (paired 1:1) | Wilcoxon signed-rank, two-sided |
| RQ3 | injected vs. non-injected index, per query category (paired) | Wilcoxon signed-rank, two-sided |

Effect size is the rank-biserial correlation *r*<sub>rb</sub> throughout; α = 0.05.


## Reproducing the Experiments

```bash
# 1. Environment (Python 3.11)
git clone https://github.com/LeoM12/temporal-semantic-retrieval.git
cd temporal-semantic-retrieval
git lfs install && git lfs pull       # fetches the FAISS indices (9 files, ~90 MB each)
uv sync
```

The indices, query sets and all experiment outputs are included, so you can
inspect the results or jump straight to step 3 without rebuilding anything.

```bash
# 2. (Optional) rebuild corpus and indices
uv run src/data_scripts/build_dataset.py                       # streams ccnews, one JSONL per topic
uv run src/data_scripts/build_index.py \
    --corpus_path path/to/financial_markets.jsonl \
    --injection_weight 1                                       # 0 = baseline, 1 = single injection

# 3. Retrieval (example: RQ1 on the baseline index)
uv run src/retrieval_scripts/retrieval.py \
    --index_dir data/processed/indexes/no_date_injection \
    --queries_path data/processed/queries/queries_ts_no_cues.json \
    --rq 1 --top-k 10

# 4. Recency metric and significance test (example: RQ1, k = 10)
uv run src/evaluation_scripts/avg_date_per_query.py \
    --results_path experiments/rq1_results/top_k_10/time_sensitive_unmarked_results.json
uv run src/evaluation_scripts/significance_test.py \
    --baseline_path   experiments/evergreen_results/top_k_10/avg_date_per_query_evergreen.jsonl \
    --comparison_path experiments/rq1_results/top_k_10/avg_date_per_query_time_sensitive_unmarked.jsonl \
    --test mwu                                                 # use 'wilcoxon' for paired designs
```

`run_evaluation.py` chains steps 4 for all `top_k_N/` folders of one experiment
(`--baseline_dir`, `--comparison_dir`, `--samples paired|unpaired`). Its final
LaTeX export relies on `src/helpers/significance_results_to_latex.py`, which is
not tracked in this repository.


## Project Documents

| Document | Path |
|---|---|
| Final report (ACM format) | [`docs/leonard_motte_temporal_relevance_in_semantic_retrieval.pdf`](docs/leonard_motte_temporal_relevance_in_semantic_retrieval.pdf) |
| Research narrative (proposal) | [`docs/motte-leonard-research-narrative.pdf`](docs/motte-leonard-research-narrative.pdf) |


## License

MIT — see [`LICENSE`](LICENSE).
