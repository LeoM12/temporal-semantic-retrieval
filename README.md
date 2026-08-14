# Temporal Relevance in Semantic Retrieval Systems

Seminar paper project (Data Science Seminar, University of Passau) investigating whether dense embedding-based retrievers exhibit implicit temporal bias.

## Research Questions

- **RQ1:** Do retrievers favor recent documents for time-sensitive vs. evergreen queries without explicit temporal cues?
- **RQ2:** Do explicit temporal markers in queries shift retrieval recency?
- **RQ3:** Does injecting publication-date metadata into document text before embedding shift retrieved-document age distributions?

## Corpus

~100k articles from `stanford-oval/ccnews` (2017–2024), balanced per topic-year across three domains: Financial Markets, Internet Platforms, UK Football.

## Pipeline

- `build_index.py`: embeds articles (`multi-qa-mpnet-base-dot-v1`), builds per-topic FAISS `IndexFlatIP` indices, saves metadata
- `retrieve.py`: runs queries against topic indices, outputs structured JSON + `_config.yaml`
- `retrieval_analysis_eg_vs_ts.ipynb`: analysis and visual comparison of experiment results.

## Status

RQ1/RQ2 pipeline complete (null findings so far). RQ3 date-injection experiment in progress.
