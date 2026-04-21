# Experiments

Use one folder per experiment or run family.

Minimum expected contents per experiment folder:

- `README.md`: intent, relation to issue, interpretation
- `config.yaml`: parameters and settings
- `results.json`: machine-readable outcomes

Example:

```text
experiments/
└── 2026-04-21_dense_reranker_preexperiment/
    ├── README.md
    ├── config.yaml
    └── results.json
```

Log all meaningful runs, not only the successful ones.
