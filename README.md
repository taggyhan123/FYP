# TATM — Trie-Aware Tool Memory

This repository contains the local, reproducible foundation for the FYP
described in [initial-research-brief.md](initial-research-brief.md).
See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the task-by-task completion and
the exact SoC cluster boundary.

Implemented locally:

- ToolRet tool/query discovery and download;
- a manageable BFCL V4 static subset;
- canonical tool-schema normalization and Qwen tokenizer accounting;
- schema-quality, frequency, co-occurrence, locality, replay, ordering, and
  tool-trie analyses;
- a prompt-level exact ToolTrie workload builder;
- a vLLM prefix-cache probe that can be run later on the SoC CUDA cluster.

The raw and normalized datasets are reproducible and ignored by Git. Compact
Markdown, JSON, and CSV reports are written to `reports/` and may be committed.

## Local setup

```bash
uv sync
uv run python scripts/download_datasets.py
uv run python scripts/run_pipeline.py
uv run pytest
```

The default tokenizer is `Qwen/Qwen3-0.6B`. Its official `tokenizer.json` is
cached under ignored `data/tokenizers/`. Override it with `--tokenizer MODEL_ID`
if the cluster experiment uses a different model.

## Main outputs

- `notes/reading-note.md` — Task A reading note and request-flow diagram;
- `reports/dataset-inventory.md` — Task C dataset and schema report;
- `reports/access-patterns.md` — Task D workload and trie analysis;
- `reports/analysis-summary.json` — machine-readable analysis;
- `reports/tables/` — compact source tables;
- `cluster/README.md` — exact steps for Task B/E on a CUDA vLLM server.

The public datasets are benchmarks, not production traces. Reports deliberately
label ToolRet relevance as **gold requirement** and BFCL function lists as
**menu exposure**; neither is called production popularity.
