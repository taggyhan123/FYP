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
- a causal recent-path ToolTrie planner and prompt-level workload builders;
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

The ToolTrie-v0 planner itself is CPU-only. Generate a base selected-tool
workload, then causally reorder it without using the current or future request:

```bash
uv run python scripts/build_cluster_workload.py \
  --partition bfcl --ordering original --menu-size 64 --limit 200 \
  --output data/processed/bfcl-base-menu64.jsonl

uv run python scripts/build_tooltrie_workload.py \
  --input data/processed/bfcl-base-menu64.jsonl \
  --fallback alphabetical --recency-window 128 \
  --output data/processed/bfcl-tooltrie-menu64.jsonl
```

The second file is directly compatible with `replay_vllm_workload.py` and
`score_bfcl_quality.py`. Real APC hits and latency still require the CUDA vLLM
server described in `cluster/README.md`.

The default tokenizer is `Qwen/Qwen3-0.6B`. Its official `tokenizer.json` is
cached under ignored `data/tokenizers/`. Override it with `--tokenizer MODEL_ID`
if the cluster experiment uses a different model.

## Main outputs

- `task_c_d.html` — polished browser report for Tasks C and D;
- `task_c_d.pdf` — print-ready version of the same report;
- `task_c_d.md` — maintainable source used to generate both versions;
- `notes/reading-note.md` — Task A reading note and request-flow diagram;
- `reports/dataset-inventory.md` — Task C dataset and schema report;
- `reports/access-patterns.md` — Task D workload and trie analysis;
- `reports/analysis-summary.json` — machine-readable analysis;
- `reports/tables/` — compact source tables;
- `cluster/README.md` — exact steps for Task B/E on a CUDA vLLM server.

The public datasets are benchmarks, not production traces. Reports deliberately
label ToolRet relevance as **gold requirement** and BFCL function lists as
**menu exposure**; neither is called production popularity.
