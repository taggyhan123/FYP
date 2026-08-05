# AGENTS.md

## Project

TATM is a Python 3.12 research codebase for trie-aware tool memory and prompt
prefix-cache experiments. Read `README.md`, `PROJECT_STATUS.md`, and the files
directly relevant to the assigned objective before editing.

## Commands

Set up and validate local work with:

```bash
uv sync
uv run pytest
```

Use targeted pytest invocations while iterating, then run the full suite before
declaring the task complete. Dataset regeneration is intentionally separate:

```bash
uv run python scripts/download_datasets.py
uv run python scripts/run_pipeline.py
```

## Working Rules

- Preserve existing user changes and keep each change scoped to the objective.
- Do not commit `data/raw/`, `data/processed/`, `data/tokenizers/`,
  `cluster/results/`, `.gnhf/`, `.tools/`, or `node_modules/`.
- Do not infer GPU or serving-engine results from local tests. GPU experiments
  must follow the applicable NUS runbook and record actual measurements.
- Do not use a shared GPU, change an experiment's declared method silently, or
  overwrite an existing result directory.
- Keep claims in reports traceable to generated artifacts or recorded evidence.
- Update tests and relevant documentation when behavior or commands change.
- A task is complete only when its observable stop condition is met, relevant
  checks pass, and no unrelated files were changed.
