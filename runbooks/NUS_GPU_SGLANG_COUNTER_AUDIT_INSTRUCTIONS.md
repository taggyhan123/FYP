# NUS server instructions — historical SGLang aggregate-counter audit

This is a **CPU-only raw-artifact audit**. It does not start SGLang, vLLM, or a
GPU workload, and it does not rerun a model. Its sole purpose is to determine
whether all 72 historical SGLang Phase 2 replays can be validated against the
independent aggregate `sglang:cached_tokens_total` counter already recorded in
each raw result.

The old reconciliation compared one response-derived value with another. Do
not use the old `sglang-reconciled/` directory as evidence.

## Execute at the handed-off commit

Replace the placeholder with the exact commit supplied by the analysis session.

```bash
FYP_REPO=/home/taghan/FYP
AUDIT_COMMIT=REPLACE_WITH_HANDOFF_COMMIT
PHASE2_RAW=/home/taghan/FYP/cluster/results/tooltrie-phase2-20260803-181133
AUDIT_STAMP=$(date +%Y%m%d-%H%M%S)
AUDIT_WORKTREE="/home/taghan/FYP-sglang-counter-audit-$AUDIT_STAMP"
AUDIT_OUTPUT="/home/taghan/sglang-counter-audit-$AUDIT_STAMP"

git -C "$FYP_REPO" fetch origin
git -C "$FYP_REPO" worktree add --detach "$AUDIT_WORKTREE" "$AUDIT_COMMIT"
cd "$AUDIT_WORKTREE"
test "$(git rev-parse HEAD)" = "$AUDIT_COMMIT"
test -z "$(git status --porcelain)"
uv sync
uv run pytest -q
test -d "$PHASE2_RAW"
test ! -e "$AUDIT_OUTPUT"
uv run python reports/tooltrie-phase2/reconcile_causal_sglang.py \
  --input-dir "$PHASE2_RAW" \
  --output-dir "$AUDIT_OUTPUT"
```

The command is fail-closed and all-or-nothing. It expects exactly:

- two datasets;
- 12 conditions, including `contextpilot_causal`;
- three trials per condition;
- 72 unique raw files in total.

For every run it requires request and prompt counter agreement, the first
request as the only missing zero-valued response cache field, exact equality
between the response sum and independent aggregate cached-token delta, and no
failed request. If any file is absent or any check fails, it exits nonzero and
writes **no reconciled copies**.

## Accept or quarantine

Accept the historical SGLang cleanliness claim only if
`$AUDIT_OUTPUT/aggregate-counter-audit-summary.json` reports:

```text
declared_runs = 72
accepted_runs = 72
refused_runs = 0
all_clean = true
```

If the audit refuses even one run, preserve the failed check names and keep the
entire historical SGLang arm provisional. Do not weaken the checks and do not
infer an aggregate counter from response fields. A fresh 72-run replay is a
separate, predeclared experiment and is not authorized by this audit protocol.

## Hand over compact evidence

Create a new branch in the detached worktree. Commit only:

- a handover Markdown file;
- `aggregate-counter-audit-summary.json`;
- a SHA-256 for a tar archive of `$AUDIT_OUTPUT`;
- the executed FYP commit and environment summary.

Do not commit the 72 reconciled raw copies. The handover must state explicitly
that no model server or GPU replay was run, whether 72/72 passed, and any
missing or refused files.
