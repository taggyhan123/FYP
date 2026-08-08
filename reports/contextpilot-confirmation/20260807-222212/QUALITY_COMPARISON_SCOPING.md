# Quality comparison inputs: metric-scoped score views

Section 4 of `NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md` states:

> Function/full metrics apply to 640 relevance cases; no-tool applies to 160
> irrelevance cases.

`score_bfcl_quality.py` emits all 800 task-level rows in one `scores` list, and
the two case families carry different keys:

| Rows | Domain | Keys present |
| --- | --- | --- |
| 640 | multiple, parallel, parallel_multiple, simple_python | `name_correct`, `full_correct`, `call_count_correct` |
| 160 | irrelevance | `no_tool_correct` |

`compare_bfcl_quality.py` has no domain filter, and
`src/tatm/paired_quality.py:72` raises when the requested metric is absent from
any row it is handed:

```
ValueError: irrelevance_0:menu_seed=42: missing binary metric name_correct
ValueError: multiple_0:menu_seed=42: missing binary metric no_tool_correct
```

So the runbook's §4 command block, run literally against the whole score file,
fails for all three metrics. The first attempt (`run_quality_8b.sh`, logged in
`run-quality-8b.log`) did exactly that and all 9 comparisons failed; the
replays and scores it produced were unaffected and were reused.

## What was done

No source file was modified. For each condition, two **views** of its score
file were written into this results directory — same format, same rows,
`scores` filtered to the family the metric applies to:

- `quality-<condition>-score-relevance.json` — 640 rows, used for
  `name_correct` and `full_correct`
- `quality-<condition>-score-irrelevance.json` — 160 rows, used for
  `no_tool_correct`

Each view records `score_view` and `score_view_source` so it is traceable back
to the unmodified `quality-<condition>-score.json`. The comparisons were then
run with the unmodified `scripts/compare_bfcl_quality.py` at the declared
`--bootstrap-samples 50000 --bootstrap-seed 42`.

Every comparison reports `paired_cases` 640 (relevance) or 160 (irrelevance),
matching the runbook's declared case counts exactly.

## What this does not change

The metrics, the bootstrap parameters, the conditions, and the pairing are the
runbook's. Filtering to the applicable case family is the runbook's own
declared scope, not a change to it. Because it is a harness limitation rather
than a measurement decision, the fix belongs in
`scripts/compare_bfcl_quality.py` (a `--domain` / metric-aware filter) and is
left to the session that owns that script.

No `--equivalence-margin-pp` was passed to any comparison, so these are
estimation-only results and carry no equivalence pass/fail claim.
