# Static-refit resume + SGLang counter audit — GPU executor handover

**Stamp** `20260808-234909`
**Executed FYP commit** `a706f6ebe7bd4c9aa7b95b06434a723d0d50b64a`
**ContextPilot commit** `1fa0a143fdeda344585666648ab2b30cb7fea77f`, alpha `0.001`
**Engine** vLLM 0.26.0, unmodified.

This run executed `NUS_GPU_CONTEXTPILOT_STATIC_REFIT_RESUME_INSTRUCTIONS.md`
and `NUS_GPU_SGLANG_COUNTER_AUDIT_INSTRUCTIONS.md`. It was **superseded
mid-flight** by the dual-model protocol at `1528570`, which instructed that the
old runbooks not run in parallel with it. Work already completed at that point
is preserved here; work not yet started was abandoned.

## Raw archives

```
/home/taghan/contextpilot-static-refit-20260808-234909.tar.gz
sha256 8473faaee1d74dbb1559ad3e035a54b146193eb7f7dc961547adbe5f501f9b12
79 entries, 12 MB

/home/taghan/sglang-counter-audit-20260808-234909.tar.gz
sha256 2a8dfb9063d4306d52390fefbbaf506a5aa16e741349629f3205e7a53c7a2e14
74 entries, 2.8 MB
```

## 1. The static-refit defect is genuinely fixed

`f4384db` was verified against pinned upstream. All seven `static_refit_causal`
workloads built cleanly with `mode=static_refit_causal`, `alpha=0.001`, commit
`1fa0a143`, `request_order_changed=false`, `full_contextpilot_system=false`,
`information_regime=causal`, `offline_transductive=false`, annotations and
eviction feedback both false. Zero violations.

**18/18 Qwen3-0.6B systems replays accepted** — 200 requests each, zero
failures, `counter_validation.clean` true, cache reset before every one,
capacity 190,896 matching the accepted confirmation run.

The six combined summaries put static-refit beside the four previously accepted
conditions; all pass the three equivalence guards with five conditions each:

| group | original | alphabetical | tooltrie_v0 | cp-online | **cp-static-refit** |
| --- | --- | --- | --- | --- | --- |
| BFCL padded-64 | 1.19% | 38.13% | 87.19% | 96.16% | **96.16%** |
| ToolRet padded-64 | 13.87% | 51.05% | 83.58% | 95.27% | **95.74%** |
| BM25 k=4 | 15.87% | 15.28% | 17.48% | 18.72% | 18.42% |
| BM25 k=16 | 6.12% | 6.27% | 7.77% | 9.93% | 9.80% |
| BM25 k=64 | 0.91% | 1.24% | 1.90% | 4.78% | 4.01% |
| BM25 k=128 | 0.37% | 0.58% | 1.13% | 1.99% | **2.96%** |

## 2. The Qwen3-8B static-refit quality cell is closed

Run on a separate idle GPU, capacity 44,656 **matching the accepted 8B run**.
800 requests, zero failures, clean counters, reset, model `Qwen/Qwen3-8B`.

`full 78.44% | name 84.84% | no_tool 84.38%`

Six comparisons committed here, all `sequence_state_dependent: true`, 50 000
bootstrap samples, seed 42:

| vs | metric | Δ pp | 95% CI | n |
| --- | --- | --- | --- | --- |
| alphabetical | name | +2.03 | [+0.47, +3.59] | 640 |
| alphabetical | full | +2.03 | [+0.16, +3.91] | 640 |
| alphabetical | no_tool | −5.00 | [−8.75, −1.88] | 160 |
| tooltrie_v0 | name | +1.09 | [+0.00, +2.19] | 640 |
| tooltrie_v0 | full | +0.78 | [−0.31, +1.88] | 640 |
| tooltrie_v0 | no_tool | −3.12 | [−6.25, −0.62] | 160 |

**These are numerically identical to the persistent-API comparisons, and that
is not a coincidence.** The two workloads differ (782 of 800 orderings match,
18 differ, different file hashes), but a per-case diff of the two score files
found **zero** differences on all three metrics. The two ContextPilot APIs are
quality-indistinguishable at 8B, so the original run's use of `fit_transform`
instead of `reorder()` did not affect any quality conclusion.

## 3. SGLang aggregate-counter audit: ACCEPTED

CPU-only, no GPU or model server involved.

```
declared_runs = 72   accepted_runs = 72   refused_runs = 0   all_clean = true
```

All 72 historical SGLang Phase 2 replays validate against the independent
`sglang:cached_tokens_total` aggregate, across 12 conditions including
`contextpilot_causal`. The historical SGLang cleanliness claim is supported.

**Scope caution:** this audit validates *counters*. It says nothing about
whether the twelve conditions are distinct — see the separate defect below.

## Not executed

The runbook's §5 regeneration of the nine accepted 8B comparisons with
`--sequence-state-dependent`, and its §6 combined archive/handover, were
abandoned when the dual-model protocol superseded this work. The six
static-refit comparisons in §2 above were generated; the nine pre-existing ones
were not regenerated.

## Deviation

`uv venv` instead of `python3.12 -m venv` — this machine's system python3.12
has no `ensurepip` and the documented fix needs `sudo`, which project rules
forbid. Already-recorded environment deviation.

No quarantined attempts. `--allow-counter-mismatch` never passed.

## Analysis-session review

The compact package was integrated and re-audited locally. All six checks in
`reports/contextpilot-static-refit-resume/local-audit.json` pass. The audit can
verify 18 systems trials, aggregate quality equality, paired-statistic equality,
and 72/72 SGLang counter decisions. It cannot verify the executor's stronger
zero per-case-difference statement or reopen either raw archive because those
files remain server-only. The historical 8B quality matrix also still lacks an
ordinary `original` fallback condition; see the reconciled `findings.md`.
