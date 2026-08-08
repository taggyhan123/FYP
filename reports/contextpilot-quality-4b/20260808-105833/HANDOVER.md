# Qwen3-4B quality point — additional model, requested after the confirmation run closed

**Run stamp** `20260808-105833`
**Branch** `gpu/contextpilot-confirmation-20260807-222212`
**FYP commit executed** `081b8cdcefcf7538bc4efc2d5c837a321408c477`
(the confirmation-run commit; harness identical to
`ece2b267fe9411c861bc6cbcb7f52a04ad0ad656`)

**This is not part of the accepted confirmation acceptance table.** The runbook
declared Qwen3-8B for section 4 and that result stands unchanged. This is a
third model point added on request, run with the same procedure, and reported
separately so the accepted set is not silently widened.

## Raw archive

```
/home/taghan/contextpilot-quality-4b-20260808-105833.tar.gz
sha256 c8883e5a268765e0d00904b1936417094979999294a5f1e4af14a0e579c3dbc5
```

6.2 MB compressed, 40 entries, verified with `tar -tzf`. Archived with `tar -h`
so the two symlinked workload inputs are stored as real content, making it
self-contained. GPU-server-only, no off-machine backup.

## Validation

| | |
| --- | --- |
| Accepted replays | 3 (one per condition), `request_count` 800 each |
| Failed requests | 0 |
| `counter_validation.clean` | true on all 3 |
| Cache reset before every replay | yes |
| Model recorded in every replay | `Qwen/Qwen3-4B` |
| Failures in the driver log | 0 |

## Setup

Same GPU 2 (RTX 3090), one server at a time, one isolated replay client,
unmodified vLLM 0.26.0 APC, `--max-tokens 128 --disable-thinking
--reset-before`, n=800 (640 relevance + 160 irrelevance).

`Qwen/Qwen3-4B` was not in the HF cache and was downloaded (7.6 GB, snapshot
`1cfa9a7208912126459214e8b04321603b3df60c`).

**Capacity differs by model, so the ToolTrie plan was rebuilt**, exactly as the
runbook requires when switching server:

| Model | `block_size` × `num_gpu_blocks` | capacity_tokens |
| --- | --- | --- |
| Qwen3-8B | 16 × 2 791 | 44 656 |
| Qwen3-4B | 16 × 6 052 | **96 832** |

At 4B the rebuilt plan hinted a prefix for 799 of 800 requests with 4 201
evictions and 96 794 retained schema tokens. The `alphabetical` and
`contextpilot-online_incremental` workloads are capacity-independent and were
reused byte-for-byte from the confirmation run, so those two arms are identical
prompts across 4B and 8B; `tooltrie_v0` is not, and 4B-vs-8B for that arm
therefore mixes model and plan.

## Results

| Condition | full (4B / 8B) | name (4B / 8B) | no_tool (4B / 8B) |
| --- | --- | --- | --- |
| alphabetical | 73.28 / 76.41 | 82.03 / 82.81 | 85.62 / 89.38 |
| tooltrie_v0 | 75.31 / 77.66 | 83.75 / 83.75 | 86.88 / 87.50 |
| contextpilot online | **77.19** / 78.44 | **84.53** / 84.84 | 85.62 / 84.38 |

Paired task-clustered bootstrap, 50 000 samples, seed 42, **no equivalence
margin declared — estimation only**:

| Comparison | metric | 4B Δ pp | 4B 95% CI | n | 8B Δ pp |
| --- | --- | --- | --- | --- | --- |
| cp-online vs alphabetical | name | +2.50 | [+0.78, +4.22] | 640 | +2.03 |
| cp-online vs alphabetical | full | +3.91 | [+1.88, +5.94] | 640 | +2.03 |
| cp-online vs alphabetical | no_tool | **+0.00** | [−4.38, +4.38] | 160 | **−5.00** |
| cp-online vs tooltrie_v0 | name | +0.78 | [−0.16, +1.88] | 640 | +1.09 |
| cp-online vs tooltrie_v0 | full | +1.88 | [+0.62, +3.28] | 640 | +0.78 |
| cp-online vs tooltrie_v0 | no_tool | −1.25 | [−5.62, +3.12] | 160 | −3.12 |
| tooltrie_v0 vs alphabetical | name | +1.72 | [+0.00, +3.44] | 640 | +0.94 |
| tooltrie_v0 vs alphabetical | full | +2.03 | [+0.00, +4.06] | 640 | +1.25 |
| tooltrie_v0 vs alphabetical | no_tool | +1.25 | [−1.88, +5.00] | 160 | −1.88 |

## What changes, and what does not

**The 8B irrelevance regression does not reproduce at 4B.** The 8B run's most
notable result was `contextpilot-online_incremental` losing 5.00 pp of no-tool
accuracy against alphabetical with a CI excluding zero and 8 one-directional
discordant pairs. At 4B the same comparison is **exactly 0.00 pp**, CI
[−4.38, +4.38], McNemar p = 1.0000. `tooltrie_v0` likewise flips sign, −1.88 pp
at 8B versus +1.25 pp at 4B, both CIs spanning zero.

So the "reordering makes the model more willing to call a tool when it should
decline" reading is **not supported across models on this evidence**. It is a
single-model observation at 8B. n=160 irrelevance cases at one seed is thin for
either conclusion; distinguishing a real 8B-specific effect from noise needs
more irrelevance cases or more seeds, not another model.

**The relevance-side gains do reproduce, and are larger at 4B.** `cp-online` vs
alphabetical is +3.91 pp full at 4B against +2.03 pp at 8B, and `cp-online` now
also beats `tooltrie_v0` on full_correct with a CI excluding zero (+1.88 pp,
p = 0.0118), which it did not at 8B. Absolute accuracy is lower at 4B for every
condition, as expected, so ordering has more headroom to matter.

Nothing here touches the systems result. Reuse was not remeasured at 4B; the
padded-menu versus BM25-retrieved-menu gap from the confirmation run is
unchanged.

## Naming

`contextpilot-online_incremental` is **ContextPilot persistent-API adaptation
(alpha=0.001; no eviction feedback or annotations)** — ordering only, one
persistent `reorder()` instance, no eviction feedback, no relevance
annotations. **Not the full ContextPilot system.** No eviction-feedback or
annotation arm was executed here either. `static_refit_causal` remains
quarantined and was not run at 4B.

## Deviations

The metric-scoped score views described in the confirmation run's
`QUALITY_COMPARISON_SCOPING.md` were applied identically here (relevance rows
for name/full, irrelevance rows for no_tool); `paired_cases` is 640/160 as
expected. No source file was modified.
