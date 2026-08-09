# Key findings

Terse summary. Full evidence, caveats and provenance:
[`consolidated-report.md`](consolidated-report.md).

```
reuse = prompt_tokens_cached / (prompt_tokens_cached + request_prefill_kv_computed_tokens_sum)
```
vLLM counter deltas over one replay. Quality is n=800 BFCL: 640 relevance cases
(function-name, full call), 160 irrelevance cases (no-tool). Differences in
percentage points.

## Qwen3-4B — primary

| Condition | padded-64 reuse | BM25 k=128 reuse | full call | no-tool |
| --- | ---: | ---: | ---: | ---: |
| Original text prefill | 1.19% | 0.34% | 76.09% | **88.12%** |
| Alphabetical | 37.99% | 0.44% | 73.28% | 85.62% |
| ToolTrie-v0 | 87.19% | 0.89% | 75.31% | 87.50% |
| ContextPilot persistent | **96.16%** | 1.35% | **77.03%** | 85.00% |
| ContextPilot static refit | **96.16%** | **2.23%** | **77.03%** | 85.00% |
| `frequency_online` | 96.27% | 2.33% | 76.88% | 83.75% |

190 accepted GPU replays, 33 audit checks passed, Qwen3-0.6B replicates.

## 1. The headline number is a workload artifact

Reuse falls from 82–96% on padded menus to **0.3–19%** on BM25-retrieved menus,
for every policy, at both models. Padded menus share 60–63 of 64 tools; real
retrieved menus do not. The often-quoted 95–96% ContextPilot figure reproduces
exactly and means far less than it appears to.

## 2. The padded benchmark cannot discriminate between methods

`bfcl-padded64` has 63 tools in all 200 menus, 198 in one, one in two. Reuse is
essentially the position of the single non-shared tool:

| ordering | mean position of that tool | reuse |
| --- | ---: | ---: |
| original | 1.0 | 1.19% |
| alphabetical | 24.1 | 38.13% |
| frequency_fitted | 25.0 | 39.69% |
| ToolTrie-v0 | 57.3 | 87.19% |
| ContextPilot persistent | 63.7 | 96.16% |
| `frequency_online` | 63.7 | 96.27% |

A ~30-line counter matches hierarchical clustering. So results on this workload
cannot be evidence for clustering, tries, or ordering sophistication.

**They are tied, not ranked.** `frequency_online` caches exactly 6,704 tokens
(419 blocks) from the third request onward — the entire shared prefix. The 0.11
point gap to ContextPilot is 95 blocks across 200 requests.

One variable predicts the table: non-shared tools per request. One → counting
ties clustering. Four or more → clustering leads.

## 3. ToolTrie-v0 is dominated

Both ContextPilot arms beat it in **all 12** systems cells; `frequency_online`
beats it in **10 of 12**, losing both BM25 k=4 cells (17.01% vs 17.48%).

Its `visit_count` field is incremented and never read — the popularity signal
its objective is missing.

## 4. No reordering is a quality win

At 4B, against ordinary text prefill: best gain is **+0.94 points** full call
(ContextPilot), and **every** policy loses no-tool accuracy. Alphabetical — the
baseline most historical comparisons used — is itself **worse than doing
nothing** (−2.81 full at 4B, −11.57 at 0.6B), so most reported "gains" are
recovering ground alphabetical lost.

Per-case, of 160 irrelevance cases at 4B:

| Policy | lost | recovered |
| --- | ---: | ---: |
| ToolTrie-v0 | 5 | 4 |
| ContextPilot persistent | 5 | **0** |
| `frequency_online` | 7 | **0** |

The high-reuse policies never repair an irrelevance case. Exact McNemar
p ≈ 0.016 and 0.063. Too thin to claim degradation scales with reuse.

## 5. Ordering-quality effects are a small-model artifact

Ordering moves full-call accuracy by **11.6 points at 0.6B** and **3.8 at 4B**.
Conclusions drawn at 0.6B measure the model, not the method. The 8B irrelevance
regression (−5.00 points) does **not** reproduce at 4B (+0.00, p=1.0).

## 6. Frequency ordering: the earlier answer was incomplete

`frequency_fitted` fits support on a task-disjoint corpus — correct anti-leakage
practice, but it cannot know which tools *this* stream shares, so it lands at
39.69%. The same signal estimated online reaches 96.27%. Verified: the frozen
fit is uncontaminated (a leaked fit would rank universal tools first and reach
~96%; it places them at mean position 31.6/64).

The brief's §7 Q3 answer — "does frequency ordering improve reusable prefix
length? no" — holds only for the frozen variant.

## 7. Defects found, and their status

| Defect | Status |
| --- | --- |
| `static_refit_causal` returned ContextPilot's internal integer IDs, so the arm could never run | **Fixed** (`f4384db`), re-run, 18/18 cells clean. Changed no conclusion: the two ContextPilot APIs give zero per-case quality differences across 800 cases |
| Five Phase-2 "learned" policies emit **byte-identical** `tool_ids` on BFCL — only the label differs | **Reporting corrected.** Brief §7 Q5 downgraded from "answered: essentially none" to "not tested on BFCL". Not a code bug: the policies degenerate when their statistics do not discriminate. Testing pair/triple properly needs a workload where co-occurrence discriminates — a new experiment |
| The 8B quality comparisons have no `original` control | **Cannot be fixed retroactively.** Those runs were unpinned, so a replay today cannot be shown to use the same snapshot. Requires a fresh pinned five-condition 8B matrix, or the comparison is omitted |
| Brief §7 Q3 recorded "frequency ordering does not help" | **Corrected**: true of the frozen estimator only; online estimation reaches 96.27% |

Separately, the historical SGLang arm was **validated**, 72/72 against the
independent aggregate counter — a clean result, not a defect.

## Not established

Latency. Planning cost is excluded entirely (workloads are pre-built), and it is
not negligible: `static_refit` costs 0.75 s/request at n=800 versus ~2–7 ms for
the persistent API. Single menu seed, single request sequence, no competing
traffic, no equivalence margins declared, reduced AST checker rather than the
official BFCL evaluator.
