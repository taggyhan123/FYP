# ToolTrie Phase 2 — external comparison and targeted no-tool evaluation

Runbook: `NUS_GPU_PHASE2_INSTRUCTIONS.md`. All cache and latency claims come from
each engine's own counters. The planner's analytical `hinted_schema_tokens` is
never used to support a cache claim.

**Headline.** ToolTrie-v0 raises measured prefix-cache reuse from 38.13% to
87.19% on BFCL and cuts aggregate TTFT 2.3×, beating every baseline **except
ContextPilot, which beats it by ~9pp under an equally causal regime** (§2a —
this corrects an earlier draft that dismissed ContextPilot as offline-only). Its
one measured quality cost is real: it declines irrelevant requests less reliably
than alphabetical ordering, −3.75pp with a 95% interval excluding zero.

## 7. Provenance (stated first, because three conditions are not upstream code)

| Condition | Implementation | Label required in any write-up |
| --- | --- | --- |
| ToolTrie-v0 | this repository's causal recent-path planner | `ToolTrie-v0` |
| CacheWeaver | faithful tool-ID transcription of paper Algorithm 1; no public implementation existed as of 2026-08-03 | **`CacheWeaver Algorithm-1 reimplementation`** |
| FP-tree conditional | training-only tool-order adaptation of FP-tree traversal | **`FP-tree-derived adaptation`**, not an FP-Growth result |
| Pair/triple conditional | training-only unordered co-occurrence statistics | **`pair/triple adaptation`** |
| ContextPilot | actual upstream code, commit `1fa0a143fdeda344585666648ab2b30cb7fea77f`, v0.4.1 | `ContextPilot offline/transductive` |
| SGLang | official `v0.5.15.post1`, commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` | `SGLang/RadixAttention engine` |

FYP commit `38566fd`. Full environment in `environment.txt`; the 230 resolved
SGLang pins are in `sglang-resolved-deps.txt`.

## 1. Targeted no-tool result — the primary analysis

All 240 BFCL irrelevance tasks under 5 fixed menu seeds, Qwen3-8B, one APC reset
per condition. The bootstrap clusters on the **240 unique tasks**, not the 1,200
menu realizations.

| | value |
| --- | --- |
| alphabetical | 87.50% |
| ToolTrie-v0 | 83.75% |
| difference | **−3.75pp** |
| task-clustered 95% CI | **[−5.67, −2.00]** — excludes zero |
| discordant pairs | alphabetical-only correct **54**, ToolTrie-only correct **9** |
| scope | 240 task clusters, 1,200 paired menu cases |

**This is a regression, not a null result.** The same effect appeared twice
before without enough power to resolve it (0.6B n=100; 8B n=1000, −4.00pp with
CI [−10.89, +2.89]). The point estimate has been stable at roughly −4pp across
three independent measurements; only the interval changed. The 54:9 discordant
split is the clearest single piece of evidence.

Exact McNemar returns p=1e-08 but its independence assumption is **not** met
here (five menu cases share each task), so it is descriptive only; the clustered
bootstrap above is primary. No equivalence margin was declared, so no
equivalence claim is made.

## 2. vLLM fixed-request-order comparison (3 trials, 95% intervals)

Qwen3-0.6B, 200 requests, 64-tool menus, matched tool sets (verified by the
summarizer: same case set, same request sequence, same selected tool sets).

| BFCL | cached ratio | TTFT s | wall s |
| --- | --- | --- | --- |
| contextpilot_intra | 96.64% | 12.33 ±0.03 | 43.9 ±0.8 |
| **tooltrie_v0** | **87.19%** | **17.35 ±0.65** | **50.3 ±0.6** |
| schema_cost_fitted | 39.69% | 39.35 ±0.77 | 71.7 ±1.8 |
| frequency_fitted | 39.69% | 39.87 ±0.26 | 73.0 ±0.5 |
| fp_tree_conditional | 39.69% | 39.85 ±0.54 | 73.0 ±0.7 |
| conditional_pair_triple | 39.69% | 39.93 ±0.14 | 72.7 ±0.5 |
| conditional_pair | 39.69% | 39.81 ±0.57 | 72.9 ±0.6 |
| alphabetical | 38.13% | 40.33 ±0.51 | 73.6 ±0.6 |
| original | 1.19% | 53.33 ±1.19 | 85.5 ±1.6 |
| **cacheweaver** | **1.19%** | 53.73 ±0.47 | 85.9 ±0.4 |

| ToolRet | cached ratio | TTFT s |
| --- | --- | --- |
| contextpilot_intra | 96.30% | 13.00 ±0.56 |
| **tooltrie_v0** | **83.58%** | **19.59 ±0.48** |
| alphabetical | 51.05% | 34.86 ±0.21 |
| schema_cost_fitted | 41.87% | 38.27 ±0.22 |
| frequency/pair/triple/fp_tree | 41.56–41.58% | ~38.0 |
| cacheweaver | 22.46% | 44.09 ±1.79 |
| original | 13.87% | 48.14 ±1.30 |

Three findings:

1. **CacheWeaver does nothing on BFCL.** Its 1.19% is identical to `original`
   because Algorithm 1 returned the **unmodified input ordering on 200/200 BFCL
   requests** (180/200 on ToolRet). Verified directly against the input files,
   so this is a property of the algorithm on this workload, not a measurement
   artifact. It is designed for overlapping retrieved text evidence; tool menus
   drawn from a shared catalog do not present the overlap structure it seeks.
2. **All five fitted baselines collapse onto alphabetical.** Frequency,
   schema-cost, FP-tree, pair and triple land within 0.01pp of each other
   (39.69% on BFCL). Five different statistical policies, trained on disjoint
   data, rediscover essentially one stable global order — consistent with Task
   E's finding that the cache rewards *identical*, not *important*.
3. **ContextPilot leads — and §2a shows this is NOT an artifact of its offline
   regime.** An earlier draft of this report claimed ContextPilot was "not a
   competitor" because it sees the whole evaluation batch. That claim was tested
   and is **wrong**; see §2a.

**Reproducibility.** `tooltrie_v0` measured 87.19% and `original` 1.19%, matching
Phase 1's independent run to the digit on a different day and GPU.

## 2a. Information regime — the comparison is now standardised

The §2 table mixed two information regimes. ToolTrie-v0 is **causal**: it may use
only already-served requests, because a deployed server cannot see the future.
ContextPilot is **offline**: `fit_transform` is fitted over the entire evaluation
batch. Comparing 87.19% against 96.64% therefore conflated *which algorithm* with
*how much it was allowed to see*.

Both missing cells were built and measured. `contextpilot_causal` recomputes the
ordering for request *n* from `fit_transform(contexts[0..n])` only.
`tooltrie_offline` grants ToolTrie the batch: a trie observes every request's
ordering, all requests are re-planned against it, iterated to a fixpoint
(converged in 2 iterations on both datasets) with the recency window disabled.

| condition | regime | BFCL cached | ToolRet cached |
| --- | --- | --- | --- |
| contextpilot_intra | offline | 96.64% | 96.30% |
| **contextpilot_causal** | **causal** | **96.16%** | **94.82%** |
| **tooltrie_v0** | **causal** | **87.19%** | **83.58%** |
| alphabetical | causal | 38.13% | 51.05% |
| tooltrie_offline | offline | 29.96% | 43.20% |

**Two corrections follow, both against the earlier draft.**

1. **ContextPilot's advantage is not the regime.** Restricting it to causal costs
   only **0.48pp** on BFCL and 1.48pp on ToolRet. It still beats ToolTrie-v0 by
   ~9pp as a legitimate causal policy. The previous framing — that 96.6% was an
   artifact of batch visibility — does not survive the test, and it was the most
   load-bearing caveat in this report.
2. **ToolTrie's causality is load-bearing, not a handicap.** Given the batch it
   *collapses*, 87.19% → **29.96%**, below plain alphabetical. The mechanism is
   self-reinforcement: early requests establish a path, later ones follow it, and
   the shared prefix converges. Treating the batch atemporally destroys exactly
   that. "Offline" is therefore not an upper bound for ToolTrie — it is a
   different, worse algorithm.

Caveat: `tooltrie_offline` is our construction, one of several possible offline
formulations. It shows *this* formulation fails, not that none could work.

**Every other condition is already causal**, verified rather than assumed: the
five fitted policies train on disjoint data (200 evaluation tasks vs 1,040
training tasks, **intersection empty**), ToolTrie and CacheWeaver are strictly
plan-before-observe, and alphabetical/original do no learning. ContextPilot was
the only non-causal condition in the study.

**Cross-arm replication.** The causal result is not specific to one engine or to
the unsanitized schemas. `contextpilot_causal` was replayed 3× on each of the
three systems arms:

| arm | BFCL cached | ToolRet cached |
| --- | --- | --- |
| vLLM (unsanitized) | 96.16% | 94.82% |
| vLLM (sanitized) | 96.18% | 94.86% |
| SGLang | 96.38% | 95.09% |

Maximum spread across arms is **0.22pp** (BFCL) and **0.27pp** (ToolRet), the
same order as every other condition's cross-arm variation, so the ~9pp gap over
ToolTrie-v0 replicates on a second engine with an independent cache
implementation. All six SGLang runs passed the same independent counter check
applied to the other 66 (index 0 the only omitted `cached_tokens`, totals
reconciling, zero failed requests).

**Still outstanding:** the n=800 quality replay for `contextpilot_causal` (§4).
Until it lands, the §4 table shows only the offline ContextPilot variant, and no
causal claim about ContextPilot's *accuracy* cost should be read from it. The §5
systems tables are complete.

## 3. ContextPilot scheduling table (kept separate)

`contextpilot_intra` preserves the request sequence; `contextpilot_intra_schedule`
enables the inter-context scheduler and does reorder requests on ToolRet.
Enabling it changed cached ratio by **+0.00pp on every dataset and both engines**
(BFCL and ToolRet; vLLM unsanitized, vLLM sanitized, SGLang).

This is a valid negative result: on these workloads the inter-context scheduler
contributes nothing beyond within-request tool ordering. No scheduling gain may
be attributed to tool ordering, and none is claimed.

## 4. BFCL correctness, n=800 (160 per domain), Qwen3-8B

| condition | function-name | full | no-tool |
| --- | --- | --- | --- |
| alphabetical | 82.81% | 76.41% | **89.38%** |
| tooltrie_v0 | 83.75% | 77.66% | 87.50% |
| fp_tree_conditional | 83.91% | 77.97% | 88.12% |
| conditional_pair_triple | 83.91% | 77.81% | 88.12% |
| cacheweaver | 84.84% | 78.91% | 82.50% |
| contextpilot_intra | **85.31%** | **78.91%** | 82.50% |

Paired differences versus alphabetical, task-clustered 95% CI, `*` = excludes zero:

| condition | name_correct | full_correct | no_tool_correct |
| --- | --- | --- | --- |
| tooltrie_v0 | +0.94 [−0.47, +2.34] | +1.25 [−0.47, +3.12] | −1.88 [−4.38, +0.00] |
| cacheweaver | +2.03 [+0.47, +3.75]* | +2.50 [+0.47, +4.53]* | −6.88 [−11.25, −2.50]* |
| fp_tree_conditional | +1.09 [+0.16, +2.19]* | +1.56 [+0.31, +2.97]* | −1.25 [−3.12, +0.00] |
| conditional_pair_triple | +1.09 [+0.16, +2.19]* | +1.41 [+0.16, +2.66]* | −1.25 [−3.12, +0.00] |
| contextpilot_intra | +2.50 [+0.94, +4.06]* | +2.50 [+0.62, +4.38]* | −6.88 [−11.25, −3.12]* |

**A systematic trade-off appears across all six orderings.** Ranking by
function-name accuracy produces almost exactly the reverse ranking on no-tool
accuracy: alphabetical is worst at selection (82.81%) and best at declining
(89.38%); ContextPilot is best at selection (85.31%) and tied-worst at declining
(82.50%). An ordering that makes the right tool easier to find also makes the
model likelier to call *something* when it should decline. This is the sharpest
evidence yet on brief question 6 (quality and safety), and it is not specific to
ToolTrie — ToolTrie sits in the middle of the trade-off, with a *smaller*
no-tool cost than either CacheWeaver or ContextPilot.

Note on scope: each metric is only defined on its applicable subset —
name/full on the 640 relevance cases, no-tool on the 160 irrelevance cases. The
runbook's §6 instruction to compare all three metrics on the mixed 800-task set
cannot be executed as written; the comparisons above are metric-scoped. The §1
targeted run remains the primary no-tool analysis, and its interval excludes
zero where this smaller 160-task arm's does not.

## 5. SGLang/RadixAttention — separate engine table

Identical sanitized **input files** on both engines (see §6), Qwen3-0.6B, 3 trials.
Both ContextPilot variants are listed; only **contextpilot_causal** shares an
information regime with ToolTrie-v0 and the other conditions, so it is the row to
compare against (§2a). `contextpilot_intra` is retained as the offline reference.

**The rendered prompts are not identical, and this bounds what may be compared.**
vLLM serves with `--tool-call-parser hermes`, SGLang with `--tool-call-parser
qwen`, so each engine serializes the same tool definitions through a different
chat template. SGLang's rendering costs **exactly +640 tokens on every request**
(6,868 → 7,508 on BFCL request 0; the delta is constant across all 200, so it is
fixed template overhead, not content drift). Totals therefore differ by ~9%
(1,380,294 vs 1,508,294 on BFCL).

Consequently **only cached *ratios* are cross-engine comparable** — each is
normalized by its own engine's prompt tokens. Absolute token counts, prefill
tokens, and TTFT are not. That the ratios still agree to ~0.2pp *despite* a
different prompt rendering strengthens rather than weakens the conclusion: the
ordering effect survives both a different cache architecture and a different
template.

| BFCL | vLLM cached | vLLM TTFT | SGLang cached | SGLang TTFT |
| --- | --- | --- | --- | --- |
| contextpilot_intra *(offline)* | 96.67% | 12.55 | 96.86% | 37.74 |
| **contextpilot_causal** | **96.18%** | **15.35** | **96.38%** | **44.25** |
| tooltrie_v0 | 87.11% | 16.82 | 87.29% | 44.68 |
| fitted policies | 39.69% | ~39 | 39.58% | ~72 |
| alphabetical | 38.13% | 39.47 | 38.02% | 72.74 |
| cacheweaver / original | 1.19% | 52.4 | 1.19% | 89.0 |

| ToolRet | vLLM cached | SGLang cached |
| --- | --- | --- |
| contextpilot_intra *(offline)* | 96.33% | 96.56% |
| **contextpilot_causal** | **94.86%** | **95.09%** |
| tooltrie_v0 | 83.55% | 83.74% |
| alphabetical | 51.06% | 49.94% |
| cacheweaver | 22.46% | 17.75% |
| original | 13.88% | 11.29% |

**The ordering effect replicates on a different engine with a different cache
architecture.** Cached ratios agree to within ~0.2pp on BFCL (87.11% vs 87.29%
for ToolTrie) and the condition ranking is identical on both engines. That the
same prompt-layer reordering yields the same reuse under vLLM's paged APC and
SGLang's radix tree is strong evidence the mechanism is a property of the
ordering, not of one implementation.

**SGLang's absolute TTFT is 2–3× vLLM's here, and that must not be attributed to
RadixAttention.** The entire engine differs — scheduler, attention backend
(flashinfer, JIT-compiled on this host), and memory configuration
(`--mem-fraction-static 0.85` versus vLLM's 0.92 default). Only the *relative*
behaviour within each engine is comparable. ToolRet shows slightly lower SGLang
reuse in the low-reuse conditions (17.75% vs 22.46% for cacheweaver), which is a
real engine difference worth noting but not diagnosed here.

## 6. Failures, contamination, and limitations

**Both benchmarks ship invalid JSON Schema, at scale.** vLLM tolerates it;
SGLang validates against the metaschema and rejects the entire request. Two
independent classes:

1. **Python type names** — `float`, `dict`, `int`, `str`, `String`,
   `int, optional`, `List[str]`, `datetime`. Affects **74 distinct BFCL tools and
   187 ToolRet tools, in 200/200 requests of both datasets**.
2. **Structural malformation** — one ToolRet tool declares a parameter named
   `title` at the schema root, where JSON Schema reserves `title` as a string
   annotation. One tool in one request, but enough to abort a whole 200-request
   replay.

§7 could not run on the original workloads at all. A sanitizer was applied
**identically to all 22 condition files**, normalizing only the `type` string
inside `function.parameters` and relocating root-level property definitions;
tool names, tool order, `tool_ids`, request order, and menu membership were
asserted unchanged. Both engines then ran the same sanitized inputs.

Sanitized and unsanitized numbers are **not comparable to each other** (repairing
types changes tool text, hence token counts). The unsanitized vLLM arm is
retained as the continuity arm and is what §2 reports; §5 uses the sanitized pair
throughout. In practice the two agree closely (87.19% vs 87.11%), so the
sanitization was near-neutral, but they are reported separately regardless.

**Quarantined:** 33 ToolRet vLLM-sanitized replays in
`quarantine-prefix-schemafix/`. They ran cleanly but predate the structural fix,
so their inputs differ from what SGLang could accept. Superseded, not
contaminated. BFCL was unaffected (the relocation never fires there), so 66 BFCL
replays were preserved rather than redone.

**SGLang counter validation.** SGLang omits
`usage.prompt_tokens_details.cached_tokens` when it is zero, so the first request
after every flush reports `None` and the replay script's "every response reported
a value" clause fails on every otherwise-clean run. Runs used
`--allow-counter-mismatch` plus a **stricter** independent check: request and
prompt counters must match, aggregate `sglang:cached_tokens_total` must equal the
sum of reported per-request values, and index 0 must be the only omission.
**66/66 runs passed.** Reconciled copies (flag corrected, justification recorded)
are in `sglang-reconciled/`; originals are untouched.

**Deviations from the runbook**, all forced and recorded in `environment.txt`:
`uv` instead of `python3.12 -m venv` (no `ensurepip`, no sudo);
`--prerelease=allow` (sglang 0.5.15.post1 hard-pins `flash-attn-4==4.0.0b15`, a
pre-release — itself a reproducibility hazard, hence the frozen dep list);
`ninja`/`cmake` added to the SGLang venv for flashinfer's JIT.

**Limitations.** One model family; one menu size (64); one GPU type. Quality runs
are single-pass, justified by `temperature=0, seed=0` but batching
nondeterminism is not formally excluded. ToolRet is used for cache/TTFT only —
its retrieval labels are not call correctness. ContextPilot is offline and not a
deployable causal policy. No equivalence margin was declared for any metric, so
nothing here is an equivalence claim.

## Positive and negative regimes

**Where ToolTrie-v0 helps:** padded menus drawn from a stable shared catalog,
above the ~4-tool/433-token crossover, where requests repeat catalog structure.
+49pp cache reuse over alphabetical on BFCL, 2.3× TTFT, replicated on two
engines.

**Where it does not:** raw benchmark menus below the crossover (median 1 tool per
task) — nothing to reorder. And on correctly declining irrelevant requests, where
it is measurably worse than plain alphabetical ordering.

**Recommended next step.** The no-tool regression is now the substantive open
problem, and §4 shows it is a general property of reuse-optimizing orderings
rather than a ToolTrie bug. The natural response is the brief's §9.2 extension —
separate the tools present in cached context from the tools that may actually be
called, via an active-tool manifest and constrained decoding — which targets
exactly this failure mode without giving up the cache win.
