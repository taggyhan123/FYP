# Initial-brief closure findings

## Verdict

The explicit Tasks A–F in `initial-research-brief.md` and the project’s stricter
gap-closure execution manifest are now **substantively complete**. BM25-retrieved
menu serving, the ordinary
fallback, direct partial reuse, exact rendered-token/block auditing, and
controlled memory pressure all have accepted GPU evidence.

The pressure criterion that initially failed 0/24 now passes 24/24 under the
separately predeclared controlled-cache protocol. This completes the planned
initial experiment stage, but formal publication-grade closure still requires
the corrected analysis paths, artifact-to-report audit, and a second durable
copy of the raw archives. Section 9 retained-tool and KV-composition work remains
a later extension rather than part of this result.

One initial experimental sub-question remains partial: schema-token-weighted
frequency was tested, but a policy weighted by separately measured per-schema
prefill time was not.

## Evidence and provenance

The measurements ran on an isolated RTX 3090 with vLLM 0.26.0 and
`Qwen/Qwen3-0.6B`. Live APC capacity was read from the server as 190,896 tokens
(16 tokens/block × 11,931 blocks).

- Measurements were executed at commit `65b86bacfbcc227c2b21223682018a7b29dd3e94`.
- The worktree advanced during the run to `917d560`, which added only
  `AGENTS.md`; no code, data, script, model, or configuration changed.
- All 84 primary replays are present and clean: four menu sizes × seven
  conditions × three trials, with 200 requests per trial.
- All seven k64 rendered-prefix audits are clean.
- The complete raw archive contains 221 readable entries and has SHA-256
  `568dedaea859056a4c3d2cb4773a810364b40b03858efc83092c5d0804d4a7d5`.
- Warmup, contended, orphaned, and parallel-session artifacts were retained in
  quarantine and excluded from the accepted summaries.
- The pressure rerun executed from pinned commit `caf8ab576eb15d95265a53ec76c772c8af6c7929`
  in a detached worktree with a separately read-back capacity of 7,680 tokens
  (16 × 480 blocks).
- Its 24/24 regime-runs passed all 384 checks, with raw archive SHA-256
  `d18c613958dbeadbb9114a7ec6c1418d1c062803075974940b470791e86a78e0`.

The corrected execution record is
[`HANDOVER.md`](20260805-222246-gpu-executor/HANDOVER.md). The raw archive is
currently only on the GPU server at
`/home/taghan/initial-brief-closure-20260805-222246.tar.gz`; it still needs an
owner-approved second physical copy. The pressure archive is likewise only at
`/home/taghan/initial-brief-pressure-rerun-20260807-005414.tar.gz` on the same
physical disk.

## 1. Retrieval is a separate bottleneck

BM25 selected tools without reading each evaluation query’s gold IDs. Gold
labels were used afterward only to score retrieval. This is a reproducible
lexical baseline, not the official ToolRet retriever and not production
popularity evidence.

| Retrieved tools | Macro recall | Any-gold hit rate | MRR |
| ---: | ---: | ---: | ---: |
| 4 | 41.71% | 51.50% | 0.3858 |
| 16 | 55.04% | 65.50% | 0.4022 |
| 64 | 64.21% | 75.50% | 0.4061 |
| 128 | 67.54% | 80.50% | 0.4066 |

Increasing the menu improves coverage, but even k=128 misses every gold tool
on 19.5% of queries. No ordering or cache method can recover a tool absent from
the retrieved set.

## 2. Exact cache reuse on BM25-retrieved menus

Every condition received the same selected tool set for each query; only its
order changed. `original` is the explicit ordinary selected-tool text-prefill
fallback in BM25 rank order. `FP-tree global` is an FP-tree-derived global
ordering, not a claim that FP-Growth itself produced these systems results.

| k | Original fallback | Alphabetical | Random seed 42 | Frequency | Schema-cost weighted | FP-tree global | ToolTrie-v0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 15.87% | 15.28% | 14.17% | 14.62% | 14.84% | 14.62% | **17.48%** |
| 16 | 6.12% | 6.27% | 5.39% | 5.59% | 6.24% | 5.59% | **7.77%** |
| 64 | 0.91% | 1.24% | 0.96% | 0.96% | 1.09% | 0.96% | **1.90%** |
| 128 | 0.37% | 0.58% | 0.59% | 0.54% | 0.57% | 0.54% | **1.13%** |

ToolTrie-v0 is best at all four sizes, but the absolute gain over ordinary text
prefill is small: +1.61, +1.65, +0.99, and +0.76 percentage points. This is a
positive ordering result and a negative magnitude result. Independently
retrieved menus do not share enough exact rendered prefix for the much larger
reuse seen in shared padded-catalog experiments.

Frequency and schema-cost weighting do not win consistently. Frequency and the
current FP-tree global adaptation are identical because both reduce to the same
global support order in this baseline.

## 3. Prompt cost and TTFT

The TTFT columns below divide each three-trial mean aggregate by 200 requests.
They are descriptive, not a paired significance test.

| k | Prompt tokens/query | Fallback reuse | Best static reuse | ToolTrie reuse | Fallback TTFT/query | ToolTrie TTFT/query |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 640 | 15.87% | 15.87% | 17.48% | 32.3 ms | 31.2 ms |
| 16 | 2,139 | 6.12% | 6.27% | 7.77% | 70.7 ms | 70.3 ms |
| 64 | 8,361 | 0.91% | 1.24% | 1.90% | 349.7 ms | 350.0 ms |
| 128 | 16,267 | 0.37% | 0.59% | 1.13% | 937.0 ms | 933.5 ms |

Prompt cost grows about 25× from k=4 to k=128 while retrieval macro recall
rises only 25.83 percentage points. The three-trial TTFT intervals overlap, and
no paired-difference interval was predeclared. These results therefore do not
establish that ToolTrie makes retrieved-menu serving faster.

At k=64, both direct partial-reuse conditions contained 199 genuinely partial
requests and one cold request in every trial:

| Condition | Partial cached ratio | Partial prefill/query | Partial TTFT/query |
| --- | ---: | ---: | ---: |
| Alphabetical | 1.37% | 309.0 ms | 351.7 ms |
| ToolTrie-v0 | 2.12% | 307.4 ms | 350.5 ms |

This confirms that the partial stratum exists and was measured directly. The
latency difference is too small to support a speed claim.

## 4. Exact rendered-token and block audit

All seven k64 audits passed every declared validation:

- cache reset before measurement;
- server `/tokenize` count equals completion prompt usage;
- cached plus computed tokens equal rendered prompt tokens per request;
- measurement and counter identities are consistent;
- block size is the live server value of 16.

The compact validation record is
[`audit-k64-validation-summary.json`](20260805-222246-gpu-executor/audit-k64-validation-summary.json).
Exact rendered token IDs, block boundaries, best-prior full-block prefixes,
cache hits, prefill, and TTFT remain in the checksummed raw archive rather than
Git. The audit closes the rendered-token/block evidence gap without treating
canonical schema-token estimates as server tokenization.

## 5. Memory-pressure result and correction

All six static orderings ran under empirical, uniform, skewed, and
session-bursty workloads. All 24 regime-runs had clean counters and cache
resets, but sampled peak occupancy was only 3.6379–3.6882%, so none met the
predeclared 90% criterion.

This is not evidence that vLLM cannot experience pressure. The sequential
client had at most one request running, none waiting, and approximately
6,945–7,041 resident tokens in a 190,896-token cache. Cumulative prompt volume
of 7.23× cache capacity is throughput, not simultaneous residency.

The original outputs remain valid low-occupancy measurements and are
quarantined only from **pressure** claims. The replacement protocol changes one
controlled variable—the cache is capped at 480 blocks/7,680 tokens—while
keeping concurrency one and the threshold at 90%. It also requires a positive
sampled eviction count in every regime. Latency may not be compared across the
two cache capacities. See
[`initial-brief-pressure-rerun-manifest.json`](../../cluster/initial-brief-pressure-rerun-manifest.json).

The controlled rerun passes every predeclared criterion:

| Controlled-pressure ordering | Empirical reuse | Uniform reuse | Skewed reuse | Session-bursty reuse | Sampled evictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 1.18% | 0.69% | 1.24% | 0.69% | 84,810–85,340 |
| Alphabetical | 29.21% | 29.35% | 28.06% | 27.76% | 60,600–61,948 |
| Random seed 42 | **32.16%** | **31.27%** | **32.67%** | **30.19%** | **57,696–59,854** |
| Frequency = schema-cost = FP-tree | 9.44% | 8.99% | 9.51% | 9.00% | 77,682–78,176 |

Across all 24 regime-runs:

- peak occupancy is 91.02–91.86% against the unchanged 90% threshold;
- sampled evictions are 57,696–85,340 at 100% metric sampling;
- preemptions are zero;
- peak running/waiting requests are 1/0, preserving the sequential client;
- all 384 validator checks pass.

The table is a **within-7,680-token controlled stress comparison**. Its reuse or
latency must not be compared numerically with the earlier 190,896-token cache,
and it is not evidence of natural production pressure. Only empirical and
session-bursty are matched permutations; uniform and skewed resample content.

Fixed random seed 42 leads reuse and minimizes eviction in every regime, but one
run per cell and one seed cannot establish that random ordering is generally
superior; this pressure matrix has no repeated-trial uncertainty interval. It is
a sensitivity result that motivates repeated random-seed sweeps, not a new
deployment recommendation.

### Four distinct orderings, not six

The compact GPU output omitted tool IDs, so the local analysis reconstructed
the exact deterministic workloads from the pinned parameters. For all four
regimes and all 200 requests, `frequency`, `schema_cost_weighted`, and
`fp_tree_global` have identical ordered tool-ID arrays and SHA-256 sequence
digests. The matrix contains 24 valid executions but only four distinct
ordering policies on this workload.

The direct audit is
[`ordering-equivalence.json`](../initial-brief-pressure-rerun/ordering-equivalence.json),
generated by
[`audit_pressure_ordering_equivalence.py`](../../scripts/audit_pressure_ordering_equivalence.py).
This strengthens the executor’s aggregate-based inference. It also narrows two
overbroad handover statements: sampler counts and one sampled occupancy value
are not identical across the three executions, and the tracked environment file
does not contain the traceback that its prose says was retained. Neither issue
changes the engine versions, emitted sequence equivalence, or 24/24 acceptance.

## 6. Answers to the initial experimental questions

1. **Tool-schema prefill materially affects TTFT above the small-prompt floor.**
   The earlier controlled sweep found no resolvable benefit at 303 tokens but a
   4.5× warm-cache TTFT speedup at 64 padded tools and 10.6× at 200 tools.
2. **Reuse is available without changing selected membership, but depends on
   set overlap.** Ordinary retrieved-order reuse falls from 15.87% at k=4 to
   0.37% at k=128; deterministic reordering can improve it without adding or
   removing tools.
3. **Frequency ordering is not reliably better.** It can destroy the shared
   prefix by placing request-specific tools first and is not the winner on the
   retrieved arm.
4. **Schema-cost weighting is not reliably better either.** It narrowly beats
   several static controls in some rows but never beats ToolTrie here.
5. **Simple pair/triple statistics add little on the measured fitted arm.** On
   BFCL, exact reconstruction shows that the Phase 2 pair, triple, frequency,
   schema-cost, and FP-tree fitted labels emit one byte-identical 200-request
   tool sequence; they are one tested behavior, not five independent policies.
   On ToolRet there are three fitted sequences: frequency/pair/triple are
   identical, while schema-cost and FP-tree each differ. Alphabetical is
   substantially higher than all of them on the measured ToolRet arm.
   The ContextPilot-derived static-refit ordering is the important historical
   competitor, not evidence that the current pair/triple adaptation is
   sufficient. A later persistent-API adaptation also leads ToolTrie on every
   retrieved menu size; neither ordering-only arm is the full ContextPilot
   system.
6. **Request order and causality matter.** ToolTrie-v0’s causal
   self-reinforcement is load-bearing; its tested offline fixpoint variant
   collapses. Under controlled pressure, empirical and session-bursty also
   produce different reuse despite being permutations of one task multiset.
7. **Ordering changes quality and can change safety.** The n=800 Qwen3-8B runs
   find a fixed-sequence reuse/selection/no-tool frontier: both
   ContextPilot-derived adaptations lead ToolTrie on reuse and relevance-side
   point estimates, while ToolTrie has higher no-tool accuracy. The persistent
   arm's −5.00-point no-tool difference against alphabetical does not reproduce
   at Qwen3-4B (0.00 points), so a universal safety penalty is not supported.
   These are ordering-only results, not full-system quality.
8. **Trie ordering provides little benefit when menus share little exact
   prefix, prompts are below the latency floor, or retrieval misses the needed
   tool.** The BM25-retrieved-menu arm demonstrates all three limits.

## 7. Decision and fallback

The initial hypothesis is refined, not rejected. Exact tool ordering can create
reuse, but reuse is not enough by itself: retrieval coverage, menu overlap,
prompt cost, no-tool safety, and cache capacity determine whether it is useful.

For now, the system should send the ordinary retrieved tool set through normal
text prefill whenever expected savings do not exceed context, decode, retrieval,
or safety costs. No inactive tools are retained and no KV tensors are composed
in that fallback. The controlled-pressure gate is now complete, so the next
publication-oriented experiment may study safe retention. It must still be
   evaluated against this fallback, the historical static-refit adapter, and
   the now-measured persistent-API adaptation at `alpha=0.001`. It should include
   a random-seed sensitivity sweep rather than promoting seed 42 as an
   algorithm.

## Traceable artifacts

- Retrieval curve: [`retrieval-bm25-sweep.json`](../retrieval-bm25-sweep.json)
- Four replay matrices: [`20260805-222246-gpu-executor/`](20260805-222246-gpu-executor/)
- Corrected GPU handover: [`HANDOVER.md`](20260805-222246-gpu-executor/HANDOVER.md)
- Pressure failure record: [`PRESSURE-QUARANTINE.md`](20260805-222246-gpu-executor/PRESSURE-QUARANTINE.md)
- Phase 2 causal/quality comparison:
  [`tooltrie-phase2/findings.md`](../tooltrie-phase2/findings.md)
- ContextPilot historical-arm provenance correction:
  [`contextpilot-causal-provenance-correction.json`](../tooltrie-phase2/contextpilot-causal-provenance-correction.json)
- Corrected ContextPilot confirmation runbook:
  [`NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md`](../../runbooks/NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md)
- Reconciled ContextPilot confirmation findings:
  [`contextpilot-confirmation/findings.md`](../contextpilot-confirmation/findings.md)
- Controlled pressure rerun instructions:
  [`NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md`](../../runbooks/NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md)
- Accepted pressure handover:
  [`20260807-005414/HANDOVER.md`](../initial-brief-pressure-rerun/20260807-005414/HANDOVER.md)
- Pressure sequence-equivalence audit:
  [`ordering-equivalence.json`](../initial-brief-pressure-rerun/ordering-equivalence.json)
