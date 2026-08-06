# Initial-brief closure findings

## Verdict

The explicit Tasks A–F in `initial-research-brief.md` are substantively
answered. The new GPU run closes the previously missing retrieved-menu,
ordinary-fallback, direct partial-reuse, and exact rendered-token/block audits.

One stricter acceptance item is still open: the project’s later gap-closure
manifest required all 24 memory-pressure regime-runs to reach 90% live KV
occupancy. The runs were clean, but the original 190,896-token cache was too
large for a sequential approximately 7k-token request, so the result was 0/24.
The initial brief can be described as **answered**, but the enhanced systems
closure package must not be described as **fully accepted** until the separately
predeclared controlled-cache rerun passes.

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

The corrected execution record is
[`HANDOVER.md`](20260805-222246-gpu-executor/HANDOVER.md). The raw archive is
currently only on the GPU server at
`/home/taghan/initial-brief-closure-20260805-222246.tar.gz`; it still needs an
owner-approved second physical copy.

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

## 2. Exact cache reuse on true retrieved menus

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
5. **Simple pair/triple statistics add little on the measured fitted arm.** The
   Phase 2 pair, triple, frequency, schema-cost, and FP-tree fitted policies
   collapse to within 0.01 percentage points of one another and alphabetical.
   ContextPilot’s stronger clustering is the important competitor, not evidence
   that the current pair/triple adaptation is sufficient.
6. **Request order and causality matter.** ToolTrie-v0’s causal
   self-reinforcement is load-bearing; its tested offline fixpoint variant
   collapses. The finite-cache pressure comparison is the remaining controlled
   systems check.
7. **Ordering changes quality and safety.** The n=800 Qwen3-8B Phase 2 run finds
   a reuse/selection/no-tool frontier: causal ContextPilot leads reuse and call
   accuracy but has a larger no-tool penalty; ToolTrie is safer on no-tool
   requests but does not lead reuse.
8. **Trie ordering provides little benefit when menus share little exact
   prefix, prompts are below the latency floor, or retrieval misses the needed
   tool.** The true retrieved-menu arm demonstrates all three limits.

## 7. Decision and fallback

The initial hypothesis is refined, not rejected. Exact tool ordering can create
reuse, but reuse is not enough by itself: retrieval coverage, menu overlap,
prompt cost, no-tool safety, and cache capacity determine whether it is useful.

For now, the system should send the ordinary retrieved tool set through normal
text prefill whenever expected savings do not exceed context, decode, retrieval,
or safety costs. No inactive tools are retained and no KV tensors are composed
in that fallback. The next publication-oriented experiment may study safe
retention, but it must be evaluated against causal ContextPilot and this
fallback after the controlled pressure acceptance run is complete.

## Traceable artifacts

- Retrieval curve: [`retrieval-bm25-sweep.json`](../retrieval-bm25-sweep.json)
- Four replay matrices: [`20260805-222246-gpu-executor/`](20260805-222246-gpu-executor/)
- Corrected GPU handover: [`HANDOVER.md`](20260805-222246-gpu-executor/HANDOVER.md)
- Pressure failure record: [`PRESSURE-QUARANTINE.md`](20260805-222246-gpu-executor/PRESSURE-QUARANTINE.md)
- Phase 2 causal/quality comparison:
  [`tooltrie-phase2/findings.md`](../tooltrie-phase2/findings.md)
- Controlled pressure rerun instructions:
  [`NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md`](../../NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md)
