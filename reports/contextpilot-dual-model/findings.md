# ContextPilot dual-model replication — reconciled findings

## Verdict

The dual-model run is accepted for its predeclared scope. Qwen3-4B is the
primary model and Qwen3-0.6B is a separately reported replication. The run
contains 190 accepted GPU replays, and all 33 GPU-side audit checks pass.

The main systems result is robust across both model arms:

- both ContextPilot-derived ordering adaptations produce more exact-prefix
  reuse than ToolTrie-v0 in every one of the 12 model/workload cells;
- the large 82–96% reuse figures occur only on padded menus with a largely
  shared 64-tool catalog;
- on independently retrieved BM25 menus, reuse is only 0.3–18.7% across the
  measured conditions. Retrieval overlap, not the choice among the three
  adaptive orderers, is therefore the dominant bottleneck.

This does **not** make ContextPilot an overall quality winner. On the 4B fixed
sequence, its two adaptations lead ToolTrie-v0 on full-call accuracy but trail
the original-order fallback on no-tool accuracy. On 0.6B, original order has
the best relevance-side accuracy while alphabetical has the best no-tool
accuracy. No utility weighting or equivalence margin was predeclared, so no
single overall winner is claimed.

## What was verified

The GPU result branch `gpu/contextpilot-dual-model-20260809-004603` records
commit `d133670f33d9937a32a731bcdd06507faa7efc40`, whose parent is the exact
predeclared experiment commit
`15285704e73af680c0125ea4bfeb0b54a14f278e`.

Local verification found:

- the copied protocol manifest is byte-identical to the predeclared manifest;
- the FYP, ContextPilot, model, tokenizer, chat-template, and vLLM pins match;
- all 81 committed JSON artifacts parse;
- all 33 recorded audit checks pass and account for all 190 replays;
- all 12 systems summaries preserve case set, request sequence, and selected
  tool membership across conditions;
- in all 60 systems cells, cached plus computed prompt tokens equals rendered
  prompt tokens and the reported cache ratio recomputes from those counters;
- both ContextPilot adaptations exceed ToolTrie-v0 reuse in all 12
  model/workload cells;
- the full repository test suite passes: 119 tests.

The reproducible compact-artifact check is:

```bash
uv run python scripts/audit_contextpilot_dual_model_compact.py
```

It passes 15/15 local checks while explicitly reporting that it cannot verify
the server-only raw archive.

The raw archive itself could not be re-opened from the local analysis machine.
Only its recorded server-side verification and SHA-256 are available here:

```text
/home/taghan/contextpilot-dual-model-20260809-004603.tar.gz
3fc5f5ec08580c22dcee65ed015fcb96ae6b36598aeaaaf3eeb8aeb8aac62012
```

It remains a single-machine artifact until copied off the GPU server.

## Systems result

Reuse is
`prompt_tokens_cached / (prompt_tokens_cached + computed_prompt_tokens)` from
vLLM's aggregate counters. Each value is the mean of three reset replays; the
cached-token counts are identical across those trials.

### Qwen3-4B primary

Native capacity: 96,832 tokens.

| Workload | Original | Alphabetical | ToolTrie-v0 | CP persistent API | CP static refit |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL padded 64 | 1.19% | 37.99% | 87.19% | **96.16%** | **96.16%** |
| ToolRet padded 64 | 6.56% | 43.81% | 82.18% | 95.27% | **95.74%** |
| ToolRet BM25 k=4 | 15.87% | 15.28% | 17.48% | **18.72%** | 18.42% |
| ToolRet BM25 k=16 | 5.14% | 5.40% | 6.73% | 9.10% | **9.36%** |
| ToolRet BM25 k=64 | 0.72% | 0.98% | 1.63% | 3.07% | **3.48%** |
| ToolRet BM25 k=128 | 0.34% | 0.44% | 0.89% | 1.35% | **2.23%** |

### Qwen3-0.6B replication

Native capacity: 188,912 tokens.

| Workload | Original | Alphabetical | ToolTrie-v0 | CP persistent API | CP static refit |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL padded 64 | 1.19% | 38.13% | 87.19% | **96.16%** | **96.16%** |
| ToolRet padded 64 | 13.73% | 50.82% | 83.58% | 95.27% | **95.74%** |
| ToolRet BM25 k=4 | 15.87% | 15.28% | 17.48% | **18.72%** | 18.42% |
| ToolRet BM25 k=16 | 6.12% | 6.27% | 7.77% | **9.93%** | 9.80% |
| ToolRet BM25 k=64 | 0.91% | 1.24% | 1.90% | **4.78%** | 4.01% |
| ToolRet BM25 k=128 | 0.37% | 0.58% | 1.13% | 1.99% | **2.96%** |

The persistent API and static-refit adaptations are close, and neither
dominates the other. This suggests that their shared context-clustering order,
rather than persistence by itself, explains most of the measured gain in this
fixed sequence. A planner-state or sequence-replication study would be needed
to make that mechanism claim causal.

The cross-model differences at k=16–128 must not be attributed to model size
alone. The models have different native KV capacities, and ToolTrie is rebuilt
from each live capacity. Compare policies within a model; do not pool models or
compare their absolute TTFT.

## Quality result

The quality arm has 640 relevance and 160 irrelevance cases per condition.
Scores come from this repository's reduced BFCL-style AST checker, not the
official BFCL leaderboard evaluator.

| Condition | 4B full | 4B name | 4B no-tool | 0.6B full | 0.6B name | 0.6B no-tool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 76.09% | 83.13% | **88.12%** | **55.16%** | **73.75%** | 86.25% |
| Alphabetical | 73.28% | 82.19% | 85.62% | 43.59% | 60.47% | **94.37%** |
| ToolTrie-v0 | 75.31% | 83.75% | 87.50% | 53.28% | 69.06% | 93.13% |
| CP persistent API | **77.03%** | **84.38%** | 85.00% | 51.72% | 68.91% | 91.25% |
| CP static refit | **77.03%** | **84.38%** | 85.00% | 51.88% | 68.91% | 91.25% |

On 4B, the persistent adaptation is +1.72 points in full-call accuracy against
ToolTrie-v0 on this fixed sequence (descriptive 95% interval +0.31 to +3.12),
but only +0.94 against the original-order fallback (interval −0.78 to +2.81).
Its no-tool accuracy is −3.12 points against original (interval −6.25 to
−0.62), while its no-tool differences against alphabetical and ToolTrie both
include zero.

On 0.6B, the persistent adaptation is −3.44 full-accuracy points and −4.84
name-accuracy points against original, while improving no-tool accuracy by
5.00 points. It strongly beats alphabetical on relevance accuracy but does not
resolve a quality advantage over ToolTrie-v0.

These intervals resample tasks from one emitted planner sequence. Because
ToolTrie and both ContextPilot adaptations depend on earlier requests, they do
not generalize over alternative request orders. Each condition also has only
one complete quality replay, so the intervals do not include run-to-run model
generation variation.

The earlier Qwen3-4B addendum and this fresh 4B run differ by at most one scored
case per shared condition and metric. The direction of the main comparisons is
unchanged, but the small mismatch confirms that a fixed seed does not make the
whole GPU/tool-calling stack bit-identical. Publication-grade quality claims
should use complete sequence replicates or multiple inference seeds.

## Corrections to the executor handover wording

The executor handover says the 4B result shows the 0.6B quality behavior was a
"small-model artifact." That is stronger than this design supports and
conflicts with the manifest's reporting constraint. The defensible statement
is:

> Ordering effects on quality are model-sensitive and substantially larger in
> the 0.6B arm. Qwen3-4B is the predeclared primary model; the 0.6B results
> should be reported as a small-model replication, not generalized to larger
> models.

Only one model checkpoint was tested at each size, and the capacity-dependent
ToolTrie condition differs across them. This run is not a pure causal estimate
of model-size effects.

The handover also states that a separate Qwen3-8B static-refit cell and a 72/72
SGLang audit were completed. Neither compact handover is present in commit
`d133670`, and no fetched remote branch contains a newer supporting commit.
Those two claims must remain unintegrated until the GPU session pushes their
branch, commit, summaries, and provenance.

## Scope limits

- Neither ContextPilot condition is the full ContextPilot system. Both omit
  eviction feedback, relevance annotations, and de-duplication.
- Planning was performed before serving. The cache comparison is valid for the
  emitted orders, but it is not an end-to-end latency comparison including
  planner overhead.
- ToolRet retrieval uses deterministic BM25, not the official dense retriever.
- Quality is measured on padded BFCL menus, not on an end-to-end retrieved-menu
  tool-selection pipeline.
- A separate Qwen3-8B job ran on GPU 3 for 23 minutes during the 4B systems
  arm. Clean per-server counters protect cache-ratio claims, but latency samples
  from the overlap should not be treated as an uncontaminated host-isolation
  experiment without identifying the affected raw replay windows.
- No inactive-tool retention, active-tool manifest, constrained invocation, or
  full ContextPilot condition was executed.

## Research consequence

ToolTrie-v0 remains a valid, simple causal baseline, but it is not the best
measured ordering rule. Redesigning the trie solely to win the padded-menu
ordering contest is unlikely to be the strongest next contribution. The more
important result is that all orderers lose most reuse after realistic
retrieval. The next publishable experiment should therefore target the gap
between retrieval and cache retention: safe, budgeted retention of inactive
tools with an explicit active-tool manifest and the ordinary selected-tool
text path as a predeclared fallback.

## Immediate actions

1. Copy all raw archives off the GPU server and verify their SHA-256 values and
   `tar -tzf` readability at the destination.
2. Ask the GPU session to push separate compact evidence for the claimed 8B
   static-refit closure and SGLang 72/72 audit.
3. Do not rerun the accepted 190-replay dual-model matrix.
4. Before a publication-quality quality claim, predeclare complete request-
   sequence replicates or multiple inference seeds and, ideally, validate with
   the official BFCL evaluator.

## Provenance

- GPU executor record: [`20260809-004603/HANDOVER.md`](20260809-004603/HANDOVER.md)
- Predeclared protocol: [`protocol-manifest.json`](20260809-004603/protocol-manifest.json)
- GPU audit: [`dual-model-audit.json`](20260809-004603/dual-model-audit.json)
- Qwen3-4B compact scores: [`quality-scores-compact.json`](20260809-004603/qwen3-4b/quality-scores-compact.json)
- Qwen3-0.6B compact scores: [`quality-scores-compact.json`](20260809-004603/qwen3-0.6b/quality-scores-compact.json)
