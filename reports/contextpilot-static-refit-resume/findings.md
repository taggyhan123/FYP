# ContextPilot static-refit resume and SGLang audit — reconciled findings

## Bottom line

The two previously untracked handoffs are now integrated and their compact
artifacts pass a local six-check audit:

- the corrected `alpha=0.001` static-refit adapter has 18/18 accepted
  Qwen3-0.6B systems replays across six workloads;
- its missing Qwen3-8B n=800 quality cell is present;
- the independent aggregate-counter audit accepts all 72 historical SGLang
  replays and refuses none.

This closes the static-refit implementation gap and the SGLang counter-cleanliness
gap. It does not create an ordinary-original quality control at 8B, and it does
not make either ordering-only ContextPilot adaptation the full ContextPilot
system.

## Static-refit systems result

Qwen3-0.6B, 200 requests, three clean reset trials per workload:

| Workload | Original | Alphabetical | ToolTrie-v0 | CP persistent API | CP static refit |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL padded-64 | 1.19% | 38.13% | 87.19% | **96.16%** | **96.16%** |
| ToolRet padded-64 | 13.87% | 51.05% | 83.58% | 95.27% | **95.74%** |
| BM25 k=4 | 15.87% | 15.28% | 17.48% | **18.72%** | 18.42% |
| BM25 k=16 | 6.12% | 6.27% | 7.77% | **9.93%** | 9.80% |
| BM25 k=64 | 0.91% | 1.24% | 1.90% | **4.78%** | 4.01% |
| BM25 k=128 | 0.37% | 0.58% | 1.13% | 1.99% | **2.96%** |

Every combined summary preserves the case set, request sequence, and selected
tool membership across all five conditions. The static builder records causal
prefix refitting, pinned ContextPilot commit `1fa0a143`, `alpha=0.001`, no
future batch, no request scheduling, no annotations, and no eviction feedback.

## Qwen3-8B quality cell

The static-refit replay scores 78.44% full-call accuracy, 84.84%
function-name accuracy, and 84.38% no-tool accuracy on 800 cases. Its aggregate
and six paired-comparison statistics are numerically identical to the
persistent-API replay:

| Comparator | Metric | Difference | Fixed-sequence 95% interval |
| --- | --- | ---: | ---: |
| Alphabetical | function name | +2.03 pp | +0.47 … +3.59 |
| Alphabetical | full call | +2.03 pp | +0.16 … +3.91 |
| Alphabetical | no tool | −5.00 pp | −8.75 … −1.88 |
| ToolTrie-v0 | function name | +1.09 pp | 0.00 … +2.19 |
| ToolTrie-v0 | full call | +0.78 pp | −0.31 … +1.88 |
| ToolTrie-v0 | no tool | −3.12 pp | −6.25 … −0.62 |

The executor reports zero per-case score differences between the two
ContextPilot APIs. The Git package proves identical aggregate, domain, and
paired statistics, but the two raw per-case score files remain only in the
server archive, so the local audit does not elevate the stronger zero-difference
claim beyond that provenance.

There is still no `original` condition in this historical Qwen3-8B n=800
matrix. Alphabetical is not a neutral fallback: in the accepted dual-model run
it lowers full accuracy relative to original by 2.81 points at 4B and 11.57
points at 0.6B. Therefore the 8B “ContextPilot gain” versus alphabetical cannot
be translated into a gain over ordinary selected-tool text prefill. The old 8B
server command did not pin a model revision, so one new control row is valid
only if the exact cached snapshot used by the old rows can be proved; otherwise
the defensible choices are a fresh pinned matrix or no 8B fallback comparison.

The −5.00-point 8B no-tool result also fails to reproduce as a resolved effect
at 4B: the accepted 4B difference from alphabetical is −0.62 point with an
interval spanning zero. The 8B fixed-sequence observation remains real; a
universal or model-independent safety claim does not.

## SGLang audit

The CPU-only audit reads the independent `sglang:cached_tokens_total` aggregate
recorded in each historical raw result. It accepts 72/72 files across 12
conditions, with three trials on each of two datasets. For every run:

- request and prompt counters agree;
- request 0 is the only missing zero-valued response cache field;
- the response-derived sum equals the independent aggregate cached-token delta;
- no request failed.

This replaces the old circular reconciliation and clears the counter-cleanliness
caveat. Its scope is deliberately narrow: it does not prove that twelve policy
labels emitted twelve different orderings. The exact fitted-policy audit shows
that the five BFCL fitted labels emitted one byte-identical sequence.

## Remaining evidence risk

The compact audit is reproducible from `scripts/audit_contextpilot_static_refit_compact.py`
and `local-audit.json`. The underlying 12 MB static-refit archive and 2.8 MB
SGLang audit archive still have only one known copy on the GPU server. Their
recorded SHA-256 values have not been re-opened from this local machine.
