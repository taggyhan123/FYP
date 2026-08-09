# ContextPilot dual-model replication — GPU executor handover

**Run stamp** `20260809-004603`
**Executed FYP commit** `15285704e73af680c0125ea4bfeb0b54a14f278e`
**ContextPilot commit** `1fa0a143fdeda344585666648ab2b30cb7fea77f`, alpha `0.001`
**Engine** vLLM `0.26.0`, unmodified, `--gpu-memory-utilization 0.92`
**Physical GPU** GPU 2 (NVIDIA RTX 3090, driver 580.173.02), one server at a
time, one sequential client. Both models ran on the **same physical GPU**.

## Acceptance

```
status                = accepted
accepted_gpu_replays  = 190
all_checks_passed     = true
```

33 audit checks, none failed (`dual-model-audit.json`). Per model: 90 systems
replays + 5 quality replays accepted, 1 diagnostic warmup excluded, all replays
reset with clean counters, raw outputs preserved. CPU analysis accepted 27
comparisons per model (54 total), all `sequence_state_dependent: true`.

| | Qwen3-4B (primary) | Qwen3-0.6B (replication) |
| --- | --- | --- |
| revision | `1cfa9a7208912126459214e8b04321603b3df60c` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| native capacity tokens | 96,832 | 188,912 |
| systems / quality accepted | 90 / 5 | 90 / 5 |

Tokenizer gate passed before execution: `tokenizer_json_identical` and
`chat_template_identical` both true across the two pinned revisions, so the
shared schema-token accounting used by ToolTrie is valid for both.

## Raw archive

```
/home/taghan/contextpilot-dual-model-20260809-004603.tar.gz
sha256 3fc5f5ec08580c22dcee65ed015fcb96ae6b36598aeaaaf3eeb8aeb8aac62012
```

58 MB compressed, 596 MB expanded, 666 entries, verified with `tar -tzf`.
GPU-server-only; **no off-machine backup exists.**

## Systems results — reuse %, mean of 3 trials

Reuse = `vllm:prompt_tokens_cached / (cached + request_prefill_kv_computed_tokens_sum)`.
All six summaries per model pass all three equivalence guards (same case set,
request sequence, selected tool sets across all five conditions).

### Qwen3-4B — primary, native capacity 96,832

| group | original | alphabetical | tooltrie_v0 | cp-online | cp-static-refit |
| --- | --- | --- | --- | --- | --- |
| BFCL padded-64 | 1.19% | 37.99% | 87.19% | **96.16%** | **96.16%** |
| ToolRet padded-64 | 6.56% | 43.81% | 82.18% | 95.27% | **95.74%** |
| BM25 k=4 | 15.87% | 15.28% | 17.48% | **18.72%** | 18.42% |
| BM25 k=16 | 5.14% | 5.40% | 6.73% | 9.10% | **9.36%** |
| BM25 k=64 | 0.72% | 0.98% | 1.63% | 3.07% | **3.48%** |
| BM25 k=128 | 0.34% | 0.44% | 0.89% | 1.35% | **2.23%** |

### Qwen3-0.6B — replication, native capacity 188,912

| group | original | alphabetical | tooltrie_v0 | cp-online | cp-static-refit |
| --- | --- | --- | --- | --- | --- |
| BFCL padded-64 | 1.19% | 38.13% | 87.19% | **96.16%** | **96.16%** |
| ToolRet padded-64 | 13.73% | 50.82% | 83.58% | 95.27% | **95.74%** |
| BM25 k=4 | 15.87% | 15.28% | 17.48% | **18.72%** | 18.42% |
| BM25 k=16 | 6.12% | 6.27% | 7.77% | **9.93%** | 9.80% |
| BM25 k=64 | 0.91% | 1.24% | 1.90% | **4.78%** | 4.01% |
| BM25 k=128 | 0.37% | 0.58% | 1.13% | 1.99% | **2.96%** |

**Do not pool the models or compare TTFT across them.** Capacities differ by
roughly 2×, ToolTrie is rebuilt from each live capacity, and the manifest
excludes a pure model-size causal comparison.

Two observations that hold in **both** models:

1. **The padded-versus-retrieved gap is the dominant effect.** Padded 64-tool
   menus reach 82–96% reuse; genuinely retrieved BM25 menus reach 0.3–18.7%.
   The high figures are a property of menus that share 60–63 of 64 tools, not
   of any ordering policy under retrieval.
2. **Both ContextPilot arms beat ToolTrie-v0 in all 12 systems cells**, and the
   two ContextPilot arms are close to each other throughout. ToolTrie beats both
   static baselines everywhere but does not reach either ContextPilot arm.

## Quality results — n=800 BFCL, 640 relevance + 160 irrelevance

| condition | 4B full | 4B name | 4B no_tool | 0.6B full | 0.6B name | 0.6B no_tool |
| --- | --- | --- | --- | --- | --- | --- |
| original | 76.09% | 83.13% | 88.12% | 55.16% | 73.75% | 86.25% |
| alphabetical | 73.28% | 82.19% | 85.62% | 43.59% | 60.47% | 94.37% |
| tooltrie_v0 | 75.31% | 83.75% | 87.50% | 53.28% | 69.06% | 93.13% |
| cp-online | **77.03%** | **84.38%** | 85.00% | 51.72% | 68.91% | 91.25% |
| cp-static-refit | **77.03%** | **84.38%** | 85.00% | 51.88% | 68.91% | 91.25% |

54 paired comparisons are committed under each model's `comparisons/`, all with
`sequence_state_dependent: true`, `cluster_bootstrap_generalizes_across_request_sequences: false`
and `mcnemar_independence_assumption_met: false`, at 50 000 bootstrap samples,
seed 42.

**The 4B-primary decision is vindicated by the size of the small-model effect.**
At 0.6B, ordering moves `full_accuracy` by 11.6 pp (alphabetical 43.59% against
original 55.16%) and `no_tool` by 8.1 pp in the opposite direction. At 4B the
same spread is 3.8 pp and 3.1 pp. Ordering-quality conclusions drawn at 0.6B
would be dominated by a small-model artifact.

**No equivalence margin was declared**, so every quality result here is
estimation only and supports no equivalence claim.

## Scope — what was NOT done

Per `protocol-manifest.json` `does_not_include`: no SGLang replay, no
ContextPilot eviction feedback, no ContextPilot relevance annotations, no
de-duplication, no cross-model pooling, no pure model-size causal comparison,
and not the historical Qwen3-8B static-refit closure cell.

**Neither ContextPilot arm is the full ContextPilot system.** Both are
ordering-only adaptations at alpha=0.001 with no eviction feedback and no
relevance annotations. `original` is retained as ordinary selected-tool text
prefill, the fallback condition.

## Exact server commands

Qwen3-4B (primary, run first):

```
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES=2 \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000
```

Qwen3-0.6B (replication, same GPU, after the 4B server was fully stopped and
GPU memory confirmed released) is identical except
`serve Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca`.

## Deviations and concurrency disclosure

1. **`uv venv` instead of `python3.12 -m venv`** — this machine's system
   python3.12 has no `ensurepip` and the documented fix needs `sudo`, which
   project rules forbid. Already-recorded environment deviation; same
   interpreter, same pinned upstream, same editable target.
2. **A separate Qwen3-8B job ran concurrently on GPU 3** between 02:31 and
   02:54 while the 4B arm was on GPU 2. It was the historical 8B static-refit
   quality cell, which this manifest explicitly excludes, kept in its own
   directory and port (8003). GPU 2 held sustained 100% utilisation throughout,
   so no measurable degradation was observed, but the overlap is disclosed here
   because the 4B latency columns were collected during it. Reuse ratios are
   deterministic and unaffected.
3. **Two idle gaps.** The 4B driver finished at 08:36 and the 0.6B arm did not
   start until 11:32; the 0.6B driver finished at 13:46 and CPU analysis did not
   start until 19:36. Both gaps were caused by a background monitor being killed
   without notice, not by any failure of the run. No measurement is affected —
   servers were idle, and every replay resets the cache before it starts.

No quarantined attempts. `--allow-counter-mismatch` was never passed. No
acceptance check was lowered and no model revision was substituted.

## Interpretation, fixed in advance

Qwen3-4B is primary; Qwen3-0.6B is a replication. Compare policies **within**
each model. Neither ContextPilot arm is the full system.
