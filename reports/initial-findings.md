# Initial research findings and recommendation

## Result in one paragraph

The public workloads contain a measurable analytical prefix-locality signal,
but it is workload- and evidence-dependent. Across 45,815 canonical
tools, median schema length is 70 Qwen tokens and P95 is
220. ToolRet's median relevance set and BFCL's median
exposed menu both contain one tool, which limits what reordering can achieve on
many tasks. On empirical dataset order, schema-cost-weighted ordering raises the
ToolRet tool-prefix block-reuse estimate from
26.91% to
31.80% (4.89
percentage points). Frequency ordering raises the BFCL menu estimate from
10.98% to
18.88% (7.90
points). These are analytical estimates, not vLLM hit-rate or latency results.

## Evidence obtained locally

- ToolRet: 7,961 tasks, 7,652
  tools appearing in gold relevance labels, with every label resolved.
- BFCL subset: 1,240 tasks and
  1,362 distinct canonical functions appearing in
  exposed menus.
- Schema diagnostics flag 19,921/45,815 tools. Most flags are
  empty parameter objects, which can be legitimate; malformed and missing-field
  counts are reported separately.
- Controlled support-skewed replay produces much more reuse than uniform or
  empirical access, confirming that any systems claim must state its workload
  locality rather than assume production-like skew.
- Classic FP-tree global order equals the frequency order in this baseline.
  Conditional pattern mining is still untested.

## Prefix-cache sanity status

| Check | Local status | Required cluster evidence |
| --- | --- | --- |
| Identical prompt reuse | Pending CUDA vLLM | cached/computed prompt tokens and TTFT |
| Changed second tool | Pending CUDA vLLM | first differing block boundary |
| Reordered tools | Pending CUDA vLLM | rendered token IDs and hit boundary |
| Cache on/off equivalence | Pending CUDA vLLM | identical generated token sequence |
| GPU/KV memory | Unavailable on this Mac | GPU type, cache usage, eviction settings |

The runnable probe and procedure are in `cluster/README.md`; no synthetic GPU
number is substituted here.

## Recommendation

Proceed with the exact prompt-level ToolTrie baseline on the cluster, focusing
on multi-tool tasks and session-local replay. Use schema-cost-weighted ordering
as the strongest ToolRet candidate and frequency ordering as the strongest BFCL
candidate from this first analysis. Report the original order and fixed-random
controls alongside them.

The likely publishable refinement is not a generic “reorder context into a
trie” claim, because closely related cache-aware context ordering already
exists. The more defensible direction is tool-specific cache admission using
schema cost plus workflow co-occurrence, while preserving an active/authorized
tool manifest and measuring function-call quality.

Do not pursue arbitrary independent KV concatenation yet. First establish that
native exact APC converts the local token-reuse signal into repeatable TTFT
benefit without a BFCL quality regression. If it does not, narrow the project to
characterizing the crossover regimes or pivot toward retrieval/menu reduction.

## Immediate cluster run

1. Freeze model revision, tokenizer, chat template, vLLM version, GPU, block
   size, dtype, and server flags.
2. Run the cache-enabled and cache-disabled sanity probe.
3. Render and save complete prompt token IDs for each ordering.
4. Replay original, frequency, and schema-cost-weighted workloads with
   documented cold/warm policies and repeated trials.
5. Add BFCL name/argument/no-tool scores before interpreting latency.
