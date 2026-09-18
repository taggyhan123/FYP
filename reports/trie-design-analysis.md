# ToolTrie-v1: design review and where it can still improve

What the trie does, where its gains come from, what the closest published
methods do differently, and which candidate improvements hold up when
simulated on our own workloads. Simulation numbers come from
`scripts/simulate_trie_variants.py`, which re-renders every prompt as vLLM
does and replays it through the validated cache simulator
(`src/tatm/prefix_cache_sim.py`); it reproduces the measured ToolTrie-v1 hit
rates to within 0.03 points (6.19 / 4.33 / 2.41% simulated vs 6.22 / 4.35 /
2.41% served at 0.6B / 4B / 8B). Source:
`cluster/results/trie-variants-sim-20260918-134120/results.json`.

## 1. What ToolTrie-v1 is

`src/tatm/tooltrie.py` (base) and `src/tatm/tooltrie_v1.py` (the shipped
policy). Per request:

1. **Walk a prefix tree of orders already served.** From the root, among the
   child tools this request contains and that were served within the last 128
   requests, take the one whose subtree offers the most reusable schema tokens.
   Repeat until no child matches. That path is the *matched prefix*.
2. **Append every unmatched tool in the retriever's order.** This is the one
   change from v0 (which sorted them alphabetically) and it is why v1 costs no
   accuracy: the retriever's ranking is preserved wherever the trie has nothing
   to say.
3. **Record the served order** in the trie. The trie is bounded (190,896
   schema tokens, 100,000 nodes) and evicts least-recently-used leaves first,
   mimicking the KV cache it stands in for.

It is causal (only past requests), needs no training data (the trie starts
empty; the `support` statistics the base class accepts are never passed), and
never changes the tool set.

**Where the gains come from, measured.** On 64-tool dense menus (0.6B cache):

| fact | value | source |
|---|---|---|
| requests where the trie matches nothing | **126 of 200 (63%)** | `tooltrie_plan.matched_prefix_ids` in the k64 workload |
| mean matched prefix when it does match | 4.1 tools (max 40) | same |
| reuse lost to v1's own per-request order choice | **0.00 points** | `cache-hit-miss.md` §3 |
| reuse lost to lack of coordination with earlier requests (upper bound) | 13.9 points | same |
| reuse lost to cache capacity (upper bound) | 8.3 points | same |
| tokens no ordering could ever reuse | **71.5%** | same |

So v1 is already the best *per-request* order given what is cached. Every
remaining point of reuse must come from one of three places: ordering earlier
requests with later ones in mind (coordination), a bigger or better-used cache,
or changing the prompt layout itself. That is the frame for everything below.

## 2. What the literature does differently

| method | what it is | relation to v1 |
|---|---|---|
| [CacheWeaver](https://arxiv.org/html/2606.19667) (Jun 2026) | trie over recently served *document* sequences, greedy longest reusable prefix first, rest in retrieval rank | **the same algorithm as v1**, on RAG passages. Reports 20-33% lower median TTFT vs retrieval order and that greedy reaches 97.5% of its oracle-ordering gain. Cite it; v1's contribution is the tool-menu setting and the show-10 accuracy result, not the walk |
| [Prefix Sharing Is a Sorting Problem](https://arxiv.org/html/2609.13692) (Sep 2026) | proves chunk ordering = choosing a hierarchy over requests; agglomerative clustering by common intersection is a 1/2-approximation within 0.05% of exact on real traces | evaluates **online greedy longest-prefix** (v1's family) at 0.749-0.800 of the no-sharing token bill vs **0.591-0.643 for clustering** (NFCorpus/SciFact, k=8-16). Clustering needs a batch to permute; greedy is causal. This is the coordination lever quantified |
| [ContextPilot](https://arxiv.org/html/2511.03475v3) | context index, moves tools shared with similar earlier requests to the front | hoists a *global core*; wins where every request shares one (padded menus, BM25 show-10), loses where overlap is local. v1 never hoists - its boundary condition |
| [ReCache](https://arxiv.org/html/2608.19662) (Aug 2026) | position-independent KV blocks per tool (resource-wise attention), so order stops mattering | orthogonal and stronger, but needs a modified attention kernel. Out of scope under this project's unmodified-vLLM constraint; the right "future work" citation |
| LPM scheduling (SGLang), [PEEK](https://arxiv.org/pdf/2607.02525), [CacheRoute](https://arxiv.org/html/2608.19677) | the *scheduler* picks which waiting request runs next by cache affinity | v1 only orders tools within a request; pairing it with an affinity scheduler is the untested combination |

## 3. Candidate improvements, simulated

Hit rate = prompt tokens served from cache, 200 dense ToolRet requests, the
same completion lengths for every variant. "gold in shown" is how often a
correct tool is among the 10 shown (the accuracy ceiling; F1 itself needs a
GPU run).

### 64 tools shown

| variant | 0.6B cache | 4B cache | 8B cache | verdict |
|---|---|---|---|---|
| no reordering | 0.82 | 0.72 | 0.64 | control |
| **v1 as shipped** | **6.19** | **4.33** | **2.41** | baseline |
| v1, trie capacity = real cache | 6.19 | 4.20 | 1.80 | **worse**: the trie's token accounting is not the cache's block accounting, so a tighter budget evicts paths the cache still holds |
| v1 + lead with 1 / 2 / 3 recurring tools when nothing matches | 6.00 / 5.58 / 5.73 | 4.07 / 3.80 / 3.96 | 2.41 / 2.28 / 2.31 | **worse**: hoisting breaks exact-prefix matches on menus with little overlap (the frequency baseline's failure in miniature) |
| v1 + scheduler, window 8, chain by set overlap | 6.72 | 4.46 | 2.46 | small gain |
| v1 + scheduler, window 32, chain by set overlap | 4.87 | 4.41 | 3.94 | helps only the small cache |
| **v1 + LPM scheduler, window 8** | **6.65** | **4.52** | 2.52 | +7% / +4% / +5% |
| **v1 + LPM scheduler, window 32** | 6.49 | **5.35** | **4.46** | +5% / **+24%** / **+85%** |
| v1 + LPM scheduler, whole stream (offline bound) | 7.59 | 7.59 | 7.59 | +23% / +75% / +215% |

LPM = the dispatcher serves next the waiting request whose v1 plan would reuse
the most trie tokens right now, within a window of W arrivals. It is the
scheduler-side counterpart of v1's own walk, and the pending requests it
chooses among are only ones that have already arrived, so it is causal.

### Retrieve 64, show 10

| variant | 0.6B | 4B | 8B | gold in shown | verdict |
|---|---|---|---|---|---|
| no reordering | 6.38 | 6.22 | 5.11 | 62.0% | control |
| **v1 as shipped** | **27.64** | **27.64** | 27.54 | 61.5% | baseline |
| v1, trie capacity = real cache | - | 20.22 | 11.44 | 59.5 / 61.5% | **much worse** |
| **v1 records only the 10 served** (`p10obs`) | **31.59** | **31.58** | 26.71 | **59.0%** | **+14% reuse at 0.6B/4B, −2.5 points gold-in-shown**; needs an F1 run |
| v1 + lead with recurring tools | 25.7-26.9 | 25.7-26.8 | 25.2-26.6 | 59.0-60.5% | worse on both |
| v1 + LPM scheduler, window 4 / 8 | 27.85 / 27.05 | 27.85 / 27.05 | 27.74 / 26.87 | 61.5% | no gain |
| v1 + LPM scheduler, window 32 | 26.12 | 26.12 | 26.03 | 61.5% | slightly worse |

## 4. What this says

1. **Within a request, v1 is done.** Its own-order loss is zero, and every
   attempt to be cleverer inside the request (anchoring recurring tools,
   matching the trie budget to the cache) made reuse worse. The trie's
   optimistic residency hint is a feature: it keeps paths the cache may still
   partly hold.

2. **The next real gain is scheduling, and it grows as the cache shrinks.**
   Letting v1 also choose *which* waiting request to serve next raises reuse
   by 4-7% with a window of 8 and by 24% (4B) to 85% (8B) with a window of 32
   on 64-tool menus; the whole-stream bound is 7.59% at every cache size,
   i.e. the ordering becomes cache-size-independent because each request
   reuses the one just served. This matches the Sorting paper's finding that
   batch-aware hierarchies beat online greedy by a wide margin, and our own
   coordination bound (13.9 points). It is the one lever with double-digit
   upside on the data we have.
   - Cost: a window delays some requests. Window 8 at 0.7 req/s is about 11 s
     of extra queueing worst case; at 32 in flight it is a batch. This must be
     measured as p95 latency, not just reuse, and needs a fairness bound
     (`--max-delay`, which the replay driver already supports for its
     `affinity` dispatcher).
   - On show-10 it does nothing: those prompts are short and already reuse 86%
     of what any ordering could, so there is no queue-ordering to exploit.

3. **On show-10, record what was served.** Observing only the 10 tools the
   server actually saw lifts reuse from 27.6% to 31.6% at 0.6B and 4B. It also
   lowers the share of requests whose 10 contain a correct tool from 61.5% to
   59.0%, because the trie then matches on the served prefix more aggressively
   and pushes one more retriever pick out. Whether F1 moves needs one GPU
   run (~15 minutes, 4B). At 8B the gain vanishes (26.7%), the small cache
   again.

4. **Do not add a global-core hoist to v1.** Every hoisting variant lost. If
   the padded-menu regime matters for a deployment, use ContextPilot there;
   the two policies are complementary, and a switch on measured overlap (core
   size ≥ half the menu → ContextPilot) is a cleaner answer than a hybrid.

5. **The ceiling is the workload, not the algorithm.** 71.5% of prompt tokens
   on 64-tool menus can never be reused by any ordering, and 63% of requests
   share no leading tool with anything recent. Larger gains need more shared
   structure (multi-tenant agents, repeated tool sets) or a layout change that
   makes tool KV position-independent (ReCache), which is outside the
   unmodified-vLLM constraint of this project.

## 5. Recommended next steps, in order

| step | what it settles | cost |
|---|---|---|
| GPU run of v1 + LPM window 8 and 32 on 64 tools, 4B, with a max-delay bound | whether the simulated +24% reuse becomes throughput and what it costs in p95 | ~1 h, one GPU |
| GPU accuracy run of `p10obs` on show-10, 4B | whether +14% reuse costs F1 | ~15 min |
| cite CacheWeaver, the Sorting paper and ReCache in related work | positions v1 correctly: same walk as CacheWeaver, applied to tool menus with the show-10 accuracy result as the new finding | writing only |

## 6. Limits of this analysis

- One 200-request stream from 2 of ToolRet's 35 sources; the scheduling gains
  depend on how much locality the stream has.
- The simulator scores reuse, not latency; scheduling changes queueing, which
  only a GPU run can measure.
- "gold in shown" is a ceiling for accuracy, not F1.
- Windows of 32 assume 32 requests are waiting, i.e. saturation; at light load
  the window never fills and LPM degrades to arrival order.
