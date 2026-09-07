# How ToolTrie-v0 works

> **ToolTrie-v1** changes exactly one thing in what follows — the *fallback*
> ordering in `plan()`. See [`tooltrie-v1-design.md`](tooltrie-v1-design.md).

The Task E deliverable: `src/tatm/tooltrie.py`, 283 lines, no dependencies
beyond the standard library and `CanonicalTool`. This document explains the
mechanism and quotes the code that implements it. Results are in
[`reports/key-findings.md`](../reports/key-findings.md).

## The problem it solves

vLLM's automatic prefix caching reuses KV state only for an **exact** shared
token prefix. Two requests exposing the same 64 tools in different orders share
nothing after the first differing schema. So if request N serialises
`[search, weather, calendar]` and request N+1 serialises
`[weather, calendar, search]`, the cache hit ends at the system preamble even
though the tool sets are identical.

ToolTrie-v0 reorders the **already-selected** tool set so that requests
converge on a common prefix. It is prompt-layer metadata only: it never touches
KV tensors, never changes which tools are exposed, and keeps ordinary text
prefill as the fallback path.

## The structure

One node per tool-ID edge. A root-to-node path is a tool sequence that was
actually served.

```python
@dataclass(eq=False)
class ToolTrieNode:
    tool_id: str | None
    parent: "ToolTrieNode | None" = None
    children: dict[str, "ToolTrieNode"] = field(default_factory=dict)
    schema_tokens: int = 0
    cumulative_schema_tokens: int = 0
    visit_count: int = 0
    last_seen: int = -1
    live: bool = True
```

Nodes are **planner metadata, not vLLM KV blocks**. `last_seen` is a
conservative guess at whether the engine still holds that prefix; the engine's
own Prometheus counters remain the authority on what was actually reused. That
separation is deliberate — every reuse figure in the reports comes from the
counters, never from this structure.

## `plan()` — choosing an order

Greedy descent from the root. At each step, consider the children that are both
in the current menu and plausibly still cached, and take the best one.

```python
while remaining:
    frozen_remaining = frozenset(remaining)
    candidates = [
        child
        for tool_id, child in node.children.items()
        if tool_id in remaining and self._is_resident_hint(child)
    ]
    if not candidates:
        break

    chosen = min(candidates, key=lambda child: self._selection_key(child, frozen_remaining))
    matched.append(chosen.tool_id)
    hinted_tokens += chosen.schema_tokens
    remaining.remove(chosen.tool_id)
    node = chosen

fallback_ids = self._fallback_order(remaining)
ordered_ids = (*matched, *fallback_ids)
```

Two things make this more than a popularity sort.

**Residency gating.** A node only counts as a candidate if it was seen recently
enough to plausibly still be cached:

```python
def _is_resident_hint(self, node: ToolTrieNode) -> bool:
    if not node.live or node.last_seen < 0:
        return False
    if self.recency_window is None:
        return True
    return self.request_index - node.last_seen < self.recency_window
```

**Lookahead, not greedy-next.** The ranking key's leading term is not "how often
was this child visited" but "how many cacheable schema tokens are reachable down
this branch", computed recursively over tools still in the menu:

```python
def _reachable_cached_cost(self, node, remaining) -> int:
    """Maximum hinted-resident path cost reachable through ``node``."""
    tool_id = node.tool_id
    if tool_id is None or tool_id not in remaining or not self._is_resident_hint(node):
        return 0
    next_remaining = remaining - {tool_id}
    downstream = max(
        (self._reachable_cached_cost(child, next_remaining)
         for child in node.children.values()),
        default=0,
    )
    return node.schema_tokens + downstream
```

The full key, ties broken deterministically:

```python
def _selection_key(self, node, remaining) -> tuple[int, int, str]:
    return (
        -self._reachable_cached_cost(node, remaining),
        -self.support.get(node.tool_id or "", 0),
        node.tool_id or "",
    )
```

**This is the design decision that matters most, and it is why the weighted
variant changed nothing.** The first term dominates: it almost always separates
the candidates on its own, so the second term — training support here,
`visit_count` in `tooltrie_weighted.py` — is rarely consulted.

Tools with no resident path fall through to `_fallback_order`, which is plain
alphabetical by function name (or frozen frequency, if a separate training
corpus was supplied). That fallback is what keeps the ordering deterministic for
tools the planner has never seen, and it is why request 0 is pure alphabetical.

## `observe()` — learning, strictly afterwards

```python
def observe(self, ordered_ids):
    ids = self._validated_ids(ordered_ids)
    self.request_index += 1
    node = self.root
    for tool_id in ids:
        child = node.children.get(tool_id)
        if child is None:
            ...  # create node, add schema_tokens to the retained budget
        child.visit_count += 1
        self._touch(child)
        node = child

    while self._over_budget() and self._evict_one_leaf():
        pass
```

**Causality is the load-bearing property.** `plan()` never mutates the trie;
`observe()` is called only after the request has been served. A planner that
sees the batch it is ordering can trivially win, and it would be measuring
nothing. The split is enforced in tests
(`test_online_frequency_plan_cannot_see_the_current_request` and the equivalent
for the weighted variant), and the workload builders call `plan()` before `observe()` per
record.

The project measured what happens when causality is dropped: granting ToolTrie
the whole evaluation batch and iterating to a fixpoint collapses it from
**87.19% to 29.96%** — below plain alphabetical. "Offline" is not an upper bound
for this planner, it is a different and worse algorithm.

## The budget

The planner's metadata is bounded by `capacity_tokens` and `max_nodes`, with
leaf-first LRU eviction — the same shape as a radix cache, so the metadata
cannot grow without limit while the engine's cache is evicting underneath it.

```python
def _over_budget(self) -> bool:
    return (
        self.capacity_tokens is not None
        and self.retained_schema_tokens > self.capacity_tokens
    ) or (self.max_nodes is not None and len(self._live_nodes) > self.max_nodes)
```

This budget governs **how much served history the planner remembers**. It is not
KV-cache retention — deciding which tools stay resident in GPU memory is brief
§9.1/§9.3 and was never built.

Until the controlled-pressure run of 2026-08-11 this path had effectively never
fired: at 190,896-token capacity a ~6,900-token menu never approaches the
budget. At 7,680 tokens it fired 506–1,494 times per regime and ran clean.

## Parameters

| parameter | value used throughout | what it controls |
| --- | --- | --- |
| `recency_window` | 128 | how many requests back a node still counts as resident |
| `capacity_tokens` | the live cache capacity | metadata budget in schema tokens |
| `max_nodes` | 100,000 | hard node ceiling |
| `fallback` | `alphabetical` | order for tools with no resident path |

`recency_window` has been fixed at 128 for every run with no sensitivity
analysis — it is suggested next step 4 in the key findings.

## What it deliberately does not do

- **No KV tensor manipulation.** Ordering only; the engine does the rest.
- **No change to tool selection.** It receives a selected set and permutes it.
- **No retention of inactive tools.** Brief §9.1, unbuilt.
- **No use of `visit_count`.** The field is incremented on every observation and
  read nowhere. That is what makes v0 an *unweighted* trie, and
  `src/tatm/tooltrie_weighted.py` is the variant that reads it — which changes 0
  of 200 emitted orderings, for the reason given above.

## Where to look next

| | |
| --- | --- |
| Implementation | `src/tatm/tooltrie.py` |
| ToolTrie-v1 | `src/tatm/tooltrie_v1.py`, [design note](tooltrie-v1-design.md) |
| Weighted variant | `src/tatm/tooltrie_weighted.py` |
| Workload builder | `scripts/build_tooltrie_workload.py` |
| Tests | `tests/test_tooltrie.py`, `tests/test_tooltrie_v1.py`, `tests/test_tooltrie_weighted.py` |
| First GPU measurement | `reports/tooltrie-v0/` |
| Under a limited cache budget | `reports/tooltrie-pressure/20260811-001032/` |
| Weighted variant measured | `reports/tooltrie-weighted/20260811-144741/` |
| Early design proposal, 2026-07-31 | [`exact-tooltrie-proposal.md`](exact-tooltrie-proposal.md) |
