"""ToolTrie-v1 — the weighted variant brief §2 Q3 asks for.

Brief §2 Q3 asks whether frequently occurring tool sequences can be represented
as a **weighted** trie or prefix memory under a limited cache budget.
:class:`~tatm.tooltrie.ToolTrie` builds and bounds the prefix memory, and its
nodes already carry a ``visit_count`` that ``observe`` increments on every
served request — but nothing ever reads it. Selection tie-breaks on frozen
training support, and eviction is pure recency. The trie is therefore bounded
and budget-tested but *unweighted*, and the weighting clause of that question
has never been answered.

``WeightedToolTrie`` is the minimal policy that answers it. It changes exactly
the two places where a weight could act:

1. **Selection.** The tie-break after reachable cached cost becomes the node's
   own observed ``visit_count`` instead of support frozen on a task-disjoint
   training corpus. A held-out corpus cannot know which paths *this* stream
   walks; the trie has measured exactly that.
2. **Eviction.** The least-*visited* leaf is dropped first, ties broken by least
   recently used, replacing pure leaf-first LRU. This matters only when the
   budget binds — which, before the controlled-pressure run of 2026-08-11, it
   never did in this project.

Everything else — causality, the reachable-cached-cost objective, the recency
window, leaf-first eviction order, the alphabetical fallback — is inherited
unchanged, so a difference in measured reuse is attributable to the weighting
and nothing else.

**This is a proposal, not a replacement.** ToolTrie-v0 remains the Task E
deliverable that every published figure describes. Adopting v1 would require
regenerating those figures.
"""

from __future__ import annotations

from heapq import heappush

from .tooltrie import ToolTrie, ToolTrieNode


class WeightedToolTrie(ToolTrie):
    """ToolTrie-v0 with its own ``visit_count`` read during selection and eviction."""

    def _selection_key(
        self,
        node: ToolTrieNode,
        remaining: frozenset[str],
    ) -> tuple[int, int, str]:
        # Same leading term as v0, so the objective is unchanged; only the
        # tie-break differs. Observed visits replace frozen training support.
        return (
            -self._reachable_cached_cost(node, remaining),
            -node.visit_count,
            node.tool_id or "",
        )

    def _evict_one_leaf(self) -> bool:
        """Drop the least-visited evictable leaf, ties broken by least-recently-used.

        v0 pops a min-heap keyed on ``last_seen``. Recency is a poor proxy for
        value when many nodes are touched every request: under a binding budget
        a path walked once and a path walked two hundred times can share a
        ``last_seen`` and be evicted with equal probability.

        The heap v0 maintains is keyed for LRU and cannot answer "least
        visited", so this scans live evictable leaves directly. The scan is
        O(live nodes) per eviction against v0's amortised O(log n); with
        ``max_nodes`` bounded at 100k and evictions numbering in the low
        thousands per run, the cost is not measurable beside prefill.
        """

        best: ToolTrieNode | None = None
        best_key: tuple[int, int] | None = None
        for node in self._live_nodes:
            if node.children or node.parent is None or not node.live:
                continue
            key = (node.visit_count, node.last_seen)
            if best_key is None or key < best_key:
                best, best_key = node, key

        if best is None:
            return False

        parent = best.parent
        assert parent is not None and best.tool_id is not None
        parent.children.pop(best.tool_id, None)
        best.live = False
        self._live_nodes.remove(best)
        self.retained_schema_tokens -= best.schema_tokens
        self.evictions += 1

        if parent is not self.root and not parent.children:
            # Keep the inherited heap consistent for any code still reading it,
            # and preserve the parent's own last-use time exactly as v0 does.
            self._heap_serial += 1
            heappush(self._eviction_heap, (parent.last_seen, self._heap_serial, parent))
        return True

__all__ = ["WeightedToolTrie"]
