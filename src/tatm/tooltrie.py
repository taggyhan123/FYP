from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any

from tatm.models import CanonicalTool


@dataclass(eq=False)
class ToolTrieNode:
    """One tool-ID edge in the prompt-layer ToolTrie.

    Nodes are deliberately tool-level planner metadata, not physical vLLM KV
    blocks. ``last_seen`` is a conservative residency hint; vLLM's measured APC
    counters remain the authority on whether rendered token blocks were reused.
    """

    tool_id: str | None
    parent: "ToolTrieNode | None" = None
    children: dict[str, "ToolTrieNode"] = field(default_factory=dict)
    schema_tokens: int = 0
    cumulative_schema_tokens: int = 0
    visit_count: int = 0
    last_seen: int = -1
    live: bool = True


@dataclass(frozen=True)
class ToolTriePlan:
    """Causal ordering decision made before the current request is observed."""

    ordered_ids: tuple[str, ...]
    matched_prefix_ids: tuple[str, ...]
    fallback_ids: tuple[str, ...]
    hinted_schema_tokens: int

    def to_record(self) -> dict[str, Any]:
        return {
            "ordered_ids": list(self.ordered_ids),
            "matched_prefix_ids": list(self.matched_prefix_ids),
            "fallback_ids": list(self.fallback_ids),
            "hinted_schema_tokens": self.hinted_schema_tokens,
        }


class ToolTrie:
    """Bounded recent-path planner over previously served tool-ID sequences.

    The planner adapts CacheWeaver's greedy recent-path walk to tool IDs and
    uses SGLang-like leaf-first LRU semantics for its *metadata* budget. It is
    synchronous and therefore needs no active-path locks; a concurrent serving
    integration would need to add those before sharing one instance.

    ``plan`` never mutates request history. Call ``observe`` only after the
    corresponding request has been served (or accepted into the replay). This
    separation prevents accidental look-ahead leakage in offline experiments.
    """

    def __init__(
        self,
        tools: Mapping[str, CanonicalTool],
        *,
        fallback: str = "alphabetical",
        support: Mapping[str, int] | None = None,
        recency_window: int | None = 128,
        capacity_tokens: int | None = None,
        max_nodes: int | None = 100_000,
    ) -> None:
        if fallback not in {"alphabetical", "frequency"}:
            raise ValueError(f"Unknown ToolTrie fallback: {fallback}")
        if fallback == "frequency" and support is None:
            raise ValueError(
                "frequency fallback requires support fitted on a separate "
                "training workload"
            )
        if recency_window is not None and recency_window < 1:
            raise ValueError("recency_window must be >= 1 or None")
        if capacity_tokens is not None and capacity_tokens < 1:
            raise ValueError("capacity_tokens must be >= 1 or None")
        if max_nodes is not None and max_nodes < 1:
            raise ValueError("max_nodes must be >= 1 or None")

        self.tools = tools
        self.fallback = fallback
        self.support = dict(support or {})
        self.recency_window = recency_window
        self.capacity_tokens = capacity_tokens
        self.max_nodes = max_nodes

        self.root = ToolTrieNode(tool_id=None)
        self.request_index = 0
        self.retained_schema_tokens = 0
        self.evictions = 0
        self._live_nodes: set[ToolTrieNode] = set()
        self._eviction_heap: list[tuple[int, int, ToolTrieNode]] = []
        self._heap_serial = 0

    def _validated_ids(self, tool_ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(dict.fromkeys(tool_ids))
        unknown = [tool_id for tool_id in ids if tool_id not in self.tools]
        if unknown:
            sample = ", ".join(repr(item) for item in unknown[:3])
            raise ValueError(f"ToolTrie received unknown tool IDs: {sample}")
        return ids

    def _is_resident_hint(self, node: ToolTrieNode) -> bool:
        if not node.live or node.last_seen < 0:
            return False
        if self.recency_window is None:
            return True
        return self.request_index - node.last_seen < self.recency_window

    def _reachable_cached_cost(
        self,
        node: ToolTrieNode,
        remaining: frozenset[str],
    ) -> int:
        """Maximum hinted-resident path cost reachable through ``node``."""

        tool_id = node.tool_id
        if (
            tool_id is None
            or tool_id not in remaining
            or not self._is_resident_hint(node)
        ):
            return 0
        next_remaining = remaining - {tool_id}
        downstream = max(
            (
                self._reachable_cached_cost(child, next_remaining)
                for child in node.children.values()
            ),
            default=0,
        )
        return node.schema_tokens + downstream

    def _selection_key(
        self,
        node: ToolTrieNode,
        remaining: frozenset[str],
    ) -> tuple[int, int, str]:
        """Rank one candidate child during ``plan``.

        Greater reachable cached cost wins, then greater training support, then
        lexicographically smaller tool ID for complete determinism. Subclasses
        override this to weight the choice differently; ``visit_count`` is
        deliberately *not* consulted here, which is what makes this planner an
        unweighted trie.
        """

        return (
            -self._reachable_cached_cost(node, remaining),
            -self.support.get(node.tool_id or "", 0),
            node.tool_id or "",
        )

    def _fallback_order(self, tool_ids: set[str]) -> tuple[str, ...]:
        if self.fallback == "alphabetical":
            key = lambda item: (self.tools[item].name.casefold(), item)
        else:
            key = lambda item: (-self.support.get(item, 0), item)
        return tuple(sorted(tool_ids, key=key))

    def plan(self, selected_ids: Sequence[str]) -> ToolTriePlan:
        """Order a selected set using only paths observed before this request."""

        ids = self._validated_ids(selected_ids)
        remaining = set(ids)
        node = self.root
        matched: list[str] = []
        hinted_tokens = 0

        while remaining:
            frozen_remaining = frozenset(remaining)
            candidates = [
                child
                for tool_id, child in node.children.items()
                if tool_id in remaining and self._is_resident_hint(child)
            ]
            if not candidates:
                break

            chosen = min(
                candidates,
                key=lambda child: self._selection_key(child, frozen_remaining),
            )
            assert chosen.tool_id is not None
            matched.append(chosen.tool_id)
            hinted_tokens += chosen.schema_tokens
            remaining.remove(chosen.tool_id)
            node = chosen

        fallback_ids = self._fallback_order(remaining)
        ordered_ids = (*matched, *fallback_ids)
        assert set(ordered_ids) == set(ids)
        assert len(ordered_ids) == len(ids)
        return ToolTriePlan(
            ordered_ids=ordered_ids,
            matched_prefix_ids=tuple(matched),
            fallback_ids=fallback_ids,
            hinted_schema_tokens=hinted_tokens,
        )

    def _touch(self, node: ToolTrieNode) -> None:
        node.last_seen = self.request_index
        self._heap_serial += 1
        heappush(
            self._eviction_heap,
            (node.last_seen, self._heap_serial, node),
        )

    def _over_budget(self) -> bool:
        return (
            self.capacity_tokens is not None
            and self.retained_schema_tokens > self.capacity_tokens
        ) or (self.max_nodes is not None and len(self._live_nodes) > self.max_nodes)

    def _evict_one_leaf(self) -> bool:
        while self._eviction_heap:
            last_seen, _, node = heappop(self._eviction_heap)
            if (
                not node.live
                or node.children
                or node.last_seen != last_seen
                or node.parent is None
            ):
                continue

            parent = node.parent
            assert node.tool_id is not None
            parent.children.pop(node.tool_id, None)
            node.live = False
            self._live_nodes.remove(node)
            self.retained_schema_tokens -= node.schema_tokens
            self.evictions += 1

            if parent is not self.root and not parent.children:
                # The parent has become an evictable leaf. Preserve its own
                # last-use time, matching leaf-first LRU radix-cache semantics.
                self._heap_serial += 1
                heappush(
                    self._eviction_heap,
                    (parent.last_seen, self._heap_serial, parent),
                )
            return True
        return False

    def observe(self, ordered_ids: Sequence[str]) -> dict[str, int | None]:
        """Insert one served sequence and enforce the planner metadata budget."""

        ids = self._validated_ids(ordered_ids)
        if len(ids) != len(ordered_ids):
            raise ValueError("A served ToolTrie sequence must not repeat tool IDs")

        self.request_index += 1
        node = self.root
        for tool_id in ids:
            child = node.children.get(tool_id)
            if child is None:
                schema_tokens = self.tools[tool_id].schema_tokens
                child = ToolTrieNode(
                    tool_id=tool_id,
                    parent=node,
                    schema_tokens=schema_tokens,
                    cumulative_schema_tokens=(
                        node.cumulative_schema_tokens + schema_tokens
                    ),
                )
                node.children[tool_id] = child
                self._live_nodes.add(child)
                self.retained_schema_tokens += schema_tokens
            child.visit_count += 1
            self._touch(child)
            node = child

        while self._over_budget() and self._evict_one_leaf():
            pass
        if self._over_budget():
            raise RuntimeError("ToolTrie is over budget but has no evictable leaf")
        return self.snapshot()

    def snapshot(self) -> dict[str, int | None]:
        resident_nodes = sum(
            self._is_resident_hint(node) for node in self._live_nodes
        )
        return {
            "requests_observed": self.request_index,
            "nodes": len(self._live_nodes),
            "resident_hint_nodes": resident_nodes,
            "retained_schema_tokens": self.retained_schema_tokens,
            "capacity_tokens": self.capacity_tokens,
            "max_nodes": self.max_nodes,
            "recency_window": self.recency_window,
            "evictions": self.evictions,
        }
