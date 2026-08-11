"""Ordering baselines derived from the project's cited prior work.

These objects only arrange an already-selected set of tool IDs. They never
alter schemas, authorization, user messages, or model state. Actual KV-cache
residency remains the serving engine's responsibility.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from tatm.models import CanonicalTool


@dataclass(frozen=True)
class OrderingPlan:
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


@dataclass(eq=False)
class CacheWeaverNode:
    tool_id: str | None
    parent: "CacheWeaverNode | None" = None
    children: dict[str, "CacheWeaverNode"] = field(default_factory=dict)
    active_paths: int = 0
    schema_tokens: int = 0


class CacheWeaverPlanner:
    """Faithful tool-ID transcription of CacheWeaver Algorithm 1.

    Candidate tools are inspected in their original retrieval order. The first
    one that continues a path retained in the recent-history tree is selected.
    If no continuation exists, the untouched remainder is appended in original
    order. Unlike ToolTrie-v0, this baseline performs no schema-cost or support
    scoring. A bounded path history implements the paper's O(Hk) tree.
    """

    def __init__(
        self,
        tools: Mapping[str, CanonicalTool],
        *,
        history_window: int = 128,
    ) -> None:
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        self.tools = tools
        self.history_window = history_window
        self.root = CacheWeaverNode(tool_id=None)
        self.history: deque[tuple[str, ...]] = deque()
        self.request_index = 0
        self.node_count = 0
        self.retained_schema_tokens = 0

    def _validated_ids(self, tool_ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(tool_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("A CacheWeaver sequence must not repeat tool IDs")
        unknown = [tool_id for tool_id in ids if tool_id not in self.tools]
        if unknown:
            sample = ", ".join(repr(item) for item in unknown[:3])
            raise ValueError(f"CacheWeaver received unknown tool IDs: {sample}")
        return ids

    def plan(self, selected_ids: Sequence[str]) -> OrderingPlan:
        ids = self._validated_ids(selected_ids)
        remaining = list(ids)
        node = self.root
        matched: list[str] = []
        hinted_tokens = 0

        while remaining:
            found_index: int | None = None
            found_child: CacheWeaverNode | None = None
            # Algorithm 1 deliberately checks R in retrieval-rank order.
            for index, tool_id in enumerate(remaining):
                child = node.children.get(tool_id)
                if child is not None and child.active_paths > 0:
                    found_index = index
                    found_child = child
                    break
            if found_index is None or found_child is None:
                break
            tool_id = remaining.pop(found_index)
            matched.append(tool_id)
            hinted_tokens += found_child.schema_tokens
            node = found_child

        fallback_ids = tuple(remaining)
        ordered_ids = (*matched, *fallback_ids)
        return OrderingPlan(
            ordered_ids=ordered_ids,
            matched_prefix_ids=tuple(matched),
            fallback_ids=fallback_ids,
            hinted_schema_tokens=hinted_tokens,
        )

    def _remove_path(self, path: tuple[str, ...]) -> None:
        node = self.root
        nodes: list[CacheWeaverNode] = []
        for tool_id in path:
            child = node.children.get(tool_id)
            if child is None:
                raise RuntimeError("CacheWeaver history/tree state diverged")
            nodes.append(child)
            node = child

        for child in nodes:
            child.active_paths -= 1
            if child.active_paths < 0:
                raise RuntimeError("CacheWeaver path reference count became negative")
        for child in reversed(nodes):
            if child.active_paths != 0:
                break
            if child.children:
                raise RuntimeError("Inactive CacheWeaver node still has active children")
            assert child.parent is not None and child.tool_id is not None
            child.parent.children.pop(child.tool_id, None)
            self.node_count -= 1
            self.retained_schema_tokens -= child.schema_tokens

    def observe(self, ordered_ids: Sequence[str]) -> dict[str, int]:
        ids = self._validated_ids(ordered_ids)
        self.request_index += 1
        node = self.root
        for tool_id in ids:
            child = node.children.get(tool_id)
            if child is None:
                child = CacheWeaverNode(
                    tool_id=tool_id,
                    parent=node,
                    schema_tokens=self.tools[tool_id].schema_tokens,
                )
                node.children[tool_id] = child
                self.node_count += 1
                self.retained_schema_tokens += child.schema_tokens
            child.active_paths += 1
            node = child

        self.history.append(ids)
        if len(self.history) > self.history_window:
            self._remove_path(self.history.popleft())
        return self.snapshot()

    def snapshot(self) -> dict[str, int]:
        return {
            "requests_observed": self.request_index,
            "nodes": self.node_count,
            "retained_paths": len(self.history),
            "retained_schema_tokens": self.retained_schema_tokens,
            "history_window": self.history_window,
        }


@dataclass
class FPTreeNode:
    tool_id: str | None
    count: int = 0
    children: dict[str, "FPTreeNode"] = field(default_factory=dict)


class FittedOrderingPlanner:
    """Frozen training-only ordering baselines.

    ``fp_tree_conditional`` inserts globally support-sorted training
    transactions into a conventional FP-tree, then greedily follows the most
    supported available child for an evaluation menu. The traversal is a
    tool-ordering adaptation; FP-Growth itself mines itemsets rather than
    prescribing an online serving order.

    ``conditional_pair`` and ``conditional_pair_triple`` directly test whether
    pair/triple co-occurrence adds value beyond global support. Their statistics
    are unordered same-transaction evidence, not observed execution transitions.
    Every unresolved tie falls back to function-name alphabetical order and
    then canonical tool ID, so unseen tools remain deterministic.
    """

    POLICIES = {
        "frequency_fitted",
        "schema_cost_fitted",
        "fp_tree_conditional",
        "conditional_pair",
        "conditional_pair_triple",
    }

    def __init__(
        self,
        tools: Mapping[str, CanonicalTool],
        training_transactions: Sequence[Sequence[str]],
        *,
        policy: str,
        max_cooccurrence_transaction_size: int = 25,
    ) -> None:
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown fitted ordering policy: {policy}")
        if not training_transactions:
            raise ValueError("At least one training transaction is required")
        if max_cooccurrence_transaction_size < 2:
            raise ValueError("max_cooccurrence_transaction_size must be >= 2")
        self.tools = tools
        self.policy = policy
        self.max_cooccurrence_transaction_size = max_cooccurrence_transaction_size
        self.support: Counter[str] = Counter()
        self.pair_support: Counter[frozenset[str]] = Counter()
        self.triple_support: Counter[frozenset[str]] = Counter()
        self.skipped_large_transactions = 0

        transactions: list[tuple[str, ...]] = []
        for raw_transaction in training_transactions:
            transaction = tuple(
                tool_id
                for tool_id in dict.fromkeys(raw_transaction)
                if tool_id in tools
            )
            transactions.append(transaction)
            self.support.update(transaction)
            if len(transaction) > max_cooccurrence_transaction_size:
                self.skipped_large_transactions += 1
                continue
            self.pair_support.update(
                frozenset(items) for items in combinations(transaction, 2)
            )
            self.triple_support.update(
                frozenset(items) for items in combinations(transaction, 3)
            )

        self.root = FPTreeNode(tool_id=None)
        for transaction in transactions:
            node = self.root
            for tool_id in sorted(transaction, key=self._frequency_key):
                child = node.children.get(tool_id)
                if child is None:
                    child = FPTreeNode(tool_id=tool_id)
                    node.children[tool_id] = child
                child.count += 1
                node = child

    def _validated_ids(self, tool_ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(tool_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("A fitted ordering menu must not repeat tool IDs")
        unknown = [tool_id for tool_id in ids if tool_id not in self.tools]
        if unknown:
            sample = ", ".join(repr(item) for item in unknown[:3])
            raise ValueError(f"Fitted ordering received unknown tool IDs: {sample}")
        return ids

    def _frequency_key(self, tool_id: str) -> tuple[int, str, str]:
        return (
            -self.support[tool_id],
            self.tools[tool_id].name.casefold(),
            tool_id,
        )

    def _schema_cost_key(self, tool_id: str) -> tuple[int, int, str, str]:
        return (
            -(self.support[tool_id] * self.tools[tool_id].schema_tokens),
            -self.support[tool_id],
            self.tools[tool_id].name.casefold(),
            tool_id,
        )

    def _conditional_order(self, ids: tuple[str, ...], use_triples: bool) -> tuple[str, ...]:
        remaining = set(ids)
        ordered: list[str] = []
        while remaining:
            if not ordered:
                chosen = min(remaining, key=self._frequency_key)
            else:
                previous = ordered[-1]

                def key(candidate: str) -> tuple[int, int, int, str, str]:
                    pair = self.pair_support[frozenset((previous, candidate))]
                    triple = 0
                    if use_triples and len(ordered) >= 2:
                        triple = self.triple_support[
                            frozenset((ordered[-2], previous, candidate))
                        ]
                    return (
                        -triple,
                        -pair,
                        -self.support[candidate],
                        self.tools[candidate].name.casefold(),
                        candidate,
                    )

                chosen = min(remaining, key=key)
            ordered.append(chosen)
            remaining.remove(chosen)
        return tuple(ordered)

    def _fp_tree_order(self, ids: tuple[str, ...]) -> OrderingPlan:
        remaining = set(ids)
        node = self.root
        matched: list[str] = []
        while remaining:
            candidates = [
                child
                for tool_id, child in node.children.items()
                if tool_id in remaining
            ]
            if not candidates:
                break

            def child_key(child: FPTreeNode) -> tuple[int, int, str, str]:
                if child.tool_id is None:
                    raise RuntimeError("An FP-tree child has no tool ID")
                return (
                    -child.count,
                    -self.support[child.tool_id],
                    self.tools[child.tool_id].name.casefold(),
                    child.tool_id,
                )

            chosen = min(
                candidates,
                key=child_key,
            )
            assert chosen.tool_id is not None
            matched.append(chosen.tool_id)
            remaining.remove(chosen.tool_id)
            node = chosen
        fallback = tuple(sorted(remaining, key=self._frequency_key))
        ordered = (*matched, *fallback)
        return OrderingPlan(
            ordered_ids=ordered,
            matched_prefix_ids=tuple(matched),
            fallback_ids=fallback,
            hinted_schema_tokens=sum(
                self.tools[tool_id].schema_tokens for tool_id in matched
            ),
        )

    def plan(self, selected_ids: Sequence[str]) -> OrderingPlan:
        ids = self._validated_ids(selected_ids)
        if self.policy == "frequency_fitted":
            ordered = tuple(sorted(ids, key=self._frequency_key))
        elif self.policy == "schema_cost_fitted":
            ordered = tuple(sorted(ids, key=self._schema_cost_key))
        elif self.policy == "conditional_pair":
            ordered = self._conditional_order(ids, use_triples=False)
        elif self.policy == "conditional_pair_triple":
            ordered = self._conditional_order(ids, use_triples=True)
        else:
            return self._fp_tree_order(ids)
        return OrderingPlan(
            ordered_ids=ordered,
            matched_prefix_ids=(),
            fallback_ids=ordered,
            hinted_schema_tokens=0,
        )

    def snapshot(self) -> dict[str, int | str]:
        def count_nodes(node: FPTreeNode) -> int:
            return sum(1 + count_nodes(child) for child in node.children.values())

        return {
            "policy": self.policy,
            "training_tools": len(self.support),
            "fp_tree_nodes": count_nodes(self.root),
            "pair_patterns": len(self.pair_support),
            "triple_patterns": len(self.triple_support),
            "skipped_large_transactions": self.skipped_large_transactions,
            "max_cooccurrence_transaction_size": self.max_cooccurrence_transaction_size,
        }


class OnlineFrequencyPlanner:
    """Causal online presence-frequency ordering.

    Orders a selected set by how many *already-served* requests contained each
    tool, descending. This is the adaptive counterpart to
    ``FittedOrderingPlanner``'s ``frequency_fitted`` policy, which freezes its
    support on a task-disjoint training corpus and never updates it.

    The distinction matters because every ordering policy benchmarked against
    the fitted family — ToolTrie, CacheWeaver, both ContextPilot adaptations —
    updates state from the served stream, while the fitted family cannot. This
    planner isolates that difference: it uses the same frequency signal, but
    estimated online.

    ``plan`` reads only counts accumulated from strictly earlier requests, so it
    never sees the request it is ordering. Call ``observe`` only after the
    corresponding request has been served. Tools never seen before have count
    zero and therefore sort last, which keeps novel tools out of the shared
    prefix without a special case.
    """

    def __init__(self, tools: Mapping[str, CanonicalTool]) -> None:
        self.tools = tools
        self.presence: Counter[str] = Counter()
        self.request_index = 0

    def _validated_ids(self, tool_ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(tool_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("An online-frequency menu must not repeat tool IDs")
        unknown = [tool_id for tool_id in ids if tool_id not in self.tools]
        if unknown:
            sample = ", ".join(repr(item) for item in unknown[:3])
            raise ValueError(f"Online frequency received unknown tool IDs: {sample}")
        return ids

    def _key(self, tool_id: str) -> tuple[int, str, str]:
        return (
            -self.presence[tool_id],
            self.tools[tool_id].name.casefold(),
            tool_id,
        )

    def plan(self, selected_ids: Sequence[str]) -> OrderingPlan:
        ids = self._validated_ids(selected_ids)
        ordered = tuple(sorted(ids, key=self._key))
        if set(ordered) != set(ids) or len(ordered) != len(ids):
            raise RuntimeError("Online frequency ordering is not a permutation")
        return OrderingPlan(
            ordered_ids=ordered,
            matched_prefix_ids=(),
            fallback_ids=ordered,
            hinted_schema_tokens=0,
        )

    def observe(self, ordered_ids: Sequence[str]) -> dict[str, int]:
        ids = self._validated_ids(ordered_ids)
        self.request_index += 1
        self.presence.update(ids)
        return self.snapshot()

    def snapshot(self) -> dict[str, int]:
        return {
            "requests_observed": self.request_index,
            "tools_seen": len(self.presence),
            "max_presence": max(self.presence.values(), default=0),
        }


class OnlinePairTriplePlanner:
    """Causal online pair/triple co-occurrence ordering.

    The adaptive counterpart to ``FittedOrderingPlanner``'s ``conditional_pair``
    and ``conditional_pair_triple`` policies, which freeze their co-occurrence
    statistics on a task-disjoint training corpus and never update them.

    That freezing is why brief §7 Q5 has never been answered: both fitted
    policies emit orderings byte-identical to ``frequency_fitted`` on every
    workload measured, because the training corpus never observed the pairs that
    appear in the evaluation menus, so every lookup returns zero and the key
    falls through to plain frequency.

    A second reason applies to padded menus specifically, and it is structural
    rather than a property of any planner: when a menu holds a fixed core plus
    one varying tool, pair support is a deterministic function of the two tools'
    presence counts, so a pair key carries no signal a frequency key does not.
    ``scripts/audit_pair_triple_information.py`` measures this — 100.00% of
    ``bfcl-padded64`` pairs satisfy ``support(a, b) == min(presence(a),
    presence(b))`` with zero violations, against 67.34% on ``toolret-bm25-k128``.
    This planner can only discriminate where that identity breaks, which means
    retrieved menus.

    Statistics accumulate from **already-served** menus. ``plan`` reads only
    counts from strictly earlier requests; call ``observe`` after the request has
    been served. Ordering is a greedy chain: the first tool by presence, then
    each next tool by triple support with the previous two, then pair support
    with the previous one, then presence, then function name, then tool ID.
    """

    def __init__(
        self,
        tools: Mapping[str, CanonicalTool],
        *,
        use_triples: bool = True,
    ) -> None:
        self.tools = tools
        self.use_triples = use_triples
        self.presence: Counter[str] = Counter()
        self.pair_support: Counter[frozenset[str]] = Counter()
        self.triple_support: Counter[frozenset[str]] = Counter()
        self.request_index = 0

    def _validated_ids(self, tool_ids: Sequence[str]) -> tuple[str, ...]:
        ids = tuple(tool_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("A pair/triple menu must not repeat tool IDs")
        unknown = [tool_id for tool_id in ids if tool_id not in self.tools]
        if unknown:
            sample = ", ".join(repr(item) for item in unknown[:3])
            raise ValueError(f"Online pair/triple received unknown tool IDs: {sample}")
        return ids

    def plan(self, selected_ids: Sequence[str]) -> OrderingPlan:
        ids = self._validated_ids(selected_ids)
        remaining = set(ids)
        ordered: list[str] = []

        while remaining:
            if not ordered:
                chosen = min(
                    remaining,
                    key=lambda item: (
                        -self.presence[item],
                        self.tools[item].name.casefold(),
                        item,
                    ),
                )
            else:
                previous = ordered[-1]

                def key(candidate: str) -> tuple[int, int, int, str, str]:
                    pair = self.pair_support[frozenset((previous, candidate))]
                    triple = 0
                    if self.use_triples and len(ordered) >= 2:
                        triple = self.triple_support[
                            frozenset((ordered[-2], previous, candidate))
                        ]
                    return (
                        -triple,
                        -pair,
                        -self.presence[candidate],
                        self.tools[candidate].name.casefold(),
                        candidate,
                    )

                chosen = min(remaining, key=key)
            ordered.append(chosen)
            remaining.remove(chosen)

        result = tuple(ordered)
        if set(result) != set(ids) or len(result) != len(ids):
            raise RuntimeError("Online pair/triple ordering is not a permutation")
        return OrderingPlan(
            ordered_ids=result,
            matched_prefix_ids=(),
            fallback_ids=result,
            hinted_schema_tokens=0,
        )

    def observe(self, ordered_ids: Sequence[str]) -> dict[str, int]:
        ids = self._validated_ids(ordered_ids)
        self.request_index += 1
        self.presence.update(ids)
        self.pair_support.update(frozenset(items) for items in combinations(ids, 2))
        if self.use_triples:
            self.triple_support.update(
                frozenset(items) for items in combinations(ids, 3)
            )
        return {
            "requests_observed": self.request_index,
            "tools_seen": len(self.presence),
            "pairs_seen": len(self.pair_support),
            "triples_seen": len(self.triple_support),
        }

    def snapshot(self) -> dict[str, int]:
        return {
            "requests_observed": self.request_index,
            "tools_seen": len(self.presence),
            "pairs_seen": len(self.pair_support),
            "triples_seen": len(self.triple_support),
        }
