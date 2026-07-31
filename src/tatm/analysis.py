from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

from tatm.io import read_jsonl
from tatm.models import CanonicalTool, TaskRecord


# Cache capacities to sweep, as fractions of a partition's distinct-tool token
# working set. 1.0 means the whole working set fits and nothing is ever evicted.
CAPACITY_FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05)


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)


def load_processed(
    processed_dir: Path,
) -> tuple[dict[str, CanonicalTool], list[TaskRecord]]:
    tools = {
        record["tool_id"]: CanonicalTool.from_record(record)
        for record in read_jsonl(processed_dir / "tools.jsonl")
    }
    tasks = [
        TaskRecord.from_record(record)
        for record in read_jsonl(processed_dir / "tasks.jsonl")
    ]
    return tools, tasks


def percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def distribution(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p25": round(percentile(values, 0.25), 3),
        "median": round(float(median(values)), 3),
        "p75": round(percentile(values, 0.75), 3),
        "p90": round(percentile(values, 0.90), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": max(values),
        "mean": round(mean(values), 3),
    }


def deduplicated_existing_ids(
    task: TaskRecord,
    tools: dict[str, CanonicalTool],
) -> tuple[str, ...]:
    return tuple(
        tool_id
        for tool_id in dict.fromkeys(task.tool_ids)
        if tool_id in tools
    )


def occurrence_counts(
    tasks: Sequence[TaskRecord],
    tools: dict[str, CanonicalTool],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for task in tasks:
        counts.update(deduplicated_existing_ids(task, tools))
    return counts


def cooccurrence_counts(
    tasks: Sequence[TaskRecord],
    tools: dict[str, CanonicalTool],
    *,
    width: int,
    max_tools_per_task: int = 25,
) -> tuple[Counter[tuple[str, ...]], int]:
    counts: Counter[tuple[str, ...]] = Counter()
    skipped = 0
    for task in tasks:
        unique_ids = deduplicated_existing_ids(task, tools)
        if len(unique_ids) > max_tools_per_task:
            skipped += 1
            continue
        counts.update(combinations(sorted(unique_ids), width))
    return counts, skipped


def adjacency_counts(
    tasks: Sequence[TaskRecord],
    tools: dict[str, CanonicalTool],
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for task in tasks:
        ids = deduplicated_existing_ids(task, tools)
        counts.update(zip(ids, ids[1:]))
    return counts


def _stable_random_rank(tool_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{tool_id}".encode()).hexdigest()


def ordering_functions(
    tools: dict[str, CanonicalTool],
    support: Counter[str],
) -> dict[str, Callable[[tuple[str, ...]], tuple[str, ...]]]:
    alphabetical_rank = {
        tool_id: (tool.name.casefold(), tool_id)
        for tool_id, tool in tools.items()
    }

    def original(ids: tuple[str, ...]) -> tuple[str, ...]:
        return ids

    def by_key(
        key: Callable[[str], Any],
    ) -> Callable[[tuple[str, ...]], tuple[str, ...]]:
        return lambda ids: tuple(sorted(ids, key=key))

    functions = {
        "original": original,
        "alphabetical": by_key(lambda item: alphabetical_rank[item]),
        "random_seed_7": by_key(lambda item: _stable_random_rank(item, 7)),
        "random_seed_42": by_key(lambda item: _stable_random_rank(item, 42)),
        "random_seed_101": by_key(lambda item: _stable_random_rank(item, 101)),
        "frequency": by_key(lambda item: (-support[item], item)),
        "schema_cost_weighted": by_key(
            lambda item: (
                -(support[item] * tools[item].schema_tokens),
                -support[item],
                item,
            )
        ),
        # The classic FP-tree global order is descending support.
        "fp_tree_global": by_key(lambda item: (-support[item], item)),
    }
    return functions


def replay_workloads(
    tasks: Sequence[TaskRecord],
    tools: dict[str, CanonicalTool],
    seed: int = 2026,
) -> dict[str, list[TaskRecord]]:
    if not tasks:
        return {"empirical": [], "uniform": [], "skewed": [], "session_bursty": []}

    generator = random.Random(seed)
    uniform = [generator.choice(tasks) for _ in range(len(tasks))]

    support = occurrence_counts(tasks, tools)
    weights = [
        1.0 + sum(support[item] for item in deduplicated_existing_ids(task, tools))
        for task in tasks
    ]
    skewed = generator.choices(tasks, weights=weights, k=len(tasks))

    groups: dict[str, list[TaskRecord]] = defaultdict(list)
    for task in tasks:
        groups[task.domain or task.source].append(task)
    group_names = sorted(groups)
    generator.shuffle(group_names)
    session_bursty: list[TaskRecord] = []
    for group_name in group_names:
        group = list(groups[group_name])
        generator.shuffle(group)
        session_bursty.extend(group)

    return {
        "empirical": list(tasks),
        "uniform": uniform,
        "skewed": skewed,
        "session_bursty": session_bursty,
    }


def trie_metrics(
    sequences: Iterable[tuple[str, ...]],
    tools: dict[str, CanonicalTool],
    *,
    block_size: int = 16,
) -> dict[str, int | float]:
    root = TrieNode()
    node_count = 0
    naive_nodes = 0
    request_count = 0
    total_tool_tokens = 0
    reusable_tool_tokens = 0
    cacheable_block_tokens = 0

    for sequence in sequences:
        request_count += 1
        naive_nodes += len(sequence)
        request_tokens = sum(tools[item].schema_tokens for item in sequence)
        total_tool_tokens += request_tokens

        node = root
        shared_tokens = 0
        matching = True
        for tool_id in sequence:
            if matching and tool_id in node.children:
                node = node.children[tool_id]
                shared_tokens += tools[tool_id].schema_tokens
                continue
            matching = False
            child = node.children.get(tool_id)
            if child is None:
                child = TrieNode()
                node.children[tool_id] = child
                node_count += 1
            node = child

        reusable_tool_tokens += shared_tokens
        cacheable_block_tokens += (shared_tokens // block_size) * block_size

    compression = 1.0 - (node_count / naive_nodes) if naive_nodes else 0.0
    reuse_ratio = (
        cacheable_block_tokens / total_tool_tokens if total_tool_tokens else 0.0
    )
    return {
        "requests": request_count,
        "trie_nodes": node_count,
        "naive_nodes": naive_nodes,
        "node_compression_ratio": round(compression, 6),
        "total_tool_tokens": total_tool_tokens,
        "reusable_tool_tokens": reusable_tool_tokens,
        "cacheable_block_tokens": cacheable_block_tokens,
        "estimated_block_reuse_ratio": round(reuse_ratio, 6),
        "block_size": block_size,
    }


@dataclass
class _CacheNode:
    key: str | None = None
    parent: "_CacheNode | None" = None
    children: dict[str, "_CacheNode"] = field(default_factory=dict)
    tokens: int = 0
    last_used: int = 0
    live: bool = True


def bounded_trie_metrics(
    sequences: Iterable[tuple[str, ...]],
    tools: dict[str, CanonicalTool],
    *,
    block_size: int = 16,
    capacity_tokens: int | None = None,
) -> dict[str, int | float | None]:
    """Trie reuse under a finite cache with leaf-first LRU eviction.

    `trie_metrics` retains every node forever, which makes its output depend
    only on the multiset of sequences and not on their order: a bursty replay
    and a uniformly interleaved replay of the same requests score identically.
    Request ordering can only matter once capacity forces eviction, so locality
    questions need this function rather than `trie_metrics`.

    A node is evictable only when it has no children, mirroring a radix tree
    that cannot free a block while a longer path still depends on it.
    `capacity_tokens=None` disables eviction and reproduces `trie_metrics`.
    """
    root = _CacheNode()
    created = 0
    live_nodes = 0
    naive_nodes = 0
    request_count = 0
    total_tool_tokens = 0
    reusable_tool_tokens = 0
    cacheable_block_tokens = 0
    retained_tokens = 0
    evictions = 0
    clock = 0
    # (last_used, tiebreak, node); stale entries are skipped at pop time.
    eviction_heap: list[tuple[int, int, _CacheNode]] = []
    sequence_number = 0

    def touch(node: _CacheNode) -> None:
        nonlocal clock, sequence_number
        clock += 1
        sequence_number += 1
        node.last_used = clock
        heappush(eviction_heap, (node.last_used, sequence_number, node))

    def evict_one() -> bool:
        nonlocal retained_tokens, evictions, live_nodes, sequence_number
        while eviction_heap:
            last_used, _, node = heappop(eviction_heap)
            if not node.live or node.children or node.last_used != last_used:
                continue  # stale entry, already evicted, or not a leaf
            node.live = False
            live_nodes -= 1
            retained_tokens -= node.tokens
            evictions += 1
            parent = node.parent
            if parent is not None and node.key is not None:
                parent.children.pop(node.key, None)
                if parent is not root and not parent.children:
                    # The parent just became a leaf and is now evictable.
                    sequence_number += 1
                    heappush(
                        eviction_heap,
                        (parent.last_used, sequence_number, parent),
                    )
            return True
        return False

    for sequence in sequences:
        request_count += 1
        naive_nodes += len(sequence)
        total_tool_tokens += sum(tools[item].schema_tokens for item in sequence)

        node = root
        shared_tokens = 0
        matching = True
        for tool_id in sequence:
            if matching and tool_id in node.children:
                node = node.children[tool_id]
                shared_tokens += tools[tool_id].schema_tokens
                touch(node)
                continue
            matching = False
            child = node.children.get(tool_id)
            if child is None:
                child = _CacheNode(
                    key=tool_id,
                    parent=node,
                    tokens=tools[tool_id].schema_tokens,
                )
                node.children[tool_id] = child
                created += 1
                live_nodes += 1
                retained_tokens += child.tokens
            node = child
            touch(node)

        reusable_tool_tokens += shared_tokens
        cacheable_block_tokens += (shared_tokens // block_size) * block_size

        if capacity_tokens is not None:
            while retained_tokens > capacity_tokens and evict_one():
                pass

    compression = 1.0 - (created / naive_nodes) if naive_nodes else 0.0
    reuse_ratio = (
        cacheable_block_tokens / total_tool_tokens if total_tool_tokens else 0.0
    )
    return {
        "requests": request_count,
        "trie_nodes": created,
        "naive_nodes": naive_nodes,
        "node_compression_ratio": round(compression, 6),
        "total_tool_tokens": total_tool_tokens,
        "reusable_tool_tokens": reusable_tool_tokens,
        "cacheable_block_tokens": cacheable_block_tokens,
        "estimated_block_reuse_ratio": round(reuse_ratio, 6),
        "block_size": block_size,
        "capacity_tokens": capacity_tokens,
        "evictions": evictions,
        "live_nodes": live_nodes,
        "final_retained_tokens": retained_tokens,
    }


def locality_metrics(
    replay: Sequence[TaskRecord],
    tools: dict[str, CanonicalTool],
) -> dict[str, float | int]:
    if len(replay) < 2:
        return {
            "adjacent_pairs": 0,
            "same_domain_ratio": 0.0,
            "shared_tool_ratio": 0.0,
            "mean_tool_jaccard": 0.0,
        }
    same_domain = 0
    shared_tool = 0
    jaccards: list[float] = []
    for left, right in zip(replay, replay[1:]):
        if (left.domain or left.source) == (right.domain or right.source):
            same_domain += 1
        left_ids = set(deduplicated_existing_ids(left, tools))
        right_ids = set(deduplicated_existing_ids(right, tools))
        intersection = left_ids & right_ids
        union = left_ids | right_ids
        if intersection:
            shared_tool += 1
        jaccards.append(len(intersection) / len(union) if union else 0.0)
    denominator = len(replay) - 1
    return {
        "adjacent_pairs": denominator,
        "same_domain_ratio": round(same_domain / denominator, 6),
        "shared_tool_ratio": round(shared_tool / denominator, 6),
        "mean_tool_jaccard": round(mean(jaccards), 6),
    }


def _named_item(
    ids: tuple[str, ...],
    count: int,
    tools: dict[str, CanonicalTool],
) -> dict[str, Any]:
    return {
        "tool_ids": list(ids),
        "tool_names": [tools[item].name for item in ids],
        "count": count,
    }


def _named_adjacency(
    ids: tuple[str, str],
    count: int,
    outgoing: Counter[str],
    tools: dict[str, CanonicalTool],
) -> dict[str, Any]:
    record = _named_item(ids, count, tools)
    record["p_next_given_current"] = round(count / outgoing[ids[0]], 6)
    return record


def analyze_partition(
    name: str,
    tasks: Sequence[TaskRecord],
    tools: dict[str, CanonicalTool],
) -> dict[str, Any]:
    support = occurrence_counts(tasks, tools)
    task_tool_counts = [
        len(deduplicated_existing_ids(task, tools)) for task in tasks
    ]
    task_token_counts = [
        sum(
            tools[item].schema_tokens
            for item in deduplicated_existing_ids(task, tools)
        )
        for task in tasks
    ]
    pairs, pair_skipped = cooccurrence_counts(tasks, tools, width=2)
    triples, triple_skipped = cooccurrence_counts(tasks, tools, width=3)
    adjacency = adjacency_counts(tasks, tools)
    outgoing = Counter()
    for (current, _next), count in adjacency.items():
        outgoing[current] += count

    orderers = ordering_functions(tools, support)
    replays = replay_workloads(tasks, tools)
    ordering_results: list[dict[str, Any]] = []
    locality_results: list[dict[str, Any]] = []
    capacity_results: list[dict[str, Any]] = []

    observed_ids = {
        tool_id for task in tasks for tool_id in deduplicated_existing_ids(task, tools)
    }
    working_set_tokens = sum(tools[tool_id].schema_tokens for tool_id in observed_ids)

    for replay_name, replay in replays.items():
        locality_results.append(
            {
                "partition": name,
                "replay": replay_name,
                **locality_metrics(replay, tools),
            }
        )
        base_sequences = [
            deduplicated_existing_ids(task, tools) for task in replay
        ]
        for ordering_name, orderer in orderers.items():
            ordered_sequences = [orderer(sequence) for sequence in base_sequences]
            ordering_results.append(
                {
                    "partition": name,
                    "replay": replay_name,
                    "ordering": ordering_name,
                    **trie_metrics(ordered_sequences, tools),
                }
            )
            # Unbounded retention makes reuse order-invariant, so the replays
            # above cannot express locality. Repeat under finite capacity.
            for fraction in CAPACITY_FRACTIONS:
                capacity = max(1, int(working_set_tokens * fraction))
                capacity_results.append(
                    {
                        "partition": name,
                        "replay": replay_name,
                        "ordering": ordering_name,
                        "capacity_fraction": fraction,
                        "working_set_tokens": working_set_tokens,
                        **bounded_trie_metrics(
                            ordered_sequences,
                            tools,
                            capacity_tokens=capacity,
                        ),
                    }
                )

    top_tools = [
        {
            "tool_id": tool_id,
            "name": tools[tool_id].name,
            "source": tools[tool_id].source,
            "domain": tools[tool_id].domain,
            "occurrences": count,
            "schema_tokens": tools[tool_id].schema_tokens,
            "weighted_tokens": count * tools[tool_id].schema_tokens,
        }
        for tool_id, count in support.most_common(50)
    ]
    top_weighted = sorted(
        (
            {
                "tool_id": tool_id,
                "name": tools[tool_id].name,
                "source": tools[tool_id].source,
                "domain": tools[tool_id].domain,
                "occurrences": count,
                "schema_tokens": tools[tool_id].schema_tokens,
                "weighted_tokens": count * tools[tool_id].schema_tokens,
            }
            for tool_id, count in support.items()
        ),
        key=lambda row: (-row["weighted_tokens"], row["tool_id"]),
    )
    top_pairs = [
        _named_item(ids, count, tools)
        for ids, count in pairs.most_common(30)
    ]
    top_triples = [
        _named_item(ids, count, tools)
        for ids, count in triples.most_common(30)
    ]
    top_adjacency = [
        _named_adjacency(ids, count, outgoing, tools)
        for ids, count in adjacency.most_common(30)
    ]

    return {
        "partition": name,
        "tasks": len(tasks),
        "evidence_types": dict(
            sorted(Counter(task.evidence_type for task in tasks).items())
        ),
        "task_tool_count": distribution(task_tool_counts),
        "task_schema_tokens": distribution(task_token_counts),
        "unique_observed_tools": len(support),
        "task_domain_counts": dict(
            Counter(task.domain or task.source for task in tasks).most_common()
        ),
        "top_tools": top_tools,
        "top_schema_weighted_tools": top_weighted[:50],
        "top_pairs": top_pairs,
        "top_triples": top_triples,
        "top_ordered_adjacency_proxy": top_adjacency,
        "cooccurrence_tasks_skipped_over_25_tools": {
            "pairs": pair_skipped,
            "triples": triple_skipped,
        },
        "locality": locality_results,
        "ordering_results": ordering_results,
        "capacity_results": capacity_results,
        "working_set_tokens": working_set_tokens,
    }


def analyze_all(
    tools: dict[str, CanonicalTool],
    tasks: Sequence[TaskRecord],
    *,
    tokenizer: str,
) -> dict[str, Any]:
    issue_counts = Counter(
        issue for tool in tools.values() for issue in tool.issues
    )
    source_counts = Counter(tool.source for tool in tools.values())
    domain_counts = Counter(tool.domain or "unknown" for tool in tools.values())
    schema_tokens = [tool.schema_tokens for tool in tools.values()]
    source_schema_stats = {
        source: distribution(
            [
                tool.schema_tokens
                for tool in tools.values()
                if tool.source == source
            ]
        )
        for source in sorted(source_counts)
    }
    longest_tools = [
        {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "source": tool.source,
            "domain": tool.domain,
            "schema_tokens": tool.schema_tokens,
        }
        for tool in sorted(
            tools.values(),
            key=lambda item: (-item.schema_tokens, item.tool_id),
        )[:25]
    ]
    partitions = {
        "toolret_gold": [
            task for task in tasks if task.evidence_type == "gold_relevance"
        ],
        "bfcl_exposed": [
            task for task in tasks if task.evidence_type == "exposed_menu"
        ],
    }

    return {
        "format_version": 1,
        "tokenizer": tokenizer,
        "methodology": {
            "toolret_evidence": "gold relevance labels",
            "bfcl_evidence": "exposed function menus, not gold calls",
            "token_measure": "canonical JSON serialization without chat-template wrapper",
            "cache_estimate": (
                "tool-unit trie with 16-token block rounding; it excludes constant "
                "system/user prefixes and must be validated against vLLM"
            ),
            "replays": {
                "empirical": "dataset file/config order",
                "uniform": "uniform task sampling with replacement",
                "skewed": (
                    "controlled sampling weighted by aggregate benchmark tool support"
                ),
                "session_bursty": "same-domain tasks grouped into contiguous sessions",
            },
        },
        "inventory": {
            "tools": len(tools),
            "tasks": len(tasks),
            "tool_sources": dict(sorted(source_counts.items())),
            "source_schema_token_distribution": source_schema_stats,
            "top_tool_domains": dict(domain_counts.most_common(50)),
            "schema_token_distribution": distribution(schema_tokens),
            "longest_tools": longest_tools,
            "schema_issue_counts": dict(sorted(issue_counts.items())),
            "tools_with_any_issue": sum(bool(tool.issues) for tool in tools.values()),
        },
        "partitions": {
            name: analyze_partition(name, partition_tasks, tools)
            for name, partition_tasks in partitions.items()
        },
    }
