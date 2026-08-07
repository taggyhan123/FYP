"""Validation and materialization for external ContextPilot baselines.

ContextPilot exposes distinct offline and online APIs.  Keeping their adapters
separate prevents an offline ``fit_transform`` loop from being reported as the
official persistent online algorithm.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _payloads_by_id(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tool_ids = record.get("tool_ids")
    payloads = record.get("tools")
    if not isinstance(tool_ids, list) or not isinstance(payloads, list):
        raise ValueError("tool_ids/tools must be aligned lists")
    if len(tool_ids) != len(payloads) or len(tool_ids) != len(set(tool_ids)):
        raise ValueError("tool_ids/tools are not a one-to-one aligned menu")
    return dict(zip(tool_ids, payloads, strict=True))


def _validate_ordering(
    source_ids: Sequence[str], ordered_ids: Sequence[str], record_index: int
) -> None:
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError(f"ContextPilot duplicated a tool in input record {record_index}")
    if len(ordered_ids) != len(source_ids) or set(ordered_ids) != set(source_ids):
        raise ValueError(
            "ContextPilot changed the selected tool set for input record "
            f"{record_index}"
        )


def build_static_refit_causal_orderings(
    records: Sequence[Mapping[str, Any]],
    index_factory: Callable[[], Any],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Refit ``ContextIndex`` on each causal prefix and retain the newest order.

    This is a no-future-information adaptation, but it is deliberately named
    ``static_refit``: it is not ContextPilot's persistent online API.  String
    tool IDs are passed directly so the planner does not need a vocabulary
    constructed from future requests.
    """

    history: list[list[str]] = []
    orderings: list[list[str]] = []
    plans: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        selected = [str(tool_id) for tool_id in record.get("tool_ids", [])]
        if not selected:
            raise ValueError(f"Input record {index} has no tool_ids")
        history.append(selected)
        indexer = index_factory()
        started = clock()
        with contextlib.redirect_stdout(io.StringIO()):
            result = indexer.fit_transform([list(context) for context in history])
        planning_seconds = clock() - started
        if len(result.reordered_contexts) != len(history):
            raise ValueError(
                "ContextPilot refit output length does not match causal history"
            )
        ordered = [str(tool_id) for tool_id in result.reordered_contexts[-1]]
        _validate_ordering(selected, ordered, index)
        search_paths = getattr(result, "search_paths", None)
        search_path = (
            list(search_paths[-1])
            if isinstance(search_paths, Sequence) and search_paths
            else []
        )
        orderings.append(ordered)
        plans.append(
            {
                "history_requests": len(history),
                "planning_seconds": round(planning_seconds, 9),
                "search_path": search_path,
            }
        )
    return orderings, plans


def build_online_incremental_orderings(
    records: Sequence[Mapping[str, Any]],
    planner: Any,
    *,
    conversation_prefix: str = "tatm-evaluation",
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Plan requests through ContextPilot's persistent ``reorder`` API.

    Each benchmark request receives its own conversation ID.  The context index
    remains shared for cross-request cache alignment, while conversation-level
    de-duplication state cannot leak between independent benchmark cases.
    """

    orderings: list[list[str]] = []
    plans: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        selected = [str(tool_id) for tool_id in record.get("tool_ids", [])]
        if not selected:
            raise ValueError(f"Input record {index} has no tool_ids")
        task_id = str(record.get("task_id", index))
        conversation_id = f"{conversation_prefix}:{index}:{task_id}"
        started = clock()
        with contextlib.redirect_stdout(io.StringIO()):
            reordered, original_indices = planner.reorder(
                selected,
                conversation_id=conversation_id,
            )
        planning_seconds = clock() - started
        if list(original_indices) != [0] or len(reordered) != 1:
            raise ValueError(
                "A single-request ContextPilot online call changed request order"
            )
        ordered = [str(tool_id) for tool_id in reordered[0]]
        _validate_ordering(selected, ordered, index)
        orderings.append(ordered)
        plans.append(
            {
                "history_requests": index + 1,
                "planning_seconds": round(planning_seconds, 9),
                "conversation_id": conversation_id,
            }
        )
    return orderings, plans


def materialize_contextpilot_causal_workload(
    records: Sequence[Mapping[str, Any]],
    reordered_contexts: Sequence[Sequence[str]],
    plans: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply a causal ContextPilot ordering without changing request membership."""

    ordering_by_mode = {
        "static_refit_causal": "contextpilot_static_refit_causal",
        "online_incremental": "contextpilot_online_incremental",
    }
    if mode not in ordering_by_mode:
        raise ValueError(f"Unknown causal ContextPilot mode: {mode}")
    if len(records) != len(reordered_contexts) or len(records) != len(plans):
        raise ValueError("ContextPilot causal output length does not match input")

    output: list[dict[str, Any]] = []
    for index, (source, ordering, plan) in enumerate(
        zip(records, reordered_contexts, plans, strict=True)
    ):
        source_ids = source.get("tool_ids")
        if not isinstance(source_ids, list):
            raise ValueError(f"Input record {index} has no tool_ids list")
        ordered_ids = list(ordering)
        _validate_ordering(source_ids, ordered_ids, index)
        payload_by_id = _payloads_by_id(source)
        reordered = dict(source)
        reordered["base_ordering"] = source.get("ordering")
        reordered["ordering"] = ordering_by_mode[mode]
        reordered["tool_ids"] = ordered_ids
        reordered["tools"] = [payload_by_id[tool_id] for tool_id in ordered_ids]
        reordered["contextpilot_plan"] = {
            "mode": mode,
            "information_regime": "causal",
            "offline_transductive": False,
            "official_online_api_used": mode == "online_incremental",
            "request_order_changed": False,
            **dict(plan),
            "reference": dict(provenance),
        }
        output.append(reordered)
    return output


def materialize_contextpilot_workload(
    records: Sequence[Mapping[str, Any]],
    reordered_contexts: Sequence[Sequence[str]],
    original_indices: Sequence[int],
    search_paths: Sequence[Sequence[int]],
    *,
    mode: str,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply ContextPilot output without changing any selected tool set.

    ``reordered_contexts[i]`` must correspond to
    ``records[original_indices[i]]``. The explicit mapping matters because the
    full ContextPilot offline algorithm controls both the within-request tool
    order and the between-request execution schedule (which can be identity).
    """

    if mode not in {"intra", "intra_schedule"}:
        raise ValueError(f"Unknown ContextPilot mode: {mode}")
    count = len(records)
    if len(reordered_contexts) != count or len(original_indices) != count:
        raise ValueError("ContextPilot output length does not match input")
    if len(search_paths) != count:
        raise ValueError("ContextPilot search-path count does not match input")
    if sorted(original_indices) != list(range(count)):
        raise ValueError("ContextPilot request mapping must be a permutation")
    if mode == "intra" and list(original_indices) != list(range(count)):
        raise ValueError("intra mode must preserve request order")
    request_order_changed = list(original_indices) != list(range(count))

    output: list[dict[str, Any]] = []
    for scheduled_index, original_index in enumerate(original_indices):
        source = records[original_index]
        source_ids = source.get("tool_ids")
        if not isinstance(source_ids, list):
            raise ValueError(f"Input record {original_index} has no tool_ids list")
        ordered_ids = list(reordered_contexts[scheduled_index])
        _validate_ordering(source_ids, ordered_ids, original_index)

        payload_by_id = _payloads_by_id(source)
        reordered = dict(source)
        reordered["base_ordering"] = source.get("ordering")
        reordered["ordering"] = f"contextpilot_{mode}"
        reordered["tool_ids"] = ordered_ids
        reordered["tools"] = [payload_by_id[tool_id] for tool_id in ordered_ids]
        reordered["contextpilot_plan"] = {
            "mode": mode,
            "information_regime": "offline_transductive",
            "offline_transductive": True,
            "official_online_api_used": False,
            "scheduler_enabled": mode == "intra_schedule",
            "request_order_changed": request_order_changed,
            "original_request_index": original_index,
            "scheduled_request_index": scheduled_index,
            "search_path": list(search_paths[original_index]),
            "reference": dict(provenance),
        }
        output.append(reordered)
    return output
