"""Validation and materialization for the external ContextPilot baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _payloads_by_id(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tool_ids = record.get("tool_ids")
    payloads = record.get("tools")
    if not isinstance(tool_ids, list) or not isinstance(payloads, list):
        raise ValueError("tool_ids/tools must be aligned lists")
    if len(tool_ids) != len(payloads) or len(tool_ids) != len(set(tool_ids)):
        raise ValueError("tool_ids/tools are not a one-to-one aligned menu")
    return dict(zip(tool_ids, payloads, strict=True))


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
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError(
                f"ContextPilot duplicated a tool in input record {original_index}"
            )
        if len(ordered_ids) != len(source_ids) or set(ordered_ids) != set(source_ids):
            raise ValueError(
                f"ContextPilot changed the selected tool set for input record "
                f"{original_index}"
            )

        payload_by_id = _payloads_by_id(source)
        reordered = dict(source)
        reordered["base_ordering"] = source.get("ordering")
        reordered["ordering"] = f"contextpilot_{mode}"
        reordered["tool_ids"] = ordered_ids
        reordered["tools"] = [payload_by_id[tool_id] for tool_id in ordered_ids]
        reordered["contextpilot_plan"] = {
            "mode": mode,
            "offline_transductive": True,
            "scheduler_enabled": mode == "intra_schedule",
            "request_order_changed": request_order_changed,
            "original_request_index": original_index,
            "scheduled_request_index": scheduled_index,
            "search_path": list(search_paths[original_index]),
            "reference": dict(provenance),
        }
        output.append(reordered)
    return output
