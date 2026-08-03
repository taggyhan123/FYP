"""Engine-aware summaries for repeated ordering replays."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from tatm.stats import describe


def summarize_trial(payload: Mapping[str, Any]) -> dict[str, Any]:
    validation = payload.get("counter_validation") or {}
    if validation.get("clean") is not True:
        raise ValueError(
            f"Replay {payload.get('run_label', '<unknown>')} has no clean "
            "counter validation"
        )
    engine = str(payload.get("engine") or "vllm")
    if engine not in {"vllm", "sglang"}:
        raise ValueError(f"Unknown replay engine: {engine}")
    if engine == "vllm" and payload.get("cache_reset_before") is not True:
        raise ValueError("vLLM replay did not record a pre-run cache reset")
    if engine == "sglang" and payload.get("cache_flushed_before") is not True:
        raise ValueError("SGLang replay did not record a pre-run cache flush")

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Replay has no request results")
    prompt_tokens = sum(
        int((row.get("usage") or {}).get("prompt_tokens", 0)) for row in results
    )
    delta = payload.get("aggregate_metric_delta") or {}
    if engine == "vllm":
        cached_tokens = float(delta["vllm:prompt_tokens_cached"])
        computed_tokens = float(delta["vllm:request_prefill_kv_computed_tokens_sum"])
        ttft_seconds = float(delta["vllm:time_to_first_token_seconds_sum"])
        prefill_seconds = float(delta["vllm:request_prefill_time_seconds_sum"])
        e2e_seconds = None
    else:
        cached_tokens = float(delta["sglang:cached_tokens_total"])
        computed_tokens = prompt_tokens - cached_tokens
        ttft_seconds = float(delta["sglang:time_to_first_token_seconds_sum"])
        prefill_seconds = None
        e2e_seconds = float(delta["sglang:e2e_request_latency_seconds_sum"])

    case_ids = [str(row.get("case_id") or row["task_id"]) for row in results]
    menu_by_case = {
        case_id: sorted(str(tool_id) for tool_id in row.get("tool_ids", []))
        for case_id, row in zip(case_ids, results, strict=True)
    }
    return {
        "engine": engine,
        "run_label": payload.get("run_label"),
        "request_count": len(results),
        "case_ids": case_ids,
        "menu_by_case": menu_by_case,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "cached_ratio": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        "computed_prompt_tokens": computed_tokens,
        "ttft_seconds": ttft_seconds,
        "prefill_seconds": prefill_seconds,
        "e2e_seconds": e2e_seconds,
        "wall_seconds": float(payload["wall_seconds"]),
    }


def summarize_labeled_replays(
    labeled_payloads: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label, payload in labeled_payloads:
        grouped[label].append(summarize_trial(payload))
    if not grouped:
        raise ValueError("No replay files supplied")

    engines = {trial["engine"] for trials in grouped.values() for trial in trials}
    if len(engines) != 1:
        raise ValueError("vLLM and SGLang must be summarized separately")
    engine = engines.pop()
    reference_case_set: set[str] | None = None
    reference_case_sequence: list[str] | None = None
    same_case_sets = True
    same_request_sequence = True
    same_selected_tool_sets = True
    reference_menus: dict[str, list[str]] | None = None
    summary: dict[str, Any] = {}

    for label, trials in sorted(grouped.items()):
        first_sequence = trials[0]["case_ids"]
        if any(trial["case_ids"] != first_sequence for trial in trials[1:]):
            raise ValueError(f"Trials for {label} use different request sequences")
        if any(
            trial["menu_by_case"] != trials[0]["menu_by_case"]
            for trial in trials[1:]
        ):
            raise ValueError(f"Trials for {label} use different selected tool sets")
        case_set = set(first_sequence)
        if len(case_set) != len(first_sequence):
            raise ValueError(f"Replay {label} contains duplicate case IDs")
        if reference_case_set is None:
            reference_case_set = case_set
            reference_case_sequence = first_sequence
            reference_menus = trials[0]["menu_by_case"]
        else:
            same_case_sets &= case_set == reference_case_set
            same_request_sequence &= first_sequence == reference_case_sequence
            same_selected_tool_sets &= trials[0]["menu_by_case"] == reference_menus

        fixed_fields = ("request_count", "prompt_tokens")
        for field in fixed_fields:
            values = {trial[field] for trial in trials}
            if len(values) != 1:
                raise ValueError(f"Trials for {label} disagree on {field}")
        measurements = {}
        for field in (
            "cached_prompt_tokens",
            "cached_ratio",
            "computed_prompt_tokens",
            "ttft_seconds",
            "prefill_seconds",
            "e2e_seconds",
            "wall_seconds",
        ):
            values = [float(trial[field]) for trial in trials if trial[field] is not None]
            if values:
                measurements[field] = describe(values)
        summary[label] = {
            "trials": len(trials),
            "requests": trials[0]["request_count"],
            "prompt_tokens": trials[0]["prompt_tokens"],
            "measurements": measurements,
        }

    return {
        "format_version": 1,
        "engine": engine,
        "all_conditions_have_same_case_set": same_case_sets,
        "all_conditions_have_same_selected_tool_sets": same_selected_tool_sets,
        "all_conditions_have_same_request_sequence": same_request_sequence,
        "conditions": summary,
    }
