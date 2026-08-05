from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from tatm.stats import describe


REPLAY_CONDITION_ROLES = (
    "ordering_candidate",
    "ordinary_text_prefill_fallback",
)


def selected_tool_text_condition(
    role: str,
    orderings: Sequence[str],
) -> dict[str, Any]:
    """Describe the semantics-preserving materialization used by a replay.

    Every prompt-layer baseline in this repository sends the selected tools as
    ordinary text through the OpenAI-compatible ``tools`` field.  The explicit
    fallback condition additionally preserves the selected workload's original
    order (BM25 rank in the retrieved-menu arm), retains no inactive tools, and
    never composes or edits KV tensors.
    """
    if role not in REPLAY_CONDITION_ROLES:
        raise ValueError(f"unknown replay condition role: {role}")
    normalized_orderings = tuple(sorted(set(orderings)))
    if not normalized_orderings:
        raise ValueError("a replay condition must contain at least one ordering")
    if (
        role == "ordinary_text_prefill_fallback"
        and normalized_orderings != ("original",)
    ):
        raise ValueError(
            "ordinary_text_prefill_fallback requires an original-order workload"
        )
    return {
        "role": role,
        "materialization": "ordinary_selected_tool_text_prefill",
        "workload_orderings": list(normalized_orderings),
        "selected_tool_membership": "input_workload",
        "preserves_input_order": normalized_orderings == ("original",),
        "retains_inactive_tools": False,
        "modifies_kv_tensors": False,
    }


def project_request_measurement(
    metric_delta: Mapping[str, float],
    usage: Mapping[str, Any] | None,
) -> dict[str, float | int | None]:
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    cached_tokens = float(metric_delta.get("vllm:prompt_tokens_cached", 0.0))
    computed_tokens = metric_delta.get("vllm:request_prefill_kv_computed_tokens_sum")
    inter_token_sum = float(
        metric_delta.get("vllm:inter_token_latency_seconds_sum", 0.0)
    )
    inter_token_count = float(
        metric_delta.get("vllm:inter_token_latency_seconds_count", 0.0)
    )
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_prefill_tokens": computed_tokens,
        "cached_ratio": round(cached_tokens / prompt_tokens, 8)
        if prompt_tokens
        else 0.0,
        "prefill_seconds": metric_delta.get("vllm:request_prefill_time_seconds_sum"),
        "ttft_seconds": metric_delta.get("vllm:time_to_first_token_seconds_sum"),
        "decode_seconds": metric_delta.get("vllm:request_decode_time_seconds_sum"),
        "mean_inter_token_latency_seconds": (
            round(inter_token_sum / inter_token_count, 8)
            if inter_token_count
            else None
        ),
    }


def reuse_bucket(cached_tokens: float, prompt_tokens: int) -> str:
    if prompt_tokens <= 0 or cached_tokens <= 0:
        return "cold"
    if cached_tokens >= prompt_tokens:
        return "full"
    ratio = cached_tokens / prompt_tokens
    if ratio < 0.25:
        return "partial_0_25"
    if ratio < 0.50:
        return "partial_25_50"
    if ratio < 0.75:
        return "partial_50_75"
    return "partial_75_100"


def summarize_request_measurements(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in measurements:
        grouped[
            reuse_bucket(
                float(row.get("cached_tokens", 0.0) or 0.0),
                int(row.get("prompt_tokens", 0) or 0),
            )
        ].append(row)

    def values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    def described(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        observed = values(rows, key)
        return describe(observed) if observed else {"n": 0}

    output: dict[str, Any] = {}
    for bucket in (
        "cold",
        "partial_0_25",
        "partial_25_50",
        "partial_50_75",
        "partial_75_100",
        "full",
    ):
        rows = grouped.get(bucket, [])
        if not rows:
            continue
        prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in rows)
        cached_tokens = sum(float(row.get("cached_tokens", 0.0) or 0.0) for row in rows)
        output[bucket] = {
            "requests": len(rows),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "aggregate_cached_ratio": round(cached_tokens / prompt_tokens, 8)
            if prompt_tokens
            else 0.0,
            "ttft_seconds": described(rows, "ttft_seconds"),
            "prefill_seconds": described(rows, "prefill_seconds"),
            "wall_seconds": described(rows, "wall_seconds"),
        }

    partial_rows = [
        row
        for bucket, rows in grouped.items()
        if bucket.startswith("partial_")
        for row in rows
    ]
    output["partial_all"] = {
        "requests": len(partial_rows),
        "mean_cached_ratio": round(
            fmean(float(row.get("cached_ratio", 0.0)) for row in partial_rows), 8
        )
        if partial_rows
        else 0.0,
        "ttft_seconds": described(partial_rows, "ttft_seconds"),
        "prefill_seconds": described(partial_rows, "prefill_seconds"),
        "wall_seconds": described(partial_rows, "wall_seconds"),
    }
    return output
