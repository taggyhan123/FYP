"""Small, dependency-free helpers for measuring an SGLang server."""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from typing import Any


COUNTER_METRICS = (
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:cached_tokens_total",
    "sglang:num_requests_total",
    "sglang:num_aborted_requests_total",
    "sglang:time_to_first_token_seconds_sum",
    "sglang:time_to_first_token_seconds_count",
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:e2e_request_latency_seconds_count",
    "sglang:inter_token_latency_seconds_sum",
    "sglang:inter_token_latency_seconds_count",
)

GAUGE_METRICS = (
    "sglang:cache_hit_rate",
    "sglang:token_usage",
    "sglang:full_token_usage",
    "sglang:num_used_tokens",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Aggregate selected SGLang metrics across their label dimensions."""

    totals: dict[str, float] = {}
    names = set((*COUNTER_METRICS, *GAUGE_METRICS))
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_with_labels, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        metric_name = metric_with_labels.split("{", 1)[0]
        if metric_name not in names:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if metric_name in GAUGE_METRICS:
            # Multi-scheduler deployments expose one gauge per scheduler. Peak
            # occupancy is more meaningful than summing percentages.
            totals[metric_name] = max(totals.get(metric_name, value), value)
        else:
            totals[metric_name] = totals.get(metric_name, 0.0) + value
    return totals


def metric_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {
        metric: (
            round(after.get(metric, 0.0) - before.get(metric, 0.0), 6)
            if metric in COUNTER_METRICS
            else round(after.get(metric, 0.0), 6)
        )
        for metric in (*COUNTER_METRICS, *GAUGE_METRICS)
        if metric in before or metric in after
    }


def cached_token_projection(response: dict[str, Any]) -> dict[str, Any]:
    """Extract standard and SGLang-specific cached-token response fields."""

    usage = response.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached = prompt_details.get("cached_tokens")
    extension = response.get("sglext") or {}
    by_source = extension.get("cached_tokens_details") or {}
    normalized_details = {
        key: value
        for key, value in by_source.items()
        if key in {"device", "host", "storage", "storage_backend"}
    }
    return {
        "cached_prompt_tokens": int(cached) if cached is not None else None,
        "cached_tokens_details": normalized_details or None,
    }


def flush_cache(base_url: str, timeout: int = 60) -> str:
    """Flush SGLang's RadixAttention cache and return the server message."""

    url = f"{base_url.rstrip('/')}/flush_cache"
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST {url}: HTTP {error.code}: {details}"
        ) from error
