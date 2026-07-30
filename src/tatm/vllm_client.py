from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any


COUNTER_METRICS = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens_cached",
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
)
GAUGE_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url}: HTTP {error.code}: {details}") from error


def fetch_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    names = (*COUNTER_METRICS, *GAUGE_METRICS)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_with_labels, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        metric_name = metric_with_labels.split("{", 1)[0]
        if metric_name.endswith("_total"):
            unsuffixed = metric_name.removesuffix("_total")
            if unsuffixed in COUNTER_METRICS:
                metric_name = unsuffixed
        if metric_name not in names:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            if metric_name in GAUGE_METRICS:
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


def served_model(base_url: str, requested: str | None = None) -> str:
    if requested:
        return requested
    model_list = request_json("GET", f"{base_url.rstrip('/')}/v1/models")
    return str(model_list["data"][0]["id"])


def response_projection(response: dict[str, Any]) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    return {
        "usage": response.get("usage", {}),
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
    }
