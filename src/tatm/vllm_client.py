from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from typing import Any


COUNTER_METRICS = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens_cached",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
    # Decode-side cost. Needed before any extension that lengthens context
    # (e.g. retained inactive tools) can be evaluated honestly.
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:request_decode_time_seconds_sum",
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


def parse_info_labels(text: str, metric_name: str) -> dict[str, str]:
    """Extract the label set of a Prometheus Info-style metric."""
    for line in text.splitlines():
        if not line.startswith(metric_name + "{"):
            continue
        label_blob = line[len(metric_name) + 1 : line.rindex("}")]
        labels: dict[str, str] = {}
        for pair in label_blob.split('",'):
            key, separator, value = pair.partition("=")
            if separator:
                labels[key.strip()] = value.strip().strip('"')
        return labels
    return {}


def server_cache_config(base_url: str) -> dict[str, Any]:
    """Read the served cache configuration, including the prefix-caching flag.

    Recording this is what makes a cache-disabled control trustworthy: omitting
    --enable-prefix-caching does not disable the feature in vLLM V1, so the flag
    must be read back from the server rather than assumed from the command line.
    """
    text = fetch_text(f"{base_url.rstrip('/')}/metrics")
    labels = parse_info_labels(text, "vllm:cache_config_info")
    flag = labels.get("enable_prefix_caching")
    enabled: bool | None
    if flag is None:
        enabled = None
    else:
        enabled = flag.strip().lower() == "true"
    return {
        "enable_prefix_caching": enabled,
        "block_size": labels.get("block_size"),
        "num_gpu_blocks": labels.get("num_gpu_blocks"),
        "raw_labels": labels,
    }


def reset_prefix_cache(base_url: str) -> bool:
    """Drop the prefix cache so a following request is genuinely cold.

    Requires the server to run with VLLM_SERVER_DEV_MODE=1.
    """
    try:
        request_json("POST", f"{base_url.rstrip('/')}/reset_prefix_cache", {})
        return True
    except RuntimeError:
        return False
    except urllib.error.URLError:
        return False


class KvUsageSampler:
    """Poll the KV-cache gauge while a request is in flight.

    vllm:kv_cache_usage_perc is an instantaneous gauge. Sampling it only after a
    request completes always reads ~0 because the blocks have already been
    released, which is why earlier runs recorded no GPU memory evidence.
    """

    def __init__(self, base_url: str, interval_seconds: float = 0.01) -> None:
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak: dict[str, float] = {}
        self.samples = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                values = parse_prometheus(fetch_text(self._url, timeout=5))
            except Exception:  # a dropped scrape must not fail the experiment
                continue
            self.samples += 1
            for metric in GAUGE_METRICS:
                if metric in values:
                    self.peak[metric] = max(
                        self.peak.get(metric, 0.0), values[metric]
                    )
            self._stop.wait(self._interval)

    def __enter__(self) -> "KvUsageSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


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
