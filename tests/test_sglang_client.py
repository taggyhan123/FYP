from tatm.sglang_client import (
    cached_token_projection,
    initial_missing_cached_reconciliation,
    metric_delta,
    parse_prometheus,
)


def test_parse_sglang_prometheus_aggregates_counters_and_peaks_gauges() -> None:
    text = """
# HELP ignored ignored
sglang:prompt_tokens_total{model_name="x"} 100
sglang:cached_tokens_total{cache_source="device"} 40
sglang:cached_tokens_total{cache_source="host"} 5
sglang:time_to_first_token_seconds_sum 1.25
sglang:time_to_first_token_seconds_count 2
sglang:token_usage{scheduler="0"} 0.4
sglang:token_usage{scheduler="1"} 0.7
unrelated_metric 999
"""
    parsed = parse_prometheus(text)
    assert parsed["sglang:prompt_tokens_total"] == 100
    assert parsed["sglang:cached_tokens_total"] == 45
    assert parsed["sglang:time_to_first_token_seconds_sum"] == 1.25
    assert parsed["sglang:token_usage"] == 0.7


def test_sglang_metric_delta_keeps_after_gauge() -> None:
    delta = metric_delta(
        {"sglang:prompt_tokens_total": 10, "sglang:token_usage": 0.8},
        {"sglang:prompt_tokens_total": 25, "sglang:token_usage": 0.2},
    )
    assert delta == {
        "sglang:prompt_tokens_total": 15,
        "sglang:token_usage": 0.2,
    }


def test_cached_token_projection_reads_standard_and_extension_fields() -> None:
    projected = cached_token_projection(
        {
            "usage": {"prompt_tokens_details": {"cached_tokens": 96}},
            "sglext": {
                "cached_tokens_details": {
                    "device": 80,
                    "host": 16,
                    "unrelated": 1,
                }
            },
        }
    )
    assert projected == {
        "cached_prompt_tokens": 96,
        "cached_tokens_details": {"device": 80, "host": 16},
    }


def test_missing_initial_cached_value_uses_independent_aggregate_counter() -> None:
    run = {
        "aggregate_metric_delta": {"sglang:cached_tokens_total": 64},
        "counter_validation": {
            "request_counter_matches": True,
            "prompt_counter_matches": True,
            # This field is also derived from responses and must not be used as
            # the independent reconciliation target.
            "response_cached_tokens": 64,
        },
        "results": [
            {
                "index": 0,
                "usage": {"prompt_tokens_details": {}},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "usage": {"prompt_tokens_details": {"cached_tokens": 64}},
                "finish_reason": "stop",
            },
        ],
    }
    accepted = initial_missing_cached_reconciliation(run)
    assert accepted["clean"] is True

    run["aggregate_metric_delta"]["sglang:cached_tokens_total"] = 32
    rejected = initial_missing_cached_reconciliation(run)
    assert rejected["clean"] is False
    assert rejected["checks"]["cached_total_equals_sum"] is False
