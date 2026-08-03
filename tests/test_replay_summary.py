import pytest

from tatm.replay_summary import summarize_labeled_replays, summarize_trial


def replay(label: str, cached: int, ttft: float = 2.0) -> dict:
    return {
        "engine": "vllm",
        "run_label": label,
        "cache_reset_before": True,
        "counter_validation": {"clean": True},
        "wall_seconds": 3.0,
        "aggregate_metric_delta": {
            "vllm:prompt_tokens_cached": cached,
            "vllm:request_prefill_kv_computed_tokens_sum": 200 - cached,
            "vllm:time_to_first_token_seconds_sum": ttft,
            "vllm:request_prefill_time_seconds_sum": 1.0,
        },
        "results": [
            {
                "case_id": "a",
                "task_id": "a",
                "tool_ids": ["x", "y"],
                "usage": {"prompt_tokens": 100},
            },
            {
                "case_id": "b",
                "task_id": "b",
                "tool_ids": ["z"],
                "usage": {"prompt_tokens": 100},
            },
        ],
    }


def test_summarize_trial_computes_reuse() -> None:
    result = summarize_trial(replay("tooltrie", 150))
    assert result["cached_ratio"] == 0.75
    assert result["computed_prompt_tokens"] == 50


def test_summarize_repeated_conditions() -> None:
    summary = summarize_labeled_replays(
        [
            ("alpha", replay("a1", 100)),
            ("alpha", replay("a2", 100, 2.2)),
            ("tooltrie", replay("t1", 150)),
            ("tooltrie", replay("t2", 150, 1.8)),
        ]
    )
    assert summary["engine"] == "vllm"
    assert summary["all_conditions_have_same_case_set"] is True
    assert summary["conditions"]["tooltrie"]["measurements"]["cached_ratio"][
        "mean"
    ] == 0.75


def test_unvalidated_replay_is_rejected() -> None:
    payload = replay("bad", 10)
    payload["counter_validation"] = {"clean": False}
    with pytest.raises(ValueError, match="no clean counter validation"):
        summarize_trial(payload)


def test_sglang_trial_uses_engine_specific_metrics() -> None:
    payload = {
        "engine": "sglang",
        "run_label": "sglang",
        "cache_flushed_before": True,
        "counter_validation": {"clean": True},
        "wall_seconds": 4.0,
        "aggregate_metric_delta": {
            "sglang:cached_tokens_total": 75,
            "sglang:time_to_first_token_seconds_sum": 1.5,
            "sglang:e2e_request_latency_seconds_sum": 2.5,
        },
        "results": [
            {
                "case_id": "a",
                "task_id": "a",
                "tool_ids": ["x"],
                "usage": {"prompt_tokens": 100},
            }
        ],
    }
    result = summarize_trial(payload)
    assert result["cached_ratio"] == 0.75
    assert result["computed_prompt_tokens"] == 25
    assert result["e2e_seconds"] == 2.5
