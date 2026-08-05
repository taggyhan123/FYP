import pytest

from tatm.replay_summary import summarize_labeled_replays, summarize_trial


def replay(label: str, cached: int, ttft: float = 2.0) -> dict:
    return {
        "engine": "vllm",
        "run_label": label,
        "execution_condition": {
            "role": "ordering_candidate",
            "materialization": "ordinary_selected_tool_text_prefill",
        },
        "cache_reset_before": True,
        "counter_validation": {"clean": True},
        "wall_seconds": 3.0,
        "aggregate_metric_delta": {
            "vllm:prompt_tokens_cached": cached,
            "vllm:request_prefill_kv_computed_tokens_sum": 200 - cached,
            "vllm:time_to_first_token_seconds_sum": ttft,
            "vllm:request_prefill_time_seconds_sum": 1.0,
        },
        "direct_measurements_by_reuse_bucket": {
            "partial_50_75": {
                "requests": 2,
                "aggregate_cached_ratio": cached / 200,
                "ttft_seconds": {"n": 2, "mean": ttft / 2},
                "prefill_seconds": {"n": 2, "mean": 0.5},
                "wall_seconds": {"n": 2, "mean": 1.5},
            }
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
    assert result["execution_condition"]["role"] == "ordering_candidate"


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
    partial = summary["conditions"]["tooltrie"]["direct_reuse_buckets"][
        "partial_50_75"
    ]
    assert partial["cached_ratio"]["mean"] == 0.75
    assert partial["mean_ttft_seconds"]["n"] == 2
    assert summary["conditions"]["tooltrie"]["execution_condition"]["role"] == (
        "ordering_candidate"
    )


def test_unvalidated_replay_is_rejected() -> None:
    payload = replay("bad", 10)
    payload["counter_validation"] = {"clean": False}
    with pytest.raises(ValueError, match="no clean counter validation"):
        summarize_trial(payload)


def test_repeated_trials_reject_condition_metadata_drift() -> None:
    first = replay("a1", 100)
    second = replay("a2", 100)
    second["execution_condition"] = {"role": "ordinary_text_prefill_fallback"}
    with pytest.raises(ValueError, match="different condition metadata"):
        summarize_labeled_replays([("alpha", first), ("alpha", second)])


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
