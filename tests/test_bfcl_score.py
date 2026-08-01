from __future__ import annotations

from tatm.bfcl_score import (
    aggregate,
    parse_predicted_calls,
    score_single_call,
    score_task,
)


def openai_call(name: str, arguments: dict) -> dict:
    return {"function": {"name": name, "arguments": __import__("json").dumps(arguments)}}


def test_parse_predicted_calls_none_and_missing_arguments() -> None:
    assert parse_predicted_calls(None) == []
    assert parse_predicted_calls([]) == []


def test_parse_predicted_calls_bad_json_marks_not_ok() -> None:
    calls = parse_predicted_calls([{"function": {"name": "f", "arguments": "{bad"}}])
    assert calls[0]["parse_ok"] is False


def test_score_single_call_exact_match() -> None:
    predicted = parse_predicted_calls([openai_call("math.factorial", {"number": 5})])[0]
    gold = {"math.factorial": {"number": [5]}}
    result = score_single_call(predicted, gold)
    assert result.name_correct
    assert result.arguments_correct


def test_score_single_call_wrong_name() -> None:
    predicted = parse_predicted_calls([openai_call("wrong_fn", {"number": 5})])[0]
    gold = {"math.factorial": {"number": [5]}}
    result = score_single_call(predicted, gold)
    assert not result.name_correct
    assert not result.arguments_correct


def test_score_single_call_optional_arg_omitted_is_acceptable() -> None:
    predicted = parse_predicted_calls([openai_call("f", {"base": 10, "height": 5})])[0]
    gold = {"f": {"base": [10], "height": [5], "unit": ["units", ""]}}
    assert score_single_call(predicted, gold).arguments_correct


def test_score_single_call_wrong_value_fails() -> None:
    predicted = parse_predicted_calls([openai_call("f", {"base": 11, "height": 5})])[0]
    gold = {"f": {"base": [10], "height": [5]}}
    assert not score_single_call(predicted, gold).arguments_correct


def test_score_single_call_extra_argument_fails() -> None:
    predicted = parse_predicted_calls([openai_call("f", {"base": 10, "extra": 1})])[0]
    gold = {"f": {"base": [10]}}
    assert not score_single_call(predicted, gold).arguments_correct


def test_score_single_call_list_argument_matches() -> None:
    predicted = parse_predicted_calls([openai_call("f", {"point": [3, 4]})])[0]
    gold = {"f": {"point": [[3, 4]]}}
    assert score_single_call(predicted, gold).arguments_correct


def test_score_single_call_numeric_tolerance() -> None:
    predicted = parse_predicted_calls([openai_call("f", {"x": 5.0})])[0]
    gold = {"f": {"x": [5]}}
    assert score_single_call(predicted, gold).arguments_correct


def test_score_task_parallel_calls_are_order_independent() -> None:
    tool_calls = [
        openai_call("spotify.play", {"artist": "Maroon 5", "duration": 15}),
        openai_call("spotify.play", {"artist": "Taylor Swift", "duration": 20}),
    ]
    ground_truth = {
        "parallel_0": [
            {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
            {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
        ]
    }
    score = score_task("parallel", "parallel_0", tool_calls, ground_truth)
    assert score["name_correct"]
    assert score["full_correct"]


def test_score_task_call_count_mismatch_fails() -> None:
    tool_calls = [openai_call("spotify.play", {"artist": "Maroon 5", "duration": 15})]
    ground_truth = {
        "parallel_0": [
            {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
            {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
        ]
    }
    score = score_task("parallel", "parallel_0", tool_calls, ground_truth)
    assert not score["name_correct"]
    assert not score["full_correct"]


def test_score_task_irrelevance_correct_when_no_call() -> None:
    score = score_task("irrelevance", "irrelevance_0", None, {})
    assert score["no_tool_correct"]


def test_score_task_irrelevance_incorrect_when_called() -> None:
    tool_calls = [openai_call("f", {})]
    score = score_task("irrelevance", "irrelevance_0", tool_calls, {})
    assert not score["no_tool_correct"]


def test_score_task_unknown_task_id_returns_none() -> None:
    assert score_task("multiple", "multiple_999", None, {}) is None


def test_aggregate_mixes_relevance_and_irrelevance_independently() -> None:
    scores = [
        {"name_correct": True, "full_correct": True},
        {"name_correct": True, "full_correct": False},
        {"no_tool_correct": True},
        {"no_tool_correct": False},
    ]
    result = aggregate(scores)
    assert result["function_name_accuracy"] == 1.0
    assert result["full_accuracy"] == 0.5
    assert result["no_tool_accuracy"] == 0.5
