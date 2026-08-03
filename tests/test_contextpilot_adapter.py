import pytest

from tatm.contextpilot_adapter import materialize_contextpilot_workload


def record(task_id: str, tool_ids: list[str]) -> dict:
    return {
        "task_id": task_id,
        "ordering": "original",
        "tool_ids": tool_ids,
        "tools": [{"function": {"name": item}} for item in tool_ids],
    }


def test_contextpilot_materialization_preserves_alignment_and_mapping() -> None:
    records = [record("first", ["a", "b"]), record("second", ["b", "c"])]
    output = materialize_contextpilot_workload(
        records,
        [["b", "c"], ["b", "a"]],
        [1, 0],
        [[0, 1], [1, 0]],
        mode="intra_schedule",
        provenance={"commit": "abc"},
    )
    assert [item["task_id"] for item in output] == ["second", "first"]
    assert output[0]["tool_ids"] == ["b", "c"]
    assert [item["function"]["name"] for item in output[1]["tools"]] == ["b", "a"]
    assert output[0]["contextpilot_plan"]["original_request_index"] == 1
    assert output[0]["contextpilot_plan"]["search_path"] == [1, 0]
    assert output[0]["contextpilot_plan"]["scheduler_enabled"] is True
    assert output[0]["contextpilot_plan"]["request_order_changed"] is True


def test_contextpilot_rejects_changed_tool_set() -> None:
    records = [record("first", ["a", "b"])]
    with pytest.raises(ValueError, match="changed the selected tool set"):
        materialize_contextpilot_workload(
            records,
            [["a", "c"]],
            [0],
            [[]],
            mode="intra",
            provenance={},
        )


def test_contextpilot_intra_mode_rejects_request_rescheduling() -> None:
    records = [record("first", ["a"]), record("second", ["b"])]
    with pytest.raises(ValueError, match="intra mode must preserve request order"):
        materialize_contextpilot_workload(
            records,
            [["b"], ["a"]],
            [1, 0],
            [[], []],
            mode="intra",
            provenance={},
        )
