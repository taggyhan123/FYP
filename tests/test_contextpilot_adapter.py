from types import SimpleNamespace

import pytest

from tatm.contextpilot_adapter import (
    build_online_incremental_orderings,
    build_static_refit_causal_orderings,
    materialize_contextpilot_causal_workload,
    materialize_contextpilot_workload,
)


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


def test_static_refit_is_causal_and_restores_upstream_integer_ids() -> None:
    observed_histories: list[list[list[str]]] = []

    class FakeIndex:
        def fit_transform(self, contexts: list[list[str]]) -> SimpleNamespace:
            observed_histories.append([list(context) for context in contexts])
            vocabulary = {
                tool_id: index
                for index, tool_id in enumerate(
                    dict.fromkeys(tool for context in contexts for tool in context)
                )
            }
            converted = [
                [vocabulary[tool_id] for tool_id in context] for context in contexts
            ]
            return SimpleNamespace(
                # This is the pinned ContextPilot contract: string inputs are
                # converted internally and fit_transform returns integer IDs.
                original_contexts=converted,
                reordered_contexts=[list(reversed(context)) for context in converted],
                search_paths=[[index] for index in range(len(contexts))],
            )

    ticks = iter((1.0, 1.1, 2.0, 2.2))
    records = [record("first", ["z", "a"]), record("second", ["q", "b"])]
    orderings, plans = build_static_refit_causal_orderings(
        records,
        FakeIndex,
        clock=lambda: next(ticks),
    )

    assert observed_histories == [
        [["z", "a"]],
        [["z", "a"], ["q", "b"]],
    ]
    assert orderings == [["a", "z"], ["b", "q"]]
    assert plans[0]["history_requests"] == 1
    assert plans[1]["search_path"] == [1]


def test_online_incremental_uses_persistent_api_and_isolated_conversations() -> None:
    class FakeOnlinePlanner:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], str]] = []

        def reorder(
            self, context: list[str], *, conversation_id: str
        ) -> tuple[list[list[str]], list[int]]:
            self.calls.append((list(context), conversation_id))
            return [list(reversed(context))], [0]

    planner = FakeOnlinePlanner()
    ticks = iter((1.0, 1.1, 2.0, 2.3))
    records = [record("first", ["a", "b"]), record("second", ["b", "c"])]
    orderings, plans = build_online_incremental_orderings(
        records,
        planner,
        conversation_prefix="audit",
        clock=lambda: next(ticks),
    )

    assert orderings == [["b", "a"], ["c", "b"]]
    assert planner.calls == [
        (["a", "b"], "audit:0:first"),
        (["b", "c"], "audit:1:second"),
    ]
    assert plans[1]["history_requests"] == 2


def test_causal_materialization_records_exact_semantics() -> None:
    records = [record("first", ["a", "b"])]
    output = materialize_contextpilot_causal_workload(
        records,
        [["b", "a"]],
        [{"planning_seconds": 0.1}],
        mode="online_incremental",
        provenance={
            "alpha": 0.001,
            "annotations_enabled": False,
            "eviction_feedback_enabled": False,
        },
    )

    assert output[0]["ordering"] == "contextpilot_online_incremental"
    plan = output[0]["contextpilot_plan"]
    assert plan["information_regime"] == "causal"
    assert plan["official_online_api_used"] is True
    assert plan["reference"]["alpha"] == 0.001
