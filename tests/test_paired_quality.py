import pytest

from tatm.paired_quality import compare_paired_binary, metric_scoped_scores


def row(case: str, task: str, correct: bool) -> dict:
    return {
        "case_id": case,
        "task_id": task,
        "no_tool_correct": correct,
    }


def test_paired_difference_and_discordant_counts() -> None:
    baseline = [
        row("a:1", "a", True),
        row("b:1", "b", True),
        row("c:1", "c", False),
        row("d:1", "d", False),
    ]
    candidate = [
        row("a:1", "a", True),
        row("b:1", "b", False),
        row("c:1", "c", True),
        row("d:1", "d", True),
    ]
    result = compare_paired_binary(
        baseline,
        candidate,
        metric="no_tool_correct",
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert result["baseline_accuracy"] == 0.5
    assert result["candidate_accuracy"] == 0.75
    assert result["difference_percentage_points"] == 25.0
    assert result["discordant_pairs"] == {
        "baseline_only_correct": 1,
        "candidate_only_correct": 2,
    }


def test_menu_repetitions_are_clustered_by_task() -> None:
    baseline = [row("a:1", "a", True), row("a:2", "a", True)]
    candidate = [row("a:1", "a", False), row("a:2", "a", False)]
    result = compare_paired_binary(
        baseline,
        candidate,
        metric="no_tool_correct",
        bootstrap_samples=20,
    )
    assert result["paired_cases"] == 2
    assert result["task_clusters"] == 1
    assert result["cluster_bootstrap_95_ci_percentage_points"] == [-100.0, -100.0]
    assert result["mcnemar_independence_assumption_met"] is False


def test_mismatched_cases_are_rejected() -> None:
    with pytest.raises(ValueError, match="Paired case IDs differ"):
        compare_paired_binary(
            [row("a", "a", True)],
            [row("b", "b", True)],
            metric="no_tool_correct",
            bootstrap_samples=10,
        )


def test_equivalence_requires_entire_interval_inside_margin() -> None:
    rows = [row(f"{i}", f"{i}", True) for i in range(20)]
    result = compare_paired_binary(
        rows,
        rows,
        metric="no_tool_correct",
        bootstrap_samples=50,
        equivalence_margin_pp=1.0,
    )
    assert result["equivalent_within_margin"] is True


def test_metric_scope_filters_undefined_mixed_domain_rows() -> None:
    rows = [
        {"case_id": "relevance", "name_correct": True},
        {"case_id": "irrelevance", "name_correct": None, "no_tool_correct": True},
    ]
    assert metric_scoped_scores(rows, "name_correct") == [rows[0]]
    assert metric_scoped_scores(rows, "no_tool_correct") == [rows[1]]


def test_stateful_sequence_is_marked_descriptive() -> None:
    rows = [row(str(index), str(index), True) for index in range(4)]
    result = compare_paired_binary(
        rows,
        rows,
        metric="no_tool_correct",
        bootstrap_samples=20,
        sequence_state_dependent=True,
    )
    assert result["cluster_bootstrap_generalizes_across_request_sequences"] is False
    assert result["mcnemar_independence_assumption_met"] is False
    assert result["inference_scope"] == (
        "descriptive for the fixed emitted planner sequence"
    )
