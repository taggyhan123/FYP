"""Simplified BFCL-style AST scoring for OpenAI-format tool calls.

This checks predicted `tool_calls` against BFCL's `possible_answer` ground
truth: function name plus, for each argument, whether the produced value is
one of the acceptable values BFCL lists (an empty string `""` in that list
means "acceptable to omit"). It is a reduced reimplementation of the official
Gorilla/BFCL checker (github.com/ShishirPatil/gorilla), not a vendored copy:
it does not execute code, does not special-case every BFCL type coercion rule,
and treats any predicted argument absent from ground truth as a mismatch.
Numbers are compared with float tolerance and bool/int are kept distinct.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

from tatm.io import read_jsonl

GroundTruthCall = dict[str, dict[str, list[Any]]]

# Configs with a possible_answer file; "irrelevance" has none by design.
SCORABLE_CONFIGS = ("simple_python", "multiple", "parallel", "parallel_multiple")
NO_TOOL_CONFIG = "irrelevance"


def load_ground_truth(raw_dir: Path) -> dict[str, list[GroundTruthCall]]:
    ground_truth: dict[str, list[GroundTruthCall]] = {}
    for config in SCORABLE_CONFIGS:
        path = raw_dir / "bfcl" / "possible_answer" / f"BFCL_v4_{config}.json"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            ground_truth[str(row["id"])] = row["ground_truth"]
    return ground_truth


def parse_predicted_calls(
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    calls: list[dict[str, Any]] = []
    for call in tool_calls:
        function = call.get("function", {})
        name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else dict(raw_arguments)
            )
        except json.JSONDecodeError:
            arguments = None
        calls.append({"name": name, "arguments": arguments, "parse_ok": arguments is not None})
    return calls


def _numbers_close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6 * max(1.0, abs(left), abs(right))


def _value_matches(actual: Any, acceptable: Any) -> bool:
    if isinstance(acceptable, bool) or isinstance(actual, bool):
        return actual is acceptable
    if isinstance(acceptable, (int, float)) and isinstance(actual, (int, float)):
        return _numbers_close(float(actual), float(acceptable))
    if isinstance(acceptable, list) and isinstance(actual, list):
        return len(actual) == len(acceptable) and all(
            _value_matches(a, b) for a, b in zip(actual, acceptable)
        )
    if isinstance(acceptable, str) and isinstance(actual, str):
        return actual.strip().casefold() == acceptable.strip().casefold()
    return actual == acceptable


def _argument_correct(actual_value: Any, acceptable_values: list[Any], provided: bool) -> bool:
    if not provided:
        return any(value == "" for value in acceptable_values)
    return any(
        value != "" and _value_matches(actual_value, value) for value in acceptable_values
    )


@dataclass
class CallScore:
    name_correct: bool
    arguments_correct: bool
    parse_ok: bool


def score_single_call(
    predicted: Mapping[str, Any], gold: GroundTruthCall
) -> CallScore:
    (gold_name, gold_args), = gold.items()
    predicted_name = predicted.get("name")
    name_correct = predicted_name == gold_name
    parse_ok = bool(predicted.get("parse_ok"))
    if not name_correct or not parse_ok:
        return CallScore(name_correct=name_correct, arguments_correct=False, parse_ok=parse_ok)

    actual_args = predicted.get("arguments") or {}
    all_correct = True
    for arg_name, acceptable_values in gold_args.items():
        provided = arg_name in actual_args
        if not _argument_correct(actual_args.get(arg_name), acceptable_values, provided):
            all_correct = False
            break
    if all_correct:
        extra_keys = set(actual_args) - set(gold_args)
        all_correct = not extra_keys
    return CallScore(name_correct=True, arguments_correct=all_correct, parse_ok=True)


def _best_matching(
    predicted_calls: list[dict[str, Any]], gold_calls: list[GroundTruthCall]
) -> tuple[bool, bool]:
    """Try every pairing (call counts are small: parallel tasks have <=3) and
    keep the assignment that maximizes correct calls, mirroring BFCL's
    order-independent matching for parallel/multiple function calls."""
    if len(predicted_calls) != len(gold_calls):
        return False, False

    best_names = 0
    best_full = 0
    for perm in permutations(range(len(gold_calls))):
        names = 0
        full = 0
        for predicted_index, gold_index in enumerate(perm):
            result = score_single_call(predicted_calls[predicted_index], gold_calls[gold_index])
            names += int(result.name_correct)
            full += int(result.name_correct and result.arguments_correct)
        best_names = max(best_names, names)
        best_full = max(best_full, full)

    count = len(gold_calls)
    return best_names == count, best_full == count


def score_task(
    domain: str,
    task_id: str,
    predicted_tool_calls: Sequence[Mapping[str, Any]] | None,
    ground_truth: dict[str, list[GroundTruthCall]],
) -> dict[str, Any] | None:
    predicted_calls = parse_predicted_calls(predicted_tool_calls)

    if domain == NO_TOOL_CONFIG:
        return {
            "task_id": task_id,
            "domain": domain,
            "scorable": True,
            "no_tool_correct": len(predicted_calls) == 0,
        }

    gold_calls = ground_truth.get(task_id)
    if gold_calls is None:
        return None

    name_correct, full_correct = _best_matching(predicted_calls, gold_calls)
    return {
        "task_id": task_id,
        "domain": domain,
        "scorable": True,
        "call_count_correct": len(predicted_calls) == len(gold_calls),
        "name_correct": name_correct,
        "full_correct": full_correct,
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    relevance_scores = [s for s in scores if "name_correct" in s]
    irrelevance_scores = [s for s in scores if "no_tool_correct" in s]

    def rate(items: list[dict[str, Any]], key: str) -> float | None:
        if not items:
            return None
        return round(sum(1 for item in items if item[key]) / len(items), 4)

    return {
        "tasks_scored": len(scores),
        "relevance_tasks": len(relevance_scores),
        "irrelevance_tasks": len(irrelevance_scores),
        "function_name_accuracy": rate(relevance_scores, "name_correct"),
        "full_accuracy": rate(relevance_scores, "full_correct"),
        "no_tool_accuracy": rate(irrelevance_scores, "no_tool_correct"),
    }
