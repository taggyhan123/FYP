from __future__ import annotations

import random

import pytest

from tatm.session_workload import (
    interleave_sessions,
    sample_change_rounds,
    trajectory_to_session,
)


def _trajectory() -> list[dict]:
    return [
        {"role": "system", "system_prompt": "SYS", "text": ""},
        {"role": "user", "system_prompt": "SYS", "text": "Solve.\nISSUE:\nIt breaks.\nINSTRUCTIONS:\nGo."},
        {"role": "ai", "system_prompt": "SYS", "text": "ls"},
        {"role": "user", "system_prompt": "SYS", "text": "a b"},
        {"role": "ai", "system_prompt": "SYS", "text": "submit"},
    ]


def test_trajectory_becomes_rounds_at_each_agent_message() -> None:
    session = trajectory_to_session("s", _trajectory())
    assert session.system_prompt == "SYS"
    assert [m["role"] for m in session.messages] == ["user", "assistant", "user", "assistant"]
    assert session.round_starts == [1, 3]
    assert session.issue_text == "It breaks."


def test_unknown_roles_are_rejected() -> None:
    with pytest.raises(ValueError):
        trajectory_to_session("s", [{"role": "system", "system_prompt": "S"}, {"role": "tool", "text": "x"}])


def test_change_rounds_never_touch_the_session_start() -> None:
    rng = random.Random(0)
    for _ in range(200):
        rounds = sample_change_rounds(10, rng, session_probability=1.0,
                                      count_samples=[1, 3], first_position_samples=[0.0, 0.5])
        assert rounds and rounds[0] >= 1 and rounds == sorted(set(rounds)) and rounds[-1] <= 9


def test_change_rounds_respect_the_session_probability() -> None:
    rng = random.Random(1)
    hits = sum(bool(sample_change_rounds(20, rng, session_probability=0.1, count_samples=[1],
                                         first_position_samples=[0.1])) for _ in range(5000))
    assert 400 < hits < 600
    assert sample_change_rounds(1, rng, session_probability=1.0, count_samples=[1],
                                first_position_samples=[0.5]) == []


def test_interleaving_keeps_each_session_in_order_and_serves_everything() -> None:
    rounds = [3, 0, 2, 4]
    order = interleave_sessions(rounds, 2, random.Random(3), wait_samples=[0.1, 2.0])
    assert sorted(order) == sorted((s, r) for s, n in enumerate(rounds) for r in range(n))
    for session in range(len(rounds)):
        served = [r for s, r in order if s == session]
        assert served == sorted(served)


def test_concurrency_one_serves_sessions_back_to_back() -> None:
    order = interleave_sessions([2, 2], 1, random.Random(0), wait_samples=[5.0])
    assert order == [(0, 0), (0, 1), (1, 0), (1, 1)]
