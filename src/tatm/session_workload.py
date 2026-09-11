"""Multi-turn session workloads built from recorded agent trajectories.

A trajectory is a recorded agent run: a system prompt, then alternating
observations (user messages) and agent actions (assistant messages). Each
assistant message is one model call ("round"); its prompt is the system prompt
plus every message before it. Everything here is pure: parsing, the sampler for
mid-session tool-set changes, and the closed-loop scheduler that interleaves
concurrent sessions into one serving order.
"""
from __future__ import annotations

import heapq
import random
from collections.abc import Sequence
from dataclasses import dataclass

ROLE_MAP = {"user": "user", "ai": "assistant", "assistant": "assistant"}


@dataclass(frozen=True)
class Session:
    session_id: str
    system_prompt: str
    messages: tuple[dict, ...]  # user/assistant chat messages after the system prompt

    @property
    def round_starts(self) -> list[int]:
        """Index of each assistant message: messages[:i] is that round's history."""
        return [i for i, message in enumerate(self.messages) if message["role"] == "assistant"]

    @property
    def issue_text(self) -> str:
        """The task statement: the first user message, trimmed to the issue itself."""
        first = next((m["content"] for m in self.messages if m["role"] == "user"), "")
        start = first.find("ISSUE:")
        end = first.find("INSTRUCTIONS:")
        if start >= 0:
            first = first[start + len("ISSUE:"):end if end > start else None]
        return first.strip()


def trajectory_to_session(session_id: str, trajectory: Sequence[dict]) -> Session:
    """Convert a SWE-agent trajectory (roles system/user/ai) into a Session."""
    system_prompt = ""
    messages: list[dict] = []
    for message in trajectory:
        role = message.get("role")
        text = message.get("text") or ""
        if role == "system":
            system_prompt = message.get("system_prompt") or text or system_prompt
            continue
        if not system_prompt and message.get("system_prompt"):
            system_prompt = message["system_prompt"]
        if role not in ROLE_MAP:
            raise ValueError(f"{session_id}: unknown role {role!r}")
        messages.append({"role": ROLE_MAP[role], "content": text})
    if not system_prompt:
        raise ValueError(f"{session_id}: no system prompt")
    return Session(session_id, system_prompt, tuple(messages))


def sample_change_rounds(
    n_rounds: int,
    rng: random.Random,
    *,
    session_probability: float,
    count_samples: Sequence[int],
    first_position_samples: Sequence[float],
) -> list[int]:
    """Rounds at which a session's tool set changes, drawn from observed rates.

    Round 0 is the session start and is never a change. Whether a session has
    any change, how many, and how far into the session the first one falls are
    each drawn from empirical samples (for example TraceLab's `ToolSearch`
    calls); later changes fall uniformly after the first.
    """
    if n_rounds < 2 or rng.random() >= session_probability:
        return []
    count = max(1, int(rng.choice(count_samples)))
    first = max(1, min(n_rounds - 1, round(rng.choice(first_position_samples) * (n_rounds - 1))))
    later_pool = list(range(first + 1, n_rounds))
    later = sorted(rng.sample(later_pool, min(count - 1, len(later_pool))))
    return [first, *later]


def interleave_sessions(
    rounds_per_session: Sequence[int],
    concurrency: int,
    rng: random.Random,
    *,
    wait_samples: Sequence[float],
    model_seconds: float = 0.5,
) -> list[tuple[int, int]]:
    """Serving order of (session index, round index) for a closed-loop system.

    `concurrency` sessions are active at once; sessions start in index order as
    slots free. After a round is served, that session's next round becomes ready
    `model_seconds` plus a sampled tool wait later. Rounds are served in ready
    time order, one at a time.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    order: list[tuple[int, int]] = []
    ready: list[tuple[float, int, int, int]] = []  # (time, tiebreak, session, round)
    tiebreak = 0
    next_session = 0
    for _ in range(min(concurrency, len(rounds_per_session))):
        if rounds_per_session[next_session] > 0:
            heapq.heappush(ready, (0.0, tiebreak, next_session, 0))
            tiebreak += 1
        next_session += 1
    while ready:
        now, _, session, round_index = heapq.heappop(ready)
        order.append((session, round_index))
        done = now + model_seconds
        if round_index + 1 < rounds_per_session[session]:
            heapq.heappush(ready, (done + rng.choice(wait_samples), tiebreak, session, round_index + 1))
            tiebreak += 1
        else:
            while next_session < len(rounds_per_session) and rounds_per_session[next_session] == 0:
                next_session += 1
            if next_session < len(rounds_per_session):
                heapq.heappush(ready, (done, tiebreak, next_session, 0))
                tiebreak += 1
                next_session += 1
    return order
