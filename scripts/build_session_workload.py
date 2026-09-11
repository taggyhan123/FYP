#!/usr/bin/env python
"""Build multi-turn session workloads from real agent trajectories.

Conversations are real (nebius/SWE-agent-trajectories); tool-set change rates,
positions and tool waits are real (TraceLab); which catalogue tools a session
gets is synthetic (dense retrieval over ToolRet), because real coding agents
barely use large tool catalogues. Stages:

  sample    (.venv, needs pyarrow)   parquet -> sessions.jsonl
  tracelab  (any python)             TraceLab trace -> tracelab_params.json
  events    (tatm env, CPU only)     retrieval, truncation, planning events

`events` writes, per (dynamics, k): a tool timeline for every session, and per
(dynamics, k, placement, concurrency): the planning events in serving order as
single-turn-style records that the existing ordering builders accept unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.session_workload import (  # noqa: E402
    Session,
    interleave_sessions,
    sample_change_rounds,
    trajectory_to_session,
)

MAX_MODEL_LEN = 32768
SAFETY_TOKENS = 256
CONCURRENCY = (1, 8, 32)
QUERY_CHARS = 1024


def stage_sample(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet, columns=["instance_id", "model_name", "trajectory"])
    rows = table.to_pylist()
    # The dataset holds many attempts per issue (~22 in shard 0). Attempts at
    # the same issue share their prompt up to the first agent action, so
    # sampling rows directly manufactures cross-session reuse that ordinary
    # traffic would not have. Keep one random attempt per issue.
    rng = random.Random(args.seed)
    by_issue: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_issue.setdefault(row["instance_id"], []).append(index)
    one_each = [rng.choice(indices) for _, indices in sorted(by_issue.items())]
    if args.sessions > len(one_each):
        raise SystemExit(f"only {len(one_each)} distinct issues available")
    picked = sorted(rng.sample(one_each, args.sessions))
    out = args.out_dir / "sessions.jsonl"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    with out.open("w") as handle:
        for index in picked:
            row = rows[index]
            session = trajectory_to_session(f"{row['instance_id']}@{index}", row["trajectory"])
            handle.write(json.dumps({
                "session_id": session.session_id, "source_row": index,
                "model_name": row["model_name"], "system_prompt": session.system_prompt,
                "messages": list(session.messages),
            }) + "\n")
    print(f"{out}: {len(picked)} sessions, one per issue (seed {args.seed}), "
          f"from {len(one_each)} distinct issues in {len(rows)} trajectories")


def stage_tracelab(args: argparse.Namespace) -> None:
    rounds: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    first: dict[str, int] = {}
    waits: list[float] = []
    with gzip.open(args.trace, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            for tool in row.get("tools") or []:
                latency = tool.get("tool_wall_latency_ms")
                if latency is not None and latency >= 0:
                    waits.append(min(latency / 1000.0, 3600.0))
            if row["provider"] != "claude":
                continue
            session = row["session_id"]
            rounds[session] += 1
            for tool in row.get("tools") or []:
                if tool["tool_name"] == "ToolSearch":
                    calls[session] += 1
                    first.setdefault(session, row["round_index"])
    params = {
        "source": "uw-syfi/TraceLab v0.0.1 syfi_coding_trace.jsonl.gz (CC BY 4.0)",
        "claude_sessions": len(rounds),
        "session_probability": len(calls) / len(rounds),
        "count_samples": sorted(calls.values()),
        "first_position_samples": sorted(
            first[s] / max(rounds[s] - 1, 1) for s in calls),
        "wait_samples_seconds": sorted(random.Random(args.seed).sample(waits, min(20000, len(waits)))),
        "tool_records_with_latency": len(waits),
    }
    out = args.out_dir / "tracelab_params.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(params))
    print(f"{out}: p(session has ToolSearch)={params['session_probability']:.3f}, "
          f"{len(calls)} sessions, {len(waits)} tool waits")


def stage_events(args: argparse.Namespace) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from transformers import AutoTokenizer

    from tatm.analysis import deduplicated_existing_ids, load_processed
    from tatm.dense_retrieval import DenseToolRetriever
    from tatm.prompting import openai_tool
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from analyze_prefix_cache import vllm_tool

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    enc_len = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
    sessions = [Session(r["session_id"], r["system_prompt"], tuple(r["messages"]))
                for r in map(json.loads, (args.out_dir / "sessions.jsonl").read_text().splitlines())]
    params = json.loads((args.out_dir / "tracelab_params.json").read_text())
    tools, tasks = load_processed(args.processed_dir)
    corpus = {i: t for i, t in tools.items() if t.source.startswith("toolret:")}
    support: Counter[str] = Counter()
    for task in tasks:
        if task.evidence_type == "gold_relevance":
            support.update(deduplicated_existing_ids(task, corpus))
    retriever = DenseToolRetriever(corpus, device="cpu")
    tool_len = {}

    def piece_len(tool_id: str) -> int:
        if tool_id not in tool_len:
            tool_len[tool_id] = enc_len("\n" + json.dumps(vllm_tool(openai_tool(corpus[tool_id])), ensure_ascii=False))
        return tool_len[tool_id]

    def retrieve(query: str, k: int) -> list[str]:
        return list(retriever.retrieve(query[:QUERY_CHARS] or "tool", k=k).tool_ids)

    # Template overhead of the tools block, measured once.
    probe = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    one = [vllm_tool(openai_tool(corpus[next(iter(corpus))]))]
    with_tool = enc_len(tok.apply_chat_template(probe, tools=one, add_generation_prompt=True, tokenize=False, enable_thinking=False))
    without = enc_len(tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False, enable_thinking=False))
    tools_overhead = with_tool - without - piece_len(next(iter(corpus)))
    appended_overhead = 40  # wrapper text and message markers of an appended-tools message, generous

    k_values = (16, 64)
    timelines: dict[tuple[str, int], dict[str, list]] = {}
    session_meta = []
    for session in sessions:
        starts = session.round_starts
        n_rounds = len(starts)
        rng = random.Random(f"{args.seed}:{session.session_id}")
        change_rounds = sample_change_rounds(
            n_rounds, rng, session_probability=params["session_probability"],
            count_samples=params["count_samples"],
            first_position_samples=params["first_position_samples"])
        queries = [session.issue_text or session.messages[0]["content"]]
        for r in range(1, n_rounds):
            last_action = session.messages[starts[r - 1]]["content"]
            observation = session.messages[starts[r] - 1]["content"] if starts[r] - 1 > starts[r - 1] else ""
            queries.append((last_action + "\n" + observation[:512]).strip())
        history = []
        for r in range(n_rounds):
            msgs = [{"role": "system", "content": session.system_prompt}, *session.messages[:starts[r]]]
            history.append(enc_len(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)))
        completion = [enc_len(session.messages[s]["content"]) for s in starts]
        per_variant_len: list[list[int]] = []
        for k in k_values:
            start_set = retrieve(queries[0], k)
            # D1: fixed set.
            d1 = [[start_set, []] for _ in range(n_rounds)]
            # D2: ToolSearch-like additions of m tools at sampled rounds.
            d2_front, d2_append, loaded, appended = [], [], list(start_set), []
            for r in range(n_rounds):
                if r in change_rounds:
                    new = [t for t in retrieve(queries[r], k + args.change_tools + 64) if t not in loaded][:args.change_tools]
                    loaded += new
                    appended.append([r, new])
                d2_front.append([list(loaded), []])
                d2_append.append([start_set, [list(a) for a in appended]])
            timelines.setdefault(("D1", k), {})[session.session_id] = d1
            timelines.setdefault(("D2-front", k), {})[session.session_id] = d2_front
            timelines.setdefault(("D2-append", k), {})[session.session_id] = d2_append
            if k == 64:
                # D3: re-retrieve the top-k every round.
                d3_front, d3_append, loaded, appended = [], [], list(start_set), []
                for r in range(n_rounds):
                    current = start_set if r == 0 else retrieve(queries[r], k)
                    d3_front.append([current, []])
                    new = [t for t in current if t not in loaded]
                    if new and len(loaded) < 2 * k:
                        new = new[: 2 * k - len(loaded)]
                        loaded += new
                        appended.append([r, new])
                    d3_append.append([start_set, [list(a) for a in appended]])
                timelines.setdefault(("D3-front", k), {})[session.session_id] = d3_front
                timelines.setdefault(("D3-append", k), {})[session.session_id] = d3_append
            for name in ("D1", "D2-front", "D2-append") + (("D3-front", "D3-append") if k == 64 else ()):
                rounds_tools = timelines[(name, k)][session.session_id]
                per_variant_len.append([
                    history[r] + tools_overhead + sum(piece_len(t) for t in rounds_tools[r][0])
                    + sum(appended_overhead + sum(piece_len(t) for t in ids) for _, ids in rounds_tools[r][1])
                    for r in range(n_rounds)])
        cap = 0
        while cap < n_rounds and max(v[cap] for v in per_variant_len) <= MAX_MODEL_LEN - SAFETY_TOKENS:
            cap += 1
        session_meta.append({"session_id": session.session_id, "rounds_available": n_rounds,
                             "rounds": cap, "round_starts": starts[:cap], "completion_tokens": completion[:cap],
                             "change_rounds": [c for c in change_rounds if c < cap]})

    meta_out = args.out_dir / "session_meta.jsonl"
    meta_out.write_text("".join(json.dumps(m) + "\n" for m in session_meta))
    caps = {m["session_id"]: m["rounds"] for m in session_meta}
    (args.out_dir / "timelines").mkdir(exist_ok=False)
    for (name, k), per_session in timelines.items():
        trimmed = {sid: rounds[:caps[sid]] for sid, rounds in per_session.items()}
        (args.out_dir / "timelines" / f"{name}-k{k}.json").write_text(json.dumps(trimmed))
    (args.out_dir / "support.json").write_text(json.dumps(dict(support)))

    # Planning events in serving order, one file per (timeline, placement rule, concurrency).
    (args.out_dir / "events").mkdir(exist_ok=False)
    ids = [m["session_id"] for m in session_meta]
    rounds = [m["rounds"] for m in session_meta]
    for concurrency in CONCURRENCY:
        order = interleave_sessions(rounds, concurrency, random.Random(args.seed),
                                    wait_samples=params["wait_samples_seconds"])
        (args.out_dir / "events" / f"order-C{concurrency}.json").write_text(json.dumps(order))
        for (name, k), per_session in timelines.items():
            modes = ["onchange"] + (["everyround"] if name == "D1" and k == 64 else [])
            for mode in modes:
                records, previous = [], {}
                for s, r in order:
                    sid = ids[s]
                    front = per_session[sid][r][0]
                    if mode == "everyround" or previous.get(sid) != front:
                        records.append({"task_id": f"{sid}#{r}", "ordering": "original", "tool_ids": list(front),
                                        "tools": [openai_tool(corpus[t]) for t in front]})
                        previous[sid] = front
                path = args.out_dir / "events" / f"{name}-k{k}-{mode}-C{concurrency}.jsonl"
                path.write_text("".join(json.dumps(x) + "\n" for x in records))
    kept = sum(caps.values()); available = sum(m["rounds_available"] for m in session_meta)
    truncated = sum(1 for m in session_meta if m["rounds"] < m["rounds_available"])
    summary = {"sessions": len(session_meta), "rounds_kept": kept, "rounds_available": available,
               "sessions_truncated": truncated, "tools_overhead_tokens": tools_overhead,
               "sessions_with_changes": sum(1 for m in session_meta if m["change_rounds"]),
               "change_tools": args.change_tools}
    (args.out_dir / "events_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("sample", "tracelab", "events"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sessions", type=int, default=300)
    parser.add_argument("--change-tools", type=int, default=2, help="tools added per ToolSearch-like event")
    parser.add_argument("--parquet", type=Path,
                        default=PROJECT_ROOT / "data/raw/swe-agent-trajectories-sample/data/train-00000-of-00012.parquet")
    parser.add_argument("--trace", type=Path, default=PROJECT_ROOT / "data/raw/tracelab/trace/syfi_coding_trace.jsonl.gz")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"sample": stage_sample, "tracelab": stage_tracelab, "events": stage_events}[args.stage](args)


if __name__ == "__main__":
    main()
