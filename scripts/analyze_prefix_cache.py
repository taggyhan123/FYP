#!/usr/bin/env python
"""Absolute prefix-cache hit and miss accounting for existing replays.

CPU only; needs `transformers` and the Qwen3 tokenizer (the `tatm` conda env).
For each one-at-a-time replay it:

1. re-renders every prompt exactly as vLLM 0.26 does. vLLM normalises each tool
   to {"type", "function": {"name", "description", "parameters"}} key order
   before applying the chat template; the rendered token counts must equal the
   served `prompt_tokens`, or the run is rejected;
2. simulates the prefix cache (`tatm.prefix_cache_sim`) with the server's real
   capacity and with unlimited capacity, and checks the simulation against the
   measured per-request cached tokens;
3. splits the hits into the fixed chat-template header (identical in every
   prompt) and tool reuse, and bounds what any ordering could have reached:
   A = best reorder of each request alone, given what earlier requests left in
       the cache;
   B = best if the earlier request had also been reordered for it (pairwise,
       ignoring that one order must serve many requests: an optimistic bound).
   Both are computed over the requests resident in the real cache ("win") and
   over every earlier request ("inf"). Candidates are ranked by an additive
   token estimate and the top three are re-tokenised exactly.

Output: one JSON line per run in <out-dir>/decomposition.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.prefix_cache_sim import simulate_prefix_cache  # noqa: E402
from tatm.prefix_evidence import common_prefix_length  # noqa: E402

BLOCK = 16
MARKER = "XML tags:\n<tools>"
_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _TOKENIZER


def encode(text: str) -> list[int]:
    return tokenizer()(text, add_special_tokens=False)["input_ids"]


def vllm_tool(tool: dict) -> dict:
    function = tool["function"]
    out = {"name": function["name"]}
    for key in ("description", "parameters"):
        if key in function:
            out[key] = function[key]
    return {"type": "function", "function": out}


def render_text(record: dict) -> str:
    return tokenizer().apply_chat_template(
        record["messages"], tools=[vllm_tool(t) for t in record["tools"]],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)


def analyse(spec: tuple[str, str]) -> dict:
    replay_path, workload_path = spec
    replay = json.loads(Path(replay_path).read_text())
    records = [json.loads(line) for line in Path(workload_path).read_text().splitlines() if line]
    results = sorted(replay["results"], key=lambda r: r["index"])
    if any(r["tool_ids"] != records[i]["tool_ids"] for i, r in enumerate(results)):
        raise SystemExit(f"{replay_path}: served order differs from workload")
    texts = [render_text(r) for r in records]
    prompts = [encode(t) for t in texts]
    if any(len(p) != r["usage"]["prompt_tokens"] for p, r in zip(prompts, results)):
        raise SystemExit(f"{replay_path}: rendered prompts differ from served prompts")

    head = texts[0].split(MARKER, 1)[0] + MARKER
    tool_text: dict[str, str] = {}
    tails: list[str] = []
    for record, text in zip(records, texts):
        pieces = ["\n" + json.dumps(vllm_tool(t), ensure_ascii=False) for t in record["tools"]]
        rest = text.split(MARKER, 1)[1]
        body = "".join(pieces)
        if not rest.startswith(body):
            raise SystemExit(f"{replay_path}: template layout differs")
        tails.append(rest[len(body):])
        tool_text.update(zip(record["tool_ids"], pieces))
    approx_len = {tool_id: len(encode(text)) for tool_id, text in tool_text.items()}

    def build(order: list[str], index: int) -> list[int]:
        return encode(head + "".join(tool_text[t] for t in order) + tails[index])

    completions = [r["usage"]["completion_tokens"] for r in results]
    measured = [int(r["measurement"]["cached_tokens"]) for r in results]
    capacity = int(replay["server_cache_config"]["num_gpu_blocks"]) - 1
    finite = simulate_prefix_cache(prompts, completions, capacity, BLOCK)
    unlimited = simulate_prefix_cache(prompts, completions, None, BLOCK)

    def floor(tokens: int, length: int) -> int:
        return min((tokens // BLOCK) * BLOCK, ((length - 1) // BLOCK) * BLOCK)

    sequences = [r["tool_ids"] for r in records]
    sets = [set(s) for s in sequences]
    # The fixed header is what every prompt shares, not just the first two:
    # two requests that start with the same tool share more than the header.
    header = floor(min(common_prefix_length(prompts[0], p) for p in prompts[1:]), 10**9)
    totals = dict.fromkeys(["A_win", "A_inf", "B_win", "B_inf"], 0)
    for i in range(1, len(records)):
        length = len(prompts[i])
        base = floor(header, length)
        window, used = [], 0
        for j in range(i - 1, -1, -1):
            used += len(prompts[j]) + completions[j]
            if used > capacity * BLOCK:
                break
            window.append(j)

        def matched_prefix(j: int) -> list[str]:
            prefix = []
            for tool_id in sequences[j]:
                if tool_id not in sets[i]:
                    break
                prefix.append(tool_id)
            return prefix

        def exact_a(j: int) -> int:
            prefix = matched_prefix(j)
            order = prefix + [t for t in sequences[i] if t not in set(prefix)]
            return floor(common_prefix_length(build(order, i), prompts[j]), length)

        def exact_b(j: int) -> int:
            common = [t for t in sequences[j] if t in sets[i]]
            shared = set(common)
            mine = common + [t for t in sequences[i] if t not in shared]
            theirs = common + [t for t in sequences[j] if t not in shared]
            return floor(common_prefix_length(build(mine, i), build(theirs, j)), length)

        def best(pool, estimate, exact, floor_value: int) -> int:
            top = sorted(pool, key=lambda j: -estimate(j))[:3]
            return max([base, floor_value] + [exact(j) for j in top])

        est_a = lambda j: sum(approx_len[t] for t in matched_prefix(j))
        est_b = lambda j: sum(approx_len[t] for t in sets[i] & sets[j])
        a_win = best(window, est_a, exact_a, finite[i])
        a_inf = best(range(i), est_a, exact_a, unlimited[i])
        totals["A_win"] += a_win
        totals["A_inf"] += a_inf
        totals["B_win"] += best(window, est_b, exact_b, a_win)
        totals["B_inf"] += best(range(i), est_b, exact_b, a_inf)

    return {
        "run": replay["run_label"], "replay": replay_path, "workload": workload_path,
        "requests": len(results), "prompt_tokens": sum(map(len, prompts)),
        "cache_capacity_tokens": capacity * BLOCK, "header_tokens_per_request": header,
        "measured_cached": sum(measured), "sim_cached": sum(finite), "unlimited_cached": sum(unlimited),
        "requests_matching_simulation": sum(int(a == b) for a, b in zip(finite, measured)),
        "header_cached": sum(floor(header, len(p)) for p in prompts[1:]),
        "requests_with_tool_reuse": sum(1 for i in range(1, len(results)) if measured[i] > floor(header, len(prompts[i]))),
        **totals,
    }


def default_runs() -> list[tuple[str, str]]:
    r = "cluster/results"
    dense = f"{r}/dense-retrieval-20260902-234630/workloads"
    gaps = f"{r}/eval-gaps-20260907-224433/workloads"
    arms = ["original", "alphabetical", "frequency", "tooltrie_v0", "cp_online", "tooltrie_v1"]
    runs = []
    for arm in arms:
        for k in ("k4", "k16"):
            runs.append((f"{r}/eval-validity-20260906-165826/replays/06-{k}-{arm}.json", f"{dense}/{k}-{arm}.jsonl"))
        for k in ("k64", "k128"):
            source = "eval-validity-20260906-165826" if arm in ("alphabetical", "frequency") else "dense-accuracy-20260903-002742"
            runs.append((f"{r}/{source}/replays/06-{k}-{arm}.json", f"{dense}/{k}-{arm}.jsonl"))
        for k in ("k10", "k64p10"):
            runs.append((f"{r}/eval-gaps-20260907-224433/replays/06-{k}-{arm}.json", f"{gaps}/{k}-{arm}.jsonl"))
    for model, source in (("4b", "dense-accuracy-20260903-002742"), ("8b", "eval-validity-20260906-165826")):
        for k in ("k64", "k128"):
            for arm in ("original", "cp_online", "tooltrie_v1"):
                runs.append((f"{r}/{source}/replays/{model}-{k}-{arm}.json", f"{dense}/{k}-{arm}.jsonl"))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory; must not exist")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with Pool(args.workers) as pool, (args.out_dir / "decomposition.jsonl").open("w") as out:
        for row in pool.imap_unordered(analyse, default_runs()):
            out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
