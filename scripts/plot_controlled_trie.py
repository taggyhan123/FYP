#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.1"]
# ///
"""Publish figures and a report from the accepted trie-improvement summary.

Run with `uv run scripts/plot_controlled_trie.py`. Never reads live replays or
launches experiments. Matplotlib is isolated from the project's dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports/trie-improvements-controlled-summary.json"
OUT = ROOT / "reports/figures"
REPORT = ROOT / "reports/trie-improvements-figures.md"
ARMS = ("original", "v1", "served_only", "bounded", "protect5", "top10_then_v1")
DATASETS = {"first200": "First200", "random2026": "Random2026"}
LABELS = ("No reordering", "ToolTrie-v1", "Observe shown ten", "Bounded prefix", "Protect first five", "Select ten, then v1")
COLORS = ("#83909c", "#3d83b8", "#075e59", "#8465a4", "#b58334", "#b96760")
FILES: list[Path] = []


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    data = json.loads(SUMMARY.read_text())
    # These checks protect the figure aggregation. The independent experimental
    # audit is the authority for acceptance; this is not a substitute for it.
    if (data["accepted_replays"], data["accepted_requests"], data["excluded_warmup_requests"]) != (96, 19200, 400):
        raise ValueError("Only the complete accepted improvement matrix may be published")
    if data["protocol"] != "trie-improvements-20260918-v2":
        raise ValueError("This report's method text applies only to protocol v2")
    cells = {(g, c["dataset"], c["arm"]): c for g, d in data["devices"].items() for c in d["cells"]}
    expected = {(g, d, a) for g in ("gpu2", "gpu3") for d in DATASETS for a in ARMS}
    if set(cells) != expected:
        raise ValueError("Missing or unexpected device/dataset/policy cells")
    cross = data["cross_device"]["cells"]
    if len(cross) != 12 or not all(c["identical_per_request_hits_all_six_systems_replays"] and c["matching_quality_outputs"] == c["requests"] == 200 for c in cross):
        raise ValueError("Shared cache/F1 panels require identical replication results")
    for d in DATASETS:
        for a in ARMS:
            x, y = (cells[g, d, a] for g in ("gpu2", "gpu3"))
            for field in ("hit_pct", "computed_tokens", "prompt_tokens"):
                if len(x[field]) != 3 or len(y[field]) != 3 or len(set(x[field] + y[field])) != 1:
                    raise ValueError(f"Cannot collapse varying {field} values")
            if x["end_to_end_f1"] != y["end_to_end_f1"]:
                raise ValueError("Cannot collapse varying F1 values")
            for c in (x, y):
                p, comp, hit = (c[k][0] for k in ("prompt_tokens", "computed_tokens", "hit_pct"))
                if not math.isclose(hit, 100 * (p - comp) / p, abs_tol=1e-9):
                    raise ValueError("Token counters disagree with hit rate")
                for key, value in (("mean_ttft_across_trials", mean(c["mean_ttft_seconds"])), ("mean_ttft_trial_sd", stdev(c["mean_ttft_seconds"]))):
                    if not math.isclose(c[key], value, abs_tol=1e-12):
                        raise ValueError("TTFT summary disagrees with trial means")
    pairs = {(p["device"], p["dataset"], p["arm"]): p for p in data["paired_quality"]}
    for d in DATASETS:
        for a in ARMS:
            if a == "v1":
                continue
            for g in ("gpu2", "gpu3"):
                p = pairs[g, d, a]
                diff = cells[g, d, a]["end_to_end_f1"] - cells[g, d, "v1"]["end_to_end_f1"]
                if not math.isclose(p["mean_difference"], diff, abs_tol=1e-12):
                    raise ValueError("Paired F1 direction disagrees with scores")
                if any(p[k] != pairs["gpu2", d, a][k] for k in ("mean_difference", "ci95_low", "ci95_high")):
                    raise ValueError("Shared quality intervals require matching device estimates")
    return data, cells, pairs


def style(ax, title, xlabel, labels=LABELS):
    ax.set_title(title, loc="left", fontweight="bold", pad=11)
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.45, -0.7)
    ax.grid(axis="x", color="#e4e9ec", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c8d1d8")
    ax.tick_params(axis="y", length=0)


def save(fig, stem):
    for ext in ("png", "svg"):
        path = OUT / f"{stem}.{ext}"
        metadata = {"Date": None} if ext == "svg" else {"Software": f"Matplotlib {matplotlib.__version__}"}
        fig.savefig(path, dpi=180, facecolor="white", metadata=metadata)
        if ext == "svg":
            # Matplotlib leaves trailing spaces inside multiline path data.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        FILES.append(path)
    plt.close(fig)


def systems(cells):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.subplots_adjust(left=.15, right=.97, bottom=.12, top=.90, wspace=.60, hspace=.62)
    fig.suptitle("Controlled trie improvements | cache and engine latency", x=.035, y=.974, ha="left", fontsize=19, fontweight="bold")
    fig.text(.035, .942, "Qwen3-4B · retrieve 64 / show 10 · fixed 97,920-token KV capacity · 200 requests per stream", fontsize=11, color="#52616d")
    for col, (dataset, name) in enumerate(DATASETS.items()):
        for row, key, scale, title, unit, fmt in (
            (0, "hit_pct", 1, "cached prompt tokens", "Token hit rate (%) — higher is better", ".2f"),
            (1, "computed_tokens", .001, "computed prompt tokens", "Total over 200 requests (thousands) — lower is better", ".1f"),
        ):
            ax = axes[row, col]
            vals = [cells["gpu2", dataset, a][key][0] * scale for a in ARMS]
            ax.barh(range(6), vals, color=COLORS, height=.62)
            style(ax, f"{name} · {title}", unit)
            ax.set_xlim(0, (36 if row == 0 else 340))
            for i, v in enumerate(vals):
                ax.text(v + ax.get_xlim()[1] * .015, i, format(v, fmt), va="center", fontsize=10)
        ax = axes[2, col]
        style(ax, f"{name} · mean engine TTFT", "Time to first token (ms) — lower is better")
        for g, offset, marker, color in (("gpu2", -.14, "o", "#1e5679"), ("gpu3", .14, "s", "#ad5c26")):
            values = [cells[g, dataset, a]["mean_ttft_across_trials"] * 1000 for a in ARMS]
            errors = [cells[g, dataset, a]["mean_ttft_trial_sd"] * 1000 for a in ARMS]
            ax.errorbar(values, [i + offset for i in range(6)], xerr=errors, fmt=marker, color=color, capsize=3, markersize=5, label=g.upper().replace("GPU", "GPU "))
        ax.set_xlim(145, 225)
        ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.text(.035, .025, "Cache/token totals match across all six systems replays. TTFT: mean ± sample SD of three trial means, shown per GPU.\nOne generated token; sequential traffic; warmup excluded. TTFT excludes planning and does not measure throughput.", fontsize=10, color="#52616d", linespacing=1.5)
    save(fig, "controlled-trie-systems")


def quality(cells, pairs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(left=.15, right=.97, bottom=.12, top=.86, wspace=.60, hspace=.65)
    fig.suptitle("Controlled trie improvements | tool-ID quality", x=.035, y=.972, ha="left", fontsize=19, fontweight="bold")
    fig.text(.035, .929, "128-token output budget · all 200 tasks scored · identical model outputs across GPUs", fontsize=11, color="#52616d")
    other = [a for a in ARMS if a != "v1"]
    other_labels = [LABELS[ARMS.index(a)] for a in other]
    for col, (dataset, name) in enumerate(DATASETS.items()):
        ax = axes[0, col]
        vals = [100 * cells["gpu2", dataset, a]["end_to_end_f1"] for a in ARMS]
        ax.barh(range(6), vals, color=COLORS, height=.62)
        style(ax, f"{name} · macro F1", "Tool-ID F1 (%) — higher is better")
        ax.set_xlim(0, 30)
        for i, v in enumerate(vals):
            ax.text(v + .4, i, f"{v:.2f}", va="center", fontsize=10)
        ax = axes[1, col]
        style(ax, f"{name} · paired difference from v1", "F1 change (percentage points) — right favors candidate", other_labels)
        ax.axvline(0, color="#7b8994", linewidth=1, linestyle="--")
        for i, a in enumerate(other):
            p = pairs["gpu2", dataset, a]
            v, lo, hi = (100 * p[k] for k in ("mean_difference", "ci95_low", "ci95_high"))
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=COLORS[ARMS.index(a)], capsize=4)
        ax.set_xlim(-3.1, 3.1)
    fig.text(.035, .035, "Intervals: paired task bootstrap, 10,000 resamples; exploratory and conditional on each frozen stream.\nAn interval crossing zero does not demonstrate equivalent quality. F1 does not grade arguments or task execution.", fontsize=10, color="#52616d", linespacing=1.5)
    save(fig, "controlled-trie-quality")


def structure():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=.025, right=.975, bottom=.04, top=.86)
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    fig.suptitle("Observe the tools actually shown | trie structure and request flow", x=.04, y=.96, ha="left", fontsize=18, fontweight="bold")
    fig.text(.04, .913, "Illustrative structure from the implementation; letters stand for tool IDs, not token blocks.", fontsize=11, color="#52616d")

    def box(x, y, w, h, text, fill="#edf3f7", edge="#c3d1db", size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=.1", facecolor=fill, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.5)

    def arrow(start, end, color="#81919d"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=color, linewidth=1.4))

    for x, t in ((.15, "Retrieve 64\nranked candidates"), (3.65, "Plan using only\nearlier observations"), (7.15, "Show first ten\nand serve request"), (10.65, "Observe only the\nten shown tools")):
        box(x, 6.6, 3, .95, t)
    for x in (3.15, 6.65, 10.15):
        arrow((x + .08, 7.07), (x + .43, 7.07))
    ax.text(.2, 5.8, "A tool-level prefix tree", fontsize=14, fontweight="bold")
    for x, y, w, t in ((2, 4.9, 1.3, "root"), (.8, 3.65, 1.2, "A"), (3.6, 3.65, 1.2, "D"), (.1, 2.4, 1.2, "B"), (1.8, 2.4, 1.2, "C"), (3.6, 2.4, 1.2, "B")):
        box(x, y, w, .65, t)
    for start, end in (((2.4, 4.85), (1.4, 4.38)), ((2.95, 4.85), (4.2, 4.38)), ((1.2, 3.57), (.7, 3.14)), ((1.6, 3.57), (2.4, 3.14)), ((4.2, 3.57), (4.2, 3.14))):
        arrow(start, end)
    ax.text(.15, 1.55, "Observed paths: [A, B], [A, C], [D, B]\nA shares a prefix; B under A and D is distinct.", fontsize=10, color="#52616d", linespacing=1.6)
    box(5.8, 3.95, 7.65, 1.6, "Each node records a tool ID, parent and children,\nschema-token cost, visit count and last-seen request.\nPlan prefers a compatible recent path with high hinted cost;\nunmatched tools retain their retrieval order.", size=11)
    box(5.8, 1.7, 7.65, 1.6, "The improvement changes observation: 64 tools → shown ten.\nIt reuses the v1 planning core and can change menu membership.\nMetadata: window 128; 100,000 nodes; 190,896 schema tokens.\nLeast-recently-used leaves are evicted when over budget.", fill="#e9f4ef", edge="#92b9a8", size=11)
    box(.15, .1, 13.3, .9, "vLLM separately caches rendered token blocks. Trie recency is a reuse hint; actual hits come from engine counters.\nPlanner metadata capacity and the server's 97,920-token KV capacity are different budgets.", fill="#fff7e7", edge="#d8c39b", size=11)
    save(fig, "controlled-trie-structure")


def report(data, cells, pairs):
    lines = [
        "# Controlled trie improvements: figures and results", "",
        "**Accepted study completed 19 September 2026: 96/96 replays, 19,200 measured requests, 400 excluded warmup requests.**", "",
        "Generated from the [audited summary](trie-improvements-controlled-summary.json). This publication covers the completed six-policy improvement study. The newer matched ContextPilot/frequency matrix and fixed-menu follow-up are separate experiments; their results are not included here.", "",
        "## Main finding", "",
        "Observing only the ten shown tools improves cache reuse and engine TTFT over v1 on both fixed streams. F1 point estimates decline slightly; the paired intervals do not establish quality preservation. These results do not select an overall winner or a new production default.", "",
        "| Dataset | Cache hit %: v1 → observe ten | Computed-token reduction | TTFT reduction: GPU 2 / GPU 3 | F1 change [95% interval], pp |",
        "|---|---:|---:|---:|---:|",
    ]
    for d, name in DATASETS.items():
        v, s = (cells["gpu2", d, a] for a in ("v1", "served_only"))
        p = pairs["gpu2", d, "served_only"]
        timing = " / ".join(f"{100*(1-cells[g,d,'served_only']['mean_ttft_across_trials']/cells[g,d,'v1']['mean_ttft_across_trials']):.2f}%" for g in ("gpu2", "gpu3"))
        lines.append(f"| {name} | {v['hit_pct'][0]:.2f} → {s['hit_pct'][0]:.2f} | {100*(1-s['computed_tokens'][0]/v['computed_tokens'][0]):.2f}% | {timing} | {100*p['mean_difference']:+.3f} [{100*p['ci95_low']:+.3f}, {100*p['ci95_high']:+.3f}] |")
    lines += ["", "## Figures", "", "### Cache, computation and latency", "", "[PNG](figures/controlled-trie-systems.png) · [SVG](figures/controlled-trie-systems.svg)", "", "![Cache, computed tokens and TTFT](figures/controlled-trie-systems.png)", "", "Cache/token totals are identical across all six systems replays for each policy/dataset. TTFT is shown separately for each GPU: error bars are the sample standard deviation of three trial means, not task confidence intervals. The TTFT axis starts at 145 ms; cache and token bars start at zero.", "", "### Quality and uncertainty", "", "[PNG](figures/controlled-trie-quality.png) · [SVG](figures/controlled-trie-quality.svg)", "", "![Tool-ID F1 and paired differences](figures/controlled-trie-quality.png)", "", "GPU quality outputs agree on all 2,400 policy requests. Repeating tasks across GPUs does not increase the number of independent tasks. Bootstrap intervals use 10,000 task resamples, seed 20260918, conditional on each frozen sequence; comparisons are exploratory.", "", "### Structure and observation rule", "", "[PNG](figures/controlled-trie-structure.png) · [SVG](figures/controlled-trie-structure.svg)", "", "![Trie structure and observation rule](figures/controlled-trie-structure.png)", "", "The tree is planner metadata over tool IDs. The shown-ten change uses the existing [v1 planner](../src/tatm/tooltrie_v1.py) and [trie core](../src/tatm/tooltrie.py). Its recorded paths end after the shown menu. This diagram is illustrative, not a dump of an experimental tree.", "", "## All measured policies", "", "| Figure label | Policy | Rule |", "|---|---|---|"]
    rules = ("Show the retriever's first ten in retrieval order", "Plan 64; show ten; observe all 64", "Plan 64; show ten; observe only shown ten", "Score rendered compatible prefixes up to ten; observe shown", "Bounded policy with the retriever's first five guaranteed inclusion", "Select the original top ten, then reorder and observe those ten")
    lines.extend(f"| {label} | `{arm}` | {rule} |" for label, arm, rule in zip(LABELS, ARMS, rules))
    lines += ["", "Except for no reordering and select-ten-then-v1, policies can change which ten tools are shown. This is a joint selection/ordering comparison. The fixed-ten arm isolates ordering relative to no reordering.", ""]
    for d, name in DATASETS.items():
        lines += [f"### {name}", "", "| Policy | Hit % | Prompt tokens | Computed tokens | Mean TTFT ± SD, GPU 2 / GPU 3 (ms) | Tool-ID F1 % |", "|---|---:|---:|---:|---:|---:|"]
        for a, label in zip(ARMS, LABELS):
            c = cells["gpu2", d, a]
            ttft = " / ".join(f"{1000*cells[g,d,a]['mean_ttft_across_trials']:.2f} ± {1000*cells[g,d,a]['mean_ttft_trial_sd']:.2f}" for g in ("gpu2", "gpu3"))
            lines.append(f"| {label} | {c['hit_pct'][0]:.2f} | {c['prompt_tokens'][0]:,.0f} | {c['computed_tokens'][0]:,.0f} | {ttft} | {100*c['end_to_end_f1']:.3f} |")
        lines += [""]
    lines += [
        "## Metric definitions and controls", "",
        "- **Hit rate:** total cached prompt tokens / total prompt tokens. It is not the percentage of requests with any hit. Prompt lengths vary with selected tools; computed-token totals capture actual prompt work.",
        "- **Latency:** engine time to first token, one generated token per request, sequential traffic. Offline planning is excluded; these measurements do not establish production throughput or full client latency.",
        "- **F1:** macro tool-ID F1 over all tasks, including menus missing the relevant tools; unknown names count as false positives. Arguments and task execution are not graded. The uniform 128-token quality budget can truncate outputs.",
        "- **Samples:** First200 contains 101 APIBank and 99 APIGen tasks; Random2026 contains 200 tasks from 27 ToolRet sources. Three IDs overlap, giving 397 distinct tasks. Both streams were previously inspected; they are exploratory rather than untouched holdouts.",
        "- **Fair requests:** identical queries, labels, order and 64 retrieved candidates; ten distinct tools shown per policy; same original schemas except a common unique function-name map. The audit checked all 2,400 policy requests. Different menu membership remains a material difference.",
        "- **Server:** Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, vLLM 0.26.0, BF16, eager, TP1, temperature/seed 0, thinking disabled. Maximum model length 32,768; one sequence; no chunked prefill.",
        "- **GPUs:** initially idle RTX 3090 devices 2 and 3, used sequentially. Every policy runs on each GPU. KV capacity is fixed and verified at 6,120 blocks × 16 = 97,920 tokens. Cache resets before each replay; first request must have zero cached tokens. Warmup is excluded; three systems trials rotate policy order, reversed on GPU 3.",
        "- **Remaining variability:** temperature, automatic clocks, and host scheduling can differ. Device timings remain separate. The interrupted GPU 3 attempt was excluded in full; this summary uses the completed retry only.",
        "- **Historical separation:** unique tool names change tokenization and potentially model behavior. Do not pool this study with historical raw-name scores, the newer 512-token quality experiment, or old concurrency results.", "",
        "## Interpretation", "",
        "The simple observation change yields a consistent systems gain. Bounded-prefix scoring adds no clear benefit here: on First200 it reports a slightly higher hit fraction but computes more tokens than observing ten with the v1 core. Select-ten-then-v1 has much smaller gains, especially on Random2026. This limits attribution of the larger gains to ordering alone.", "",
        "This study contains no fresh ContextPilot or frequency arm, so it cannot establish the improved policy's advantage over those methods. Historical figures remain available in the [figure index](figures/README.md), with their original experimental conditions and quality limitations.", "",
        "## Reproduce and trace", "",
        "From the repository root:", "", "```bash", "uv run scripts/plot_controlled_trie.py", "```", "",
        "This regenerates three PNG/SVG pairs, this report and a [SHA-256 publication manifest](figures/controlled-trie-publication.json) from the committed compact summary. It requires no GPU or raw data. Matplotlib 3.10.1 is installed by uv into an isolated script environment.", "",
        f"- Experimental manifest SHA-256: `{data['manifest_sha256']}`.",
        f"- Independent analysis script SHA-256: `{data['analysis_script_sha256']}`.",
        f"- Summary SHA-256: `{digest(SUMMARY)}`.",
        "- Raw run directory: `cluster/results/trie-improvements-controlled-20260918-04/` (ignored by Git). Its raw responses, telemetry and frozen source snapshot are required to rerun the independent experimental audit; the committed summary supports figure reproduction only.", "",
    ]
    REPORT.write_text("\n".join(lines))
    FILES.append(REPORT)


def main():
    data, cells, pairs = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12, "text.color": "#20313e", "axes.labelcolor": "#354653", "xtick.color": "#52616d", "ytick.color": "#354653", "svg.hashsalt": "tatm-controlled-trie-v1"})
    systems(cells)
    quality(cells, pairs)
    structure()
    report(data, cells, pairs)
    manifest = {"study": data["protocol"], "accepted_replays": data["accepted_replays"], "matplotlib_version": matplotlib.__version__, "inputs": {str(p.relative_to(ROOT)): digest(p) for p in (SUMMARY, Path(__file__).resolve(), ROOT / "src/tatm/tooltrie.py", ROOT / "src/tatm/tooltrie_v1.py")}, "outputs": {str(p.relative_to(ROOT)): digest(p) for p in FILES}}
    (OUT / "controlled-trie-publication.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Generated {len(FILES) - 1} figure files, {REPORT.relative_to(ROOT)}, and publication manifest")


if __name__ == "__main__":
    main()
