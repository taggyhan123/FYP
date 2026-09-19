#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.1"]
# ///
"""Reproduce the improved-trie baseline report and request-window figure.

Ordinary use reads committed compact evidence. --raw-root extracts metrics
only after the existing experiment's independent finalizer has accepted it.
This script never launches experiments or modifies experimental artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports/trie-baselines-controlled-summary.json"
METRICS = ROOT / "reports/trie-baselines-plot-data.json"
REPORT = ROOT / "reports/concurrent-latency/cache-hit-miss-f1-improved-trie.md"
FIGURE = ROOT / "reports/figures/improved-trie-cache-interactivity"
MANIFEST = ROOT / "reports/trie-baselines-publication.json"
DATASETS = {"first200": "First200", "random2026": "Random2026"}
DEVICES = ("gpu2", "gpu3")
SERIES = (
    ("served_only", "Improved ToolTrie", "#2a78d6"),
    ("contextpilot", "ContextPilot ordering", "#eb6834"),
    ("frequency_fitted", "Frequency: fitted", "#1baf7a"),
    ("frequency_online", "Frequency: online", "#9065b0"),
    ("original", "No reordering", "#8a8983"),
)
ARMS = tuple(a for a, _, _ in SERIES) + ("v1",)
LABELS = {a: label for a, label, _ in SERIES} | {"v1": "Previous ToolTrie-v1"}
WINDOW = 40


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_summary():
    data = json.loads(SUMMARY.read_text())
    require(data["protocol"] == "trie-baselines-20260919-v1", "Unexpected experimental protocol")
    require((data["accepted_replays"], data["accepted_requests"], data["excluded_warmup_requests"], data["quality_max_tokens"]) == (96, 19200, 400, 512), "Complete 512-token baseline matrix required")
    cells = {(g, c["dataset"], c["arm"]): c for g, device in data["devices"].items() for c in device["cells"]}
    require(set(cells) == {(g, d, a) for g in DEVICES for d in DATASETS for a in ARMS}, "Incomplete policy coverage")
    for d in DATASETS:
        for a in ARMS:
            left, right = (cells[g, d, a] for g in DEVICES)
            for key in ("hit_pct", "prompt_tokens", "computed_tokens"):
                require(len(left[key]) == len(right[key]) == 3 and len(set(left[key] + right[key])) == 1, f"Cannot present shared {key}")
            for key in ("end_to_end_f1", "gold_any_coverage", "gold_all_coverage", "quality_requests_at_output_limit", "quality_finish_reasons"):
                require(left[key] == right[key], f"Separate device quality tables required: {key}")
    pairs = {(p["device"], p["dataset"], p["comparator"]): p for p in data["paired_quality"]}
    for (g, d, a), p in pairs.items():
        expected = cells[g, d, "served_only"]["end_to_end_f1"] - cells[g, d, a]["end_to_end_f1"]
        require(p["difference"] == "served_only minus comparator" and math.isclose(expected, p["mean_difference"], abs_tol=1e-12), "Paired F1 direction disagrees")
    return data, cells, pairs


def extract_metrics(raw, data):
    status = json.loads((raw / "finalizer-status.json").read_text())
    receipt = json.loads((raw / "final-artifacts.json").read_text())
    require(status["complete"] and status["failure"] is None, "Independent finalizer has not accepted this run")
    require(receipt["summary_sha256"] == digest(SUMMARY), "Finalizer summary hash differs")
    require(digest(raw / "manifest.json") == data["manifest_sha256"], "Experimental manifest differs")
    replays = []
    for device in DEVICES:
        device_status = json.loads((raw / device / "status.json").read_text())
        require(device_status["complete"] and len(device_status["accepted"]) == 48, "Incomplete device receipt")
        for entry in device_status["accepted"]:
            if entry["phase"] != "systems":
                continue
            path = raw / entry["replay"]
            require(digest(path) == entry["sha256"], "Replay hash differs from accepted receipt")
            replay = json.loads(path.read_text())
            require(len(replay["results"]) == 200 and all(r["usage"]["completion_tokens"] == 1 for r in replay["results"]), "Unexpected request count or output budget")
            rows = []
            for r in replay["results"]:
                m = r["measurement"]
                require(m["prompt_tokens"] == r["usage"]["prompt_tokens"], "Prompt counters differ")
                require(m["prompt_tokens"] == m["cached_tokens"] + m["computed_prefill_tokens"], "Token partition differs")
                rows.append([m["prompt_tokens"], m["cached_tokens"], m["ttft_seconds"]])
            require(sum(r[0] for r in rows) == entry["prompt_tokens"] and sum(r[1] for r in rows) == entry["cached_tokens"], "Replay totals differ")
            replays.append({"device": device, "dataset": entry["dataset"], "arm": entry["arm"], "trial": entry["trial"], "raw_path": entry["replay"], "raw_sha256": entry["sha256"], "requests": rows})
    compact = {"protocol": data["protocol"], "summary_sha256": digest(SUMMARY), "manifest_sha256": data["manifest_sha256"], "finalizer_receipt": status, "raw_root": str(raw.relative_to(ROOT)), "fields": ["prompt_tokens", "cached_tokens", "engine_ttft_seconds"], "replays": replays}
    # One replay per line keeps the numeric artifact compact and inspectable.
    header = {k: v for k, v in compact.items() if k != "replays"}
    METRICS.write_text(json.dumps(header, indent=2)[:-2] + ',\n  "replays": [\n' + ',\n'.join('    ' + json.dumps(r, separators=(',', ':')) for r in replays) + '\n  ]\n}\n')


def load_metrics(cells):
    compact = json.loads(METRICS.read_text())
    require(compact["summary_sha256"] == digest(SUMMARY), "Plot metrics and audited summary differ")
    expected = {(g, d, a, t) for g in DEVICES for d in DATASETS for a in ARMS for t in (1, 2, 3)}
    seen = set()
    for r in compact["replays"]:
        key = (r["device"], r["dataset"], r["arm"], r["trial"])
        require(key not in seen, "Duplicate replay in plot evidence")
        seen.add(key)
        rows, cell, i = r["requests"], cells[key[:3]], r["trial"] - 1
        require(len(rows) == 200 and all(0 <= cached <= prompt and ttft > 0 for prompt, cached, ttft in rows), "Invalid request metrics")
        prompt, cached = sum(v[0] for v in rows), sum(v[1] for v in rows)
        require(prompt == cell["prompt_tokens"][i] and prompt - cached == cell["computed_tokens"][i], "Plot token totals disagree with summary")
        require(math.isclose(100 * cached / prompt, cell["hit_pct"][i], abs_tol=1e-10), "Plot hit rate differs")
        require(math.isclose(mean(v[2] for v in rows), cell["mean_ttft_seconds"][i], abs_tol=1e-12), "Plot timing differs")
    require(seen == expected, "Plot evidence lacks accepted systems replays")
    return compact


def windows(rows):
    for start in range(len(rows) - WINDOW + 1):
        w = rows[start:start + WINDOW]
        yield 1 / mean(r[2] for r in w), 100 * sum(r[1] for r in w) / sum(r[0] for r in w)


def reduction(new, old):
    return 100 * (1 - new / old)


def plot(cells, compact):
    surface, ink, secondary, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=surface)
    fig.subplots_adjust(left=.065, right=.76, bottom=.18, top=.83, hspace=.35, wspace=.18)
    fig.text(.065, .955, "Improved ToolTrie versus ContextPilot and frequency", fontsize=20, fontweight="bold", color=ink)
    fig.text(.065, .915, "Cache reuse versus first-token interactivity · matched requests · Qwen3-4B · separate GPU panels", fontsize=12, color=secondary)
    fig.text(.065, .88, "Dots: overlapping 40-request windows. Large circles: full-stream summaries over three trials. Higher and right are better systems metrics.", fontsize=10.5, color=secondary)
    plotted = {a for a, _, _ in SERIES}
    clouds = {}
    for r in compact["replays"]:
        if r["arm"] in plotted:
            clouds.setdefault((r["device"], r["dataset"], r["arm"]), []).extend(windows(r["requests"]))
    all_points = [p for pts in clouds.values() for p in pts]
    for row, (dataset, title) in enumerate(DATASETS.items()):
        extents = [p for (g, d, a), pts in clouds.items() if d == dataset for p in pts]
        extents += [(1 / c["mean_ttft_across_trials"], c["hit_pct"][0]) for (g, d, a), c in cells.items() if d == dataset and a in plotted]
        xmin = max(0, math.floor(min(p[0] for p in extents) * 2) / 2 - .25)
        xmax = math.ceil(max(p[0] for p in extents) * 2) / 2 + .25
        ymax = min(100, 5 * math.ceil(max(p[1] for p in extents) / 5))
        for col, device in enumerate(DEVICES):
            ax = axes[row, col]
            ax.set_facecolor(surface)
            ax.set(xlim=(xmin, xmax), ylim=(0, ymax))
            ax.set_title(f"{title} · GPU {device[-1]}", loc="left", fontsize=12, fontweight="bold", color=ink, pad=10)
            ax.grid(color=grid, linewidth=.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#7a7973")
            ax.tick_params(colors="#7a7973", length=0, labelsize=10)
            for arm, _, color in reversed(SERIES):
                pts = clouds[device, dataset, arm]
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=7, alpha=.12, color=color, linewidths=0)
                c = cells[device, dataset, arm]
                xs = [1 / t for t in c["mean_ttft_seconds"]]
                x, y = 1 / c["mean_ttft_across_trials"], c["hit_pct"][0]
                ax.plot([min(xs), max(xs)], [y, y], color=color, linewidth=1.4)
                ax.scatter([x], [y], s=80 if arm == "served_only" else 55, color=color, edgecolors="white", linewidths=1.5, zorder=5)
            improved, cp = (cells[device, dataset, a] for a in ("served_only", "contextpilot"))
            ax.text(.035, .95, f"Improved vs ContextPilot ordering\n+{improved['hit_pct'][0]-cp['hit_pct'][0]:.2f} pp cache · {reduction(improved['mean_ttft_across_trials'],cp['mean_ttft_across_trials']):.2f}% lower mean TTFT", transform=ax.transAxes, va="top", fontsize=9.5, color=ink, bbox={"facecolor": surface, "edgecolor": "none", "alpha": .9, "pad": 3})
            if col == 0:
                ax.set_ylabel("Cached prompt tokens (%)", color=secondary, fontsize=10.5)
            if row == 1:
                ax.set_xlabel("Interactivity: 1 / mean engine TTFT (s⁻¹)", color=secondary, fontsize=10.5, labelpad=10)
    handles = [Line2D([], [], color=c, marker="o", markeredgecolor="white", linewidth=2, markersize=7, label=label) for _, label, c in SERIES]
    fig.legend(handles=handles, title="Policy", loc="upper left", bbox_to_anchor=(.775, .85), frameon=False, labelspacing=1.1, fontsize=10.5, title_fontsize=12)
    fig.text(.79, .58, "Improved trie:\nobserve only shown ten\n\nFrequency shown separately:\nfitted training support\nonline served-menu counts\n\nRetrieve 64 / show 10\nKV: 97,920 tokens\nOne client, one output token\n200 tasks per stream", fontsize=10.5, color=secondary, linespacing=1.65, va="top")
    count = len(all_points)
    fig.text(.065, .095, f"{count:,} window points from 12,000 measured systems requests; windows overlap and do not add independent tasks.\nThis is a cache–latency plot. Sequential runs provide no throughput frontier or load curve. Planner time is excluded.\nAxis ranges match within each dataset row. Trial spans accompany large circles; quality uses separate 512-token replays.\nContextPilot is the pinned ordering-only adaptation. Source: reports/trie-baselines-controlled-summary.json and trie-baselines-plot-data.json.", fontsize=10, color="#7a7973", linespacing=1.6, va="top")
    for ext in ("png", "svg"):
        path = FIGURE.with_suffix('.' + ext)
        fig.savefig(path, dpi=180, facecolor=surface, metadata={"Date": None} if ext == "svg" else {"Software": f"Matplotlib {matplotlib.__version__}"})
        if ext == "svg":
            path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    return count


def report(data, cells, pairs, point_count):
    lines = [
        "# Improved ToolTrie: cache hit and miss rate, F1, and latency", "",
        "How much of each prompt vLLM reuses or computes under the **new trie that observes only the ten shown tools**, compared with ContextPilot ordering, fitted frequency, online frequency, no reordering, and the previous ToolTrie-v1.", "",
        "**Data:** 96/96 independently accepted replays, 19,200 measured requests, and 400 excluded warmup requests. Qwen3-4B, vLLM 0.26.0, dense retrieve-64 / show-10, with separate RTX 3090 GPU 2 and GPU 3 replication. The [audited summary](../trie-baselines-controlled-summary.json) is the source for every table.", "",
        "**Measurement pairing:** cache and latency come from three one-token systems replays per policy/dataset/GPU. F1 comes from a separate **512-token quality replay of the same frozen requests and menus**. These are matched inputs, not the same replay. The earlier 128-token improvement study is separate evidence.", "",
        "## Summary", "",
    ]
    for d, name in DATASETS.items():
        s, cp, base = (cells['gpu2', d, a] for a in ('served_only','contextpilot','original'))
        lines.append(f"- **{name}:** improved-trie hit rate is **{s['hit_pct'][0]:.2f}%** (miss **{100-s['hit_pct'][0]:.2f}%**), versus ContextPilot ordering {cp['hit_pct'][0]:.2f}% and no reordering {base['hit_pct'][0]:.2f}%. Tool-ID F1 is **{100*s['end_to_end_f1']:.3f}**, versus {100*cp['end_to_end_f1']:.3f} and {100*base['end_to_end_f1']:.3f}, respectively.")
    lines += ["- **Efficiency and quality need separate conclusions.** The new trie improves cache reuse and mean TTFT over these external baselines on both streams, but does not maximize F1. Paired intervals below describe uncertainty; an interval containing zero does not establish equivalent quality.", "- **F1 versus ContextPilot remains uncertain.** Both paired 95% intervals cross zero; higher point estimates do not establish a quality gain.", "- **Frequency has two meanings here.** Fitted frequency uses a separate training corpus; online frequency counts tools shown on earlier requests. Their selection behavior and quality differ substantially.", "", "## Where the new trie improves systems results", "", "Positive reductions mean less computed prompt work or shorter mean TTFT. F1 changes are improved trie minus comparator; intervals are paired task bootstrap 95% intervals.", ""]
    for d, name in DATASETS.items():
        lines += [f"### {name}", "", "| Comparator | Hit %: comparator → new | Computed-token reduction | Mean TTFT reduction: GPU 2 / 3 | F1 change [95% interval], pp: GPU 2 / 3 |", "|---|---:|---:|---:|---|"]
        s = cells['gpu2', d, 'served_only']
        for a in ARMS[1:]:
            b = cells['gpu2',d,a]
            timing = ' / '.join(f"{reduction(cells[g,d,'served_only']['mean_ttft_across_trials'],cells[g,d,a]['mean_ttft_across_trials']):+.2f}%" for g in DEVICES)
            estimates = ['{:+.3f} [{:+.3f}, {:+.3f}]'.format(*(100*pairs[g,d,a][k] for k in ('mean_difference','ci95_low','ci95_high'))) for g in DEVICES]
            interval = estimates[0] + (' (both)' if estimates[0] == estimates[1] else ' / ' + estimates[1])
            lines.append(f"| {LABELS[a]} | {b['hit_pct'][0]:.2f} → {s['hit_pct'][0]:.2f} | {reduction(s['computed_tokens'][0],b['computed_tokens'][0]):+.2f}% | {timing} | {interval} |")
        lines += ['']
    lines += ["**Where the new trie does not win:** no reordering has a higher F1 point estimate on both streams, and fitted frequency has a higher point estimate on First200. These data do not define a combined quality/efficiency objective or select an overall winner.", "", "## Evaluation data and request fairness", "", "| Item | Matched setting |", "|---|---|", "| First200 | 200 previously inspected tasks: 101 APIBank, 99 APIGen |", "| Random2026 | 200 previously inspected tasks from 27 ToolRet sources |", "| Overlap | Three shared task IDs; 397 distinct tasks across both streams |", "| Retrieval | Same dense-retrieved 64 candidates, query, labels and request order for every policy |", "| Model input | Ten tools per request; policy may change membership and order |", "| Names | Common stable unique function-name mapping for every policy |", "| Engine | Qwen3-4B pinned revision; vLLM 0.26.0; BF16; eager; TP1; temperature/seed 0; thinking disabled |", "| KV capacity | Fixed and checked: 6,120 × 16 = 97,920 tokens on each GPU |", "| Execution | One sequential client; cache reset before each replay; first request has zero cached tokens |", "| Replication | Every policy on GPUs 2 and 3; three rotated systems trials; reversed policy order on GPU 3 |", "", "The independent audit checked all 2,400 policy requests. These controls make the query streams and candidate pools comparable. Selected ten-tool menus and policy history rules still differ, so this comparison measures selection plus ordering.", "", "## 1. First200: cache hit/miss and tool-ID F1", ""]
    for i, (d, name) in enumerate(DATASETS.items()):
        if i:
            lines += ["## 2. Random2026: cache hit/miss and tool-ID F1", ""]
        lines += ["| Policy | Hit | Miss | Any gold tool in ten shown | All gold tools in ten shown | Tool-ID F1 ×100 | New − policy F1, pp |", "|---|---:|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            c = cells['gpu2',d,a]
            label = f"**{LABELS[a]}**" if a == 'served_only' else LABELS[a]
            lines.append(f"| {label} | {c['hit_pct'][0]:.2f}% | {100-c['hit_pct'][0]:.2f}% | {100*c['gold_any_coverage']:.1f}% | {100*c['gold_all_coverage']:.1f}% | {100*c['end_to_end_f1']:.3f} | {100*(cells['gpu2',d,'served_only']['end_to_end_f1']-c['end_to_end_f1']):+.3f} |")
        lines += ["", "Cache totals and quality metrics in this table match across the two GPUs. **New − policy F1** is the point difference in favor of the improved trie; paired intervals are reported above. Menu coverage is a selection metric, not the model's actual success rate.", ""]
    lines += ["## 3. Computed prompt tokens and latency", "", "Counts are totals over 200 requests. TTFT is the mean of three trial means; ± is their sample standard deviation. Device timings are kept separate. These are engine measurements with one output token, excluding offline planning and full response generation.", ""]
    for d, name in DATASETS.items():
        lines += [f"### {name}", "", "| Policy | Prompt tokens | Cached tokens | Computed tokens | Mean TTFT ± SD, GPU 2 / 3 (ms) |", "|---|---:|---:|---:|---:|"]
        for a in ARMS:
            c = cells['gpu2',d,a]
            timing = ' / '.join(f"{1000*cells[g,d,a]['mean_ttft_across_trials']:.2f} ± {1000*cells[g,d,a]['mean_ttft_trial_sd']:.2f}" for g in DEVICES)
            lines.append(f"| {LABELS[a]} | {c['prompt_tokens'][0]:,.0f} | {c['prompt_tokens'][0]-c['computed_tokens'][0]:,.0f} | {c['computed_tokens'][0]:,.0f} | {timing} |")
        lines += ['']
    lines += ["A higher cache percentage alone does not imply less work: policies can select schemas of different lengths. Cached and computed token counts expose that denominator difference.", "", "### Request-window figure", "", "[PNG](../figures/improved-trie-cache-interactivity.png) · [SVG](../figures/improved-trie-cache-interactivity.svg)", "", "![Improved trie compared with ContextPilot and both frequency policies](../figures/improved-trie-cache-interactivity.png)", "", f"The five requested policies contribute **{point_count:,} plotted window points from 12,000 measured systems requests**. Each point uses 40 consecutive requests (stride one): x = 1 / mean engine TTFT; y = total cached / total prompt tokens ×100. Windows stay within one policy, dataset, device and trial. They overlap and do not add independent tasks. Large circles summarize three full replays; horizontal spans show the three-trial range. For each dataset, both GPU panels share axis limits; the two datasets use different ranges to keep their distributions legible.", "", "This is a cache–latency comparison in the historical frontier's visual style. It contains no throughput measurement or fitted load curve. A throughput frontier for the improved policy requires a separately controlled load sweep.", "", "## 4. What each policy knows", "", "| Policy | Selection, order and history |", "|---|---|", "| Improved ToolTrie (`served_only`) | Plan 64, show ten, observe only shown ten; reuse the v1 planning core |", "| Previous v1 | Plan 64, show ten, observe all 64 |", "| ContextPilot ordering | Pinned official persistent API orders 64, show ten; native API observes all 64 |", "| Fitted frequency | Rank by disjoint training gold-support counts; tool-ID tie break |", "| Online frequency | Rank by counts in earlier shown menus; original-name/ID tie break; observe shown ten |", "| No reordering | Show the retriever's first ten in retrieval order |", ""]
    training = data['preflight']['frequency']
    lines += [f"Fitted frequency uses **{training['training_tasks']:,} training tasks**, excluding {training['excluded_tasks']:,} tasks whose IDs or whitespace-normalized queries overlap evaluation. The audit reconstructed training support and all 800 frequency decisions. Evaluation labels never rank the tools.", "", "Online frequency's alphabetical tie break can displace relevant tools before any useful history exists, and observing only its shown menus can reinforce those choices. Its low F1 is a result of this specific selector, not a general claim that frequency methods fail.", "", "ContextPilot is pinned to `1fa0a143fdeda344585666648ab2b30cb7fea77f`, with alpha 0.001, average linkage and persistent CPU planning. Full and prefix builds were checked for reproducibility and causality. This is an **ordering-only adaptation**: scheduling, annotations, de-duplication and engine eviction feedback are outside the comparison. It does not evaluate the full ContextPilot system.", "", "Both trie policies use a 128-request recency window, at most 100,000 nodes and 190,896 canonical schema tokens of planner metadata. This budget is distinct from the engine's physical KV allocation.", "", "## How to read the numbers", "", "- **Hit / miss:** cached / computed prompt tokens, divided by all prompt tokens; they sum to 100%. This is token-weighted reuse, not the fraction of requests with any cache hit.", "- **F1:** mean task-level tool-ID F1 across all tasks, including those whose menus omit gold tools. Unknown called names count as false positives. Arguments and execution correctness are not graded.", "- **Uncertainty:** 10,000 paired task bootstrap resamples, seed 20260919, conditional on each frozen stream; exploratory comparisons without an equivalence margin. Crossing zero does not prove quality preservation.", "- **Ratios:** hit-rate multiples are not latency speedups. Reciprocal-TTFT percentage gains differ from TTFT percentage reductions.", "", "## Limits and output-budget check", "", "| Dataset | Policy | At 512-token limit / 200 | Explicit length finish / 200 |", "|---|---|---:|---:|"]
    for d, name in DATASETS.items():
        for a in ARMS:
            c = cells['gpu2',d,a]
            lines.append(f"| {name} | {LABELS[a]} | {c['quality_requests_at_output_limit']} | {c['quality_finish_reasons'].get('length',0)} |")
    cross = data['cross_device']['cells']
    lines += ["", f"Per-request cache vectors match across all six systems replays in {sum(c['identical_per_request_hits_all_six_systems_replays'] for c in cross)}/12 policy/dataset cells. Quality outputs match on {sum(c['matching_quality_outputs'] for c in cross):,}/2,400 cross-device requests. Replays and devices do not create additional unique evaluation tasks.", "", "- Two previously inspected streams, one arrival order per stream, one model, one KV capacity and sequential traffic. No unseen-task or production-load claim follows.", "- Automatic clocks, temperature and host scheduling can differ despite identical GPU models and KV allocation; comparisons within each GPU are primary.", "- This study has a 512-token quality budget and unique function names. The earlier 128-token study and historical raw-name results are separate; their absolute scores are not pooled here.", "- The fixed-ten follow-up is separate. This report does not isolate ordering gains from menu-selection changes.", "", "## Reproduce", "", "```bash", "uv run scripts/report_improved_trie_comparison.py", "```", "", "This reproduces the report and PNG/SVG from the committed audited summary and [compact request metrics](../trie-baselines-plot-data.json), without a GPU. [Publication hashes](../trie-baselines-publication.json) identify the exact inputs and outputs. Raw responses, telemetry and the experimental source snapshot remain under the ignored raw root and are needed to repeat the independent experimental audit.", "", "To rebuild the compact request metrics from that accepted raw archive:", "", "```bash", "uv run scripts/report_improved_trie_comparison.py \\", "  --raw-root cluster/results/trie-baselines-controlled-20260919-01", "```", "", f"Experimental manifest SHA-256: `{data['manifest_sha256']}`.", "", f"Summary SHA-256: `{digest(SUMMARY)}`.", ""]
    REPORT.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path)
    args = parser.parse_args()
    data, cells, pairs = load_summary()
    if args.raw_root:
        extract_metrics(args.raw_root.resolve(), data)
    compact = load_metrics(cells)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "improved-trie-matched-baselines-v1"})
    count = plot(cells, compact)
    report(data, cells, pairs, count)
    inputs = (SUMMARY, METRICS, Path(__file__).resolve())
    outputs = (REPORT, FIGURE.with_suffix('.png'), FIGURE.with_suffix('.svg'))
    MANIFEST.write_text(json.dumps({"protocol": data['protocol'], "accepted_replays": 96, "window_size": WINDOW, "window_stride": 1, "plotted_windows": count, "matplotlib_version": matplotlib.__version__, "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs}, "outputs": {str(p.relative_to(ROOT)): digest(p) for p in outputs}}, indent=2, sort_keys=True) + '\n')
    print(f'Generated {REPORT.relative_to(ROOT)} and PNG/SVG with {count:,} measured windows')


if __name__ == '__main__':
    main()
