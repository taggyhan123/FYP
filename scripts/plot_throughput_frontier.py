#!/usr/bin/env python3
"""Throughput vs interactivity frontier for the tool-ordering policies.

One curve per ordering policy, one point per offered request rate, from an SLA
rate sweep (k=64 dense menus, Poisson arrivals, 200 requests each):

  x = interactivity  = 1 / p50 time-to-first-token   (responses per second per user)
  y = aggregate throughput = achieved requests per second

Higher and further right is better, so a policy whose curve sits up and to the
right pushes the frontier out. Writes a standalone SVG (stdlib only).

  --model 0.6b   Qwen3-0.6B sweep, offered 1.0-3.0 req/s   (eval-validity-20260906-165826)
  --model 4b     Qwen3-4B sweep,   offered 0.25-0.8 req/s  (sla-4b-single-20260912-175313)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

SERIES = [  # (arm, label, colour)  - categorical slots 1-3 + neutral baseline
    ("tooltrie_v1", "ToolTrie-v1", "#2a78d6"),
    ("cp_online", "ContextPilot", "#eb6834"),
    ("frequency", "frequency", "#1baf7a"),
    ("original", "no reordering", "#8a8983"),
]
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7973", "#e3e2de"
W, H = 940, 632
L, R, T, B = 92, 250, 92, 118

CONFIGS = {
    "0.6b": dict(
        runs="cluster/results/eval-validity-20260906-165826/replays",
        pattern="sla-k64-{arm}-rate*.json",
        out="reports/figures/throughput-interactivity-frontier.svg",
        title="ToolTrie-v1 holds the outer throughput–interactivity frontier",
        subtitle="One curve per ordering policy, swept over six offered request rates "
                 "(1.0 to 3.0 req/s). Up and to the right is better.",
        xscale="linear",
        xlim=(0.20, 2.95), ylim=(0.80, 3.05),
        xticks=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5], yticks=[1.0, 1.5, 2.0, 2.5, 3.0], yfmt=".1f",
        budget=1.0, budget_label="1 s time to first token", budget_y=2.30,
        labelled=("tooltrie_v1", "original"), label_dy={"tooltrie_v1": -26, "original": 30},
        caption=["Qwen3-0.6B, one RTX 3090,", "64 retrieved tools per request,",
                 "dense retrieval, vLLM 0.26.0."],
        rates_text="1.0, 1.5, 2.0, 2.5, 2.75, 3.0",
        source="cluster/results/eval-validity-20260906-165826 (six arms x six rates)",
        footnote_extra="",
    ),
    "4b": dict(
        runs="cluster/results/sla-4b-single-20260912-175313/replays",
        pattern="sla4b1-k64-{arm}-rate*.json",
        out="reports/figures/throughput-interactivity-frontier-4b.svg",
        focus="tooltrie_v1",
        title="ToolTrie-v1 serves the same load 1.5x faster at Qwen3-4B",
        subtitle="Shaded band: the interactivity ToolTrie-v1 gains over the best rival at each offered rate. "
                 "Fastest at all six rates; log x-axis.",
        xscale="log",
        xlim=(0.052, 0.84), ylim=(0.20, 0.79),
        xticks=[0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75], yticks=[0.25, 0.4, 0.5, 0.6, 0.7], yfmt=".2f",
        budget=0.5, budget_label="2 s time to first token", budget_y=0.66,
        labelled=("tooltrie_v1", "original"), label_dy={"tooltrie_v1": -26, "original": 30},
        caption=["Qwen3-4B, one RTX 3090 (GPU 2),", "64 retrieved tools per request,",
                 "dense retrieval, vLLM 0.26.0."],
        rates_text="0.25, 0.4, 0.5, 0.6, 0.7, 0.8",
        source="cluster/results/sla-4b-single-20260912-175313",
        xaxis_note=" - log scale",
        footnote_extra=" No arm reaches a 1 s p50 at 4B, so the 2 s budget is post-hoc.",
    ),
}


def load(cfg) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    for arm, _, _ in SERIES:
        points = []
        for path in glob.glob(f"{cfg['runs']}/{cfg['pattern'].format(arm=arm)}"):
            run = json.loads(Path(path).read_text())
            points.append((float(path.rsplit("-rate", 1)[1][:-5]),
                           1.0 / run["ttft_seconds"]["p50"], run["achieved_rate"]))
        if not points:
            raise SystemExit(f"no replays for {arm} under {cfg['runs']}")
        out[arm] = sorted(points)
    return out


def budget_rate(points, threshold):
    """Offered rate where p50 TTFT crosses the budget, interpolated linearly in
    p50 between the two bracketing rates - the same rule as
    metrics-and-latency-tradeoffs.md Sec4.2, so figure and table agree."""
    for (r0, x0, _), (r1, x1, _) in zip(points, points[1:]):
        p50_0, p50_1 = 1 / x0, 1 / x1
        if p50_0 <= 1 / threshold <= p50_1:
            return r0 + (1 / threshold - p50_0) / (p50_1 - p50_0) * (r1 - r0)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(CONFIGS), default="0.6b")
    args = parser.parse_args()
    cfg = CONFIGS[args.model]
    data = load(cfg)
    out = Path(cfg["out"])
    xmin, xmax = cfg["xlim"]
    ymin, ymax = cfg["ylim"]
    if cfg.get("xscale") == "log":
        from math import log10
        lo, hi = log10(xmin), log10(xmax)
        sx = lambda v: L + (log10(v) - lo) / (hi - lo) * (W - L - R)
    else:
        sx = lambda v: L + (v - xmin) / (xmax - xmin) * (W - L - R)
    sy = lambda v: H - B - (v - ymin) / (ymax - ymin) * (H - T - B)

    # Annotation 1 (measured, no interpolation): the highest offered load.
    top = {a: max(data[a], key=lambda q: q[0]) for a, _, _ in SERIES}
    # Annotation 2: the offered rate each policy sustains inside the TTFT budget.
    sla = {a: budget_rate(sorted(data[a]), cfg["budget"]) for a, _, _ in SERIES}

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{L}" y="38" font-size="19" font-weight="600" fill="{INK}">{cfg["title"]}</text>',
         f'<text x="{L}" y="62" font-size="13.5" fill="{INK2}">{cfg["subtitle"]}</text>']

    # grid + axes
    for v in cfg["yticks"]:
        s.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L-12}" y="{sy(v)+4:.1f}" font-size="12" fill="{INK3}" text-anchor="end">{v:{cfg["yfmt"]}}</text>')
    for v in cfg["xticks"]:
        s.append(f'<line x1="{sx(v):.1f}" y1="{T}" x2="{sx(v):.1f}" y2="{H-B}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{sx(v):.1f}" y="{H-B+22:.0f}" font-size="12" fill="{INK3}" text-anchor="middle">{v:g}</text>')
    s.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-B+50:.0f}" font-size="13" fill="{INK2}" text-anchor="middle">interactivity: responses per second per user (1 / p50 time to first token){cfg.get("xaxis_note","")}</text>')
    s.append(f'<text transform="translate(26,{(T+H-B)/2:.0f}) rotate(-90)" font-size="13" fill="{INK2}" text-anchor="middle">aggregate throughput (requests per second)</text>')

    # curves - rivals first and lighter, the focal policy heaviest and on top
    focus = cfg.get("focus")
    for arm, label, colour in sorted(SERIES, key=lambda s: s[0] == focus):
        lead = arm == focus
        pts = data[arm]
        path = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (_, x, y) in enumerate(pts))
        s.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{3.2 if lead else 1.8}" '
                 f'stroke-linejoin="round" opacity="{1 if lead or not focus else 0.62}"/>')
        for _, x, y in pts:
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{5.5 if lead else 4}" fill="{colour}" '
                     f'stroke="{SURFACE}" stroke-width="2" opacity="{1 if lead or not focus else 0.62}"/>')
        if arm in cfg["labelled"]:  # selective labels; legend carries the rest
            _, x, y = max(pts, key=lambda q: q[0])     # the saturated end
            dy = cfg["label_dy"][arm]
            s.append(f'<line x1="{sx(x):.1f}" y1="{sy(y)+(6 if dy>0 else -6):.1f}" x2="{sx(x):.1f}" y2="{sy(y)+dy*0.7:.1f}" stroke="{colour}" stroke-width="1"/>')
            anchor = "start" if sx(x) < L + 60 else "middle"
            s.append(f'<text x="{sx(x) + (-8 if anchor == "start" else 0):.1f}" y="{sy(y)+dy:.1f}" font-size="12.5" font-weight="600" fill="{INK}" text-anchor="{anchor}">{label}</text>')

    if cfg.get("focus"):  # the band v1 opens over the best-performing rival at each rate
        rival = {r: min((data[a][i][1] for a, _, _ in SERIES if a != cfg["focus"]))
                 for i, (r, _, _) in enumerate(data[cfg["focus"]])}
        lead_pts = data[cfg["focus"]]
        band = ([f"M{sx(x):.1f},{sy(y):.1f}" if i == 0 else f"L{sx(x):.1f},{sy(y):.1f}"
                 for i, (_, x, y) in enumerate(lead_pts)]
                + [f"L{sx(rival[r]):.1f},{sy(y):.1f}" for r, _, y in reversed(lead_pts)] + ["Z"])
        s.append(f'<path d="{" ".join(band)}" fill="#2a78d6" opacity="0.10"/>')

    # annotation 1: same offered load, opposite ends of the interactivity axis
    yv = (sy(top["tooltrie_v1"][2]) + sy(top["original"][2])) / 2
    x1, x2 = sx(top["original"][1]), sx(top["tooltrie_v1"][1])
    s.append(f'<line x1="{x1:.1f}" y1="{yv:.1f}" x2="{x2:.1f}" y2="{yv:.1f}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for x in (x1, x2):
        s.append(f'<line x1="{x:.1f}" y1="{yv-6:.1f}" x2="{x:.1f}" y2="{yv+6:.1f}" stroke="{INK2}" stroke-width="1.5"/>')
    gain = 100 * (top["tooltrie_v1"][1] / top["original"][1] - 1)
    tx = max((x1 + x2) / 2, L + 96)
    s.append(f'<text x="{tx:.1f}" y="{yv-16:.1f}" font-size="13" font-weight="600" fill="{INK}" text-anchor="middle">+{gain:.0f}% interactivity at {top["tooltrie_v1"][0]:.1f} req/s</text>')

    # annotation 2: the latency budget line and the throughput each policy holds within it
    xv = sx(cfg["budget"])
    s.append(f'<line x1="{xv:.1f}" y1="{T}" x2="{xv:.1f}" y2="{H-B}" stroke="{INK3}" stroke-width="1.5" stroke-dasharray="3 4"/>')
    s.append(f'<text x="{xv+8:.1f}" y="{T+16:.0f}" font-size="11.5" fill="{INK2}">{cfg["budget_label"]}</text>')
    for a, _, colour in SERIES:
        if sla[a]:
            s.append(f'<circle cx="{xv:.1f}" cy="{sy(sla[a]):.1f}" r="3" fill="{colour}"/>')
    if sla["tooltrie_v1"] and sla["original"]:
        sgain = 100 * (sla["tooltrie_v1"] / sla["original"] - 1)
        ay = sy(cfg["budget_y"])
        # keep the callout inside the plot: flip it left of the line when the
        # line sits in the right half, where the legend column would collide
        right = xv > L + 0.55 * (W - L - R)
        tx, anchor = (xv - 30, "end") if right else (xv + 30, "start")
        s.append(f'<line x1="{xv:.1f}" y1="{sy(sla["tooltrie_v1"]):.1f}" x2="{xv + (-26 if right else 26):.1f}" y2="{ay-14:.1f}" stroke="{INK3}" stroke-width="1"/>')
        s.append(f'<text x="{tx:.1f}" y="{ay-10:.1f}" font-size="12.5" font-weight="600" fill="{INK}" text-anchor="{anchor}">{sgain:+.1f}% throughput within the budget</text>')
        s.append(f'<text x="{tx:.1f}" y="{ay+8:.1f}" font-size="11.5" fill="{INK2}" text-anchor="{anchor}">{sla["tooltrie_v1"]:.2f} vs {sla["original"]:.2f} req/s sustained</text>')

    # legend
    lx0, ly0 = W - R + 24, T + 8
    s.append(f'<text x="{lx0}" y="{ly0}" font-size="12" font-weight="600" fill="{INK2}">ordering policy</text>')
    for i, (_, label, colour) in enumerate(SERIES):
        y = ly0 + 24 + i * 22
        s.append(f'<line x1="{lx0}" y1="{y-4}" x2="{lx0+22}" y2="{y-4}" stroke="{colour}" stroke-width="2"/>')
        s.append(f'<circle cx="{lx0+11}" cy="{y-4}" r="4.5" fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>')
        s.append(f'<text x="{lx0+30}" y="{y}" font-size="12.5" fill="{INK}">{label}</text>')
    for i, line in enumerate(cfg["caption"]):
        s.append(f'<text x="{lx0}" y="{ly0+140+17*i}" font-size="11.5" fill="{INK2}">{line}</text>')
    s.append(f'<text x="{L}" y="{H-30}" font-size="11.5" fill="{INK3}">Offered rates {cfg["rates_text"]} req/s, 200 requests each. At {top["tooltrie_v1"][0]:.1f} req/s the median first token arrives after {1/top["tooltrie_v1"][1]:.2f} s with ToolTrie-v1</text>')
    s.append(f'<text x="{L}" y="{H-12}" font-size="11.5" fill="{INK3}">and {1/top["original"][1]:.2f} s with no reordering. Source: {cfg["source"]}.{cfg["footnote_extra"]}</text>')
    s.append("</svg>")
    out.write_text("\n".join(s))
    print(f"{out}\n  at {top['tooltrie_v1'][0]:g} req/s offered: " + ", ".join(f"{a} p50 {1/top[a][1]:.2f}s (ach {top[a][2]:.2f})" for a, _, _ in SERIES)
          + f"\n  v1 interactivity vs no reordering: {gain:+.1f}%"
          + f"\n  max throughput within p50 < {1/cfg['budget']:g} s: "
          + ", ".join(f"{a} {sla[a]:.3f}" if sla[a] else f"{a} n/a" for a, _, _ in SERIES))


if __name__ == "__main__":
    main()
