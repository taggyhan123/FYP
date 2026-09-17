#!/usr/bin/env python3
"""Throughput vs interactivity frontier for the tool-ordering policies.

For each ordering policy, from an SLA rate sweep (k=64 dense menus, Poisson
arrivals, 200 requests per offered rate):

  cloud  one semi-transparent dot per request, describing the window of the
         K requests that finished nearest in time to it (K/2 either side):
           x = 1 / median time-to-first-token across that window
           y = completions per second across that window
         vLLM has no per-request throughput, so both axes are measured over the
         same window. The K/2 earliest and latest completions of each run have
         no full window and are left out.
  trend  one vertex per offered rate: 1 / p50 TTFT against the run's achieved
         request rate - the aggregate trace the cloud scatters around.

L-shaped brackets compare ToolTrie-v1 with no reordering at a vertex of the
no-reordering trend: the vertical leg is the throughput gain at the same
interactivity, the horizontal leg the interactivity gain at the same
throughput. Both are read off the two trend lines by linear interpolation.
Writes a standalone SVG (stdlib only).

  --model 0.6b   Qwen3-0.6B sweep, offered 1.0-3.0 req/s   (eval-validity-20260906-165826)
  --model 4b     Qwen3-4B sweep,   offered 0.25-0.8 req/s  (sla-4b-single-20260912-175313)
  --model 4b-p10-conc
                 Qwen3-4B closed-loop concurrency sweep, retrieve-64 / show-10,
                 N = 1-32 in flight (conc-4b-p10-single-20260916-214417), with a
                 dashed full-reuse ceiling (conc-4b-p10-ceiling-20260917-010805)
                 and frequency left out; log axes, as the ceiling spans ~15x
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
    ("full_reuse_ceiling", "full-reuse ceiling", "#3b3a37"),
]
REFERENCE = {"full_reuse_ceiling"}  # drawn dashed, without a cloud; not an ordering policy
FOCUS, BASELINE = "tooltrie_v1", "original"
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7973", "#e3e2de"
W, H = 940, 660
L, R, T, B = 84, 228, 96, 128
K = 40  # completions per local-throughput window

CONFIGS = {
    "0.6b": dict(
        runs="cluster/results/eval-validity-20260906-165826/replays",
        pattern="sla-k64-{arm}-rate*.json",
        out="reports/figures/throughput-interactivity-frontier.svg",
        title="ToolTrie-v1 holds the outer throughput–interactivity frontier at Qwen3-0.6B",
        xlim=(0.0, 3.5), ylim=(0.7, 6.2),
        xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], yticks=[1, 2, 3, 4, 5, 6], yfmt=".0f",
        brackets=[2.5, 2.0],
        caption=["Qwen3-0.6B, one RTX 3090,", "64 retrieved tools per request,",
                 "dense retrieval, vLLM 0.26.0."],
        rates_text="1.0, 1.5, 2.0, 2.5, 2.75, 3.0",
        source="cluster/results/eval-validity-20260906-165826",
    ),
    "4b": dict(
        runs="cluster/results/sla-4b-single-20260912-175313/replays",
        pattern="sla4b1-k64-{arm}-rate*.json",
        out="reports/figures/throughput-interactivity-frontier-4b.svg",
        title="ToolTrie-v1 holds the outer throughput–interactivity frontier at Qwen3-4B",
        xscale="log", axis_x_note=" (log scale)",
        xlim=(0.05, 1.0), ylim=(0.22, 0.82),
        xticks=[0.05, 0.1, 0.2, 0.5, 1.0], yticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], yfmt=".1f",
        brackets=[0.7, 0.5],
        caption=["Qwen3-4B, one RTX 3090", "(all arms on the same GPU),",
                 "64 retrieved tools per request,", "dense retrieval, vLLM 0.26.0."],
        rates_text="0.25, 0.4, 0.5, 0.6, 0.7, 0.8",
        source="cluster/results/sla-4b-single-20260912-175313",
    ),
    "4b-p10-conc": dict(
        arms=["tooltrie_v1", "cp_online", "original", "full_reuse_ceiling"],
        sources={
            "tooltrie_v1": ("cluster/results/conc-4b-p10-single-20260916-214417/replays", "conc4bp10-tooltrie_v1-c*.json"),
            "cp_online": ("cluster/results/conc-4b-p10-single-20260916-214417/replays", "conc4bp10-cp_online-c*.json"),
            "original": ("cluster/results/conc-4b-p10-single-20260916-214417/replays", "conc4bp10-original-c*.json"),
            "full_reuse_ceiling": ("cluster/results/conc-4b-p10-ceiling-20260917-010805/replays", "conc4bp10-full_reuse_ceiling-c*.json"),
        },
        key="-c",
        out="reports/figures/throughput-interactivity-frontier-4b-p10.svg",
        title="ToolTrie-v1 leads every ordering; a full-reuse ceiling shows the headroom",
        subtitle="Dots: one per request, over its 40-request window. Lines: run medians per concurrency level. "
                 "Brackets: ToolTrie-v1 vs no reordering.",
        xscale="log", yscale="log",
        xlim=(0.25, 40.0), ylim=(0.9, 30.0),
        xticks=[0.5, 1, 2, 5, 10, 20], yticks=[1, 2, 5, 10, 20], yfmt="g",
        brackets=[4, 16],
        caption=["Qwen3-4B, one RTX 3090", "(all arms on the same GPU),", "retrieve 64 tools, show 10,",
                 "dense retrieval, vLLM 0.26.0.", "", "Ceiling: every request carries",
                 "the same 10 tools, so its whole", "tool list is cached. Not an",
                 "ordering; accuracy not measured."],
        load_text="Closed loop, N = 1, 2, 4, 8, 16, 32 requests in flight",
        axis_note=" (log scale)",
        source="conc-4b-p10-single-20260916-214417 + conc-4b-p10-ceiling-20260917-010805",
    ),
}


def series_arms(cfg):
    return cfg.get("arms", [a for a, _, _ in SERIES if a not in REFERENCE])


def load(cfg):
    """Per arm: (trend vertices [(rate, x, y)], cloud [(x, y)])."""
    trend, cloud = {}, {}
    key = cfg.get("key", "-rate")
    for arm in series_arms(cfg):
        vertices, dots = [], []
        runs, pattern = cfg["sources"][arm] if "sources" in cfg else (cfg["runs"], cfg["pattern"].format(arm=arm))
        for path in glob.glob(f"{runs}/{pattern}"):
            run = json.loads(Path(path).read_text())
            vertices.append((float(path.rsplit(key, 1)[1][:-5]),
                             1.0 / run["ttft_seconds"]["p50"], run["achieved_rate"]))
            done = sorted((r["dispatch_offset"] + r["e2e_seconds"], r["ttft_seconds"])
                          for r in run["results"])
            for i in range(K // 2, len(done) - K // 2):
                window = done[i - K // 2:i + K // 2 + 1]
                span = window[-1][0] - window[0][0]
                if span > 0:
                    ttfts = sorted(w[1] for w in window)
                    dots.append((1.0 / ttfts[len(ttfts) // 2], K / span))
        if not vertices:
            raise SystemExit(f"no replays for {arm} under {cfg['runs']}")
        trend[arm], cloud[arm] = sorted(vertices), dots
    return trend, cloud


def interp(points, value):
    """Piecewise-linear lookup in [(key, value)] pairs; None outside the range."""
    pts = sorted(points)
    for (k0, v0), (k1, v1) in zip(pts, pts[1:]):
        if k0 <= value <= k1:
            return v0 if k1 == k0 else v0 + (value - k0) / (k1 - k0) * (v1 - v0)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(CONFIGS), default="0.6b")
    args = parser.parse_args()
    cfg = CONFIGS[args.model]
    trend, cloud = load(cfg)
    (xmin, xmax), (ymin, ymax) = cfg["xlim"], cfg["ylim"]
    from math import log10
    def axis(lo, hi, start, span, scale, flip):
        if scale == "log":
            a, b = log10(lo), log10(hi)
            f = lambda v: (log10(v) - a) / (b - a)
        else:
            f = lambda v: (v - lo) / (hi - lo)
        return (lambda v: start - f(v) * span) if flip else (lambda v: start + f(v) * span)
    sx = axis(xmin, xmax, L, W - L - R, cfg.get("xscale", "linear"), False)
    sy = axis(ymin, ymax, H - B, H - T - B, cfg.get("yscale", "linear"), True)
    series = [s_ for s_ in SERIES if s_[0] in series_arms(cfg)]
    inside = lambda x, y: xmin <= x <= xmax and ymin <= y <= ymax

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<defs><clipPath id="plot"><rect x="{L}" y="{T}" width="{W-L-R}" height="{H-T-B}"/></clipPath></defs>',
         f'<text x="{L}" y="38" font-size="19" font-weight="600" fill="{INK}">{cfg["title"]}</text>',
         f'<text x="{L}" y="62" font-size="13" fill="{INK2}">' + cfg.get("subtitle",
             "Dots: one per request, over its 40-request window. Lines: run medians per offered rate. "
             "Brackets: ToolTrie-v1 vs no reordering.") + '</text>']

    # grid + axes
    for v in cfg["yticks"]:
        s.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L-12}" y="{sy(v)+4:.1f}" font-size="12" fill="{INK3}" text-anchor="end">{v:{cfg["yfmt"]}}</text>')
    for v in cfg["xticks"]:
        s.append(f'<line x1="{sx(v):.1f}" y1="{T}" x2="{sx(v):.1f}" y2="{H-B}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{sx(v):.1f}" y="{H-B+22:.0f}" font-size="12" fill="{INK3}" text-anchor="middle">{v:g}</text>')
    s.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-B+50:.0f}" font-size="13" fill="{INK2}" text-anchor="middle">interactivity: responses per second per user (1 / time to first token){cfg.get("axis_note", cfg.get("axis_x_note", ""))}</text>')
    s.append(f'<text transform="translate(24,{(T+H-B)/2:.0f}) rotate(-90)" font-size="13" fill="{INK2}" text-anchor="middle">throughput (requests per second){cfg.get("axis_note", "")}</text>')

    # clouds: rivals first, the focal policy last so its swath sits on top
    order = sorted(series, key=lambda sr: sr[0] == FOCUS)
    s.append('<g clip-path="url(#plot)">')
    for arm, _, colour in order:
        if arm in REFERENCE:
            continue
        s.append(f'<g fill="{colour}" fill-opacity="0.22">')
        s.extend(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.3"/>' for x, y in cloud[arm] if inside(x, y))
        s.append('</g>')
    # trends on top of every cloud, each with a surface halo so it reads through the noise
    for arm, _, colour in order:
        lead = arm == FOCUS
        path = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (_, x, y) in enumerate(trend[arm]))
        s.append(f'<path d="{path}" fill="none" stroke="{SURFACE}" stroke-width="{7 if lead else 5}" stroke-linejoin="round" opacity="0.85"/>')
        if arm in REFERENCE:
            s.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.2" stroke-dasharray="7 4" stroke-linejoin="round"/>')
            for _, x, y in trend[arm]:
                s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.8" fill="{SURFACE}" stroke="{colour}" stroke-width="1.8"/>')
            continue
        s.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{3.2 if lead else 2.2}" stroke-linejoin="round"/>')
        for _, x, y in trend[arm]:
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{5 if lead else 3.8}" fill="{colour}" stroke="{SURFACE}" stroke-width="1.8"/>')
    s.append('</g>')

    # L brackets anchored on no-reordering vertices
    focal = trend[FOCUS]
    y_at_x = [(x, y) for _, x, y in focal]
    x_at_y = [(y, x) for _, x, y in focal]
    report = []
    for rate in cfg["brackets"]:
        _, ax, ay = next(v for v in trend[BASELINE] if abs(v[0] - rate) < 1e-9)
        by, cx = interp(y_at_x, ax), interp(x_at_y, ay)
        dash = f'stroke="{INK}" stroke-width="1.4" stroke-dasharray="5 3" fill="none"'
        if by is not None:   # vertical leg: throughput at the same interactivity
            s.append(f'<line x1="{sx(ax):.1f}" y1="{sy(ay):.1f}" x2="{sx(ax):.1f}" y2="{sy(by):.1f}" {dash}/>')
            s.append(f'<line x1="{sx(ax)-5:.1f}" y1="{sy(by):.1f}" x2="{sx(ax)+5:.1f}" y2="{sy(by):.1f}" stroke="{INK}" stroke-width="1.4"/>')
            dy = 100 * (by / ay - 1)
            s.append(f'<text x="{sx(ax)-8:.1f}" y="{(sy(ay)+sy(by))/2+4:.1f}" font-size="12.5" font-weight="600" fill="{INK}" '
                     f'text-anchor="end" paint-order="stroke" stroke="{SURFACE}" stroke-width="6">{dy:+.0f}% throughput</text>')
            report.append(f"at {rate:g} no-reordering vertex: throughput {dy:+.1f}%")
        if cx is not None:   # horizontal leg: interactivity at the same throughput
            s.append(f'<line x1="{sx(ax):.1f}" y1="{sy(ay):.1f}" x2="{sx(cx):.1f}" y2="{sy(ay):.1f}" {dash}/>')
            s.append(f'<line x1="{sx(cx):.1f}" y1="{sy(ay)-5:.1f}" x2="{sx(cx):.1f}" y2="{sy(ay)+5:.1f}" stroke="{INK}" stroke-width="1.4"/>')
            dx = 100 * (cx / ax - 1)
            s.append(f'<text x="{(sx(ax)+sx(cx))/2:.1f}" y="{sy(ay)+19:.1f}" font-size="12.5" font-weight="600" fill="{INK}" '
                     f'text-anchor="middle" paint-order="stroke" stroke="{SURFACE}" stroke-width="6">{dx:+.0f}% interactivity</text>')
            report.append(f"at {rate:g} no-reordering vertex: interactivity {dx:+.1f}%")
        s.append(f'<circle cx="{sx(ax):.1f}" cy="{sy(ay):.1f}" r="3" fill="{INK}"/>')

    # legend
    lx0, ly0 = W - R + 22, T + 8
    s.append(f'<text x="{lx0}" y="{ly0}" font-size="12" font-weight="600" fill="{INK2}">ordering policy</text>')
    for i, (arm, label, colour) in enumerate(series):
        y = ly0 + 24 + i * 22
        ref = arm in REFERENCE
        s.append(f'<line x1="{lx0}" y1="{y-4}" x2="{lx0+22}" y2="{y-4}" stroke="{colour}" stroke-width="{2.2 if ref else 2.6}"'
                 + (' stroke-dasharray="5 3"' if ref else '') + '/>')
        s.append(f'<circle cx="{lx0+11}" cy="{y-4}" r="4" fill="{SURFACE if ref else colour}" stroke="{colour if ref else SURFACE}" stroke-width="1.8"/>')
        s.append(f'<text x="{lx0+30}" y="{y}" font-size="12.5" fill="{INK}">{label}</text>')
    for i, line in enumerate(cfg["caption"]):
        s.append(f'<text x="{lx0}" y="{ly0+42+22*len(series)+17*i}" font-size="11.5" fill="{INK2}">{line}</text>')

    n = sum(len(c) for arm, c in cloud.items() if arm not in REFERENCE)
    s.append(f'<text x="{L}" y="{H-44}" font-size="11.5" fill="{INK3}">{cfg.get("load_text", "Offered rates " + cfg.get("rates_text", "") + " req/s")}, 200 requests each ({n:,} dots). Each dot is the {K} requests that finished nearest to one request: median</text>')
    s.append(f'<text x="{L}" y="{H-27}" font-size="11.5" fill="{INK3}">interactivity and completion rate over that window. The first and last {K//2} completions per run lack a full window. Bracket gaps are read off the</text>')
    clipped = sum(1 for arm, c in cloud.items() if arm not in REFERENCE for x, y in c if not inside(x, y))
    s.append(f'<text x="{L}" y="{H-10}" font-size="11.5" fill="{INK3}">median lines by linear interpolation. Source: {cfg["source"]}.'
             + (f' {clipped} dots off-axis.' if clipped else '') + '</text>')
    s.append("</svg>")
    Path(cfg["out"]).write_text("\n".join(s))
    print(f"{cfg['out']}: {n} dots, {clipped} outside the axes\n  " + "\n  ".join(report))


if __name__ == "__main__":
    main()
