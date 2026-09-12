#!/usr/bin/env python3
"""Throughput vs interactivity frontier for the tool-ordering policies.

One curve per ordering policy, one point per offered request rate, from the
SLA sweep (k=64 dense menus, Qwen3-0.6B, Poisson arrivals, 200 requests each):

  x = interactivity  = 1 / p50 time-to-first-token   (responses per second per user)
  y = aggregate throughput = achieved requests per second

Higher and further right is better, so a policy whose curve sits up and to the
right pushes the frontier out. Writes a standalone SVG (stdlib only).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

RUNS = "cluster/results/eval-validity-20260906-165826/replays"
OUT = Path("reports/figures/throughput-interactivity-frontier.svg")

SERIES = [  # (arm, label, colour)  - categorical slots 1-3 + neutral baseline
    ("tooltrie_v1", "ToolTrie-v1", "#2a78d6"),
    ("cp_online", "ContextPilot", "#eb6834"),
    ("frequency", "frequency", "#1baf7a"),
    ("original", "no reordering", "#8a8983"),
]
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7973", "#e3e2de"
W, H = 940, 632
L, R, T, B = 92, 250, 92, 118


def load() -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    for arm, _, _ in SERIES:
        points = []
        for path in glob.glob(f"{RUNS}/sla-k64-{arm}-rate*.json"):
            run = json.loads(Path(path).read_text())
            points.append((float(path.rsplit("-rate", 1)[1][:-5]),
                           1.0 / run["ttft_seconds"]["p50"], run["achieved_rate"]))
        out[arm] = sorted(points)
    return out


def interp(points, value, axis):
    """Linear interpolation along a monotone-in-rate curve."""
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    a, b = (xs, ys) if axis == "x" else (ys, xs)
    pairs = sorted(zip(a, b))
    for (a0, b0), (a1, b1) in zip(pairs, pairs[1:]):
        if a0 <= value <= a1:
            t = 0 if a1 == a0 else (value - a0) / (a1 - a0)
            return b0 + t * (b1 - b0)
    return None


def main() -> None:
    data = load()
    xmin, xmax = 0.20, 2.95
    ymin, ymax = 0.80, 3.05
    sx = lambda v: L + (v - xmin) / (xmax - xmin) * (W - L - R)
    sy = lambda v: H - B - (v - ymin) / (ymax - ymin) * (H - T - B)

    # Annotation 1 (measured, no interpolation): the same offered load, 3.0 req/s.
    top = {a: max(data[a], key=lambda q: q[0]) for a, _, _ in SERIES}
    # Annotation 2: the offered rate at which p50 crosses 1 s, interpolated the
    # same way as metrics-and-latency-tradeoffs.md Sec4.2 (linear in p50 between
    # the two bracketing rates), so the figure and that table agree.
    def sla_rate(points):
        for (r0, x0, _), (r1, x1, _) in zip(points, points[1:]):
            p50_0, p50_1 = 1 / x0, 1 / x1
            if p50_0 <= 1.0 <= p50_1:
                return r0 + (1.0 - p50_0) / (p50_1 - p50_0) * (r1 - r0)
        return None
    sla = {a: sla_rate(sorted(data[a])) for a, _, _ in SERIES}

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{L}" y="38" font-size="19" font-weight="600" fill="{INK}">ToolTrie-v1 holds the outer throughput–interactivity frontier</text>',
         f'<text x="{L}" y="62" font-size="13.5" fill="{INK2}">One curve per ordering policy, swept over six offered request rates (1.0 to 3.0 req/s). Up and to the right is better.</text>']

    # grid + axes
    for v in [1.0, 1.5, 2.0, 2.5, 3.0]:
        s.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L-12}" y="{sy(v)+4:.1f}" font-size="12" fill="{INK3}" text-anchor="end">{v:.1f}</text>')
    for v in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]:
        s.append(f'<line x1="{sx(v):.1f}" y1="{T}" x2="{sx(v):.1f}" y2="{H-B}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{sx(v):.1f}" y="{H-B+22:.0f}" font-size="12" fill="{INK3}" text-anchor="middle">{v:g}</text>')
    s.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-B+50:.0f}" font-size="13" fill="{INK2}" text-anchor="middle">interactivity: responses per second per user (1 / p50 time to first token)</text>')
    s.append(f'<text transform="translate(26,{(T+H-B)/2:.0f}) rotate(-90)" font-size="13" fill="{INK2}" text-anchor="middle">aggregate throughput (requests per second)</text>')

    # curves
    for arm, label, colour in SERIES:
        pts = data[arm]
        path = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (_, x, y) in enumerate(pts))
        s.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>')
        for _, x, y in pts:
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>')
        if arm in ("tooltrie_v1", "original"):  # selective labels; legend carries the rest
            _, x, y = max(pts, key=lambda q: q[0])     # the 3.0 req/s end
            dy = -26 if arm == "tooltrie_v1" else 30
            s.append(f'<line x1="{sx(x):.1f}" y1="{sy(y)+(6 if dy>0 else -6):.1f}" x2="{sx(x):.1f}" y2="{sy(y)+dy*0.7:.1f}" stroke="{colour}" stroke-width="1"/>')
            anchor = "start" if sx(x) < L + 60 else "middle"
            s.append(f'<text x="{sx(x) + (-8 if anchor == "start" else 0):.1f}" y="{sy(y)+dy:.1f}" font-size="12.5" font-weight="600" fill="{INK}" text-anchor="{anchor}">{label}</text>')

    # annotation 1: same offered load, opposite ends of the interactivity axis
    yv = (sy(top["tooltrie_v1"][2]) + sy(top["original"][2])) / 2
    x1, x2 = sx(top["original"][1]), sx(top["tooltrie_v1"][1])
    s.append(f'<line x1="{x1:.1f}" y1="{yv:.1f}" x2="{x2:.1f}" y2="{yv:.1f}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for x in (x1, x2):
        s.append(f'<line x1="{x:.1f}" y1="{yv-6:.1f}" x2="{x:.1f}" y2="{yv+6:.1f}" stroke="{INK2}" stroke-width="1.5"/>')
    gain = 100 * (top["tooltrie_v1"][1] / top["original"][1] - 1)
    tx = max((x1 + x2) / 2, L + 96)
    s.append(f'<text x="{tx:.1f}" y="{yv-16:.1f}" font-size="13" font-weight="600" fill="{INK}" text-anchor="middle">+{gain:.0f}% interactivity at 3.0 req/s</text>')

    # annotation 2: the 1-second budget line and the throughput each policy holds within it
    xv = sx(1.0)
    s.append(f'<line x1="{xv:.1f}" y1="{T}" x2="{xv:.1f}" y2="{H-B}" stroke="{INK3}" stroke-width="1.5" stroke-dasharray="3 4"/>')
    s.append(f'<text x="{xv+8:.1f}" y="{T+16:.0f}" font-size="11.5" fill="{INK2}">1 s time to first token</text>')
    for a, _, colour in SERIES:
        if sla[a]:
            s.append(f'<circle cx="{xv:.1f}" cy="{sy(sla[a]):.1f}" r="3" fill="{colour}"/>')
    sgain = 100 * (sla["tooltrie_v1"] / sla["original"] - 1)
    ay = sy(2.30)
    s.append(f'<line x1="{xv:.1f}" y1="{sy(sla["tooltrie_v1"]):.1f}" x2="{xv+26:.1f}" y2="{ay-14:.1f}" stroke="{INK3}" stroke-width="1"/>')
    s.append(f'<text x="{xv+30:.1f}" y="{ay-10:.1f}" font-size="12.5" font-weight="600" fill="{INK}">+{sgain:.1f}% throughput within a 1 s budget</text>')
    s.append(f'<text x="{xv+30:.1f}" y="{ay+8:.1f}" font-size="11.5" fill="{INK2}">{sla["tooltrie_v1"]:.1f} vs {sla["original"]:.1f} req/s sustained</text>')

    # legend
    lx0, ly0 = W - R + 24, T + 8
    s.append(f'<text x="{lx0}" y="{ly0}" font-size="12" font-weight="600" fill="{INK2}">ordering policy</text>')
    for i, (_, label, colour) in enumerate(SERIES):
        y = ly0 + 24 + i * 22
        s.append(f'<line x1="{lx0}" y1="{y-4}" x2="{lx0+22}" y2="{y-4}" stroke="{colour}" stroke-width="2"/>')
        s.append(f'<circle cx="{lx0+11}" cy="{y-4}" r="4.5" fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>')
        s.append(f'<text x="{lx0+30}" y="{y}" font-size="12.5" fill="{INK}">{label}</text>')
    s.append(f'<text x="{lx0}" y="{ly0+140}" font-size="11.5" fill="{INK2}">Qwen3-0.6B, one RTX 3090,</text>')
    s.append(f'<text x="{lx0}" y="{ly0+157}" font-size="11.5" fill="{INK2}">64 retrieved tools per request,</text>')
    s.append(f'<text x="{lx0}" y="{ly0+174}" font-size="11.5" fill="{INK2}">dense retrieval, vLLM 0.26.0.</text>')
    s.append(f'<text x="{L}" y="{H-30}" font-size="11.5" fill="{INK3}">Offered rates 1.0, 1.5, 2.0, 2.5, 2.75, 3.0 req/s, 200 requests each. At 3.0 req/s the median first token arrives after {1/top["tooltrie_v1"][1]:.2f} s with ToolTrie-v1</text>')
    s.append(f'<text x="{L}" y="{H-12}" font-size="11.5" fill="{INK3}">and {1/top["original"][1]:.2f} s with no reordering. Source: cluster/results/eval-validity-20260906-165826 (six arms x six rates).</text>')
    s.append("</svg>")
    OUT.write_text("\n".join(s))
    print(f"{OUT}\n  at 3.0 req/s offered: " + ", ".join(f"{a} p50 {1/top[a][1]:.2f}s (ach {top[a][2]:.2f})" for a, _, _ in SERIES)
          + f"\n  v1 interactivity vs no reordering: +{gain:.1f}%"
          + "\n  max throughput within p50 < 1 s: " + ", ".join(f"{a} {sla[a]:.3f}" for a, _, _ in SERIES)
          + f" | v1 +{sgain:.1f}% vs no reordering")


if __name__ == "__main__":
    main()
