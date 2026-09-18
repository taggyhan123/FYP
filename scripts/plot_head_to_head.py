#!/usr/bin/env python3
"""ToolTrie-v1 head to head with one rival per figure.

  --rival cp_online   reports/figures/head-to-head-contextpilot.svg
  --rival original    reports/figures/head-to-head-no-reordering.svg

Six panels, each comparing exactly two policies:

  1-2  p50 time to first token vs offered load, 64 tools, open loop:
       Qwen3-0.6B (eval-validity-20260906-165826) and Qwen3-4B
       (sla-4b-single-20260912-175313)
  3-4  retrieve 64 / show 10, Qwen3-4B, closed loop at N in flight
       (conc-4b-p10-single-20260916-214417): throughput and p50 TTFT
  5    prompt-token cache hit rate per setting and model
  6    end-to-end F1 per setting and model, with the paired test from
       scripts/summarize_hit_miss_f1.py

The area between the two lines is shaded in ToolTrie-v1's colour where it is
better and in the rival's colour where the rival is better, split exactly at
crossings. Rows in panels 5-6 are tinted the same way; an F1 row counts as a
win or loss only at |z| >= 2 on the paired test, otherwise it is a tie.
Writes standalone SVG (stdlib only).
"""
from __future__ import annotations

import argparse
import glob
import json
from math import log10
from pathlib import Path

SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7973", "#e3e2de"
V1 = ("tooltrie_v1", "ToolTrie-v1", "#2a78d6")
RIVALS = {"cp_online": ("ContextPilot", "#eb6834", "contextpilot"),
          "original": ("no reordering", "#8a8983", "no-reordering")}
W, H = 1180, 1250
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"

SLA = {"0.6B": ("cluster/results/eval-validity-20260906-165826/replays/sla-k64-{arm}-rate{r}.json",
                ["1.0", "1.5", "2.0", "2.5", "2.75", "3.0"]),
       "4B": ("cluster/results/sla-4b-single-20260912-175313/replays/sla4b1-k64-{arm}-rate{r}.json",
              ["0.25", "0.4", "0.5", "0.6", "0.7", "0.8"])}
CONC = "cluster/results/conc-4b-p10-single-20260916-214417/replays/conc4bp10-{arm}-c{n}.json"
LEVELS = [1, 2, 4, 8, 16, 32]
SETTINGS = [  # (label, setting, k) - rows of panels 5-6, each at 4B then 0.6B
    ("show 10 of 64, dense", "retrieve 64 / present 10, dense", "k64p10"),
    ("show 10 of 64, BM25", "retrieve 64 / present 10, BM25", "k64p10"),
    ("128 tools", "whole menu, dense", "k128"),
    ("64 tools", "whole menu, dense", "k64"),
    ("16 tools", "whole menu, dense", "k16"),
    ("4 tools", "whole menu, dense", "k4"),
]


def text(x, y, s, size=12, weight=400, fill=INK, anchor="start", halo=False):
    h = f' paint-order="stroke" stroke="{SURFACE}" stroke-width="5"' if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{h}>{s}</text>')


def scale(lo, hi, a, b, log=False):
    if log:
        lo, hi = log10(lo), log10(hi)
        return lambda v: a + (log10(v) - lo) / (hi - lo) * (b - a)
    return lambda v: a + (v - lo) / (hi - lo) * (b - a)


def line_panel(s, x0, y0, w, h, *, title, subtitle, xs, v1, rival, rcolour, better, xlim, ylim,
               xticks, yticks, xlog, ylog, xlabel, ylabel, callout):
    """Two-series line chart with the gap shaded by whichever policy is better."""
    s.append(text(x0, y0, title, 14, 700))
    s.append(text(x0, y0 + 18, subtitle, 12, 400, INK2))
    pl, pr, pt, pb = x0 + 52, x0 + w - 8, y0 + 38, y0 + h - 44
    sx = scale(*xlim, pl, pr, xlog)
    sy = scale(*ylim, pb, pt, ylog)
    for v in yticks:
        s.append(f'<line x1="{pl}" y1="{sy(v):.1f}" x2="{pr}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
        s.append(text(pl - 8, sy(v) + 4, f"{v:g}", 11, 400, INK3, "end"))
    for v in xticks:
        s.append(text(sx(v), pb + 16, f"{v:g}", 11, 400, INK3, "middle"))
    s.append(f'<line x1="{pl}" y1="{pb}" x2="{pr}" y2="{pb}" stroke="{INK3}"/>')
    s.append(f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pb}" stroke="{INK3}"/>')
    s.append(text((pl + pr) / 2, pb + 34, xlabel, 11.5, 400, INK2, "middle"))
    s.append(f'<text transform="translate({x0 + 12},{(pt + pb) / 2:.0f}) rotate(-90)" font-size="11.5" '
             f'fill="{INK2}" text-anchor="middle">{ylabel}</text>')

    P = [(sx(x), sy(a), sy(b)) for x, a, b in zip(xs, v1, rival)]
    v1_better = (lambda a, b: a > b) if better == "lower" else (lambda a, b: a < b)  # screen coords

    def fill(poly, colour):
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in poly)
        s.append(f'<polygon points="{pts}" fill="{colour}" fill-opacity="0.22"/>')

    for (xa, a0, b0), (xb, a1, b1) in zip(P, P[1:]):
        d0, d1 = a0 - b0, a1 - b1
        if d0 * d1 < 0:  # the lines cross inside this interval: split there
            t = d0 / (d0 - d1)
            xc, yc = xa + t * (xb - xa), a0 + t * (a1 - a0)
            fill([(xa, a0), (xc, yc), (xa, b0)], V1[2] if v1_better(a0, b0) else rcolour)
            fill([(xc, yc), (xb, a1), (xb, b1)], V1[2] if v1_better(a1, b1) else rcolour)
        else:
            ref = (a0, b0) if d0 != 0 else (a1, b1)
            fill([(xa, a0), (xb, a1), (xb, b1), (xa, b0)], V1[2] if v1_better(*ref) else rcolour)
    for series, colour, width in ((2, rcolour, 2), (1, V1[2], 2.8)):
        path = " ".join(("M" if i == 0 else "L") + f"{p[0]:.1f},{p[series]:.1f}" for i, p in enumerate(P))
        s.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linejoin="round"/>')
        for p in P:
            s.append(f'<circle cx="{p[0]:.1f}" cy="{p[series]:.1f}" r="{4.2 if series == 1 else 3.4}" '
                     f'fill="{colour}" stroke="{SURFACE}" stroke-width="1.5"/>')
    if callout:
        i, label, dy = callout
        cx, cy = P[i][0], min(P[i][1], P[i][2]) + dy
        s.append(text(cx, cy, label, 13, 700, INK, "end" if i == len(P) - 1 else "middle", halo=True))


def dumbbell_panel(s, x0, y0, w, *, title, subtitle, rows, rcolour, xlim, xticks, xlog, xlabel):
    """rows: (label, v1, rival, note, status) with status in win/tie/loss."""
    s.append(text(x0, y0, title, 14, 700))
    s.append(text(x0, y0 + 18, subtitle, 12, 400, INK2))
    row_h, lab_w, note_w = 25, 182, 76
    pl, pr = x0 + lab_w, x0 + w - note_w
    top = y0 + 34
    sx = scale(*xlim, pl, pr, xlog)
    bottom = top + row_h * len(rows)
    tint = {"win": (V1[2], 0.10), "loss": (rcolour, 0.14)}
    for i, (_, _, _, _, status) in enumerate(rows):  # row tints under everything
        if status in tint:
            c, o = tint[status]
            s.append(f'<rect x="{x0}" y="{top + i * row_h + 1:.1f}" width="{w}" height="{row_h - 2}" '
                     f'fill="{c}" fill-opacity="{o}" rx="3"/>')
    for v in xticks:
        s.append(f'<line x1="{sx(v):.1f}" y1="{top}" x2="{sx(v):.1f}" y2="{bottom}" stroke="{GRID}"/>')
        s.append(text(sx(v), bottom + 14, f"{v:g}", 11, 400, INK3, "middle"))
    for i, (label, a, b, note, status) in enumerate(rows):
        y = top + i * row_h + row_h / 2
        s.append(text(x0 + 6, y + 4, label, 11.5, 400, INK))
        s.append(f'<line x1="{sx(a):.1f}" y1="{y:.1f}" x2="{sx(b):.1f}" y2="{y:.1f}" stroke="{INK3}" stroke-width="1.5"/>')
        s.append(f'<circle cx="{sx(b):.1f}" cy="{y:.1f}" r="4.2" fill="{rcolour}" stroke="{SURFACE}" stroke-width="1.5"/>')
        s.append(f'<circle cx="{sx(a):.1f}" cy="{y:.1f}" r="5" fill="{V1[2]}" stroke="{SURFACE}" stroke-width="1.5"/>')
        s.append(text(x0 + w - 4, y + 4, note, 11.5, 700 if status == "win" else 400,
                      INK if status == "win" else INK2, "end"))
    s.append(text((pl + pr) / 2, bottom + 31, xlabel, 11.5, 400, INK2, "middle"))


def load_line(pattern, arm, keys, field):
    out = []
    for k in keys:
        run = json.loads(Path(pattern.format(arm=arm, r=k, n=k)).read_text())
        out.append(run["ttft_seconds"]["p50"] if field == "p50" else run["achieved_rate"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rival", choices=sorted(RIVALS), required=True)
    args = ap.parse_args()
    rname, rcolour, slug = RIVALS[args.rival]
    summary = json.loads(Path(sorted(glob.glob("cluster/results/hit-miss-f1-*/summary.json"))[-1]).read_text())
    R = {(r["setting"], r["model"], r["k"], r["arm"]): r for r in summary["rows"]}
    Z = {(p["setting"], p["model"], p["k"], p["rival"]): p for p in summary["paired_end_to_end_f1"]}

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']
    headline = {"cp_online": "ToolTrie-v1 vs ContextPilot: faster under load, more cache reuse, higher F1",
                "original": "ToolTrie-v1 vs no reordering: up to 1.5x faster and up to 10x the cache reuse, at equal accuracy"}
    s.append(text(40, 40, headline[args.rival], 20, 700))
    s.append(text(40, 64, f"Each panel compares only these two policies. Shading: blue where ToolTrie-v1 is better, "
                          f"{'orange' if args.rival == 'cp_online' else 'grey'} where {rname} is better.", 13, 400, INK2))
    lx = 40
    for label, colour, kind in ((V1[1], V1[2], "line"), (rname, rcolour, "line"),
                                ("ToolTrie-v1 better", V1[2], "fill"), (f"{rname} better", rcolour, "fill")):
        if kind == "line":
            s.append(f'<line x1="{lx}" y1="86" x2="{lx + 22}" y2="86" stroke="{colour}" stroke-width="2.6"/>')
            s.append(f'<circle cx="{lx + 11}" cy="86" r="4" fill="{colour}" stroke="{SURFACE}" stroke-width="1.5"/>')
        else:
            s.append(f'<rect x="{lx}" y="80" width="22" height="12" fill="{colour}" fill-opacity="0.22" rx="2"/>')
        s.append(text(lx + 30, 90, label, 12.5))
        lx += 44 + 7.2 * len(label)

    col = [40, 620]
    pw = 520
    # 1-2: latency vs offered load, 64 tools
    for c, (model, (pattern, names)) in zip(col, SLA.items()):
        v1 = load_line(pattern, V1[0], names, "p50")
        rv = load_line(pattern, args.rival, names, "p50")
        rates = [float(n) for n in names]
        ratio = [b / a for a, b in zip(v1, rv)]
        wins = sum(r > 1 for r in ratio)
        best = max(range(len(ratio)), key=lambda i: ratio[i])
        ylim, yt = ((0.3, 4.0), [0.3, 0.5, 1, 2, 3]) if model == "0.6B" else ((1.0, 20.0), [1, 2, 5, 10, 20])
        line_panel(s, c, 130, pw, 290,
                   title=f"Speed under load, 64 tools, Qwen3-{model}",
                   subtitle=f"v1 faster at {wins} of {len(rates)} rates; up to {ratio[best]:.2f}x at {rates[best]:g} req/s",
                   xs=rates, v1=v1, rival=rv, rcolour=rcolour, better="lower",
                   xlim=(rates[0] - 0.05 * (rates[-1] - rates[0]), rates[-1] + 0.05 * (rates[-1] - rates[0])),
                   ylim=ylim, xticks=rates, yticks=yt, xlog=False, ylog=True,
                   xlabel="offered load (requests per second)", ylabel="median time to first token, s (log)",
                   callout=(best, f"{ratio[best]:.2f}x faster", -12))

    # 3-4: retrieve 64 / show 10, closed loop
    v1t, rvt = load_line(CONC, V1[0], LEVELS, "thr"), load_line(CONC, args.rival, LEVELS, "thr")
    gain = [a / b - 1 for a, b in zip(v1t, rvt)]
    bt = max(range(len(gain)), key=lambda i: gain[i])
    line_panel(s, col[0], 450, pw, 290,
               title="Throughput, show 10 of 64, Qwen3-4B",
               subtitle=f"v1 higher at {sum(g > 0 for g in gain)} of {len(LEVELS)} concurrency levels; up to +{100 * gain[bt]:.0f}% at N = {LEVELS[bt]}",
               xs=LEVELS, v1=v1t, rival=rvt, rcolour=rcolour, better="higher",
               xlim=(0.8, 40), ylim=(0, 7.5), xticks=LEVELS, yticks=[0, 2, 4, 6], xlog=True, ylog=False,
               xlabel="requests in flight, N (log)", ylabel="throughput, requests per second",
               callout=(bt, f"+{100 * gain[bt]:.0f}%", -14))
    v1p, rvp = load_line(CONC, V1[0], LEVELS, "p50"), load_line(CONC, args.rival, LEVELS, "p50")
    ratio = [b / a for a, b in zip(v1p, rvp)]
    bp = max(range(len(ratio)), key=lambda i: ratio[i])
    line_panel(s, col[1], 450, pw, 290,
               title="Speed, show 10 of 64, Qwen3-4B",
               subtitle=f"v1 faster at {sum(r > 1 for r in ratio)} of {len(LEVELS)} concurrency levels; up to {ratio[bp]:.2f}x at N = {LEVELS[bp]}",
               xs=LEVELS, v1=v1p, rival=rvp, rcolour=rcolour, better="lower",
               xlim=(0.8, 40), ylim=(0.1, 3.5), xticks=LEVELS, yticks=[0.1, 0.2, 0.5, 1, 2], xlog=True, ylog=True,
               xlabel="requests in flight, N (log)", ylabel="median time to first token, s (log)",
               callout=(bp, f"{ratio[bp]:.2f}x faster", -12))

    # 5: cache hit rate
    rows, wins = [], 0
    for label, setting, k in SETTINGS:
        for model in ("4B", "0.6B"):
            a = R[(setting, model, k, V1[0])]["hit_pct"]
            b = R[(setting, model, k, args.rival)]["hit_pct"]
            status = "win" if a > b else "loss" if a < b else "tie"
            wins += status == "win"
            rows.append((f"{label} · {model}", a, b, f"{a / b:.2f}x" if a / b < 1.1 else f"{a / b:.1f}x", status))
    dumbbell_panel(s, col[0], 772, pw, title="Cache hit rate",
                   subtitle=f"share of prompt tokens served from cache; v1 higher in {wins} of {len(rows)}",
                   rows=rows, rcolour=rcolour, xlim=(0.2, 40), xticks=[0.3, 1, 3, 10, 30], xlog=True,
                   xlabel="prompt tokens served from cache, % (log)")

    # 6: end-to-end F1 with the paired test
    rows, wins, losses = [], 0, 0
    for label, setting, k in SETTINGS:
        for model in ("4B", "0.6B"):
            a = 100 * R[(setting, model, k, V1[0])]["end_to_end_f1"]
            b = 100 * R[(setting, model, k, args.rival)]["end_to_end_f1"]
            p = Z[(setting, model, k, args.rival)]
            status = "win" if p["z"] >= 2 else "loss" if p["z"] <= -2 else "tie"
            wins += status == "win"; losses += status == "loss"
            rows.append((f"{label} · {model}", a, b, f"{p['diff_points']:+.1f}" + (" ▲" if status == "win" else ""), status))
    f1_sub = (f"v1 significantly higher in {wins} of {len(rows)}, lower in {losses}"
              if wins else f"no significant difference in any of the {len(rows)} settings: no accuracy cost")
    dumbbell_panel(s, col[1], 772, pw, title="End-to-end F1",
                   subtitle=f1_sub, rows=rows, rcolour=rcolour, xlim=(10, 32), xticks=[10, 15, 20, 25, 30],
                   xlog=False, xlabel="end-to-end F1 (×100); ▲ = v1 higher by at least 2 standard errors")

    s.append(text(40, H - 40, "Panels 1-4: one run of 200 requests per point, every arm on one GPU. Panels 5-6: 200 ToolRet tasks per "
                              "setting, one request at a time; F1 differences are paired over the same tasks.", 11.5, 400, INK3))
    s.append(text(40, H - 22, "Sources: eval-validity-20260906-165826, sla-4b-single-20260912-175313, "
                              "conc-4b-p10-single-20260916-214417, hit-miss-f1 summary (scripts/summarize_hit_miss_f1.py).",
                  11.5, 400, INK3))
    s.append("</svg>")
    out = Path(f"reports/figures/head-to-head-{slug}.svg")
    out.write_text("\n".join(s))
    print(out)


if __name__ == "__main__":
    main()
