#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_COVER = {
    "doc_title": "Task C &amp; D — TATM Research Report",
    "title": "Task C &amp; Task D",
    "subtitle": (
        "Dataset normalization, schema-token accounting, tool-access patterns, "
        "deterministic ordering, and analytical trie reuse."
    ),
    "status": "Local analysis complete · Cluster validation pending",
    "metrics": [
        ("45,815", "Canonical tools"),
        ("9,201", "Benchmark tasks"),
        ("70", "Median schema tokens"),
        ("31.80%", "Best ToolRet estimate"),
        ("18.88%", "Best BFCL estimate"),
    ],
    "executive": (
        "Task C establishes a reproducible canonical tool corpus and token "
        "accounting pipeline. Task D finds that deterministic ordering improves "
        "analytical prefix sharing on multi-tool workloads: schema-cost "
        "weighting is strongest for ToolRet, while frequency ordering is "
        "strongest for the selected BFCL subset. GPU latency and correctness "
        "remain cluster measurements."
    ),
}

TASK_B_E_COVER = {
    "doc_title": "Task B &amp; E — TATM Research Report",
    "title": "Task B &amp; Task E",
    "subtitle": (
        "Exact prefix-cache verification, the prefill measurement floor, "
        "tool-ordering effects on real cache hits, and trie-model calibration."
    ),
    "status": "Historical GPU snapshot · See consolidated report",
    "metrics": [
        ("10.6×", "TTFT gain at 200 tools"),
        ("4", "Crossover menu size"),
        ("38.15%", "Measured reuse, alphabetical"),
        ("4.41%", "Measured reuse, frequency"),
        ("1.2–1.5×", "Model under-prediction"),
    ],
    "executive": (
        "All five Task B checks pass. Prefix reuse buys no TTFT at a 303-token "
        "prompt, but that is a measurement floor: padding tool menus to "
        "deployment-realistic sizes yields a 10.6× TTFT reduction at 200 tools, "
        "with a crossover near 4 tools. Validating the analytical trie estimate "
        "against measured cache hits inverts the earlier ordering "
        "recommendation — on shared catalogs, ordering by benchmark support "
        "front-loads task-specific tools and destroys the common prefix, making "
        "it nearly 9× worse than a stable global order."
    ),
}


INITIAL_FINDINGS_COVER = {
    "doc_title": "Initial Research Findings — TATM Report",
    "title": "Initial Research Findings",
    "subtitle": (
        "Task F deliverable: dataset inventory, schema-length and frequency "
        "figures, prefix-cache sanity results, ordering comparisons, and the "
        "recommendation for the next research stage."
    ),
    "status": "Initial experiments measured · Local audit complete",
    "metrics": [
        ("45,815", "Canonical tools"),
        ("9,201", "Benchmark tasks"),
        ("10.6×", "TTFT gain at 200 tools"),
        ("31.80%", "Best ToolRet estimate"),
        ("18.88%", "Best BFCL estimate"),
    ],
    "executive": (
        "Public tool workloads contain a measurable analytical prefix-locality "
        "signal, but it is workload- and evidence-dependent. Deterministic "
        "ordering raises estimated block reuse over empirical order for both "
        "ToolRet and BFCL. Measured on vLLM, prefix caching yields no TTFT "
        "benefit at 303-token prompts but up to a 10.6x reduction on padded "
        "shared-catalog menus. On deterministic BM25-retrieved menus, ToolTrie "
        "adds only 0.76-1.65 percentage points of reuse without a resolved TTFT "
        "gain. A corrected ContextPilot persistent-API adaptation leads "
        "ToolTrie on every retrieved menu size, while both lose most reuse as "
        "menu overlap falls. The recommendation is to keep ordinary text "
        "prefill as fallback and treat safe retention as a separate extension."
    ),
}

COVERS = {
    "task_c_d": DEFAULT_COVER,
    "task_b_e": TASK_B_E_COVER,
    "initial-findings": INITIAL_FINDINGS_COVER,
}


def render(source: Path, output: Path, cover: dict[str, Any] | None = None) -> None:
    try:
        import markdown
    except ImportError as error:
        raise SystemExit(
            "Missing report renderer. Run `uv sync` or install `markdown`."
        ) from error

    source_text = source.read_text(encoding="utf-8")
    lines = source_text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body_source = "\n".join(lines).lstrip()

    renderer = markdown.Markdown(
        extensions=[
            "extra",
            "sane_lists",
            "toc",
        ],
        extension_configs={
            "toc": {
                "permalink": True,
                "permalink_title": "Link to this section",
                "toc_depth": "2-3",
            }
        },
        output_format="html5",
    )
    body_html = renderer.convert(body_source)
    toc_html = renderer.toc

    cover = {**DEFAULT_COVER, **(cover or {})}
    doc_title = cover["doc_title"]
    hero_title = cover["title"]
    hero_subtitle = cover["subtitle"]
    status_text = cover["status"]
    executive_text = cover["executive"]
    metrics_html = "\n      ".join(
        f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in cover["metrics"]
    )
    footer_note = f"TATM FYP · Public benchmark analysis · Generated from {source.name}"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{doc_title}</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5c667a;
      --line: #dfe5ef;
      --soft: #f4f7fb;
      --paper: #ffffff;
      --navy: #102341;
      --blue: #2563eb;
      --cyan: #06b6d4;
      --green: #0f9f6e;
      --orange: #e27720;
      --shadow: 0 18px 55px rgba(21, 42, 78, 0.13);
    }}

    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 5%, rgba(37, 99, 235, 0.10), transparent 24rem),
        radial-gradient(circle at 92% 16%, rgba(6, 182, 212, 0.09), transparent 25rem),
        #eef2f8;
      font: 16px/1.68 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    a {{
      color: #1d5fd1;
      text-decoration-thickness: 1px;
      text-underline-offset: 3px;
    }}

    .page {{
      width: min(1180px, calc(100% - 40px));
      margin: 28px auto 70px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      padding: 62px 66px 55px;
      color: #fff;
      background:
        linear-gradient(125deg, rgba(16, 35, 65, 0.98), rgba(23, 72, 141, 0.96)),
        var(--navy);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }}

    .hero::before,
    .hero::after {{
      content: "";
      position: absolute;
      border-radius: 999px;
      pointer-events: none;
    }}

    .hero::before {{
      width: 390px;
      height: 390px;
      top: -260px;
      right: -60px;
      border: 56px solid rgba(94, 234, 212, 0.12);
    }}

    .hero::after {{
      width: 210px;
      height: 210px;
      right: 155px;
      bottom: -155px;
      background: rgba(37, 99, 235, 0.32);
    }}

    .eyebrow {{
      position: relative;
      z-index: 1;
      margin: 0 0 15px;
      color: #9ee7f2;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.17em;
      text-transform: uppercase;
    }}

    .hero h1 {{
      position: relative;
      z-index: 1;
      max-width: 800px;
      margin: 0;
      color: #fff;
      font-size: clamp(2.25rem, 5vw, 4.3rem);
      line-height: 1.02;
      letter-spacing: -0.045em;
    }}

    .hero-subtitle {{
      position: relative;
      z-index: 1;
      max-width: 760px;
      margin: 22px 0 0;
      color: #d9e8ff;
      font-size: 1.1rem;
    }}

    .status-pill {{
      display: inline-flex;
      position: relative;
      z-index: 1;
      align-items: center;
      gap: 8px;
      margin-top: 24px;
      padding: 8px 13px;
      color: #d9fff3;
      background: rgba(15, 159, 110, 0.18);
      border: 1px solid rgba(94, 234, 212, 0.32);
      border-radius: 999px;
      font-size: 0.86rem;
      font-weight: 700;
    }}

    .status-pill::before {{
      content: "";
      width: 8px;
      height: 8px;
      background: #5eead4;
      border-radius: 50%;
      box-shadow: 0 0 0 5px rgba(94, 234, 212, 0.10);
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin: 18px 0;
    }}

    .metric {{
      padding: 20px 18px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(215, 224, 238, 0.95);
      border-radius: 15px;
      box-shadow: 0 7px 24px rgba(21, 42, 78, 0.07);
    }}

    .metric strong {{
      display: block;
      color: var(--navy);
      font-size: 1.55rem;
      line-height: 1.1;
      letter-spacing: -0.025em;
    }}

    .metric span {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      line-height: 1.3;
      text-transform: uppercase;
      letter-spacing: 0.055em;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 255px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}

    .toc-card {{
      position: sticky;
      top: 18px;
      max-height: calc(100vh - 36px);
      overflow: auto;
      padding: 22px 20px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 8px 30px rgba(21, 42, 78, 0.08);
    }}

    .toc-card > strong {{
      display: block;
      margin-bottom: 12px;
      color: var(--navy);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}

    .toc-card ul {{
      margin: 0;
      padding-left: 0;
      list-style: none;
    }}

    .toc-card ul ul {{
      margin: 4px 0 8px 12px;
      padding-left: 10px;
      border-left: 1px solid var(--line);
    }}

    .toc-card li {{
      margin: 4px 0;
    }}

    .toc-card a {{
      display: block;
      padding: 4px 7px;
      color: #536078;
      border-radius: 6px;
      font-size: 0.82rem;
      line-height: 1.35;
      text-decoration: none;
    }}

    .toc-card a:hover {{
      color: var(--blue);
      background: #edf4ff;
    }}

    .report {{
      min-width: 0;
      padding: 50px 58px 62px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}

    .executive {{
      margin: 0 0 42px;
      padding: 24px 26px;
      background: linear-gradient(135deg, #eff6ff, #f0fdfa);
      border: 1px solid #bfdbfe;
      border-left: 5px solid var(--blue);
      border-radius: 12px;
    }}

    .executive strong {{
      display: block;
      margin-bottom: 6px;
      color: var(--navy);
      font-size: 1.05rem;
    }}

    .executive p {{
      margin: 0;
      color: #3f4c63;
    }}

    h2, h3, h4 {{
      color: var(--navy);
      letter-spacing: -0.025em;
    }}

    h2 {{
      margin: 66px 0 22px;
      padding-top: 8px;
      font-size: 2rem;
      line-height: 1.2;
      border-top: 1px solid var(--line);
    }}

    .report > h2:first-of-type {{
      margin-top: 12px;
      border-top: 0;
    }}

    h3 {{
      margin: 38px 0 14px;
      font-size: 1.35rem;
    }}

    h4 {{
      margin: 28px 0 10px;
      font-size: 1.05rem;
    }}

    h2 .headerlink,
    h3 .headerlink,
    h4 .headerlink {{
      margin-left: 8px;
      color: #a8b2c3;
      font-size: 0.72em;
      text-decoration: none;
      opacity: 0;
    }}

    h2:hover .headerlink,
    h3:hover .headerlink,
    h4:hover .headerlink {{
      opacity: 1;
    }}

    p {{
      margin: 0 0 16px;
    }}

    ul, ol {{
      padding-left: 24px;
    }}

    li {{
      margin: 6px 0;
    }}

    blockquote {{
      margin: 24px 0;
      padding: 18px 22px;
      color: #37445c;
      background: #f7f9fc;
      border-left: 4px solid var(--cyan);
      border-radius: 4px 10px 10px 4px;
    }}

    .table-wrap {{
      width: 100%;
      margin: 20px 0 30px;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 11px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.91rem;
      font-variant-numeric: tabular-nums;
    }}

    th {{
      padding: 12px 14px;
      color: #fff;
      background: var(--navy);
      font-size: 0.76rem;
      line-height: 1.3;
      text-align: left;
      text-transform: uppercase;
      letter-spacing: 0.055em;
    }}

    td {{
      padding: 11px 14px;
      border-top: 1px solid #e7ebf2;
      vertical-align: top;
    }}

    tbody tr:nth-child(even) {{
      background: #f8fafc;
    }}

    tbody tr:hover {{
      background: #eff6ff;
    }}

    code {{
      padding: 0.13em 0.36em;
      color: #a51d52;
      background: #f5f0f4;
      border: 1px solid #ebdde5;
      border-radius: 5px;
      font: 0.9em/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }}

    pre {{
      position: relative;
      margin: 18px 0 26px;
      padding: 20px 22px;
      overflow: auto;
      color: #dbeafe;
      background: #111d32;
      border: 1px solid #243653;
      border-radius: 11px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }}

    pre code {{
      padding: 0;
      color: inherit;
      background: transparent;
      border: 0;
      font-size: 0.86rem;
    }}

    strong {{
      color: #102341;
    }}

    .footer {{
      margin-top: 45px;
      padding-top: 20px;
      color: #738096;
      border-top: 1px solid var(--line);
      font-size: 0.82rem;
      text-align: center;
    }}

    .print-button {{
      position: fixed;
      right: 22px;
      bottom: 22px;
      z-index: 5;
      padding: 11px 16px;
      color: #fff;
      background: var(--blue);
      border: 0;
      border-radius: 999px;
      box-shadow: 0 10px 30px rgba(37, 99, 235, 0.3);
      cursor: pointer;
      font: 700 0.85rem/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    @media (max-width: 940px) {{
      .page {{
        width: min(100% - 22px, 820px);
        margin-top: 11px;
      }}

      .hero {{
        padding: 42px 28px;
        border-radius: 18px;
      }}

      .metrics {{
        grid-template-columns: repeat(2, 1fr);
      }}

      .layout {{
        grid-template-columns: 1fr;
      }}

      .toc-card {{
        position: relative;
        top: 0;
        max-height: none;
      }}

      .report {{
        padding: 35px 25px 45px;
      }}
    }}

    @media print {{
      @page {{
        size: A4;
        margin: 16mm 14mm 17mm;
      }}

      body {{
        background: #fff;
        font-size: 10.4pt;
      }}

      .page {{
        width: 100%;
        margin: 0;
      }}

      .hero {{
        min-height: 185mm;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 30mm 20mm;
        border-radius: 0;
        box-shadow: none;
        break-after: page;
      }}

      .hero h1 {{
        font-size: 38pt;
      }}

      .metrics {{
        grid-template-columns: repeat(5, 1fr);
        margin: 0 0 8mm;
        break-after: page;
      }}

      .metric {{
        padding: 12px 8px;
        box-shadow: none;
      }}

      .metric strong {{
        font-size: 15pt;
      }}

      .metric span {{
        font-size: 6.7pt;
      }}

      .layout {{
        display: block;
      }}

      .toc-card {{
        position: static;
        max-height: none;
        margin-bottom: 10mm;
        box-shadow: none;
        break-after: page;
      }}

      .report {{
        padding: 0;
        border: 0;
        border-radius: 0;
        box-shadow: none;
      }}

      h2 {{
        break-before: auto;
        break-after: avoid;
      }}

      h3, h4 {{
        break-after: avoid;
      }}

      p, li {{
        orphans: 3;
        widows: 3;
      }}

      table, pre, blockquote, .executive {{
        break-inside: avoid;
      }}

      .table-wrap {{
        overflow: visible;
      }}

      table {{
        font-size: 8.1pt;
      }}

      th, td {{
        padding: 6px 7px;
      }}

      a {{
        color: inherit;
        text-decoration: none;
      }}

      .headerlink,
      .print-button {{
        display: none !important;
      }}

      .footer {{
        margin-top: 16mm;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <p class="eyebrow">Trie-Aware Tool Memory · FYP Research Report</p>
      <h1>{hero_title}</h1>
      <p class="hero-subtitle">
        {hero_subtitle}
      </p>
      <span class="status-pill">{status_text}</span>
    </header>

    <section class="metrics" aria-label="Key findings">
      {metrics_html}
    </section>

    <div class="layout">
      <nav class="toc-card" aria-label="Table of contents">
        <strong>Contents</strong>
        {toc_html}
      </nav>

      <main class="report">
        <aside class="executive">
          <strong>Executive answer</strong>
          <p>
            {executive_text}
          </p>
        </aside>

        {body_html}

        <footer class="footer">
          {footer_note}
        </footer>
      </main>
    </div>
  </div>

  <button class="print-button" type="button" onclick="window.print()">
    Print / Save PDF
  </button>

  <script>
    document.querySelectorAll("table").forEach((table) => {{
      const wrapper = document.createElement("div");
      wrapper.className = "table-wrap";
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    }});
    document.querySelectorAll('a[href^="http"]').forEach((link) => {{
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }});
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the Task C/D Markdown as a standalone HTML report."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "task_c_d.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "task_c_d.html",
    )
    parser.add_argument(
        "--cover",
        choices=sorted(COVERS),
        help="Cover page preset. Inferred from the source filename if omitted.",
    )
    args = parser.parse_args()
    cover_key = args.cover or args.source.stem
    render(args.source, args.output, COVERS.get(cover_key))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
