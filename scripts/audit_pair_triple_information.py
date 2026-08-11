"""Does pair/triple co-occurrence carry information beyond tool frequency?

Brief §7 Q5 asks how much additional benefit comes from pair/triple workflow
structure. Every fitted policy that consumes that structure
(``conditional_pair``, ``conditional_pair_triple``, ``fp_tree_conditional``)
emitted byte-identical orderings to plain ``frequency_fitted`` on the padded
workloads, so the question was never actually posed there.

This audit measures how much room a pair key has to discriminate. Where observed
pair support tracks the two tools' presence counts, a pair key adds little the
frequency key does not already carry.

**It is a measurement, not a proof.** Only pairs actually observed together are
counted; an unobserved pair scores zero rather than ``min(presence)``, and
triples are skipped on menus wider than ``TRIPLE_MENU_LIMIT`` — which includes
``bfcl-padded64``. A workload reported as fully redundant can still admit a
pair-keyed policy that differs. Confirm any consequence by replaying the
planners and diffing their emitted orderings: on the real workloads the two
online planners differ on 0/200 ``bfcl-padded64`` records but 4/200
``toolret-padded64`` records, so redundancy is not total even on padded menus.

The audit reports, per workload:

* ``pair_equals_min_presence_fraction`` — how often
  ``support(a, b) == min(presence(a), presence(b))``. A menu whose tools are
  either universal or unique forces this identity.
* ``presence_signature_violations`` — observed pairs sharing a presence
  signature but disagreeing on support. Zero means support is a function of
  presence *over the pairs actually seen*, which bounds but does not eliminate
  what a pair key can do.

CPU only. Reads committed workload files; runs no model and no server.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.io import read_jsonl, write_json  # noqa: E402

# Triples are enumerated only for menus at or below this size. C(128, 3) per
# request is ~341k combinations, which buys no additional insight once the pair
# result is already degenerate.
TRIPLE_MENU_LIMIT = 16


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    presence: Counter[str] = Counter()
    for row in rows:
        presence.update(row["tool_ids"])

    pair: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for a, b in combinations(sorted(row["tool_ids"]), 2):
            pair[(a, b)] += 1

    equals_min = sum(
        1 for (a, b), v in pair.items() if v == min(presence[a], presence[b])
    )
    signature: dict[tuple[int, int], int] = {}
    violations = 0
    for (a, b), v in pair.items():
        key = (min(presence[a], presence[b]), max(presence[a], presence[b]))
        if key in signature and signature[key] != v:
            violations += 1
        signature.setdefault(key, v)

    menu_sizes = {len(row["tool_ids"]) for row in rows}
    result: dict[str, Any] = {
        "requests": len(rows),
        "menu_sizes": sorted(menu_sizes),
        "distinct_tools": len(presence),
        "tools_in_every_request": sum(
            1 for count in presence.values() if count == len(rows)
        ),
        "distinct_pairs": len(pair),
        "pair_equals_min_presence_fraction": round(equals_min / len(pair), 6),
        "presence_signature_violations": violations,
        "pair_support_is_a_function_of_presence": violations == 0,
    }

    if max(menu_sizes) <= TRIPLE_MENU_LIMIT:
        triple: Counter[tuple[str, str, str]] = Counter()
        for row in rows:
            for a, b, c in combinations(sorted(row["tool_ids"]), 3):
                triple[(a, b, c)] += 1
        t_equals_min = sum(
            1
            for (a, b, c), v in triple.items()
            if v == min(presence[a], presence[b], presence[c])
        )
        t_signature: dict[tuple[int, int, int], int] = {}
        t_violations = 0
        for (a, b, c), v in triple.items():
            key = tuple(sorted((presence[a], presence[b], presence[c])))
            if key in t_signature and t_signature[key] != v:
                t_violations += 1
            t_signature.setdefault(key, v)  # type: ignore[arg-type]
        result["distinct_triples"] = len(triple)
        result["triple_equals_min_presence_fraction"] = round(
            t_equals_min / len(triple), 6
        )
        result["triple_presence_signature_violations"] = t_violations
        result["triple_support_is_a_function_of_presence"] = t_violations == 0
    else:
        result["triples_skipped_menu_too_large"] = True

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether pair/triple support is redundant with tool frequency."
    )
    parser.add_argument(
        "--workload",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Workload to analyze, repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workloads: dict[str, Any] = {}
    for item in args.workload:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"expected NAME=PATH, got {item!r}")
        workloads[name] = analyze(list(read_jsonl(Path(raw_path))))

    degenerate = sorted(
        name
        for name, r in workloads.items()
        if r["pair_support_is_a_function_of_presence"]
    )
    summary = {
        "audit": "pair-triple-information-content",
        "format_version": 1,
        "generated_by": "scripts/audit_pair_triple_information.py",
        "question": "brief §7 Q5 — additional benefit from pair/triple workflow structure",
        "workloads": workloads,
        "workloads_where_pair_support_is_redundant_with_frequency": degenerate,
        "conclusion": (
            "On workloads listed as redundant, observed pair support tracks "
            "min(presence(a), presence(b)), so a pair key adds little a "
            "frequency key does not already carry. This is evidence about these "
            "workloads, not a proof about pair-keyed policies in general: only "
            "pairs actually observed together are counted, unobserved pairs "
            "score zero rather than min(presence), and triples are skipped on "
            "menus wider than TRIPLE_MENU_LIMIT. Confirm any claim by replaying "
            "the planners and diffing their emitted orderings."
        ),
        "scope": {
            "counts_observed_pairs_only": True,
            "triple_menu_limit": TRIPLE_MENU_LIMIT,
            "proves_no_pair_policy_can_differ": False,
        },
    }
    write_json(args.output, summary)

    for name, r in workloads.items():
        print(
            f"{name:22} universal={r['tools_in_every_request']:4} "
            f"pairs={r['distinct_pairs']:8} "
            f"pair==min(presence)={r['pair_equals_min_presence_fraction']*100:6.2f}% "
            f"violations={r['presence_signature_violations']}"
        )
    print(f"\nRedundant with frequency: {degenerate or 'none'}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
