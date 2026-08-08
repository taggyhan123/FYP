#!/usr/bin/env python3
"""Fail-closed aggregate-counter audit for every historical SGLang run.

SGLang omitted ``usage.prompt_tokens_details.cached_tokens`` when it was zero,
so the first request after each flush reports ``None``. The old reconciliation
mistakenly compared the response sum with another response-derived field. This
script validates all 72 raw Phase 2 runs against the independent aggregate
``sglang:cached_tokens_total`` counter and writes reconciled copies only if the
entire declared matrix passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.sglang_client import initial_missing_cached_reconciliation

DEFAULT_INPUT = Path(
    "/home/taghan/FYP/cluster/results/tooltrie-phase2-20260803-181133"
)
CONDITIONS = (
    "original",
    "alphabetical",
    "tooltrie_v0",
    "cacheweaver",
    "frequency_fitted",
    "schema_cost_fitted",
    "fp_tree_conditional",
    "conditional_pair",
    "conditional_pair_triple",
    "contextpilot_intra",
    "contextpilot_intra_schedule",
    "contextpilot_causal",
)
RECONCILIATION_NOTE = (
    "Accepted by aggregate-counter audit: request and prompt counters match; "
    "index 0 is the only response missing zero-valued cached_tokens; the sum "
    "of reported response cached tokens equals the independent aggregate "
    "sglang:cached_tokens_total delta; no request failed."
)


def expected_paths(input_dir: Path) -> list[Path]:
    return [
        input_dir / f"{dataset}-{condition}-sglang-trial-{trial}.json"
        for dataset in ("bfcl", "toolret")
        for condition in CONDITIONS
        for trial in (1, 2, 3)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit all 72 Phase 2 SGLang runs against aggregate counters."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output_dir}")

    accepted: list[tuple[Path, dict[str, object], dict[str, object]]] = []
    failures: list[dict[str, object]] = []
    for source in expected_paths(args.input_dir):
        if not source.is_file():
            failures.append({"file": source.as_posix(), "failed_checks": ["exists"]})
            print(f"REFUSE {source.name}: missing")
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        reconciliation = initial_missing_cached_reconciliation(payload)
        checks = reconciliation["checks"]
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            failures.append(
                {"file": source.as_posix(), "failed_checks": failed_checks}
            )
            print(f"REFUSE {source.name}: {failed_checks}")
            continue
        accepted.append((source, payload, reconciliation))
        print(
            f"OK {source.name}  "
            f"cached={reconciliation['reported_cached_tokens']}  "
            f"aggregate={reconciliation['aggregate_cached_tokens']}"
        )

    if failures:
        print(
            f"\nAudit failed closed: accepted={len(accepted)}, "
            f"refused={len(failures)}. No reconciled copies were written."
        )
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True)
    audit_rows: list[dict[str, object]] = []
    for source, payload, reconciliation in accepted:
        validation = payload["counter_validation"]
        validation["clean"] = True
        validation["reconciliation_note"] = RECONCILIATION_NOTE
        validation["aggregate_counter_reconciliation"] = reconciliation
        output_path = args.output_dir / source.name
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_rows.append(
            {
                "file": source.name,
                "reported_cached_tokens": reconciliation["reported_cached_tokens"],
                "aggregate_cached_tokens": reconciliation["aggregate_cached_tokens"],
                "checks": reconciliation["checks"],
            }
        )

    summary = {
        "format_version": 1,
        "input_dir": args.input_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "declared_runs": len(expected_paths(args.input_dir)),
        "accepted_runs": len(accepted),
        "refused_runs": 0,
        "all_clean": True,
        "conditions": list(CONDITIONS),
        "runs": audit_rows,
    }
    (args.output_dir / "aggregate-counter-audit-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nAggregate-counter audit accepted {len(accepted)}/72 runs.")


if __name__ == "__main__":
    main()
