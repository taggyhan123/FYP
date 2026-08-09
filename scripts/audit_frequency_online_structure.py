#!/usr/bin/env python3
"""Reconstruct the menu structure behind the online-frequency result.

Inputs are the six original workloads copied into the dual-model experiment.
They are intentionally not tracked because workload JSONL belongs with the raw
artifacts.  The expected SHA-256 values are tracked, so regenerated inputs can
be checked byte-for-byte before this script derives support and causal ordering
statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SHA256 = {
    "bfcl-padded64": "0145ed5a228e4aa414c9b8d101d612213ba6ab17c8428ccc6ba994632e6cb3c2",
    "toolret-padded64": "182c20f567f16a91d16df35e7b9a204793a76044add5069ad53a5ee23118bb96",
    "toolret-bm25-k4": "b228588eee43fcdaf8fda4f2f6e90e113201b898f46c676305221e1157507ae8",
    "toolret-bm25-k16": "c89c3a68e4cfbc212639a023ccb882d2825d4a676760590849ca8c95cae8abec",
    "toolret-bm25-k64": "905eaca6fe6578d462c772bf519328050efb0498c78146d8a1d068b0026e4eed",
    "toolret-bm25-k128": "19a5814f380869189094767bb2ab340f93dd7b5887420793db6bd377aa4f41b7",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support: Counter[str] = Counter(
        tool_id for row in rows for tool_id in row["tool_ids"]
    )
    support_histogram = Counter(support.values())
    request_count = len(rows)
    universal_ids = {tool_id for tool_id, count in support.items() if count == request_count}

    presence: Counter[str] = Counter()
    non_universal_positions: list[int] = []
    one_non_universal_per_request = True
    for row in rows:
        names = {
            tool_id: tool["function"]["name"].casefold()
            for tool_id, tool in zip(row["tool_ids"], row["tools"], strict=True)
        }
        ordered = sorted(
            row["tool_ids"],
            key=lambda tool_id: (-presence[tool_id], names[tool_id], tool_id),
        )
        non_universal = [
            tool_id for tool_id in row["tool_ids"] if tool_id not in universal_ids
        ]
        if len(non_universal) == 1:
            non_universal_positions.append(ordered.index(non_universal[0]) + 1)
        else:
            one_non_universal_per_request = False
        # The current request is observed only after its ordering is fixed.
        presence.update(row["tool_ids"])

    result: dict[str, Any] = {
        "requests": request_count,
        "menu_sizes": sorted({len(row["tool_ids"]) for row in rows}),
        "distinct_tools": len(support),
        "universal_tools": len(universal_ids),
        "max_support": max(support.values(), default=0),
        "support_histogram": {
            str(count): tools for count, tools in sorted(support_histogram.items())
        },
        "one_non_universal_tool_per_request": one_non_universal_per_request,
    }
    if one_non_universal_per_request and non_universal_positions:
        menu_size = len(rows[0]["tool_ids"])
        result["causal_online_frequency_non_universal_position"] = {
            "first_request": non_universal_positions[0],
            "mean": sum(non_universal_positions) / len(non_universal_positions),
            "last_position_requests": sum(
                position == menu_size for position in non_universal_positions
            ),
            "requests": len(non_universal_positions),
        }
    return result


def audit(inputs: dict[str, Path]) -> dict[str, Any]:
    if set(inputs) != set(EXPECTED_SHA256):
        missing = sorted(set(EXPECTED_SHA256) - set(inputs))
        extra = sorted(set(inputs) - set(EXPECTED_SHA256))
        raise ValueError(f"workload set mismatch; missing={missing}, extra={extra}")

    workloads: dict[str, Any] = {}
    hashes_match = True
    for name, path in sorted(inputs.items()):
        digest = file_sha256(path)
        matches = digest == EXPECTED_SHA256[name]
        hashes_match = hashes_match and matches
        workloads[name] = {
            "sha256": digest,
            "matches_accepted_dual_model_workload": matches,
            **analyze_rows(read_jsonl(path)),
        }

    bfcl = workloads["bfcl-padded64"]
    bfcl_position = bfcl.get("causal_online_frequency_non_universal_position") or {}
    bfcl_structure_matches = (
        bfcl["requests"] == 200
        and bfcl["menu_sizes"] == [64]
        and bfcl["distinct_tools"] == 262
        and bfcl["universal_tools"] == 63
        and bfcl["support_histogram"].get("1") == 198
        and bfcl["support_histogram"].get("2") == 1
        and bfcl["support_histogram"].get("200") == 63
        and bfcl["one_non_universal_tool_per_request"] is True
        and bfcl_position.get("first_request") == 13
        and bfcl_position.get("last_position_requests") == 199
        and abs(float(bfcl_position.get("mean", 0.0)) - 63.745) < 1e-12
    )
    retrieved_has_no_universal_set = all(
        workloads[name]["universal_tools"] == 0
        for name in EXPECTED_SHA256
        if name.startswith("toolret-bm25")
    )
    all_checks_passed = (
        hashes_match and bfcl_structure_matches and retrieved_has_no_universal_set
    )
    return {
        "format_version": 1,
        "status": "accepted" if all_checks_passed else "failed",
        "all_checks_passed": all_checks_passed,
        "checks": {
            "all_input_hashes_match": hashes_match,
            "bfcl_padded64_structure_matches": bfcl_structure_matches,
            "retrieved_workloads_have_no_universal_tool_set": retrieved_has_no_universal_set,
        },
        "workloads": workloads,
    }


def parse_workloads(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--workload must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate workload: {name}")
        result[name] = Path(raw_path).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        inputs = parse_workloads(args.workload)
        result = audit(inputs)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
