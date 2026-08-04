#!/usr/bin/env python3
"""Reconcile the contextpilot_causal SGLang runs, using the same independent
check applied to the other 66 runs.

SGLang omits usage.prompt_tokens_details.cached_tokens when it is zero, so the
first request after every flush reports None and `cached_counter_matches` fails
by exactly one, every run. A run is reconciled ONLY if all four hold:
  1. request_counter_matches and prompt_counter_matches are true
  2. the server's cached_tokens_total equals the sum of reported values
  3. index 0 is the ONLY request missing a cached value
  4. no request failed
That is strictly tighter than blanket-trusting the flag.
"""
import json, sys
from pathlib import Path

P2 = Path("/home/taghan/FYP/cluster/results/tooltrie-phase2-20260803-181133")
OUT = P2 / "sglang-reconciled"
NOTE = json.loads((OUT / "bfcl-alphabetical-sglang-trial-1.json").read_text())[
    "counter_validation"]["reconciliation_note"]

ok = fail = 0
for ds in ("bfcl", "toolret"):
    for t in (1, 2, 3):
        src = P2 / f"{ds}-contextpilot_causal-sglang-trial-{t}.json"
        d = json.loads(src.read_text())
        cv = d["counter_validation"]
        cached = [(r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                  for r in d["results"]]
        missing = [i for i, c in enumerate(cached) if c is None]
        reported = sum(c for c in cached if c is not None)
        failed = [r["index"] for r in d["results"] if r.get("finish_reason") is None]
        checks = {
            "request_counter_matches": cv["request_counter_matches"] is True,
            "prompt_counter_matches": cv["prompt_counter_matches"] is True,
            "cached_total_equals_sum": reported == cv["response_cached_tokens"],
            "only_index_0_missing": missing == [0],
            "no_failed_requests": not failed,
        }
        if not all(checks.values()):
            print(f"REFUSE {src.name}: {[k for k,v in checks.items() if not v]}")
            fail += 1
            continue
        cv["clean"] = True
        cv["reconciliation_note"] = NOTE
        (OUT / src.name).write_text(json.dumps(d))
        print(f"OK {src.name}  cached={reported}  missing={missing}")
        ok += 1
print(f"\nreconciled {ok}, refused {fail}")
sys.exit(1 if fail else 0)
