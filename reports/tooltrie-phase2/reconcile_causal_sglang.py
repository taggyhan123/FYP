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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.sglang_client import initial_missing_cached_reconciliation

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
        reconciliation = initial_missing_cached_reconciliation(d)
        checks = reconciliation["checks"]
        if not reconciliation["clean"]:
            print(f"REFUSE {src.name}: {[k for k,v in checks.items() if not v]}")
            fail += 1
            continue
        cv["clean"] = True
        cv["reconciliation_note"] = NOTE
        (OUT / src.name).write_text(json.dumps(d))
        print(
            f"OK {src.name}  "
            f"cached={reconciliation['reported_cached_tokens']}  "
            f"missing={reconciliation['missing_cached_indices']}"
        )
        ok += 1
print(f"\nreconciled {ok}, refused {fail}")
sys.exit(1 if fail else 0)
