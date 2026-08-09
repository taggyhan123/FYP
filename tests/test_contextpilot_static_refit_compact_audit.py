import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_contextpilot_static_refit_compact.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("static_refit_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_refit_and_sglang_compact_evidence_passes() -> None:
    module = load_audit_module()
    result = module.audit()

    assert result["all_checks_passed"] is True
    assert result["raw_archives_verified_by_this_audit"] is False
    assert result["per_case_static_vs_persistent_identity_verified_by_this_audit"] is False
    checks = {row["name"]: row for row in result["checks"]}
    assert checks["static_refit_systems_compact"]["details"]["accepted_replays"] == 18
    assert checks["sglang_independent_aggregate_counter_audit"]["details"] == {
        "declared": 72,
        "accepted": 72,
        "refused": 0,
        "unique_files": 72,
        "conditions": 12,
    }
