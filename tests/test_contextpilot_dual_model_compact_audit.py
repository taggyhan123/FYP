import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_contextpilot_dual_model_compact.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("dual_model_compact_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_dual_model_compact_evidence_passes() -> None:
    module = load_audit_module()
    result = module.audit()

    assert result["all_checks_passed"] is True
    assert result["status"] == "compact_evidence_valid_raw_archive_not_locally_verified"
    assert result["raw_archive_verified_by_this_audit"] is False
    checks = {row["name"]: row for row in result["checks"]}
    assert checks[
        "both_contextpilot_arms_exceed_tooltrie_in_all_systems_cells"
    ]["details"] == {"model_workload_cells": 12}
