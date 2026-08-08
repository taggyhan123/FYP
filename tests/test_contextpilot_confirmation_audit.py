import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_contextpilot_confirmation.py"


def test_tracked_contextpilot_measurements_pass_but_run_is_incomplete() -> None:
    spec = importlib.util.spec_from_file_location("confirmation_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.audit(module.DEFAULT_CONFIRMATION, module.DEFAULT_QUALITY_4B)

    assert result["tracked_measurement_artifact_checks_passed"] is True
    assert result["tracked_analysis_metadata_checks_passed"] is False
    assert result["static_refit_causal_gpu_arm_present"] is False
    assert result["missing_static_refit_gpu_replays"] == 19
    assert len(result["files_needing_sequence_state_regeneration"]) == 18
    assert result["status"] == "partial_requires_cluster_followup"
