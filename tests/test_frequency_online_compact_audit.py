import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_frequency_online_compact.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("frequency_online_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frequency_online_compact_audit_catches_the_two_k4_losses() -> None:
    module = load_audit_module()
    result = module.audit()

    assert result["all_checks_passed"] is True
    assert result["raw_replay_counter_cleanliness_verified_by_this_audit"] is False
    checks = {row["name"]: row for row in result["checks"]}
    comparison = checks["frequency_online_vs_tooltrie_count"]["details"]
    assert comparison["wins"] == 10
    assert comparison["cells"] == 12
    assert comparison["losses"] == [
        ("Qwen/Qwen3-0.6B", "toolret-bm25-k4"),
        ("Qwen/Qwen3-4B", "toolret-bm25-k4"),
    ]
