import runpy
from pathlib import Path


verify_configurations = runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "compare_probe_runs.py"
)["verify_configurations"]


def probe_run(*, cache_enabled: bool, reset: bool) -> dict:
    return {
        "format_version": 2,
        "server_cache_config": {"enable_prefix_caching": cache_enabled},
        "prefix_cache_reset_between_trials": reset,
        "trials": [],
    }


def test_probe_comparison_requires_successful_resets() -> None:
    enabled = probe_run(cache_enabled=True, reset=False)
    disabled = probe_run(cache_enabled=False, reset=True)
    problems = verify_configurations(enabled, disabled)
    assert any("Cache-enabled run" in problem for problem in problems)


def test_probe_comparison_accepts_valid_controls() -> None:
    enabled = probe_run(cache_enabled=True, reset=True)
    disabled = probe_run(cache_enabled=False, reset=True)
    assert verify_configurations(enabled, disabled) == []
