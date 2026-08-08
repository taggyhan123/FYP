import json
import runpy
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "tooltrie-phase2"
    / "reconcile_causal_sglang.py"
)


def _script_namespace() -> dict[str, object]:
    return runpy.run_path(SCRIPT.as_posix())


def test_declared_sglang_audit_matrix_has_all_72_runs(tmp_path: Path) -> None:
    namespace = _script_namespace()
    paths = namespace["expected_paths"](tmp_path)  # type: ignore[operator]

    assert len(paths) == 72
    assert len(set(paths)) == 72
    assert tmp_path / "bfcl-original-sglang-trial-1.json" in paths
    assert tmp_path / "toolret-contextpilot_causal-sglang-trial-3.json" in paths


def test_sglang_audit_fails_without_writing_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_namespace()
    output = tmp_path / "audit-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            SCRIPT.as_posix(),
            "--input-dir",
            (tmp_path / "missing-input").as_posix(),
            "--output-dir",
            output.as_posix(),
        ],
    )

    with pytest.raises(SystemExit) as error:
        namespace["main"]()  # type: ignore[operator]

    assert error.value.code == 1
    assert not output.exists()


def test_sglang_audit_writes_only_after_all_72_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_namespace()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    payload = {
        "aggregate_metric_delta": {"sglang:cached_tokens_total": 5},
        "counter_validation": {
            "request_counter_matches": True,
            "prompt_counter_matches": True,
        },
        "results": [
            {
                "index": 0,
                "finish_reason": "stop",
                "usage": {"prompt_tokens_details": {}},
            },
            {
                "index": 1,
                "finish_reason": "stop",
                "usage": {"prompt_tokens_details": {"cached_tokens": 5}},
            },
        ],
    }
    for path in namespace["expected_paths"](input_dir):  # type: ignore[operator]
        path.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "audit-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            SCRIPT.as_posix(),
            "--input-dir",
            input_dir.as_posix(),
            "--output-dir",
            output.as_posix(),
        ],
    )
    namespace["main"]()  # type: ignore[operator]

    summary = json.loads(
        (output / "aggregate-counter-audit-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["declared_runs"] == 72
    assert summary["accepted_runs"] == 72
    assert summary["all_clean"] is True
    assert len(list(output.glob("*-sglang-trial-*.json"))) == 72
