import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_contextpilot_dual_model.py"
TOKENIZER_AUDIT = PROJECT_ROOT / "scripts" / "audit_qwen_tokenizer_compatibility.py"
MANIFEST = PROJECT_ROOT / "cluster" / "contextpilot-dual-model-manifest.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("dual_model_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tokenizer_audit():
    spec = importlib.util.spec_from_file_location("tokenizer_audit", TOKENIZER_AUDIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predeclared_dual_model_plan_is_complete_and_unique() -> None:
    runner = load_runner()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = runner.systems_plan()

    assert len(plan) == 90
    assert len({(item.stem, item.condition, item.trial) for item in plan}) == 90
    assert Counter(item.condition for item in plan) == {
        condition: 18 for condition in runner.CONDITIONS
    }
    assert Counter(item.stem for item in plan) == {
        stem: 15 for stem, _ in runner.SYSTEM_WORKLOADS
    }
    assert all(
        set(order) == set(runner.CONDITIONS)
        for order in runner.CONDITION_ORDER_BY_TRIAL.values()
    )
    assert manifest["systems"]["condition_order_by_trial"] == {
        str(trial): list(order)
        for trial, order in runner.CONDITION_ORDER_BY_TRIAL.items()
    }
    assert manifest["systems"]["expected_replays_per_model"] == len(plan)
    assert manifest["systems"]["expected_replays_total"] == 2 * len(plan)
    assert manifest["quality"]["expected_replays_per_model"] == len(
        runner.QUALITY_CONDITION_ORDER
    )
    assert manifest["quality"]["expected_replays_total"] == 2 * len(
        runner.QUALITY_CONDITION_ORDER
    )
    assert manifest["acceptance"]["expected_gpu_replays_total"] == 190
    assert len(runner.QUALITY_PAIRS) * len(runner.QUALITY_METRICS) == 27


def test_replay_validation_is_fail_closed() -> None:
    runner = load_runner()
    rows = [
        {
            "case_id": f"case-{index}",
            "task_id": f"task-{index}",
            "ordering": "alphabetical",
        }
        for index in range(2)
    ]
    clean = {
        "engine": "vllm",
        "model": "Qwen/Qwen3-4B",
        "request_count": 2,
        "cache_reset_before": True,
        "counter_validation": {
            "clean": True,
            "query_counter_matches_response_prompt_tokens": True,
            "cached_plus_computed_matches_queries": True,
        },
        "execution_condition": {"role": "ordering_candidate"},
        "results": rows,
    }

    assert not runner.replay_errors(
        clean,
        expected_model="Qwen/Qwen3-4B",
        expected_requests=2,
        expected_ordering="alphabetical",
    )

    contaminated = dict(clean)
    contaminated["counter_validation"] = {"clean": False}
    contaminated["model"] = "Qwen/Qwen3-0.6B"
    errors = runner.replay_errors(
        contaminated,
        expected_model="Qwen/Qwen3-4B",
        expected_requests=2,
        expected_ordering="alphabetical",
    )
    assert "counter_validation.clean is not true" in errors
    assert any(error.startswith("model is") for error in errors)


def test_shared_workload_preflight_checks_hashes_and_membership(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    runner.SYSTEM_WORKLOADS = (("demo", 1),)

    def rows(condition: str, count: int) -> list[dict]:
        ordering = runner.EXPECTED_ORDERING[condition]
        return [
            {
                "case_id": f"case-{index}",
                "task_id": f"task-{index}",
                "ordering": ordering,
                "tool_ids": ["tool-a", f"tool-{index % 3}"],
                "tools": [{"type": "function"}, {"type": "function"}],
            }
            for index in range(count)
        ]

    for stem, count in (("demo", 200), ("quality", 800)):
        for condition in runner.CONDITIONS:
            if condition == "tooltrie_v0":
                continue
            path = tmp_path / f"{stem}-{condition}.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows(condition, count)),
                encoding="utf-8",
            )
        for condition, mode in (
            ("contextpilot-static_refit_causal", "static_refit_causal"),
            ("contextpilot-online_incremental", "online_incremental"),
        ):
            online = mode == "online_incremental"
            summary = {
                "mode": mode,
                "information_regime": "causal",
                "request_order_changed": False,
                "official_online_api_used": online,
                "full_contextpilot_system": False,
                "annotations_enabled": False,
                "eviction_feedback_enabled": False,
                "input_sha256": runner.file_sha256(
                    tmp_path / f"{stem}-original.jsonl"
                ),
                "output_sha256": runner.file_sha256(
                    tmp_path / f"{stem}-{condition}.jsonl"
                ),
                "reference": {
                    "commit": runner.EXPECTED_CONTEXTPILOT_COMMIT,
                    "alpha": 0.001,
                    "persistent_index": online,
                },
            }
            (tmp_path / f"{stem}-{condition}-summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

    runner.validate_shared_workloads(tmp_path)

    static_path = tmp_path / "demo-contextpilot-static_refit_causal.jsonl"
    static_path.write_text(static_path.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="provenance validation failed"):
        runner.validate_shared_workloads(tmp_path)


def test_tokenizer_compatibility_requires_both_vocab_and_template(
    tmp_path: Path,
) -> None:
    audit = load_tokenizer_audit()
    files = {}
    for index, model in enumerate(audit.MODELS):
        model_dir = tmp_path / str(index)
        model_dir.mkdir()
        tokenizer = model_dir / "tokenizer.json"
        config = model_dir / "tokenizer_config.json"
        tokenizer.write_text('{"same": true}', encoding="utf-8")
        config.write_text('{"chat_template": "same"}', encoding="utf-8")
        files[model] = {
            "tokenizer.json": tokenizer,
            "tokenizer_config.json": config,
        }

    report = audit.compatibility_report(files)
    assert report["compatible_for_shared_schema_token_accounting"] is True

    second = files["Qwen/Qwen3-0.6B"]["tokenizer_config.json"]
    second.write_text('{"chat_template": "different"}', encoding="utf-8")
    report = audit.compatibility_report(files)
    assert report["tokenizer_json_identical"] is True
    assert report["chat_template_identical"] is False
    assert report["compatible_for_shared_schema_token_accounting"] is False
