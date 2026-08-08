#!/usr/bin/env python3
"""Run one model arm of the predeclared ContextPilot dual-model protocol.

The model server is started separately so its exact command and logs remain
visible.  This driver refuses an existing output directory, builds the
capacity-dependent ToolTrie workloads from the live server capacity, runs the
complete systems and quality matrices, and validates every output before it is
accepted.  A failed attempt is preserved and must not be resumed or
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.vllm_client import served_model, server_cache_config


MODEL_REVISIONS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
}
MODEL_SLUGS = {
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
}
EXPECTED_CONTEXTPILOT_COMMIT = "1fa0a143fdeda344585666648ab2b30cb7fea77f"
SYSTEM_WORKLOADS = (
    ("bfcl-padded64", 48),
    ("toolret-padded64", 48),
    ("toolret-bm25-k4", 1),
    ("toolret-bm25-k16", 1),
    ("toolret-bm25-k64", 1),
    ("toolret-bm25-k128", 1),
)
CONDITIONS = (
    "original",
    "alphabetical",
    "tooltrie_v0",
    "contextpilot-static_refit_causal",
    "contextpilot-online_incremental",
)
EXPECTED_ORDERING = {
    "original": "original",
    "alphabetical": "alphabetical",
    "tooltrie_v0": "tooltrie_v0",
    "contextpilot-static_refit_causal": "contextpilot_static_refit_causal",
    "contextpilot-online_incremental": "contextpilot_online_incremental",
}
# Rotate the condition that runs first so server-age drift is not assigned to
# one policy in all three systems trials.
CONDITION_ORDER_BY_TRIAL = {
    1: CONDITIONS,
    2: (
        "contextpilot-online_incremental",
        "original",
        "alphabetical",
        "tooltrie_v0",
        "contextpilot-static_refit_causal",
    ),
    3: (
        "contextpilot-static_refit_causal",
        "contextpilot-online_incremental",
        "original",
        "alphabetical",
        "tooltrie_v0",
    ),
}
QUALITY_CONDITION_ORDER = CONDITIONS
QUALITY_PAIRS = (
    ("original", "tooltrie_v0"),
    ("original", "contextpilot-static_refit_causal"),
    ("original", "contextpilot-online_incremental"),
    ("alphabetical", "tooltrie_v0"),
    ("alphabetical", "contextpilot-static_refit_causal"),
    ("alphabetical", "contextpilot-online_incremental"),
    ("tooltrie_v0", "contextpilot-static_refit_causal"),
    ("tooltrie_v0", "contextpilot-online_incremental"),
    ("contextpilot-static_refit_causal", "contextpilot-online_incremental"),
)
QUALITY_METRICS = ("name_correct", "full_correct", "no_tool_correct")


class ReplayPlan(NamedTuple):
    stem: str
    condition: str
    trial: int
    max_tokens: int


def systems_plan() -> list[ReplayPlan]:
    return [
        ReplayPlan(stem, condition, trial, max_tokens)
        for stem, max_tokens in SYSTEM_WORKLOADS
        for trial, order in CONDITION_ORDER_BY_TRIAL.items()
        for condition in order
    ]


def run_command(command: list[str], log_path: Path) -> None:
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite log: {log_path}")
    print(f"+ {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(
            f"Command failed with exit {result.returncode}; see {log_path}"
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def validate_shared_workloads(shared: Path) -> None:
    stems = [item[0] for item in SYSTEM_WORKLOADS] + ["quality"]
    for stem in stems:
        expected_count = 800 if stem == "quality" else 200
        payloads: dict[str, list[dict[str, Any]]] = {}
        for condition in CONDITIONS:
            if condition == "tooltrie_v0":
                continue
            path = shared / f"{stem}-{condition}.jsonl"
            rows = read_jsonl(path)
            if len(rows) != expected_count:
                raise ValueError(
                    f"{path}: expected {expected_count} rows, got {len(rows)}"
                )
            orderings = {row.get("ordering") for row in rows}
            if orderings != {EXPECTED_ORDERING[condition]}:
                raise ValueError(f"{path}: unexpected ordering labels {orderings}")
            payloads[condition] = rows

        reference = payloads["original"]
        reference_cases = [
            str(row.get("case_id") or row.get("task_id")) for row in reference
        ]
        reference_menus = [
            sorted(str(tool_id) for tool_id in row.get("tool_ids", []))
            for row in reference
        ]
        for condition, rows in payloads.items():
            cases = [str(row.get("case_id") or row.get("task_id")) for row in rows]
            menus = [
                sorted(str(tool_id) for tool_id in row.get("tool_ids", []))
                for row in rows
            ]
            if cases != reference_cases or menus != reference_menus:
                raise ValueError(
                    f"{stem}-{condition}: request sequence or membership changed"
                )

        for condition, mode in (
            ("contextpilot-static_refit_causal", "static_refit_causal"),
            ("contextpilot-online_incremental", "online_incremental"),
        ):
            summary_path = shared / f"{stem}-{condition}-summary.json"
            if not summary_path.is_file():
                raise ValueError(f"Missing ContextPilot summary: {summary_path}")
            summary = load_json(summary_path)
            reference_metadata = summary.get("reference") or {}
            output_path = shared / f"{stem}-{condition}.jsonl"
            online = mode == "online_incremental"
            if not (
                summary.get("mode") == mode
                and summary.get("information_regime") == "causal"
                and summary.get("request_order_changed") is False
                and summary.get("official_online_api_used") is online
                and summary.get("full_contextpilot_system") is False
                and summary.get("annotations_enabled") is False
                and summary.get("eviction_feedback_enabled") is False
                and reference_metadata.get("commit")
                == EXPECTED_CONTEXTPILOT_COMMIT
                and reference_metadata.get("alpha") == 0.001
                and reference_metadata.get("persistent_index") is online
                and summary.get("output_sha256") == file_sha256(output_path)
                and summary.get("input_sha256")
                == file_sha256(shared / f"{stem}-original.jsonl")
            ):
                raise ValueError(f"{summary_path}: provenance validation failed")


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replay_errors(
    payload: dict[str, Any],
    *,
    expected_model: str,
    expected_requests: int,
    expected_ordering: str,
) -> list[str]:
    errors: list[str] = []
    results = payload.get("results")
    if payload.get("engine") != "vllm":
        errors.append("engine is not vllm")
    if payload.get("model") != expected_model:
        errors.append(f"model is {payload.get('model')!r}")
    if payload.get("request_count") != expected_requests:
        errors.append(f"request_count is {payload.get('request_count')!r}")
    if not isinstance(results, list) or len(results) != expected_requests:
        errors.append("result-row count does not match")
        results = []
    if payload.get("cache_reset_before") is not True:
        errors.append("cache_reset_before is not true")
    validation = payload.get("counter_validation") or {}
    if validation.get("clean") is not True:
        errors.append("counter_validation.clean is not true")
    if validation.get("query_counter_matches_response_prompt_tokens") is not True:
        errors.append("query counter does not match response prompt tokens")
    if validation.get("cached_plus_computed_matches_queries") is not True:
        errors.append("cached plus computed tokens do not match query tokens")
    expected_role = (
        "ordinary_text_prefill_fallback"
        if expected_ordering == "original"
        else "ordering_candidate"
    )
    if (payload.get("execution_condition") or {}).get("role") != expected_role:
        errors.append(f"execution-condition role is not {expected_role}")
    orderings = {row.get("ordering") for row in results}
    if orderings != {expected_ordering}:
        errors.append(f"unexpected row orderings: {sorted(map(str, orderings))}")
    case_ids = [str(row.get("case_id") or row.get("task_id")) for row in results]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case IDs")
    return errors


def validate_replay(
    path: Path,
    *,
    expected_model: str,
    expected_requests: int,
    expected_ordering: str,
) -> dict[str, Any]:
    payload = load_json(path)
    errors = replay_errors(
        payload,
        expected_model=expected_model,
        expected_requests=expected_requests,
        expected_ordering=expected_ordering,
    )
    if errors:
        raise ValueError(f"{path}: {'; '.join(errors)}")
    return payload


def workload_path(
    shared: Path, output: Path, stem: str, condition: str
) -> Path:
    if condition == "tooltrie_v0":
        return output / "workloads" / f"{stem}-{condition}.jsonl"
    return shared / f"{stem}-{condition}.jsonl"


def build_tooltrie_workloads(
    shared: Path,
    output: Path,
    capacity_tokens: int,
) -> None:
    for stem in [item[0] for item in SYSTEM_WORKLOADS] + ["quality"]:
        target = output / "workloads" / f"{stem}-tooltrie_v0.jsonl"
        log_path = output / "logs" / f"build-{stem}-tooltrie_v0.log"
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_tooltrie_workload.py"),
                "--input",
                str(shared / f"{stem}-original.jsonl"),
                "--policy",
                "tooltrie_v0",
                "--fallback",
                "alphabetical",
                "--recency-window",
                "128",
                "--capacity-tokens",
                str(capacity_tokens),
                "--output",
                str(target),
            ],
            log_path,
        )
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        expected = 800 if stem == "quality" else 200
        if len(rows) != expected:
            raise ValueError(f"{target}: expected {expected} records, got {len(rows)}")
        final_state = (rows[-1].get("tooltrie_state_after") or {}) if rows else {}
        if final_state.get("capacity_tokens") != capacity_tokens:
            raise ValueError(f"{target}: ToolTrie capacity does not match live server")


def run_replay(
    *,
    source: Path,
    target: Path,
    log_path: Path,
    label: str,
    model: str,
    condition: str,
    requests: int,
    max_tokens: int,
    base_url: str,
    limit: int | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "replay_vllm_workload.py"),
        "--input",
        str(source),
        "--run-label",
        label,
        "--base-url",
        base_url,
        "--model",
        model,
        "--max-tokens",
        str(max_tokens),
        "--disable-thinking",
        "--reset-before",
        "--output",
        str(target),
    ]
    if condition == "original":
        command.extend(["--condition-role", "ordinary_text_prefill_fallback"])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    run_command(command, log_path)
    return validate_replay(
        target,
        expected_model=model,
        expected_requests=requests,
        expected_ordering=EXPECTED_ORDERING[condition],
    )


def run_systems(
    *, shared: Path, output: Path, model: str, slug: str, base_url: str
) -> list[Path]:
    replay_paths: list[Path] = []
    for item in systems_plan():
        source = workload_path(shared, output, item.stem, item.condition)
        target = (
            output
            / "systems"
            / f"{item.stem}-{item.condition}-trial-{item.trial}.json"
        )
        run_replay(
            source=source,
            target=target,
            log_path=output / "logs" / f"{target.stem}.log",
            label=f"{slug}-{target.stem}",
            model=model,
            condition=item.condition,
            requests=200,
            max_tokens=item.max_tokens,
            base_url=base_url,
        )
        replay_paths.append(target)

    for stem, _ in SYSTEM_WORKLOADS:
        summary = output / "summaries" / f"{stem}-summary.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "summarize_ordering_replays.py"),
        ]
        for condition in CONDITIONS:
            for trial in (1, 2, 3):
                path = (
                    output
                    / "systems"
                    / f"{stem}-{condition}-trial-{trial}.json"
                )
                command.extend(["--run", f"{condition}={path}"])
        command.extend(["--output", str(summary)])
        run_command(command, output / "logs" / f"summarize-{stem}.log")
        payload = load_json(summary)
        if set(payload.get("conditions") or {}) != set(CONDITIONS):
            raise ValueError(f"{summary}: incomplete condition set")
        if not all(
            payload.get(field) is True
            for field in (
                "all_conditions_have_same_case_set",
                "all_conditions_have_same_request_sequence",
                "all_conditions_have_same_selected_tool_sets",
            )
        ):
            raise ValueError(f"{summary}: an equivalence guard failed")
    return replay_paths


def validate_quality_score(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    overall = payload.get("overall") or {}
    if payload.get("skipped_no_ground_truth") != 0:
        raise ValueError(f"{path}: skipped one or more ground-truth rows")
    if len(payload.get("scores") or []) != 800:
        raise ValueError(f"{path}: expected 800 task-level scores")
    expected = {
        "tasks_scored": 800,
        "relevance_tasks": 640,
        "irrelevance_tasks": 160,
    }
    if any(overall.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{path}: quality task counts do not match {expected}")
    return payload


def run_quality(
    *, shared: Path, output: Path, model: str, slug: str, base_url: str
) -> tuple[list[Path], list[Path]]:
    replay_paths: list[Path] = []
    score_paths: list[Path] = []
    compact_conditions: dict[str, Any] = {}
    for condition in QUALITY_CONDITION_ORDER:
        source = workload_path(shared, output, "quality", condition)
        replay = output / "quality" / f"quality-{condition}-replay.json"
        score = output / "quality" / f"quality-{condition}-score.json"
        run_replay(
            source=source,
            target=replay,
            log_path=output / "logs" / f"quality-{condition}-replay.log",
            label=f"{slug}-quality-{condition}",
            model=model,
            condition=condition,
            requests=800,
            max_tokens=128,
            base_url=base_url,
        )
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "score_bfcl_quality.py"),
                "--replay-result",
                str(replay),
                "--output",
                str(score),
            ],
            output / "logs" / f"quality-{condition}-score.log",
        )
        score_payload = validate_quality_score(score)
        compact_conditions[condition] = {
            "overall": score_payload["overall"],
            "by_domain": score_payload["by_domain"],
        }
        replay_paths.append(replay)
        score_paths.append(score)

    write_new_json(
        output / "summaries" / "quality-scores-compact.json",
        {
            "format_version": 1,
            "model": model,
            "n_per_condition": 800,
            "conditions": compact_conditions,
        },
    )
    return replay_paths, score_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one complete model arm of the ContextPilot dual-model matrix."
    )
    parser.add_argument("--shared-workloads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_REVISIONS), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the shared workloads and exit without contacting a server.",
    )
    args = parser.parse_args()

    expected_revision = MODEL_REVISIONS[args.model]
    if args.model_revision != expected_revision:
        parser.error(
            f"--model-revision must be the predeclared {expected_revision}"
        )
    shared = args.shared_workloads.resolve()
    if not shared.is_dir():
        parser.error(f"Shared workload directory does not exist: {shared}")

    required_shared = [
        shared / f"{stem}-{condition}.jsonl"
        for stem in [item[0] for item in SYSTEM_WORKLOADS] + ["quality"]
        for condition in CONDITIONS
        if condition != "tooltrie_v0"
    ]
    missing = [path for path in required_shared if not path.is_file()]
    if missing:
        parser.error(f"Missing shared workload: {missing[0]}")
    try:
        validate_shared_workloads(shared)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.preflight_only:
        print(f"Shared workload preflight passed: {shared}")
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight-only is used")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(
            f"Output already exists and will not be resumed or overwritten: {output}"
        )

    live_model = served_model(args.base_url.rstrip("/"))
    if live_model != args.model:
        parser.error(f"Server reports {live_model!r}, expected {args.model!r}")
    cache_config = server_cache_config(args.base_url.rstrip("/"))
    if cache_config.get("enable_prefix_caching") is not True:
        parser.error("Live server does not report prefix caching enabled")
    try:
        block_size = int(cache_config["block_size"])
        num_gpu_blocks = int(cache_config["num_gpu_blocks"])
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"Live server has no valid cache capacity: {error}")
    capacity_tokens = block_size * num_gpu_blocks
    cache_config["capacity_tokens"] = capacity_tokens
    if block_size != 16 or capacity_tokens <= 0:
        parser.error(
            f"Expected block size 16 and positive capacity; got {cache_config}"
        )

    output.mkdir(parents=True)
    write_new_json(output / "server-cache-config.json", cache_config)
    write_new_json(
        output / "run-provenance.json",
        {
            "format_version": 1,
            "model": args.model,
            "model_revision": args.model_revision,
            "model_role": (
                "primary" if args.model == "Qwen/Qwen3-4B" else "replication"
            ),
            "base_url": args.base_url,
            "block_size": block_size,
            "capacity_tokens": capacity_tokens,
            "systems_plan": [item._asdict() for item in systems_plan()],
            "quality_condition_order": list(QUALITY_CONDITION_ORDER),
        },
    )

    build_tooltrie_workloads(shared, output, capacity_tokens)

    slug = MODEL_SLUGS[args.model]
    run_replay(
        source=shared / "toolret-bm25-k4-original.jsonl",
        target=output / "diagnostics" / "unmeasured-warmup.json",
        log_path=output / "logs" / "unmeasured-warmup.log",
        label=f"{slug}-unmeasured-warmup",
        model=args.model,
        condition="original",
        requests=1,
        max_tokens=1,
        base_url=args.base_url,
        limit=1,
    )

    system_paths = run_systems(
        shared=shared,
        output=output,
        model=args.model,
        slug=slug,
        base_url=args.base_url,
    )
    quality_replays, quality_scores = run_quality(
        shared=shared,
        output=output,
        model=args.model,
        slug=slug,
        base_url=args.base_url,
    )

    write_new_json(
        output / "model-run-summary.json",
        {
            "format_version": 1,
            "status": "gpu_complete_pending_cpu_analysis",
            "model": args.model,
            "model_revision": args.model_revision,
            "model_role": (
                "primary" if args.model == "Qwen/Qwen3-4B" else "replication"
            ),
            "native_capacity_tokens": capacity_tokens,
            "accepted_systems_replays": len(system_paths),
            "accepted_quality_replays": len(quality_replays),
            "quality_score_files": len(quality_scores),
            "diagnostic_warmups_excluded": 1,
            "all_replays_reset_and_counter_clean": True,
            "raw_outputs_preserved": True,
        },
    )
    print(
        f"Accepted {len(system_paths)} systems and {len(quality_replays)} quality "
        f"replays for {args.model}."
    )


if __name__ == "__main__":
    main()
