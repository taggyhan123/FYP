from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tatm.io import read_jsonl, write_json, write_jsonl
from tatm.models import CanonicalTool, TaskRecord
from tatm.normalize import bfcl_tool_id, normalize_tool, parse_maybe_structured
from tatm.serialization import Counter as TokenCounterProtocol


def _parquet_records(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048):
        yield from batch.to_pylist()


def _label_ids(raw: Any) -> tuple[str, ...]:
    parsed, valid = parse_maybe_structured(raw)
    if not valid or not isinstance(parsed, list):
        return ()
    result: list[str] = []
    for label in parsed:
        if isinstance(label, str):
            result.append(label)
        elif isinstance(label, Mapping) and label.get("id"):
            result.append(str(label["id"]))
    return tuple(dict.fromkeys(result))


def _question_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    pieces: list[str] = []
    if isinstance(raw, list):
        for turn in raw:
            messages = turn if isinstance(turn, list) else [turn]
            for message in messages:
                if isinstance(message, Mapping) and message.get("content"):
                    pieces.append(str(message["content"]))
    return "\n".join(pieces)


def process_toolret(
    raw_dir: Path,
    counter: TokenCounterProtocol,
) -> tuple[dict[str, CanonicalTool], list[TaskRecord], dict[str, Any]]:
    tools: dict[str, CanonicalTool] = {}
    duplicate_ids = 0
    tools_by_config: Counter[str] = Counter()
    for path in sorted((raw_dir / "toolret" / "tools").glob("*.parquet")):
        config = path.stem
        for row in _parquet_records(path):
            raw_id = str(row.get("id", ""))
            tool = normalize_tool(
                raw_id,
                f"toolret:{config}",
                row.get("documentation"),
                counter,
                domain=raw_id.split("_tool_", 1)[0],
            )
            if tool.tool_id in tools:
                duplicate_ids += 1
                continue
            tools[tool.tool_id] = tool
            tools_by_config[config] += 1

    tasks: list[TaskRecord] = []
    tasks_by_config: Counter[str] = Counter()
    missing_label_references = 0
    empty_labels = 0
    for path in sorted((raw_dir / "toolret" / "queries").glob("*.parquet")):
        config = path.stem
        for row in _parquet_records(path):
            labels = _label_ids(row.get("labels"))
            if not labels:
                empty_labels += 1
            missing_label_references += sum(label not in tools for label in labels)
            query = str(row.get("query") or row.get("instruction") or "")
            tasks.append(
                TaskRecord(
                    task_id=str(row.get("id", f"{config}:{len(tasks)}")),
                    source=f"toolret:{config}",
                    query=query,
                    tool_ids=labels,
                    evidence_type="gold_relevance",
                    domain=str(row.get("category") or config),
                )
            )
            tasks_by_config[config] += 1

    stats = {
        "tools": len(tools),
        "tasks": len(tasks),
        "tools_by_config": dict(sorted(tools_by_config.items())),
        "tasks_by_config": dict(sorted(tasks_by_config.items())),
        "duplicate_tool_ids": duplicate_ids,
        "empty_label_tasks": empty_labels,
        "missing_label_references": missing_label_references,
    }
    return tools, tasks, stats


def process_bfcl(
    raw_dir: Path,
    counter: TokenCounterProtocol,
) -> tuple[dict[str, CanonicalTool], list[TaskRecord], dict[str, Any]]:
    tools: dict[str, CanonicalTool] = {}
    tasks: list[TaskRecord] = []
    tasks_by_config: Counter[str] = Counter()
    duplicate_definitions = 0

    for path in sorted((raw_dir / "bfcl").glob("BFCL_v4_*.json")):
        config = path.stem.removeprefix("BFCL_v4_")
        for row in read_jsonl(path):
            task_tool_ids: list[str] = []
            raw_functions = row.get("function") or []
            if isinstance(raw_functions, Mapping):
                raw_functions = [raw_functions]
            for raw_function in raw_functions:
                if not isinstance(raw_function, Mapping):
                    continue
                provisional = normalize_tool(
                    "bfcl:provisional",
                    f"bfcl:{config}",
                    raw_function,
                    counter,
                    domain=config,
                )
                tool_id = bfcl_tool_id(raw_function, provisional.canonical_json)
                tool = replace(provisional, tool_id=tool_id)
                if tool_id in tools:
                    duplicate_definitions += 1
                else:
                    tools[tool_id] = tool
                task_tool_ids.append(tool_id)

            tasks.append(
                TaskRecord(
                    task_id=str(row.get("id", f"{config}:{len(tasks)}")),
                    source=f"bfcl:{config}",
                    query=_question_text(row.get("question")),
                    tool_ids=tuple(dict.fromkeys(task_tool_ids)),
                    evidence_type="exposed_menu",
                    domain=config,
                )
            )
            tasks_by_config[config] += 1

    stats = {
        "tools": len(tools),
        "tasks": len(tasks),
        "tasks_by_config": dict(sorted(tasks_by_config.items())),
        "duplicate_definitions_merged": duplicate_definitions,
    }
    return tools, tasks, stats


def process_all(
    raw_dir: Path,
    output_dir: Path,
    counter: TokenCounterProtocol,
) -> dict[str, Any]:
    toolret_tools, toolret_tasks, toolret_stats = process_toolret(raw_dir, counter)
    bfcl_tools, bfcl_tasks, bfcl_stats = process_bfcl(raw_dir, counter)

    combined_tools = {**toolret_tools, **bfcl_tools}
    combined_tasks = [*toolret_tasks, *bfcl_tasks]
    write_jsonl(
        output_dir / "tools.jsonl",
        (tool.to_record() for tool in combined_tools.values()),
    )
    write_jsonl(
        output_dir / "tasks.jsonl",
        (task.to_record() for task in combined_tasks),
    )
    metadata = {
        "format_version": 1,
        "tokenizer": counter.model_id,
        "toolret": toolret_stats,
        "bfcl": bfcl_stats,
        "combined": {
            "tools": len(combined_tools),
            "tasks": len(combined_tasks),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata
