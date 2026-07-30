#!/usr/bin/env python3
"""Black-box exact-prefix sanity probe for an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.vllm_client import (
    fetch_text,
    metric_delta,
    parse_prometheus,
    request_json,
    response_projection,
    served_model,
)


def functions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_temperature",
                "description": "Convert a temperature between Celsius and Fahrenheit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "target_unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["value", "target_unit"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_note",
                "description": "Save a short text note.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ]


def payload(model: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Singapore? Use a tool.",
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "seed": 0,
        "max_tokens": 48,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", help="Defaults to the first served model.")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = served_model(base_url, args.model)

    original = functions()
    changed = json.loads(json.dumps(original))
    changed[1]["function"]["description"] += " Return an exact numeric result."
    reordered = [original[1], original[0], original[2]]
    scenarios = [
        ("original_cold", original),
        ("original_identical", original),
        ("changed_second_tool", changed),
        ("reordered_first_two", reordered),
        ("original_restored", original),
    ]

    results: list[dict[str, Any]] = []
    for scenario, tools in scenarios:
        before = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        start = time.perf_counter()
        response = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload(model, tools),
        )
        wall_seconds = time.perf_counter() - start
        after = parse_prometheus(fetch_text(f"{base_url}/metrics"))
        results.append(
            {
                "scenario": scenario,
                "wall_seconds": round(wall_seconds, 6),
                **response_projection(response),
                "metric_delta": metric_delta(before, after),
            }
        )
        time.sleep(args.pause_seconds)

    output = {
        "format_version": 1,
        "run_label": args.run_label,
        "server": base_url,
        "model": model,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
