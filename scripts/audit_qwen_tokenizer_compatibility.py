#!/usr/bin/env python3
"""Verify that the pinned Qwen models share tokenizer and chat-template data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compatibility_report(files: dict[str, dict[str, Path]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    tokenizer_hashes: set[str] = set()
    chat_templates: set[str] = set()
    for model, paths in files.items():
        tokenizer_hash = file_sha256(paths["tokenizer.json"])
        config_hash = file_sha256(paths["tokenizer_config.json"])
        config = json.loads(paths["tokenizer_config.json"].read_text(encoding="utf-8"))
        chat_template = json.dumps(
            config.get("chat_template"), ensure_ascii=False, sort_keys=True
        )
        tokenizer_hashes.add(tokenizer_hash)
        chat_templates.add(chat_template)
        models[model] = {
            "revision": MODELS[model],
            "tokenizer_json_sha256": tokenizer_hash,
            "tokenizer_config_json_sha256": config_hash,
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
        }
    tokenizer_identical = len(tokenizer_hashes) == 1
    chat_template_identical = len(chat_templates) == 1
    return {
        "format_version": 1,
        "models": models,
        "tokenizer_json_identical": tokenizer_identical,
        "chat_template_identical": chat_template_identical,
        "compatible_for_shared_schema_token_accounting": (
            tokenizer_identical and chat_template_identical
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        parser.error(f"huggingface_hub is required: {error}")

    files: dict[str, dict[str, Path]] = {}
    for model, revision in MODELS.items():
        files[model] = {
            filename: Path(
                hf_hub_download(
                    repo_id=model,
                    filename=filename,
                    revision=revision,
                )
            )
            for filename in ("tokenizer.json", "tokenizer_config.json")
        }
    report = compatibility_report(files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["compatible_for_shared_schema_token_accounting"]:
        raise SystemExit(
            "Pinned tokenizers/templates differ; retokenize schemas and split "
            "the capacity-dependent planner inputs before running the GPU matrix"
        )


if __name__ == "__main__":
    main()
