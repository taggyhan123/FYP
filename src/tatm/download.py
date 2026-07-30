from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from tatm.io import write_json


TOOLRET_TOOLS_API = (
    "https://huggingface.co/api/datasets/mangopy/ToolRet-Tools/parquet"
)
TOOLRET_QUERIES_API = (
    "https://huggingface.co/api/datasets/mangopy/ToolRet-Queries/parquet"
)
BFCL_RAW_ROOT = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
BFCL_FILES = (
    "BFCL_v4_simple_python.json",
    "BFCL_v4_multiple.json",
    "BFCL_v4_parallel.json",
    "BFCL_v4_parallel_multiple.json",
    "BFCL_v4_irrelevance.json",
)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": "TATM-FYP/0.1 dataset research"},
    )


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(_request(url), timeout=60) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, force: bool = False) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        if temporary.exists():
            temporary.unlink()
        with urllib.request.urlopen(_request(url), timeout=120) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    return {
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "url": url,
    }


def _parquet_urls(api_url: str) -> dict[str, str]:
    payload = fetch_json(api_url)
    result: dict[str, str] = {}
    for config, splits in payload.items():
        for split_urls in splits.values():
            if not split_urls:
                continue
            if len(split_urls) != 1:
                raise ValueError(
                    f"Expected one parquet shard for {config}, got {len(split_urls)}"
                )
            result[config] = split_urls[0]
    return result


def download_all(data_dir: Path, force: bool = False) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for dataset, api_url in (
        ("tools", TOOLRET_TOOLS_API),
        ("queries", TOOLRET_QUERIES_API),
    ):
        for config, url in sorted(_parquet_urls(api_url).items()):
            destination = data_dir / "toolret" / dataset / f"{config}.parquet"
            entry = download_file(url, destination, force)
            entry.update(dataset=f"toolret-{dataset}", config=config)
            entries.append(entry)

    for filename in BFCL_FILES:
        destination = data_dir / "bfcl" / filename
        entry = download_file(f"{BFCL_RAW_ROOT}/{filename}", destination, force)
        entry.update(dataset="bfcl", config=filename.removesuffix(".json"))
        entries.append(entry)

    manifest = {
        "format_version": 1,
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    write_json(data_dir / "manifest.json", manifest)
    return manifest
