#!/usr/bin/env python3
"""Read back vLLM's live prefix-cache configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.vllm_client import server_cache_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a live vLLM cache.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--capacity-only", action="store_true")
    args = parser.parse_args()

    config = server_cache_config(args.base_url)
    if config["enable_prefix_caching"] is not True:
        raise SystemExit(
            "Server does not report enable_prefix_caching=True: "
            f"{config['enable_prefix_caching']}"
        )
    try:
        block_size = int(config["block_size"])
        gpu_blocks = int(config["num_gpu_blocks"])
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Incomplete vLLM cache metrics: {config}") from error
    config["capacity_tokens"] = block_size * gpu_blocks
    if args.capacity_only:
        print(config["capacity_tokens"])
    else:
        print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
