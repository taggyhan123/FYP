#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.download import download_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public ToolRet and BFCL inputs used by TATM."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = download_all(args.data_dir, force=args.force)
    mib = manifest["total_bytes"] / (1024 * 1024)
    print(f"Downloaded/verified {len(manifest['files'])} files ({mib:.2f} MiB).")
    print(f"Manifest: {args.data_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
