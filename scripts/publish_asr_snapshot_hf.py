"""Publish and round-trip verify an ASR development snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from offline.asr_snapshot_hf import publish_snapshot_and_verify


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=os.environ.get("AIC_ASR_HF_REPO", ""))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()
    if not args.repo_id:
        raise ValueError("--repo-id or AIC_ASR_HF_REPO is required")
    print(json.dumps(publish_snapshot_and_verify(args.snapshot_dir, repo_id=args.repo_id, token=args.token), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
