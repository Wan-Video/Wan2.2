#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Read a JSON job file and run ``generate.generate`` (for torchrun / serving workers)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import generate as wan_generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Wan2.2 JSON job runner (use with torchrun).")
    parser.add_argument(
        "--job_json",
        type=str,
        required=True,
        help="Path to JSON job spec (see README.md).",
    )
    args_ns = parser.parse_args()
    path = Path(args_ns.job_json)
    if not path.is_file():
        raise FileNotFoundError(f"job_json not found: {path}")
    job = json.loads(path.read_text(encoding="utf-8"))
    gen_args = wan_generate.args_from_job_dict(job)
    wan_generate.generate(gen_args)


if __name__ == "__main__":
    main()
