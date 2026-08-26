#!/usr/bin/env python3
"""Compare smoke-test summaries produced by smoke_matrix.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample", type=int)
    selection.add_argument("--scene-ids", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-types", nargs="+")
    args = parser.parse_args()

    sample = args.sample if args.sample is not None else (None if args.scene_ids else 10)
    if args.scene_ids:
        digest = hashlib.sha1(",".join(args.scene_ids).encode()).hexdigest()[:8]
        selection_label = f"ids{len(args.scene_ids)}-{digest}"
    else:
        selection_label = f"n{sample}-seed{args.seed}"
    if args.test_types:
        selection_label += "-types-" + slug("-".join(args.test_types))
    label = f"smoke_{slug(args.model)}_{selection_label}"
    rows = []
    for run in args.runs:
        path = run / f"summary_{label}.json"
        if not path.exists():
            raise SystemExit(f"Missing summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        usage = summary.get("judge_usage", {})
        rows.append(
            {
                "run": run.name,
                "pass_rate": summary.get("unit_test_pass_rate"),
                "pass_rate_without_basic": summary.get(
                    "unit_test_pass_rate_without_basic_validity"
                ),
                "unsupported": summary.get("num_unsupported"),
                "errors": summary.get("num_error"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": usage.get("cost_usd"),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
