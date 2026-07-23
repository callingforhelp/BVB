#!/usr/bin/env python3
"""Merge independent unit-test shard JSONL files and regenerate a summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_utils import read_jsonl
from unit_test_metric import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in read_jsonl(path)]
    by_id = {}
    for row in rows:
        scene_id = str(row.get("id"))
        if scene_id in by_id:
            raise SystemExit(f"Duplicate scene id across shards: {scene_id}")
        by_id[scene_id] = row
    merged = [by_id[key] for key in sorted(by_id)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    summary = summarize(merged)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"scenes": len(merged), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
