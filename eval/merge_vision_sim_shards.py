#!/usr/bin/env python3
"""Merge independent vision_sim shard JSONL files and regenerate a summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval_utils import read_jsonl
from vjepa_sim_metric import summarize

_SHARD_JSONL_RE = re.compile(r"^vision_sim\.shard-(\d+)-of-(\d+)\.jsonl$")


def discover_shard_inputs(run_dir: Path, shard_count: int | None = None) -> list[Path]:
    matches: list[tuple[int, int, Path]] = []
    for path in sorted(run_dir.glob("vision_sim.shard-*-of-*.jsonl")):
        m = _SHARD_JSONL_RE.match(path.name)
        if not m:
            continue
        index, count = int(m.group(1)), int(m.group(2))
        if shard_count is not None and count != shard_count:
            continue
        matches.append((index, count, path))
    if not matches:
        raise SystemExit(f"No vision_sim shard JSONL files found under {run_dir}")
    counts = {count for _, count, _ in matches}
    if len(counts) != 1:
        raise SystemExit(f"Mixed shard counts under {run_dir}: {sorted(counts)}")
    count = next(iter(counts))
    by_index = {index: path for index, _, path in matches}
    missing = [i for i in range(1, count + 1) if i not in by_index]
    if missing:
        raise SystemExit(f"Missing shards for of-{count}: {missing}")
    return [by_index[i] for i in range(1, count + 1)]


def cleanup_shard_artifacts(inputs: list[Path]) -> list[str]:
    deleted: list[str] = []
    for path in inputs:
        # vision_sim.shard-i-of-N.jsonl -> vision_sim_summary.shard-i-of-N.json
        summary = Path(
            str(path).replace("vision_sim.shard-", "vision_sim_summary.shard-", 1).replace(
                ".jsonl", ".json"
            )
        )
        for item in (path, summary):
            if item.is_file():
                item.unlink()
                deleted.append(str(item))
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        help="Run directory containing vision_sim.shard-*-of-N.jsonl files.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Expected shard count N when using --run (optional; auto-detected).",
    )
    parser.add_argument("--inputs", type=Path, nargs="+", help="Explicit shard JSONL paths.")
    parser.add_argument("--output", type=Path, help="Merged vision_sim.jsonl path.")
    parser.add_argument("--summary-output", type=Path, help="Merged vision_sim_summary.json path.")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete shard JSONL/summary files after a successful merge.",
    )
    args = parser.parse_args()

    if args.run is not None:
        inputs = discover_shard_inputs(args.run, args.shard_count)
        output = args.output or (args.run / "vision_sim.jsonl")
        summary_output = args.summary_output or (args.run / "vision_sim_summary.json")
    elif args.inputs:
        inputs = list(args.inputs)
        if args.output is None or args.summary_output is None:
            raise SystemExit("--output and --summary-output are required with --inputs")
        output = args.output
        summary_output = args.summary_output
    else:
        raise SystemExit("Provide --run or --inputs")

    rows = [row for path in inputs for row in read_jsonl(path)]
    by_id: dict[str, dict] = {}
    for row in rows:
        scene_id = str(row.get("id"))
        if scene_id in by_id:
            raise SystemExit(f"Duplicate scene id across shards: {scene_id}")
        by_id[scene_id] = row
    merged = [by_id[key] for key in sorted(by_id)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )

    # Pull common metadata from the first non-empty shard summary if present.
    meta: dict = {}
    first_summary = Path(
        str(inputs[0]).replace("vision_sim.shard-", "vision_sim_summary.shard-").replace(
            ".jsonl", ".json"
        )
    )
    if first_summary.is_file():
        try:
            meta = json.loads(first_summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    summary = summarize(merged)
    for key in ("model", "device", "num_frames"):
        if key in meta:
            summary[key] = meta[key]
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    deleted: list[str] = []
    if args.cleanup:
        deleted = cleanup_shard_artifacts(inputs)

    print(
        json.dumps(
            {
                "scenes": len(merged),
                "inputs": [str(path) for path in inputs],
                "output": str(output),
                "summary_output": str(summary_output),
                "deleted": deleted,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
