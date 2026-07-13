#!/usr/bin/env python3
"""Bridge Stage-1 output to Stage-2 evaluation.

Scans a run's ``blends/`` directory and writes a ``code_pairs.jsonl`` that
``eval/unit_test_metric.py --mode execute`` can consume directly (it introspects
the .blend in Blender, so no bpy export is required first).

    python make_pairs.py --run results/run_001
    python ../eval/unit_test_metric.py \
        --pairs results/run_001/code_pairs.jsonl \
        --mode execute --output results/run_001/unit_tests.jsonl ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GT_BPY_DIR = REPO_ROOT / "bpy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate code_pairs.jsonl from a Stage-1 run.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing blends/.")
    parser.add_argument("--output", type=Path, help="Defaults to <run>/code_pairs.jsonl")
    args = parser.parse_args()

    blends_dir = args.run / "blends"
    if not blends_dir.is_dir():
        raise SystemExit(f"No blends directory: {blends_dir}")
    output = args.output or (args.run / "code_pairs.jsonl")

    rows = []
    for blend in sorted(blends_dir.glob("*.blend")):
        scene = blend.stem
        gt_bpy = GT_BPY_DIR / f"{scene}.py"
        row = {"id": scene, "pred_bpy_path": str(blend.resolve())}
        if gt_bpy.exists():
            row["gt_bpy_path"] = str(gt_bpy.resolve())
        rows.append(row)

    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} pairs to {output}")


if __name__ == "__main__":
    main()
