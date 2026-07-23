#!/usr/bin/env python3
"""Run the same deterministic scene sample across multiple Stage-1 runs."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path


EVALUATOR = Path(__file__).resolve().parent / "unit_test_metric.py"


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
    parser.add_argument("--mode", choices=("static", "execute"), default="execute")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--blender")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    sample = args.sample if args.sample is not None else (None if args.scene_ids else 10)
    if args.scene_ids:
        digest = hashlib.sha1(",".join(args.scene_ids).encode()).hexdigest()[:8]
        selection_label = f"ids{len(args.scene_ids)}-{digest}"
    else:
        selection_label = f"n{sample}-seed{args.seed}"
    if args.test_types:
        selection_label += "-types-" + slug("-".join(args.test_types))
    model_slug = slug(args.model)
    for run in args.runs:
        label = f"smoke_{model_slug}_{selection_label}"
        command = [
            sys.executable,
            str(EVALUATOR),
            "--run",
            str(run),
            "--mode",
            args.mode,
            "--model",
            args.model,
            "--reasoning-effort",
            "none",
            "--cache-dir",
            str(run / "introspection-cache"),
            "--output",
            str(run / f"unit_tests_{label}.jsonl"),
            "--summary-output",
            str(run / f"summary_{label}.json"),
        ]
        if args.scene_ids:
            command += ["--scene-ids", *args.scene_ids]
        else:
            command += ["--sample", str(sample), "--seed", str(args.seed)]
        if args.test_types:
            command += ["--test-types", *args.test_types]
        if args.dry_run:
            command.append("--dry-run")
        if args.blender:
            command += ["--blender", args.blender]
        if args.stop_on_error:
            command.append("--stop-on-error")
        print(f"=== {run.name} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
