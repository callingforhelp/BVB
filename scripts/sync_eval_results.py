#!/usr/bin/env python3
"""Upload Stage-2 eval artifacts to Hugging Face.

Mirrors into the same run directories under yunlong10/BVB-results, alongside any
existing Stage-1 config/agent_meta/blends files.

By default uploads ``summary.json`` / ``unit_tests.jsonl``. Pass
``--include-camera-renders`` to also push Mac-precomputed ``camera_renders/``
for cluster-side V-JEPA scoring without re-rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "yunlong10/BVB-results"
DEFAULT_RESULTS_DIR = Path("sandbox/results")
EVAL_PATTERNS = ["summary.json", "unit_tests.jsonl"]
CAMERA_PATTERNS = ["camera_renders/*.mp4", "camera_renders/*.error.json"]


def discover_runs(results_dir: Path, *, include_camera_renders: bool) -> list[Path]:
    runs = []
    for path in sorted(results_dir.glob("mini-harness-*")):
        if not path.is_dir():
            continue
        has_eval = (path / "summary.json").is_file()
        has_renders = (path / "camera_renders").is_dir()
        if has_eval or (include_camera_renders and has_renders):
            runs.append(path)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional run directory name; repeatable. Default: evaluated runs.",
    )
    parser.add_argument(
        "--include-camera-renders",
        action="store_true",
        help="Also upload <run>/camera_renders/*.mp4 (and *.error.json).",
    )
    parser.add_argument(
        "--camera-renders-only",
        action="store_true",
        help="Upload only camera_renders/** (implies --include-camera-renders).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    include_renders = args.include_camera_renders or args.camera_renders_only
    if args.camera_renders_only:
        allow_patterns = list(CAMERA_PATTERNS)
    elif include_renders:
        allow_patterns = EVAL_PATTERNS + CAMERA_PATTERNS
    else:
        allow_patterns = list(EVAL_PATTERNS)

    if args.run:
        runs = [args.results_dir / name for name in args.run]
    else:
        runs = discover_runs(args.results_dir, include_camera_renders=include_renders)
    if not runs:
        raise SystemExit(f"No matching runs found under {args.results_dir}")

    api = HfApi()
    uploaded_runs = 0
    for run_dir in runs:
        if not run_dir.is_dir():
            raise SystemExit(f"Missing run directory: {run_dir}")
        run_name = run_dir.name
        present = []
        for pattern in allow_patterns:
            if pattern == "camera_renders/*.mp4":
                if any((run_dir / "camera_renders").glob("*.mp4")):
                    present.append(pattern)
            elif pattern == "camera_renders/*.error.json":
                if any((run_dir / "camera_renders").glob("*.error.json")):
                    present.append(pattern)
            elif pattern.endswith("/**"):
                root = run_dir / pattern[:-3]
                if root.is_dir() and any(root.rglob("*")):
                    present.append(pattern)
            elif (run_dir / pattern).is_file():
                present.append(pattern)
        if not present:
            print(f"[skip] nothing to upload: {run_name}")
            continue
        print(f"[upload] {run_name}/ {{{', '.join(present)}}}")
        if args.dry_run:
            continue
        api.upload_folder(
            folder_path=str(run_dir),
            path_in_repo=run_name,
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            commit_message=(
                f"Upload camera_renders for {run_name}"
                if args.camera_renders_only
                else f"Upload Stage-2 eval artifacts for {run_name}"
            ),
        )
        uploaded_runs += 1
    print(f"Done. uploaded_runs={uploaded_runs} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
