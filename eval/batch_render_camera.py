#!/usr/bin/env python3
"""Batch-render scene-camera videos for Stage-1 runs (CPU / Blender only).

Writes ``<run>/camera_renders/<scene_id>.mp4`` (64 sparse frames by default).
Does not modify ``blends/``, ``summary.json``, or ``unit_tests.jsonl``.

Mac → HF → cluster workflow::

  python eval/batch_render_camera.py --all-runs --resume --workers 6
  python scripts/sync_eval_results.py --camera-renders-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from render_blend_video import encode_png_sequence_to_mp4, render_blend_to_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "sandbox" / "results"


def records_from_run(run_dir: Path) -> list[dict[str, str]]:
    blends_dir = run_dir / "blends"
    if not blends_dir.is_dir():
        return []
    return [
        {"id": blend.stem, "submission_path": str(blend.resolve())}
        for blend in sorted(blends_dir.glob("*.blend"))
    ]


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        index_str, count_str = value.split("/", 1)
        index = int(index_str)
        count = int(count_str)
    except ValueError as exc:
        raise SystemExit("--shard must look like i/N, e.g. 1/8") from exc
    if index < 1 or count < 1 or index > count:
        raise SystemExit("--shard must satisfy 1 <= i <= N")
    return index, count


def video_path_for(renders_dir: Path, scene_id: str) -> Path:
    return renders_dir / f"{scene_id}.mp4"


def error_path_for(renders_dir: Path, scene_id: str) -> Path:
    return renders_dir / f"{scene_id}.error.json"


def video_ready(renders_dir: Path, scene_id: str) -> bool:
    video = video_path_for(renders_dir, scene_id)
    return video.is_file() and video.stat().st_size > 0


def render_one_job(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint (must be top-level for ProcessPoolExecutor)."""
    scene_id = job["id"]
    blend_path = Path(job["submission_path"])
    renders_dir = Path(job["renders_dir"])
    num_frames = int(job["num_frames"])
    resolution = int(job["resolution"])
    timeout = float(job["timeout"])
    blender = job.get("blender")
    keep_pngs = bool(job.get("keep_pngs"))
    resume = bool(job.get("resume"))

    out: dict[str, Any] = {
        "run": job["run_name"],
        "id": scene_id,
        "status": "error",
    }
    try:
        renders_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_path_for(renders_dir, scene_id)
        err_path = error_path_for(renders_dir, scene_id)
        if resume and video_ready(renders_dir, scene_id):
            out["status"] = "skip"
            return out

        with tempfile.TemporaryDirectory(prefix=f"bvb_render_{scene_id}_") as tmp:
            tmp_root = Path(tmp)
            frames_dir = tmp_root / "frames"
            meta = render_blend_to_frames(
                blend_path,
                frames_dir,
                blender=blender,
                resolution=resolution,
                num_samples=num_frames,
                timeout=timeout,
            )
            if not meta.get("ok"):
                out["status"] = "render_error"
                out["error"] = (meta.get("stderr_tail") or meta.get("stdout_tail") or "")[-500:]
                err_path.write_text(json.dumps(meta, indent=2)[:8000] + "\n", encoding="utf-8")
                return out

            fps = int((meta.get("render") or {}).get("fps") or 24)
            tmp_mp4 = tmp_root / "video.mp4"
            encode_png_sequence_to_mp4(frames_dir, tmp_mp4, fps=fps)
            # Atomic replace so resume never sees a partial mp4.
            shutil.move(str(tmp_mp4), str(video_path))

            if keep_pngs:
                keep_dir = renders_dir / f"{scene_id}_frames"
                if keep_dir.exists():
                    shutil.rmtree(keep_dir, ignore_errors=True)
                shutil.copytree(frames_dir, keep_dir)

        if err_path.exists():
            err_path.unlink()

        out["status"] = "ok"
        out["video"] = str(video_path)
        out["bytes"] = video_path.stat().st_size
        return out
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def discover_runs(results_dir: Path, run_args: list[Path] | None) -> list[Path]:
    if run_args:
        return [path.resolve() for path in run_args]
    runs = []
    for path in sorted(results_dir.glob("mini-harness-*")):
        if path.is_dir() and (path / "blends").is_dir() and any((path / "blends").glob("*.blend")):
            runs.append(path.resolve())
    return runs


def build_jobs(
    runs: list[Path],
    *,
    num_frames: int,
    resolution: int,
    timeout: float,
    blender: str | None,
    keep_pngs: bool,
    resume: bool,
    limit: int | None,
    scene_ids: list[str] | None,
    shard: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for run_dir in runs:
        records = records_from_run(run_dir)
        if scene_ids is not None:
            wanted = set(scene_ids)
            records = [row for row in records if row["id"] in wanted]
        renders_dir = run_dir / "camera_renders"
        for row in records:
            jobs.append(
                {
                    "run_name": run_dir.name,
                    "id": row["id"],
                    "submission_path": row["submission_path"],
                    "renders_dir": str(renders_dir),
                    "num_frames": num_frames,
                    "resolution": resolution,
                    "timeout": timeout,
                    "blender": blender,
                    "keep_pngs": keep_pngs,
                    "resume": resume,
                }
            )
    if limit is not None:
        jobs = jobs[:limit]
    if shard is not None:
        index, count = shard
        jobs = [job for i, job in enumerate(jobs) if i % count == (index - 1)]
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="Stage-1 run directory. Repeatable. Default with --all-runs: every mini-harness-*.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=f"Render every mini-harness-* under {DEFAULT_RESULTS_DIR}.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--blender", default=os.getenv("BLENDER_BIN"))
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel Blender workers.")
    parser.add_argument("--limit", type=int, help="Global cap on jobs (smoke tests).")
    parser.add_argument("--scene-ids", nargs="+")
    parser.add_argument("--shard", help="Global shard i/N over the job list.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenes with camera_renders/<id>.mp4 present.",
    )
    parser.add_argument(
        "--keep-pngs",
        action="store_true",
        help="Keep frame PNGs under camera_renders/<id>_frames/ (default: delete).",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print aggregate progress every N completions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run and not args.all_runs:
        raise SystemExit("Pass --run <dir> and/or --all-runs")

    runs = discover_runs(args.results_dir, args.run if args.run else None)
    if args.all_runs and args.run:
        extra = [path.resolve() for path in args.run]
        runs = sorted(set(runs) | set(extra), key=lambda p: p.name)
    elif args.run and not args.all_runs:
        runs = [path.resolve() for path in args.run]

    shard = parse_shard(args.shard)
    jobs = build_jobs(
        runs,
        num_frames=args.num_frames,
        resolution=args.resolution,
        timeout=args.timeout,
        blender=args.blender,
        keep_pngs=args.keep_pngs,
        resume=args.resume,
        limit=args.limit,
        scene_ids=args.scene_ids,
        shard=shard,
    )

    plan = {
        "num_runs": len(runs),
        "num_jobs": len(jobs),
        "workers": args.workers,
        "num_frames": args.num_frames,
        "resolution": args.resolution,
        "resume": args.resume,
        "keep_pngs": args.keep_pngs,
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        for job in jobs[:30]:
            print(f"  {job['run_name']}/{job['id']}")
        if len(jobs) > 30:
            print(f"  ... {len(jobs) - 30} more")
        return

    counts = {"ok": 0, "skip": 0, "render_error": 0, "error": 0}
    workers = max(1, int(args.workers))

    def handle_result(result: dict[str, Any], done: int) -> None:
        status = str(result.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status not in {"ok", "skip"}:
            print(
                f"[{done}/{len(jobs)}] FAIL {result.get('run')}/{result.get('id')}: "
                f"{result.get('error', status)}",
                flush=True,
            )
            if args.stop_on_error:
                raise SystemExit(1)
        elif done % args.progress_every == 0 or done == len(jobs):
            print(
                f"[{done}/{len(jobs)}] ok={counts.get('ok', 0)} skip={counts.get('skip', 0)} "
                f"err={counts.get('render_error', 0) + counts.get('error', 0)}",
                flush=True,
            )

    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            handle_result(render_one_job(job), index)
    else:
        # Submit in waves so we do not hold 10k+ Futures in memory on macOS.
        wave = max(workers * 8, 32)
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(jobs), wave):
                chunk = jobs[start : start + wave]
                futures = [pool.submit(render_one_job, job) for job in chunk]
                for future in as_completed(futures):
                    done += 1
                    handle_result(future.result(), done)

    summary = {"counts": counts, "num_jobs": len(jobs)}
    print(json.dumps(summary, indent=2), flush=True)
    if counts.get("render_error", 0) + counts.get("error", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
