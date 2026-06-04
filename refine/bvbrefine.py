#!/usr/bin/env python3
"""bvbrefine — mechanical helpers for the BVB human↔Claude refinement loop.

Subcommands: qa, frames, start, checkpoints, gate.
All per-scene state lives under blend/refine_<id>/ (git-ignored).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bvb_lib  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "Private & Shared" / "BVB Scene Tracker a61a82bfca12838b9c9c817395be4718.csv"
UNIT_TESTS = REPO / "eval" / "unit_tests.jsonl"
DATASET = REPO / "VSI-Bench"


def refine_dir(scene_id: str) -> Path:
    return REPO / "blend" / f"refine_{scene_id}"


def cmd_qa(scene_id: str) -> int:
    qa = bvb_lib.load_qa(CSV, scene_id)
    if qa is None:
        print(f"❌ scene {scene_id} not found in tracker CSV")
        return 1
    bar = "=" * 78
    print(bar)
    print(f"  Scene: {scene_id}  |  Source: {qa['source'] or '(unknown)'}")
    print(bar)
    print("\nQA AXES:")
    for a in qa["axes"]:
        print(f"  • {a}: {bvb_lib.focus_hint(a)}")
    print("\nQUESTION(S):")
    for line in qa["qa"].split("\n"):
        print(f"  {line}")
    tests = bvb_lib.load_unit_tests(UNIT_TESTS, scene_id)
    print(f"\nUNIT TESTS ({len(tests)}):")
    for t in tests:
        print(f"  • [{t.get('test_id')}] {t.get('test_type')} ({t.get('evaluator')})")
        print(f"      {t.get('statement')}")
        print(f"      expected={t.get('expected')}")
    print(bar)
    return 0


def cmd_frames(scene_id: str, n: int) -> int:
    video = bvb_lib.resolve_video(DATASET, scene_id)
    if video is None:
        print(f"❌ no source video for {scene_id} under {DATASET}")
        return 1
    out = refine_dir(scene_id) / "frames"
    out.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.run(
        bvb_lib.probe_duration_cmd(video), capture_output=True, text=True, check=True
    ).stdout.strip())
    subprocess.run(bvb_lib.frame_extract_cmd(video, out, n, dur), check=True)
    print(f"✓ extracted {n} frames from {video.name} → {out}")
    return 0


def cmd_checkpoints(scene_id: str) -> int:
    ckpts = bvb_lib.list_checkpoints(refine_dir(scene_id))
    if not ckpts:
        print(f"(no checkpoints yet in {refine_dir(scene_id)})")
        return 0
    print(f"Checkpoints for {scene_id}:")
    for v, desc, path in ckpts:
        print(f"  v{v:02d}  {desc:24s}  {path.name}")
    return 0


def cmd_start(scene_id: str, n: int) -> int:
    rc = cmd_qa(scene_id)
    if rc:
        return rc
    cmd_frames(scene_id, n)
    refine_dir(scene_id).mkdir(parents=True, exist_ok=True)
    baseline = refine_dir(scene_id) / f"{scene_id}_v00_baseline.blend"
    print("\nNEXT (Claude runs via BlenderMCP to snapshot the baseline):")
    print("  import bpy")
    print(f"  bpy.ops.wm.save_as_mainfile(filepath=r'{baseline}', copy=True)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="bvbrefine")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("qa", "checkpoints"):
        sp = sub.add_parser(name)
        sp.add_argument("scene_id")
    for name in ("frames", "start"):
        sp = sub.add_parser(name)
        sp.add_argument("scene_id")
        sp.add_argument("--n", "--frames", type=int, default=12, dest="n")
    gp = sub.add_parser("gate")
    gp.add_argument("scene_id")
    gp.add_argument("--judge", action="store_true")
    gp.add_argument("--floor-group", default=None)
    args = p.parse_args()

    if args.cmd == "qa":
        return cmd_qa(args.scene_id)
    if args.cmd == "frames":
        return cmd_frames(args.scene_id, args.n)
    if args.cmd == "checkpoints":
        return cmd_checkpoints(args.scene_id)
    if args.cmd == "start":
        return cmd_start(args.scene_id, args.n)
    if args.cmd == "gate":
        from gate import cmd_gate  # implemented in a later task
        return cmd_gate(args.scene_id, judge=args.judge, floor_group=args.floor_group)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
