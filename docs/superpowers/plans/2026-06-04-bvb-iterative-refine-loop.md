# BVB Iterative Refinement Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable CLI (`refine/bvbrefine.py`) plus a protocol skill (`bvb-refine-loop`) that make human↔Claude iterative refinement of BVB scenes fast, safe, and repeatable across all scenes.

**Architecture:** Pure logic lives in `refine/bvb_lib.py` (unit-tested with pytest), a thin CLI dispatch in `refine/bvbrefine.py` wires those functions to stdin/stdout/subprocess, and the `gate` subcommand reuses the existing `eval/` harness's deterministic pieces for an offline official check (with an optional `--judge` passthrough to the full LLM-backed runner). The skill encodes the collaboration loop, gates, checkpoint convention, and a growing heuristics list.

**Tech Stack:** Python 3.13 stdlib (csv, json, argparse, subprocess, pathlib, re), pytest 9.0.3, ffmpeg/ffprobe, the existing `eval/` modules (`eval_utils`, `unit_test_metric`), BlenderMCP for the live-edit half (not built here).

**Spec:** `docs/superpowers/specs/2026-06-04-bvb-iterative-refine-loop-design.md`

**Branch:** create `feat/bvb-refine-loop` before Task 1 (repo norm: never commit to `main` unprompted).

---

## File Structure

| File | Responsibility |
|---|---|
| `refine/bvb_lib.py` | pure logic: QA/unit-test loading, focus hints, video resolution, ffmpeg command construction, checkpoint name parsing/listing, pairs row, floor-group heuristic, function-param grounding |
| `refine/bvbrefine.py` | CLI: argparse subcommands (`qa`, `frames`, `start`, `checkpoints`, `gate`) wiring `bvb_lib` to I/O and subprocess |
| `refine/test_bvb_lib.py` | pytest unit tests for every pure function in `bvb_lib.py` |
| `.cursor/skills/bvb-refine-loop/SKILL.md` | the collaboration protocol skill + growing Heuristics section |

Constants reused from the codebase (do not re-derive): CSV path `Private & Shared/BVB Scene Tracker a61a82bfca12838b9c9c817395be4718.csv`; unit tests `eval/unit_tests.jsonl`; dataset root `VSI-Bench/`; sources `("arkitscenes", "scannet", "scannetpp")`.

---

### Task 0: Branch

- [ ] **Step 1: Create the feature branch**

Run:
```bash
cd /Users/pinxinliu/Downloads/BVB && git checkout -b feat/bvb-refine-loop
```
Expected: `Switched to a new branch 'feat/bvb-refine-loop'`

---

### Task 1: Focus hints + checkpoint-name parsing

**Files:**
- Create: `refine/bvb_lib.py`
- Test: `refine/test_bvb_lib.py`

- [ ] **Step 1: Write the failing tests**

Create `refine/test_bvb_lib.py`:
```python
import bvb_lib


def test_focus_hint_known_axis():
    assert bvb_lib.focus_hint("room_size_estimation") == "wall positions, room footprint (length × width)"


def test_focus_hint_unknown_axis():
    assert bvb_lib.focus_hint("made_up_axis") == "(see project guide for this axis)"


def test_parse_checkpoint_name_valid():
    assert bvb_lib.parse_checkpoint_name("41125731_v03_remove_bed1.blend") == (3, "remove_bed1")


def test_parse_checkpoint_name_baseline():
    assert bvb_lib.parse_checkpoint_name("41125731_v00_baseline.blend") == (0, "baseline")


def test_parse_checkpoint_name_rejects_non_checkpoint():
    assert bvb_lib.parse_checkpoint_name("41125731.blend") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bvb_lib'`

- [ ] **Step 3: Write minimal implementation**

Create `refine/bvb_lib.py`:
```python
"""Pure logic for the BVB iterative-refinement CLI. No side effects here."""
from __future__ import annotations

import re
from pathlib import Path

FOCUS_HINTS = {
    "room_size_estimation": "wall positions, room footprint (length × width)",
    "object_counting": "exact number of each object type",
    "object_abs_distance": "real distance between specific objects (meters)",
    "object_rel_direction_easy": "left/right/front/back relations between objects",
    "object_rel_direction_medium": "left/right/front/back relations between objects",
    "object_rel_direction_hard": "left/right/front/back relations between objects",
    "object_rel_distance": "which object is closer/farther to a reference",
    "object_size_estimation": "physical dimensions (length × width × height) of objects",
    "route_planning": "furniture layout & free walking paths (no blockers, correct positions)",
    "appearance_order": "order in which objects appear in the video",
}

_CKPT_RE = re.compile(r"^.+?_v(?P<nn>\d+)_(?P<desc>.+)\.blend$")


def focus_hint(axis: str) -> str:
    return FOCUS_HINTS.get(axis, "(see project guide for this axis)")


def parse_checkpoint_name(filename: str):
    """Return (version:int, desc:str) for '<id>_vNN_<desc>.blend', else None."""
    m = _CKPT_RE.match(filename)
    if not m:
        return None
    return int(m.group("nn")), m.group("desc")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add refine/bvb_lib.py refine/test_bvb_lib.py
git commit -m "feat(refine): focus hints + checkpoint name parsing"
```

---

### Task 2: Load QA + unit tests

**Files:**
- Modify: `refine/bvb_lib.py`
- Test: `refine/test_bvb_lib.py`

- [ ] **Step 1: Write the failing tests** (append to `refine/test_bvb_lib.py`)

```python
import json


def _write_csv(tmp_path):
    p = tmp_path / "tracker.csv"
    p.write_text(
        "Scene ID,Source,QA Axes,QA\n"
        "41125731,arkitscenes,\"room_size_estimation, route_planning\",\"What is the size?\n---\n23.6\"\n",
        encoding="utf-8-sig",
    )
    return p


def test_load_qa_found(tmp_path):
    qa = bvb_lib.load_qa(_write_csv(tmp_path), "41125731")
    assert qa["source"] == "arkitscenes"
    assert qa["axes"] == ["room_size_estimation", "route_planning"]
    assert "23.6" in qa["qa"]


def test_load_qa_missing(tmp_path):
    assert bvb_lib.load_qa(_write_csv(tmp_path), "does_not_exist") is None


def test_load_unit_tests_filters_by_scene(tmp_path):
    p = tmp_path / "ut.jsonl"
    rows = [
        {"test_id": "a", "scene_name": "41125731", "evaluator": "function"},
        {"test_id": "b", "scene_name": "other", "evaluator": "rule"},
        {"test_id": "c", "scene_name": "*", "evaluator": "rule"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    got = bvb_lib.load_unit_tests(p, "41125731")
    assert [r["test_id"] for r in got] == ["a"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: FAIL — `AttributeError: module 'bvb_lib' has no attribute 'load_qa'`

- [ ] **Step 3: Write minimal implementation** (append to `refine/bvb_lib.py`)

```python
import csv
import json


def load_qa(csv_path, scene_id: str):
    """Return dict(scene_id, source, axes, qa) for the row, else None."""
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sid = r["Scene ID"].strip().rstrip("‹").strip()
            if sid == scene_id:
                return {
                    "scene_id": sid,
                    "source": r.get("Source", "").strip(),
                    "axes": [a.strip() for a in r.get("QA Axes", "").split(",") if a.strip()],
                    "qa": r.get("QA", "").strip(),
                }
    return None


def load_unit_tests(jsonl_path, scene_id: str):
    """Return the list of unit-test rows whose scene_name == scene_id."""
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if str(r.get("scene_name", "")) == scene_id:
                out.append(r)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add refine/bvb_lib.py refine/test_bvb_lib.py
git commit -m "feat(refine): load QA from CSV tracker and per-scene unit tests"
```

---

### Task 3: Video resolution + ffmpeg command builders

**Files:**
- Modify: `refine/bvb_lib.py`
- Test: `refine/test_bvb_lib.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_resolve_video_picks_existing_source(tmp_path):
    (tmp_path / "scannet").mkdir()
    vid = tmp_path / "scannet" / "scene0077_01.mp4"
    vid.write_bytes(b"x")
    assert bvb_lib.resolve_video(tmp_path, "scene0077_01") == vid


def test_resolve_video_missing(tmp_path):
    assert bvb_lib.resolve_video(tmp_path, "nope") is None


def test_probe_duration_cmd():
    cmd = bvb_lib.probe_duration_cmd("/v/x.mp4")
    assert cmd[0] == "ffprobe" and cmd[-1] == "/v/x.mp4"
    assert "format=duration" in cmd


def test_frame_extract_cmd():
    cmd = bvb_lib.frame_extract_cmd("/v/x.mp4", "/out", 12, 85.0)
    assert cmd[0] == "ffmpeg"
    assert "fps=12/85.0" in cmd
    assert cmd[-1].endswith("/out/frame_%02d.jpg")
    assert "-frames:v" in cmd and "12" in cmd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: FAIL — `AttributeError: module 'bvb_lib' has no attribute 'resolve_video'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
SOURCES = ("arkitscenes", "scannet", "scannetpp")


def resolve_video(dataset_root, scene_id: str):
    """Find VSI-Bench/<source>/<id>.mp4 across the known sources, else None."""
    for s in SOURCES:
        p = Path(dataset_root) / s / f"{scene_id}.mp4"
        if p.is_file():
            return p
    return None


def probe_duration_cmd(video) -> list:
    return [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1", str(video),
    ]


def frame_extract_cmd(video, out_dir, n: int, duration: float) -> list:
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps={n}/{duration}", "-frames:v", str(n),
        str(Path(out_dir) / "frame_%02d.jpg"),
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add refine/bvb_lib.py refine/test_bvb_lib.py
git commit -m "feat(refine): video resolution + ffmpeg command builders"
```

---

### Task 4: Checkpoint listing, pairs row, floor grounding

**Files:**
- Modify: `refine/bvb_lib.py`
- Test: `refine/test_bvb_lib.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_list_checkpoints_sorted(tmp_path):
    for name in ["41125731_v02_walls.blend", "41125731_v00_baseline.blend", "notes.txt"]:
        (tmp_path / name).write_text("x")
    got = bvb_lib.list_checkpoints(tmp_path)
    assert [(v, d) for v, d, _ in got] == [(0, "baseline"), (2, "walls")]


def test_list_checkpoints_missing_dir(tmp_path):
    assert bvb_lib.list_checkpoints(tmp_path / "nope") == []


def test_build_pairs_row():
    assert bvb_lib.build_pairs_row("41125731", "bpy/41125731.py") == {
        "id": "41125731",
        "pred_bpy_path": "bpy/41125731.py",
    }


def test_pick_floor_group_prefers_floor():
    assert bvb_lib.pick_floor_group({"Wall_Back": 1, "Floor": 2, "Ceiling": 3}) == "Floor"


def test_pick_floor_group_falls_back_to_room():
    assert bvb_lib.pick_floor_group({"Wall_Back": 1, "Room_Boundary": 2}) == "Room_Boundary"


def test_pick_floor_group_none():
    assert bvb_lib.pick_floor_group({"Wall_Back": 1, "Bed": 2}) is None


def test_ground_function_params_room_area():
    test = {"function": "room_area"}
    assert bvb_lib.ground_function_params(test, "Floor") == {"object_group": "Floor"}


def test_ground_function_params_other_returns_empty():
    assert bvb_lib.ground_function_params({"function": "count_objects"}, "Floor") == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: FAIL — `AttributeError: module 'bvb_lib' has no attribute 'list_checkpoints'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def list_checkpoints(refine_dir):
    """Return sorted [(version, desc, Path), ...] for *.blend checkpoints in dir."""
    d = Path(refine_dir)
    out = []
    if not d.is_dir():
        return out
    for p in d.glob("*.blend"):
        parsed = parse_checkpoint_name(p.name)
        if parsed:
            out.append((parsed[0], parsed[1], p))
    return sorted(out, key=lambda t: t[0])


def build_pairs_row(scene_id: str, bpy_path: str) -> dict:
    """One pairs-file row for eval/unit_test_metric.py."""
    return {"id": scene_id, "pred_bpy_path": bpy_path}


def pick_floor_group(groups) -> str | None:
    """Heuristic: the group key naming the floor/room boundary, else None."""
    keys = list(groups)
    for k in keys:
        if "floor" in k.lower():
            return k
    for k in keys:
        if "room" in k.lower():
            return k
    return None


def ground_function_params(test: dict, floor_key) -> dict:
    """Supply grounded_params for offline deterministic function evaluation.

    Only room_area is grounded automatically (the floor group). Other function
    types need object-specific grounding we do not infer offline; return {} so
    the deterministic evaluator reports them as unresolved.
    """
    if test.get("function") == "room_area":
        return {"object_group": floor_key}
    return {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refine && python -m pytest test_bvb_lib.py -q`
Expected: PASS (20 passed)

- [ ] **Step 5: Commit**

```bash
git add refine/bvb_lib.py refine/test_bvb_lib.py
git commit -m "feat(refine): checkpoint listing, pairs row, floor grounding"
```

---

### Task 5: Offline function-test evaluator (gate core)

This wraps the existing `eval/` harness's deterministic pieces so a `function` test can be scored with no LLM. It is integration code (imports `eval_utils` + `unit_test_metric`), so it gets a smoke test against the real exported bpy rather than a pure unit test.

**Files:**
- Modify: `refine/bvb_lib.py`
- Test: `refine/test_bvb_lib.py`

- [ ] **Step 1: Write the smoke test** (append)

```python
import subprocess


def test_evaluate_function_offline_room_area_on_real_scene(tmp_path):
    """Export the real 41125731.blend → bpy, score its room_area test offline."""
    import shutil
    import pytest
    repo = Path(__file__).resolve().parent.parent
    blend = repo / "blend" / "41125731.blend"
    if not blend.is_file():
        pytest.skip("scene 41125731.blend not present")
    # batch_export has no single-file flag, so isolate one .blend in its own dir.
    iso = tmp_path / "in"
    iso.mkdir()
    shutil.copy2(blend, iso / "41125731.blend")
    rc = subprocess.run(
        ["python", str(repo / "eval" / "batch_export_blend_to_bpy.py"),
         "--input-dir", str(iso), "--output-dir", str(tmp_path), "--overwrite"],
        capture_output=True, text=True,
    )
    bpy_out = tmp_path / "41125731.py"
    if not bpy_out.is_file():
        pytest.skip(f"bpy export unavailable in this env: {rc.stderr[-300:]}")
    test = {"scene_name": "41125731", "function": "room_area", "evaluator": "function",
            "expected": {"value": 23.6, "tolerance": 2.36}}
    res = bvb_lib.evaluate_function_offline(repo, bpy_out, test, floor_group_key=None)
    assert res["status"] in {"pass", "fail"}
    assert isinstance(res.get("actual"), (int, float))
```

> Note for the implementer: this smoke test is intentionally skip-tolerant — headless
> Blender export may be unavailable in CI. Its job is to prove the wiring when Blender IS
> available. The deterministic behavior is already covered by Task 4's pure tests on
> `ground_function_params` / `pick_floor_group`.

- [ ] **Step 2: Run test to verify it fails (or skips meaningfully)**

Run: `cd refine && python -m pytest test_bvb_lib.py::test_evaluate_function_offline_room_area_on_real_scene -q`
Expected: FAIL — `AttributeError: ... 'evaluate_function_offline'` (not a skip yet, since the attribute is missing before the skip guards are reached)

- [ ] **Step 3: Write minimal implementation** (append)

```python
import sys


def _import_eval(repo_root):
    eval_dir = str(Path(repo_root) / "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import eval_utils  # noqa: E402
    import unit_test_metric  # noqa: E402
    return eval_utils, unit_test_metric


def evaluate_function_offline(repo_root, bpy_path, test: dict, floor_group_key=None):
    """Score one `function` unit test deterministically, no LLM.

    Reuses the harness: parse the bpy, build object groups, ground params
    (floor group), then call the harness's deterministic evaluator.
    """
    eu, utm = _import_eval(repo_root)
    scene_index = eu.parse_bpy_scene(Path(bpy_path))
    groups = eu.build_groups(scene_index)
    floor_key = floor_group_key or pick_floor_group(groups)
    params = ground_function_params(test, floor_key)
    scene_id = str(test.get("scene_name"))
    return utm.evaluate_function_test_with_params(scene_id, test, groups, params)
```

- [ ] **Step 4: Run test to verify it passes (or skips when Blender absent)**

Run: `cd refine && python -m pytest test_bvb_lib.py::test_evaluate_function_offline_room_area_on_real_scene -q`
Expected: PASS or SKIP (skip is acceptable if headless export is unavailable). If PASS, the printed `actual` for the current 6×6 floor should be ~36 with `status == "fail"`, confirming the room_size mismatch.

- [ ] **Step 5: Commit**

```bash
git add refine/bvb_lib.py refine/test_bvb_lib.py
git commit -m "feat(refine): offline deterministic function-test evaluator"
```

---

### Task 6: CLI dispatch — qa / frames / start / checkpoints

**Files:**
- Create: `refine/bvbrefine.py`

- [ ] **Step 1: Write the CLI**

Create `refine/bvbrefine.py`:
```python
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
        from gate import cmd_gate  # implemented in Task 7
        return cmd_gate(args.scene_id, judge=args.judge, floor_group=args.floor_group)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test qa + checkpoints (no Blender needed)**

Run:
```bash
cd /Users/pinxinliu/Downloads/BVB/refine && python bvbrefine.py qa 41125731
```
Expected: prints the two QA axes (room_size_estimation, route_planning), the QA text incl. `23.6`, and the two unit tests `ut_000538` (room_size_estimation/function) and `ut_005009` (route_planning/proposition).

Run:
```bash
python bvbrefine.py checkpoints 41125731
```
Expected: lists `v00  baseline  41125731_v00_baseline.blend` (created earlier this session) — or "(no checkpoints yet)" if the refine dir was cleared.

- [ ] **Step 3: Smoke-test frames**

Run:
```bash
python bvbrefine.py frames 41125731 --n 6
```
Expected: `✓ extracted 6 frames from 41125731.mp4 → .../blend/refine_41125731/frames`

- [ ] **Step 4: Commit**

```bash
git add refine/bvbrefine.py
git commit -m "feat(refine): bvbrefine CLI — qa/frames/start/checkpoints"
```

---

### Task 7: gate subcommand — export + offline eval + --judge passthrough

**Files:**
- Create: `refine/gate.py`

- [ ] **Step 1: Write the gate module**

Create `refine/gate.py`:
```python
"""`bvbrefine gate` — export the live .blend to bpy, then score unit tests.

Default: offline deterministic scoring of `function` tests (no LLM); `proposition`
tests are reported as self-judged. With --judge, defer to eval/unit_test_metric.py
(LLM grounding + judging) for fully benchmark-faithful results.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bvb_lib  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
UNIT_TESTS = REPO / "eval" / "unit_tests.jsonl"
ADDON = REPO / "addons" / "export_bpy_code.py"


def export_bpy(scene_id: str, out_dir: Path) -> Path | None:
    """Export blend/<id>.blend → <out_dir>/<id>.py via background Blender."""
    blend = REPO / "blend" / f"{scene_id}.blend"
    if not blend.is_file():
        print(f"❌ {blend} not found")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        one = Path(td) / f"{scene_id}.blend"
        shutil.copy2(blend, one)
        subprocess.run(
            ["python", str(REPO / "eval" / "batch_export_blend_to_bpy.py"),
             "--input-dir", str(Path(td)), "--output-dir", str(out_dir), "--overwrite"],
            check=True,
        )
    bpy_path = out_dir / f"{scene_id}.py"
    return bpy_path if bpy_path.is_file() else None


def cmd_gate(scene_id: str, judge: bool = False, floor_group: str | None = None) -> int:
    refine = REPO / "blend" / f"refine_{scene_id}"
    bpy_dir = refine / "bpy"
    bpy_path = export_bpy(scene_id, bpy_dir)
    if bpy_path is None:
        print("❌ bpy export failed — is Blender + export addon available?")
        return 1
    tests = bvb_lib.load_unit_tests(UNIT_TESTS, scene_id)

    if judge:
        pairs = refine / "pairs.jsonl"
        pairs.write_text(
            json.dumps(bvb_lib.build_pairs_row(scene_id, f"bpy/{scene_id}.py")) + "\n",
            encoding="utf-8",
        )
        out = refine / "gate_results.jsonl"
        rc = subprocess.run(
            ["python", str(REPO / "eval" / "unit_test_metric.py"),
             "--pairs", str(pairs), "--unit-tests", str(UNIT_TESTS),
             "--output", str(out)],
            cwd=str(refine),
        ).returncode
        print(f"\n→ official (judged) results written to {out} (rc={rc})")
        return rc

    print(f"\nOFFLINE GATE for {scene_id} ({len(tests)} tests):")
    failed = 0
    for t in tests:
        if t.get("evaluator") == "function":
            res = bvb_lib.evaluate_function_offline(REPO, bpy_path, t, floor_group)
            status = res.get("status")
            print(f"  [{t['test_id']}] {t['test_type']}: {status.upper()} "
                  f"(actual={res.get('actual')}, expected={t.get('expected')})")
            failed += status != "pass"
        else:
            print(f"  [{t['test_id']}] {t['test_type']}: SELF-JUDGED "
                  f"(proposition; rerun with --judge for official)")
    print(f"\n{len(tests) - failed}/{len(tests)} function tests passing offline "
          f"(proposition tests need --judge).")
    return 1 if failed else 0
```

- [ ] **Step 2: Smoke-test the offline gate**

Run:
```bash
cd /Users/pinxinliu/Downloads/BVB/refine && python bvbrefine.py gate 41125731
```
Expected (with current un-refined scene): `ut_000538 room_size_estimation: FAIL (actual≈36.0, expected ...23.6...)` and `ut_005009 route_planning: SELF-JUDGED`. If Blender export is unavailable, it prints the export-failure message and returns 1 — that is the signal to run the gate from a machine with Blender.

- [ ] **Step 3: Commit**

```bash
git add refine/gate.py
git commit -m "feat(refine): gate — bpy export + offline/official unit-test scoring"
```

---

### Task 8: The `bvb-refine-loop` skill

**Files:**
- Create: `.cursor/skills/bvb-refine-loop/SKILL.md`

- [ ] **Step 1: Write the skill**

Create `.cursor/skills/bvb-refine-loop/SKILL.md` with this content:
```markdown
---
name: bvb-refine-loop
description: Human-in-the-loop iterative refinement of an existing BVB Blender scene against its QA via BlenderMCP. Use when refining/optimizing a .blend so the evaluation model answers the scene's QA correctly, when the user runs ./start.sh or bvbrefine, or mentions BVB refinement, checkpoints, or the refinement tracker. Pairs with the bvbrefine CLI (qa/frames/start/checkpoints/gate). For building a scene from scratch use bvb-scene-builder instead.
---

# BVB Refine Loop

Drive the human↔Claude refinement of ONE existing BVB scene. Mechanical work is the
`refine/bvbrefine.py` CLI; this skill is the collaboration protocol Claude follows.

## Loop

1. `python refine/bvbrefine.py start <id>` — prints QA + focus hints + the scene's unit
   tests, extracts reference frames to `blend/refine_<id>/frames/`, and prints a baseline
   snapshot snippet. Run that snippet via BlenderMCP `execute_blender_code` to save
   `blend/refine_<id>/<id>_v00_baseline.blend` (`save_as_mainfile(..., copy=True)`).
2. Inventory: `get_scene_info` + read the frames. Classify EACH QA axis as **local** (color,
   small position, lighting, missing detail) or **structural** (room size, topology, major
   furniture, scene identity). Build an ordered edit plan.
3. **GATE — structural changes:** stop and get the human's OK before any structural edit.
   Local patches proceed autonomously.
4. Per edit: `execute_blender_code` (one object group, radians via `math.radians`, deselect
   first, set `dimensions` then apply scale) → `get_viewport_screenshot` to verify → live
   function-check (compute floor area / relative positions directly) → snapshot a new
   `<id>_vNN_<desc>.blend` checkpoint (`copy=True`, so the live file stays `blend/<id>.blend`).
5. `python refine/bvbrefine.py gate <id>` — offline deterministic scoring of `function`
   tests; add `--judge` (needs `OPENAI_API_KEY` + `BVB_JUDGE_MODEL`) for the official
   proposition judge.
6. **GATE — pre-push:** show official results + final screenshots. On approval, the final
   checkpoint is already the live `blend/<id>.blend`; verify exportability (the gate's bpy
   export is the same path), then commit, push, and update the Notion tracker's Refiner column.

## Verification rule (hybrid)

- `function` tests (room_area, counts, distances): compute live in Blender each step;
  `bvbrefine gate` re-scores offline at the end. room_area = floor-group bbox `X*Y`.
- `proposition` tests (route_planning): reason from geometry + frames live; `--judge` for
  the official check.

## Checkpoint convention

- Dir `blend/refine_<id>/` (git-ignored). Names `<id>_vNN_<desc>.blend`, `v00_baseline` first.
- Always `save_as_mainfile(filepath=..., copy=True)` — never switch the live file off
  `blend/<id>.blend`. Roll back by `open_mainfile` on a `vNN`.

## Viewpoint / left-right discipline

Never infer left/right/front/back from Blender world axes. Anchor to the reference
viewpoint first (doorways, sight lines); define the viewer-facing vector; verify
left/right placements from that viewpoint. Do not add doors/openings not seen in the video.

## Heuristics (append one line per scene as you learn them)

- arkitscenes bedrooms with two beds usually have one spurious bed → keep the bed that
  clusters with the route objects (nightstand + wardrobe); delete the other.
- A window in a sloped ceiling is a skylight (roof window), not a vertical-wall window.
- room_size too large → shrink Floor + the four walls to the target footprint, apply scale=1
  so mesh scale ends (1,1,1). Target check: floor bbox X*Y within the test tolerance.
- "wardrobe with two mirrors" means two distinct mirror panels on the door — model both.
```

- [ ] **Step 2: Sanity-check the skill renders**

Run: `sed -n '1,5p' .cursor/skills/bvb-refine-loop/SKILL.md`
Expected: the YAML frontmatter with `name: bvb-refine-loop`.

- [ ] **Step 3: Commit**

```bash
git add .cursor/skills/bvb-refine-loop/SKILL.md
git commit -m "feat(skill): bvb-refine-loop collaboration protocol"
```

---

### Task 9: Full-suite test run + end-to-end smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the whole unit suite**

Run: `cd /Users/pinxinliu/Downloads/BVB/refine && python -m pytest -q`
Expected: all pure tests pass (20+); the offline-eval smoke either passes or skips.

- [ ] **Step 2: End-to-end dry pass on 41125731**

Run:
```bash
cd /Users/pinxinliu/Downloads/BVB && python refine/bvbrefine.py start 41125731 --n 8
```
Expected: QA + unit tests printed, 8 frames written, baseline snippet shown. Confirms the loop entry point works on a real scene.

- [ ] **Step 3: Final commit (docs/loop wiring)**

```bash
git add -A
git commit -m "chore(refine): end-to-end loop verified on 41125731"
```

---

## Self-Review Notes

- **Spec coverage:** §2 responsibilities → Tasks 6/7 (CLI) + Task 8 (skill); §3 loop → Task 8;
  §4 hybrid verification → Tasks 4/5/7; §5 CLI subcommands → Tasks 6/7; §6 checkpoint
  convention → Task 8 + the `copy=True` snippet in Task 6's `cmd_start`; §7 skill incl.
  Heuristics → Task 8; §8 propagation → `start` entry point (Task 6). All covered.
- **Known environment dependency:** the `gate` export and the offline-eval smoke require a
  working headless Blender with `addons/export_bpy_code.py`; tests skip gracefully when it is
  absent so the pure suite stays green in any environment.
- **Type consistency:** `build_pairs_row` emits `{"id", "pred_bpy_path"}` exactly matching the
  fields `eval/unit_test_metric.py` reads (`get_first(record, ("id","scene_id"))` and
  `("pred_bpy_path", ...)`). `evaluate_function_offline` calls the harness's
  `evaluate_function_test_with_params(scene_id, test, groups, params)` with the same signature.
```
