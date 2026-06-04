"""Pure logic for the BVB iterative-refinement CLI. No side effects here."""
from __future__ import annotations

import csv
import json
import re
import sys
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

SOURCES = ("arkitscenes", "scannet", "scannetpp")


def focus_hint(axis: str) -> str:
    return FOCUS_HINTS.get(axis, "(see project guide for this axis)")


def parse_checkpoint_name(filename: str):
    """Return (version:int, desc:str) for '<id>_vNN_<desc>.blend', else None."""
    m = _CKPT_RE.match(filename)
    if not m:
        return None
    return int(m.group("nn")), m.group("desc")


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


def pick_floor_group(groups) -> "str | None":
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
