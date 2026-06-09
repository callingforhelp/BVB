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


# Sub-part name tokens: when an object_ref matches several component groups
# (e.g. "Washer_Body" + "Washer_Door"), prefer the main body over these parts.
_SUBPART_TOKENS = ("door", "panel", "handle", "frame", "leg", "divider",
                   "mirror", "knob", "hinge", "lid", "drawer", "shelf")
_REF_STOPWORDS = {"the", "a", "an", "or", "and", "of", "room", "boundary"}


def _match_group(group_keys, object_ref, size_of=None):
    """Pick the group key that best matches a unit test's `object_ref` phrase.

    Matches by keyword (e.g. object_ref "washer" -> "Washer_Body"), preferring
    the main component over sub-parts (door/panel/handle/...). When `size_of`
    (a callable key->float) is given, pick the LARGEST matching group — the body
    of a multi-part object (a sofa's body, not its arm). Returns None if nothing
    matches.
    """
    ref = (object_ref or "").lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", ref) if t and t not in _REF_STOPWORDS]
    if not tokens:
        return None
    candidates = [k for k in group_keys if any(t in k.lower() for t in tokens)]
    if not candidates:
        return None
    main = [k for k in candidates if not any(s in k.lower() for s in _SUBPART_TOKENS)]
    pool = main or candidates
    if size_of is not None:
        return max(pool, key=lambda k: size_of(k) or 0.0)
    return pool[0]


def _group_longest_dim(group):
    """Longest bbox edge (meters) of a parsed SceneGroup, or 0.0 if unknown."""
    bmin = getattr(group, "bbox_min", None)
    bmax = getattr(group, "bbox_max", None)
    if not bmin or not bmax:
        return 0.0
    return max(bmax[i] - bmin[i] for i in range(3))


def ground_function_params(test: dict, groups, floor_key=None) -> dict:
    """Supply grounded_params for offline deterministic function evaluation.

    `room_area` grounds to the floor group; `longest_dimension` grounds to the
    object named by the test's `object_ref` (e.g. "washer" -> "Washer_Body").
    `count_objects` / `closest_distance` need multi-object grounding we don't
    infer offline -> return {} so the evaluator reports them unresolved (use
    --judge for those).
    """
    fn = test.get("function")
    params = test.get("params", {}) if isinstance(test.get("params"), dict) else {}
    ref = params.get("object_ref")
    keys = list(groups)
    largest = lambda r: _match_group(keys, r, size_of=lambda k: _group_longest_dim(groups.get(k)))
    if fn == "room_area":
        return {"object_group": floor_key or pick_floor_group(groups) or _match_group(keys, ref)}
    if fn == "longest_dimension":
        return {"object_group": largest(ref)}
    if fn == "closest_distance":
        refs = params.get("object_refs") or []
        if len(refs) >= 2:
            return {"object_a_group": largest(refs[0]), "object_b_group": largest(refs[1])}
        return {}
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
    params = ground_function_params(test, groups, floor_key)
    scene_id = str(test.get("scene_name"))
    return utm.evaluate_function_test_with_params(scene_id, test, groups, params)
