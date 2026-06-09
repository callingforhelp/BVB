import json
import subprocess
from pathlib import Path

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
    groups = {"Floor": None, "Wall_Back": None}
    test = {"function": "room_area", "params": {"object_ref": "floor or room boundary"}}
    assert bvb_lib.ground_function_params(test, groups, floor_key="Floor") == {"object_group": "Floor"}


def test_ground_function_params_longest_dimension_picks_body():
    groups = {"Washer_Body": None, "Washer_Door": None, "Sink_Basin": None}
    test = {"function": "longest_dimension", "params": {"object_ref": "washer"}}
    assert bvb_lib.ground_function_params(test, groups) == {"object_group": "Washer_Body"}


def test_ground_function_params_count_returns_empty():
    assert bvb_lib.ground_function_params({"function": "count_objects"}, {"X": None}) == {}


def test_match_group_prefers_body_over_subpart():
    assert bvb_lib._match_group(["Washer_Body", "Washer_Door"], "washer") == "Washer_Body"


def test_match_group_none_when_no_match():
    assert bvb_lib._match_group(["Floor", "Wall_Back"], "washer") is None


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
