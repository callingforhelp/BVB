import json
import subprocess

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
    test = {"function": "room_area"}
    assert bvb_lib.ground_function_params(test, "Floor") == {"object_group": "Floor"}


def test_ground_function_params_other_returns_empty():
    assert bvb_lib.ground_function_params({"function": "count_objects"}, "Floor") == {}
