from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from eval_utils import SceneGroup, SceneIndex  # noqa: E402
from exec_scene import scene_manifest  # noqa: E402
from scene_geometry import (  # noqa: E402
    appearance_order,
    bbox_distance,
    bbox_in_camera_frustum,
    closest_candidate,
    grounded_union,
    infer_floor_group,
    relative_direction,
    required_route_turns,
    route_turns_from_camera,
    union_groups,
)
from unit_test_metric import (  # noqa: E402
    build_batch_prompt,
    evaluate_function_test_with_params,
    prepare_grounding_requests,
)


def group(
    key: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float] = (0.4, 0.4, 0.4),
) -> SceneGroup:
    half = tuple(value / 2.0 for value in size)
    return SceneGroup(
        key=key,
        component_count=1,
        centroid=center,
        bbox_min=tuple(center[i] - half[i] for i in range(3)),
        bbox_max=tuple(center[i] + half[i] for i in range(3)),
        object_names=[key],
    )


class SceneGeometryTest(unittest.TestCase):
    def test_union_groups_and_bbox_distance(self) -> None:
        groups = {
            "seat": group("seat", (0.0, 0.0, 0.0)),
            "back": group("back", (0.0, 0.5, 0.5)),
            "table": group("table", (2.0, 0.0, 0.0)),
        }
        sofa = union_groups(groups, ["seat", "back"])
        self.assertIsNotNone(sofa)
        self.assertEqual(sofa.component_count, 2)
        self.assertAlmostEqual(bbox_distance(sofa, groups["table"]), 1.6)

    def test_collection_alias_unions_all_members(self) -> None:
        seat = group("SofaSeat", (0.0, 0.0, 0.0))
        back = group("SofaBack", (0.0, 1.0, 0.5))
        seat.object_names.append("SofaCollection")
        back.object_names.append("SofaCollection")
        combined = union_groups(
            {"SofaSeat": seat, "SofaBack": back},
            "SofaCollection",
        )
        self.assertEqual(combined.component_count, 2)
        self.assertEqual(set(combined.object_names), {"SofaSeat", "SofaBack", "SofaCollection"})
        automatic, report = grounded_union(
            {"SofaSeat": seat, "SofaBack": back},
            None,
            "sofa",
        )
        self.assertIsNone(automatic)
        self.assertFalse(report["auto_matched"])

    def test_union_footprint_does_not_double_count_overlap(self) -> None:
        first = group("FloorA", (0.0, 0.0, 0.0), (4.0, 4.0, 0.05))
        second = group("FloorB", (0.0, 0.0, 0.0), (4.0, 4.0, 0.05))
        first.footprint_area = 16.0
        second.footprint_area = 16.0
        combined = union_groups({"FloorA": first, "FloorB": second}, ["FloorA", "FloorB"])
        self.assertEqual(combined.footprint_area, 16.0)

    def test_semantic_grounding_accepts_synonyms_and_rejects_substitutions(self) -> None:
        groups = {
            "dark mattress": group("dark mattress", (0.0, 0.0, 0.0)),
            "monitor screen": group("monitor screen", (1.0, 0.0, 0.0)),
            "board joint": group("board joint", (2.0, 0.0, 0.0)),
            "Kitchen": group("Kitchen", (3.0, 0.0, 0.0)),
        }
        bed, bed_report = grounded_union(groups, "dark mattress", "bed")
        self.assertIsNotNone(bed)
        self.assertEqual(bed_report["accepted_group_keys"], ["dark mattress"])
        auto_bed, auto_report = grounded_union(groups, None, "bed")
        self.assertIsNotNone(auto_bed)
        self.assertTrue(auto_report["auto_matched"])
        strict_bed, strict_report = grounded_union(
            groups,
            None,
            "bed",
            allow_auto_match=False,
        )
        self.assertIsNone(strict_bed)
        self.assertFalse(strict_report["auto_matched"])
        for key, semantic in [
            ("monitor screen", "computer_mouse"),
            ("board joint", "cutting_board"),
            ("Kitchen", "kettle"),
        ]:
            grounded, report = grounded_union(groups, key, semantic)
            self.assertIsNone(grounded)
            self.assertEqual(report["rejected_groundings"][0]["reason"], "semantic_mismatch")

    def test_relative_direction_modes(self) -> None:
        anchor = group("anchor", (0.0, 0.0, 0.0))
        facing = group("facing", (0.0, 1.0, 0.0))
        self.assertEqual(
            relative_direction(
                anchor=anchor,
                facing=facing,
                query=group("query", (-1.0, 1.0, 0.0)),
                difficulty="easy",
                reference=facing,
            ),
            "left",
        )
        self.assertEqual(
            relative_direction(
                anchor=anchor,
                facing=facing,
                query=group("query", (1.0, 0.0, 0.0)),
                difficulty="medium",
            ),
            "right",
        )
        self.assertEqual(
            relative_direction(
                anchor=anchor,
                facing=facing,
                query=group("query", (-1.0, -1.0, 0.0)),
                difficulty="hard",
            ),
            "back-left",
        )

    def test_closest_candidate_and_tie(self) -> None:
        anchor = group("anchor", (0.0, 0.0, 0.0))
        label, distances = closest_candidate(
            anchor,
            {
                "near": group("near", (1.0, 0.0, 0.0)),
                "far": group("far", (3.0, 0.0, 0.0)),
            },
        )
        self.assertEqual(label, "near")
        self.assertLess(distances["near"], distances["far"])
        tied, _ = closest_candidate(
            anchor,
            {
                "left": group("left", (-1.0, 0.0, 0.0)),
                "right": group("right", (1.0, 0.0, 0.0)),
            },
        )
        self.assertIsNone(tied)

    def test_floor_inference_accepts_generic_plane_geometry(self) -> None:
        inferred = infer_floor_group(
            {
                "Plane": group("Plane", (0.0, 0.0, 0.0), (6.0, 5.0, 0.05)),
                "Table": group("Table", (0.0, 0.0, 1.0), (2.0, 1.0, 2.0)),
            }
        )
        self.assertEqual(inferred.key, "Plane")
        carpet = infer_floor_group(
            {"carpet": group("carpet", (0.0, 0.0, 0.0), (6.0, 5.0, 0.05))}
        )
        self.assertEqual(carpet.key, "carpet")
        self.assertIsNone(
            infer_floor_group(
                {
                    "Rug": group("Rug", (0.0, 0.0, 0.0), (6.0, 5.0, 0.05)),
                    "TableGlass": group(
                        "TableGlass",
                        (0.0, 0.0, 1.0),
                        (6.0, 5.0, 0.05),
                    ),
                }
            )
        )

    def test_room_area_prefers_mesh_footprint_over_bbox(self) -> None:
        floor = group("Floor", (0.0, 0.0, 0.0), (10.0, 10.0, 0.05))
        floor.footprint_area = 30.0
        result = evaluate_function_test_with_params(
            "scene",
            {
                "test_id": "room",
                "test_type": "room_size_estimation",
                "function": "room_area",
                "expected": {"value": 30.0, "tolerance": 1.0},
                "params": {"object_ref": "floor or room boundary"},
            },
            {"Floor": floor},
            {"object_group": "Floor"},
            temporal=None,
            execution_mode="execute",
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["actual"], 30.0)
        self.assertEqual(
            result["evidence"]["grounding_validation"]["area_method"],
            "mesh_projected_footprint",
        )

    def test_object_size_prefers_evaluated_dimensions(self) -> None:
        rotated_box = group("Box", (0.0, 0.0, 0.0), (3.0, 3.0, 1.0))
        rotated_box.dimensions = (2.0, 1.0, 1.0)
        result = evaluate_function_test_with_params(
            "scene",
            {
                "test_id": "size",
                "test_type": "object_size_estimation",
                "function": "longest_dimension",
                "expected": {"value": 200.0, "tolerance": 1.0},
                "params": {"object_ref": "box"},
            },
            {"Box": rotated_box},
            {"object_group": "Box"},
            temporal=None,
            execution_mode="execute",
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["actual"], 200.0)

    def test_route_turn_sequence(self) -> None:
        turns = required_route_turns(
            start=group("start", (0.0, 0.0, 0.0)),
            facing=group("facing", (0.0, 1.0, 0.0)),
            waypoints=[
                group("first", (0.0, 2.0, 0.0)),
                group("second", (-2.0, 2.0, 0.0)),
                group("third", (-2.0, 0.0, 0.0)),
            ],
        )
        self.assertEqual(turns, ["turn_left", "turn_left"])

    def test_route_uses_camera_trajectory_and_right_relation(self) -> None:
        def camera_frame(frame: int, x: float, y: float) -> dict:
            return {
                "frame": frame,
                "camera_position": [x, y, 0.0],
                "camera_forward": [0.0, 1.0, 0.0],
                "world_to_camera": [
                    [1.0, 0.0, 0.0, -x],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, y],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }

        turns, evidence = route_turns_from_camera(
            start=group("start", (0.0, 0.0, 0.0)),
            facing=group("facing", (0.0, 1.0, 0.0)),
            route_steps=[
                {"group": group("first", (0.0, 2.0, 0.0)), "relation": "near"},
                {"group": group("bin", (3.0, 3.0, 0.0)), "relation": "right"},
            ],
            temporal={
                "camera_motion": True,
                "frames": [
                    camera_frame(1, 0.0, 0.0),
                    camera_frame(2, 0.0, 2.0),
                    camera_frame(3, 2.0, 2.0),
                ],
            },
        )
        self.assertEqual(turns, ["turn_right"])
        self.assertEqual(evidence["matched_steps"][-1]["relation"], "right")
        missing_turns, missing_evidence = route_turns_from_camera(
            start=group("start", (0.0, 0.0, 0.0)),
            facing=group("facing", (0.0, 1.0, 0.0)),
            route_steps=[
                {"group": group("far", (100.0, 100.0, 0.0)), "relation": "pass"}
            ],
            temporal={
                "camera_motion": True,
                "frames": [
                    camera_frame(1, 0.0, 0.0),
                    camera_frame(2, 0.0, 2.0),
                ],
            },
        )
        self.assertIsNone(missing_turns)
        self.assertTrue(missing_evidence["reason"].startswith("route_landmark_not_reached"))

    def test_frustum_and_appearance_order(self) -> None:
        camera = {"angle_x": 1.0, "angle_y": 1.0, "clip_start": 0.1, "clip_end": 100.0}
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        visible = group("visible", (0.0, 0.0, -5.0))
        behind = group("behind", (0.0, 0.0, 5.0))
        frame = {"frame": 1, "camera_position": [0.0, 0.0, 0.0], "world_to_camera": identity}
        self.assertTrue(bbox_in_camera_frustum(visible, frame, camera))
        self.assertFalse(bbox_in_camera_frustum(behind, frame, camera))

        temporal = {
            "camera_motion": True,
            "camera": camera,
            "frames": [frame],
            "hidden_objects": [],
            "animated_objects": [],
        }
        result = appearance_order(
            category_groups={"visible": visible},
            all_groups={"visible": visible},
            temporal=temporal,
        )
        self.assertEqual(result.order, ["visible"])
        static = appearance_order(
            category_groups={"visible": visible},
            all_groups={"visible": visible},
            temporal={**temporal, "camera_motion": False},
        )
        self.assertEqual(static.reason, "static_camera")

        animated_temporal = {
            **temporal,
            "frames": [
                {
                    **frame,
                    "animated_bboxes": {
                        "animated": {
                            "bbox_min": [-0.2, -0.2, -5.2],
                            "bbox_max": [0.2, 0.2, -4.8],
                            "centroid": [0.0, 0.0, -5.0],
                        }
                    },
                }
            ],
        }
        animated = group("animated", (0.0, 0.0, 5.0))
        animated_result = appearance_order(
            category_groups={"animated": animated},
            all_groups={"animated": animated},
            temporal=animated_temporal,
        )
        self.assertEqual(animated_result.order, ["animated"])

    def test_missing_camera_motion_is_submission_failure(self) -> None:
        test = {
            "test_id": "appearance",
            "test_type": "obj_appearance_order",
            "function": "appearance_order",
            "expected": {"order": ["chair"]},
            "params": {"category_refs": ["chair"]},
        }
        result = evaluate_function_test_with_params(
            "scene",
            test,
            {"Chair": group("Chair", (0.0, 0.0, 0.0))},
            {"category_groups": {"chair": "Chair"}},
            temporal={"camera_motion": False, "frames": []},
            execution_mode="execute",
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["evidence"]["reason"], "static_camera")

    def test_prompt_uses_manifest_not_binary_source(self) -> None:
        groups = {"Chair": group("Chair", (1.0, 2.0, 0.5))}
        index = SceneIndex(path="candidate.blend", parse_ok=True, parse_error=None, objects=[], materials={})
        manifest = scene_manifest(index, groups)
        requests, _ = prepare_grounding_requests(
            [{"test_id": "ut_1", "function": "count_objects", "params": {"object_ref": "chair"}}]
        )
        prompt = build_batch_prompt("scene", manifest, requests)
        self.assertNotIn("Blender Python scene code", prompt)
        self.assertNotIn("BLENDER-v", prompt)
        self.assertNotIn("\x00", prompt)
        json.dumps(manifest)

    def test_duplicate_grounding_requests_are_deduplicated(self) -> None:
        requests, mapping = prepare_grounding_requests(
            [
                {
                    "test_id": "easy",
                    "function": "relative_direction",
                    "params": {"anchor_ref": "chair", "facing_ref": "table", "query_ref": "tv"},
                },
                {
                    "test_id": "hard",
                    "function": "relative_direction",
                    "params": {"anchor_ref": "chair", "facing_ref": "table", "query_ref": "tv"},
                },
            ]
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(mapping["easy"], mapping["hard"])


if __name__ == "__main__":
    unittest.main()
