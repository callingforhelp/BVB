from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from exec_scene import introspect_scene, resolve_blender_executable  # noqa: E402


class SceneIntrospectIntegrationTest(unittest.TestCase):
    def test_camera_trajectory_payload(self) -> None:
        try:
            blender = resolve_blender_executable(None)
        except SystemExit:
            self.skipTest("Blender is not installed")
        source = """
import bpy
import math
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.object
cube.name = "TargetCube"
cube.location = (0, 0, 0)
cube.keyframe_insert(data_path="location", frame=1)
cube.location = (1, 0, 0)
cube.keyframe_insert(data_path="location", frame=3)
bpy.ops.mesh.primitive_plane_add(
    size=4,
    location=(0, -2.5, 0),
    rotation=(math.radians(90), 0, 0),
)
glass = bpy.context.object
glass.name = "GlassPane"
material = bpy.data.materials.new("TransparentGlass")
material.use_nodes = True
principled = material.node_tree.nodes.get("Principled BSDF")
principled.inputs["Alpha"].default_value = 0.2
if principled.inputs.get("Transmission Weight") is not None:
    principled.inputs["Transmission Weight"].default_value = 1.0
glass.data.materials.append(material)
bpy.ops.object.camera_add(location=(0, -5, 1))
camera = bpy.context.object
camera.name = "Camera"
bpy.context.scene.camera = camera
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 3
camera.location = (0, -5, 1)
camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.keyframe_insert(data_path="location", frame=1)
camera.keyframe_insert(data_path="rotation_euler", frame=1)
camera.location = (2, -5, 1)
camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.keyframe_insert(data_path="location", frame=3)
camera.keyframe_insert(data_path="rotation_euler", frame=3)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "animated_scene.py"
            path.write_text(source, encoding="utf-8")
            cache_dir = Path(tmp) / "cache"
            payload = introspect_scene(
                path,
                blender_bin=blender,
                timeout=60,
                cache_dir=cache_dir,
                profile="appearance",
            )
            geometry_payload = introspect_scene(
                path,
                blender_bin=blender,
                timeout=60,
                cache_dir=cache_dir,
                profile="geometry",
            )
            failing = Path(tmp) / "failing_scene.py"
            failing.write_text("raise RuntimeError('expected failure')", encoding="utf-8")
            failed_payload = introspect_scene(
                failing,
                blender_bin=blender,
                timeout=60,
                cache_dir=cache_dir,
                profile="geometry",
            )
            cache_files = list(cache_dir.glob("*.json"))
        self.assertTrue(payload["exec_ok"], payload.get("exec_error"))
        cube_entry = next(item for item in payload["objects"] if item["name"] == "TargetCube")
        self.assertAlmostEqual(cube_entry["footprint_area"], 4.0)
        self.assertEqual(cube_entry["dimensions"], [2.0, 2.0, 2.0])
        temporal = payload["temporal"]
        self.assertTrue(temporal["camera_motion"])
        self.assertEqual([item["frame"] for item in temporal["frames"]], [1, 2, 3])
        self.assertEqual(len(temporal["frames"][0]["world_to_camera"]), 4)
        self.assertIn("TargetCube", temporal["frames"][0]["animated_bboxes"])
        self.assertNotEqual(
            temporal["frames"][0]["animated_bboxes"]["TargetCube"]["centroid"],
            temporal["frames"][-1]["animated_bboxes"]["TargetCube"]["centroid"],
        )
        self.assertIsNotNone(temporal["first_visible_frames"]["TargetCube"])
        self.assertNotIn("temporal", geometry_payload)
        self.assertFalse(failed_payload["exec_ok"])
        self.assertEqual(len(cache_files), 2)


if __name__ == "__main__":
    unittest.main()
