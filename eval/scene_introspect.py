#!/usr/bin/env python3
"""Blender-side scene introspection for execution-based BVB evaluation.

Run inside Blender, NOT with the host Python:

    blender --background --factory-startup \
        --python scene_introspect.py -- \
        --input <scene.blend|scene.py> --output <introspect.json>

Instead of statically parsing bpy source, this loads/executes the submission
and reads the *real* scene geometry through the evaluated dependency graph:
each object's world-space axis-aligned bounding box is computed from
``matrix_world @ bound_box`` on the evaluated object, so modifiers (ARRAY,
MIRROR, SOLIDIFY, ...), rotations, and parent/collection transforms are all
reflected. The result is emitted as JSON between marker lines so the host can
recover it from Blender's noisy stdout.

Any bpy submission can be evaluated this way, regardless of how it was authored.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


RESULT_START = "===BVB_INTROSPECT_START==="
RESULT_END = "===BVB_INTROSPECT_END==="


def arg_after(name: str, default: str | None = None) -> str | None:
    argv = sys.argv
    if "--" not in argv:
        return default
    args = argv[argv.index("--") + 1 :]
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
    return default


def reset_to_empty_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_scene(input_path: Path) -> None:
    suffix = input_path.suffix.lower()
    if suffix == ".blend":
        # use_scripts=False disables auto-run of any Python embedded in the
        # .blend (drivers/handlers), so loading an untrusted submission does
        # not execute arbitrary code during evaluation.
        bpy.ops.wm.open_mainfile(filepath=str(input_path), use_scripts=False)
        return
    # Treat anything else as a bpy Python script: start from a clean empty
    # scene and execute it in a fresh namespace.
    reset_to_empty_scene()
    code = input_path.read_text(encoding="utf-8", errors="replace")
    namespace = {"__name__": "__main__", "__file__": str(input_path)}
    exec(compile(code, str(input_path), "exec"), namespace)


def round_vec(vec, dp: int = 6) -> list[float]:
    return [round(float(component), dp) for component in vec]


def object_kind(obj) -> str:
    if obj.type == "CAMERA":
        return "camera"
    if obj.type == "LIGHT":
        return "light"
    return "mesh"


def world_bbox(obj_eval):
    """World-space AABB from the evaluated object's bound_box, or None."""
    try:
        corners = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
    except Exception:
        return None, None, None
    if not corners:
        return None, None, None
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    bbox_min = [min(xs), min(ys), min(zs)]
    bbox_max = [max(xs), max(ys), max(zs)]
    # An object with no geometry (e.g. an empty) yields a degenerate box at the
    # origin; treat that as "no bbox" so it does not pollute group bounds.
    if all(abs(a - b) < 1e-9 for a, b in zip(bbox_min, bbox_max)) and all(
        abs(v) < 1e-9 for v in bbox_min
    ):
        return None, None, None
    centroid = [(bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3)]
    return bbox_min, bbox_max, centroid


def object_world_points(obj_eval) -> list:
    try:
        corners = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
    except Exception:
        return []
    if not corners:
        return []
    centroid = sum(corners, Vector()) / len(corners)
    return [centroid, *corners]


def projected_footprint_area(obj_eval) -> float | None:
    if obj_eval.type != "MESH":
        return None
    mesh = obj_eval.to_mesh()
    try:
        matrix = obj_eval.matrix_world
        normal_matrix = matrix.to_3x3().inverted().transposed()
        upward_area = 0.0
        downward_area = 0.0
        for polygon in mesh.polygons:
            normal = (normal_matrix @ polygon.normal).normalized()
            if abs(normal.z) < 0.5:
                continue
            points = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
            area = abs(
                sum(
                    points[index].x * points[(index + 1) % len(points)].y
                    - points[(index + 1) % len(points)].x * points[index].y
                    for index in range(len(points))
                )
            ) / 2.0
            if normal.z >= 0.0:
                upward_area += area
            else:
                downward_area += area
        area = max(upward_area, downward_area)
        return round(float(area), 6) if area > 0.0 else None
    finally:
        obj_eval.to_mesh_clear()


def principled_base_color(material):
    if material is None or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            try:
                value = node.inputs["Base Color"].default_value
                return [round(float(value[i]), 6) for i in range(3)]
            except Exception:
                return None
    return None


def material_is_transparent(material) -> bool:
    if material is None:
        return False
    if len(material.diffuse_color) >= 4 and float(material.diffuse_color[3]) < 0.95:
        return True
    if not material.use_nodes:
        return False
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        alpha = node.inputs.get("Alpha")
        if alpha is not None and float(alpha.default_value) < 0.95:
            return True
        transmission = node.inputs.get("Transmission Weight")
        if transmission is None:
            transmission = node.inputs.get("Transmission")
        if transmission is not None and float(transmission.default_value) > 0.05:
            return True
    return False


def object_is_transparent(obj) -> bool:
    return any(
        slot.material is not None and material_is_transparent(slot.material)
        for slot in obj.material_slots
    )


def ray_reaches_object(
    scene,
    depsgraph,
    origin,
    point,
    target_name: str,
    transparent_names: set[str],
) -> bool:
    direction = point - origin
    remaining = direction.length
    if remaining <= 1e-9:
        return False
    direction.normalize()
    current_origin = origin.copy()
    for _ in range(16):
        hit, location, _, _, hit_obj, _ = scene.ray_cast(
            depsgraph,
            current_origin,
            direction,
            distance=remaining + 1e-4,
        )
        if not hit or hit_obj is None:
            return False
        if hit_obj.name == target_name:
            return True
        if hit_obj.name not in transparent_names:
            return False
        travelled = (location - current_origin).length + 1e-4
        remaining -= travelled
        if remaining <= 0.0:
            return False
        current_origin = location + direction * 1e-4
    return False


def object_visible_from_camera(
    scene,
    depsgraph,
    camera_eval,
    camera_data,
    camera_position,
    obj,
    transparent_names: set[str],
) -> bool:
    obj_eval = obj.evaluated_get(depsgraph)
    for point in object_world_points(obj_eval):
        projected = world_to_camera_view(scene, camera_eval, point)
        distance = (point - camera_position).length
        if (
            projected.z <= 0.0
            or not (0.0 <= projected.x <= 1.0 and 0.0 <= projected.y <= 1.0)
            or distance < camera_data.clip_start
            or distance > camera_data.clip_end
        ):
            continue
        if ray_reaches_object(
            scene,
            depsgraph,
            camera_position,
            point,
            obj.name,
            transparent_names,
        ):
            return True
    return False


def collect_materials() -> dict:
    materials = {}
    for material in bpy.data.materials:
        color = principled_base_color(material)
        if color is not None:
            materials[material.name] = color
    return materials


def matrix_rows(matrix) -> list[list[float]]:
    return [[round(float(matrix[row][column]), 8) for column in range(4)] for row in range(4)]


def has_animation(obj) -> bool:
    current = obj
    while current is not None:
        animation_data = getattr(current, "animation_data", None)
        if animation_data and (animation_data.action or animation_data.drivers):
            return True
        current = current.parent
    return False


def collect_temporal(profile: str, max_samples: int = 600) -> dict:
    scene = bpy.context.scene
    camera_obj = scene.camera
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    fps = float(scene.render.fps) / max(1.0, float(scene.render.fps_base))
    if camera_obj is None:
        return {
            "frame_start": frame_start,
            "frame_end": frame_end,
            "fps": fps,
            "sample_step": 1,
            "camera_motion": False,
            "camera": None,
            "frames": [],
            "hidden_objects": [],
            "animated_objects": [],
        }

    span = max(1, frame_end - frame_start + 1)
    sample_step = max(1, int(math.ceil(span / max_samples)))
    sample_frames = list(range(frame_start, frame_end + 1, sample_step))
    if sample_frames[-1] != frame_end:
        sample_frames.append(frame_end)

    original_frame = int(scene.frame_current)
    include_visibility = profile == "appearance"
    animated_meshes = [
        obj for obj in scene.objects if obj.type == "MESH" and has_animation(obj)
    ] if include_visibility else []
    visible_meshes = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and not obj.hide_render and not obj.hide_viewport
    ] if include_visibility else []
    transparent_names = {
        obj.name for obj in visible_meshes if object_is_transparent(obj)
    }
    camera_data = camera_obj.data
    first_visible_frames = {obj.name: None for obj in visible_meshes}
    frames = []
    for frame in sample_frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        camera_eval = camera_obj.evaluated_get(depsgraph)
        matrix_world = camera_eval.matrix_world.copy()
        camera_forward = matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        camera_right = matrix_world.to_quaternion() @ Vector((1.0, 0.0, 0.0))
        camera_position = matrix_world.translation.copy()
        for obj in visible_meshes:
            if first_visible_frames[obj.name] is not None:
                continue
            if object_visible_from_camera(
                scene,
                depsgraph,
                camera_eval,
                camera_data,
                camera_position,
                obj,
                transparent_names,
            ):
                first_visible_frames[obj.name] = frame
        animated_bboxes = {}
        for obj in animated_meshes:
            obj_eval = obj.evaluated_get(depsgraph)
            bbox_min, bbox_max, centroid = world_bbox(obj_eval)
            animated_bboxes[obj.name] = {
                "bbox_min": round_vec(bbox_min) if bbox_min else None,
                "bbox_max": round_vec(bbox_max) if bbox_max else None,
                "centroid": round_vec(centroid) if centroid else None,
            }
        frames.append(
            {
                "frame": frame,
                "camera_position": round_vec(camera_position, 8),
                "camera_forward": round_vec(camera_forward, 8),
                "camera_right": round_vec(camera_right, 8),
                "world_to_camera": matrix_rows(matrix_world.inverted()),
                "animated_bboxes": animated_bboxes,
            }
        )
    if include_visibility and sample_step > 1:
        frame_positions = {frame: index for index, frame in enumerate(sample_frames)}
        for obj in visible_meshes:
            coarse_frame = first_visible_frames[obj.name]
            if coarse_frame is None:
                continue
            coarse_index = frame_positions[coarse_frame]
            previous_frame = (
                sample_frames[coarse_index - 1]
                if coarse_index > 0
                else frame_start - 1
            )
            for frame in range(previous_frame + 1, coarse_frame + 1):
                scene.frame_set(frame)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                camera_eval = camera_obj.evaluated_get(depsgraph)
                camera_position = camera_eval.matrix_world.translation.copy()
                if object_visible_from_camera(
                    scene,
                    depsgraph,
                    camera_eval,
                    camera_data,
                    camera_position,
                    obj,
                    transparent_names,
                ):
                    first_visible_frames[obj.name] = frame
                    break
    scene.frame_set(original_frame)

    first_matrix = frames[0]["world_to_camera"] if frames else None
    camera_motion = any(
        any(abs(value - first_matrix[row][column]) > 1e-6 for row, values in enumerate(matrix) for column, value in enumerate(values))
        for matrix in (entry["world_to_camera"] for entry in frames[1:])
    ) if first_matrix is not None else False
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "fps": fps,
        "sample_step": sample_step,
        "camera_motion": camera_motion,
        "camera": {
            "name": camera_obj.name,
            "type": camera_data.type,
            "lens": round(float(getattr(camera_data, "lens", 0.0)), 6),
            "sensor_width": round(float(getattr(camera_data, "sensor_width", 0.0)), 6),
            "clip_start": round(float(camera_data.clip_start), 6),
            "clip_end": round(float(camera_data.clip_end), 6),
            "angle_x": round(float(camera_data.angle_x), 8),
            "angle_y": round(float(camera_data.angle_y), 8),
        },
        "frames": frames,
        "hidden_objects": sorted(
            obj.name for obj in scene.objects if bool(obj.hide_render) or bool(obj.hide_viewport)
        ),
        "animated_objects": sorted(obj.name for obj in scene.objects if has_animation(obj)),
        "first_visible_frames": first_visible_frames,
    }


def introspect(profile: str = "geometry") -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects = []
    for obj in bpy.context.scene.objects:
        obj_eval = obj.evaluated_get(depsgraph)
        bbox_min, bbox_max, centroid = world_bbox(obj_eval)
        material_name = None
        if obj.type == "MESH":
            for slot in obj.material_slots:
                if slot.material is not None:
                    material_name = slot.material.name
                    break
        collection = obj.users_collection[0].name if obj.users_collection else None
        entry = {
            "name": obj.name,
            "kind": object_kind(obj),
            "collection": collection,
            "location": round_vec(obj.matrix_world.translation),
            "rotation": round_vec(obj.rotation_euler),
            "scale": round_vec(obj.scale),
            "world_bbox_min": round_vec(bbox_min) if bbox_min else None,
            "world_bbox_max": round_vec(bbox_max) if bbox_max else None,
            "centroid": round_vec(centroid) if centroid else None,
            "dimensions": round_vec(obj_eval.dimensions),
            "footprint_area": projected_footprint_area(obj_eval),
            "material": material_name,
            "hide_render": bool(obj.hide_render),
            "hide_viewport": bool(obj.hide_viewport),
            "animated": has_animation(obj),
        }
        if obj.type == "LIGHT":
            entry["light_type"] = obj.data.type
            entry["light_energy"] = float(getattr(obj.data, "energy", 0.0))
        objects.append(entry)
    result = {
        "exec_ok": True,
        "exec_error": None,
        "blender_version": bpy.app.version_string,
        "objects": objects,
        "materials": collect_materials(),
    }
    if profile in {"route", "appearance"}:
        result["temporal"] = collect_temporal(profile)
    return result


def main() -> None:
    input_arg = arg_after("--input")
    output_arg = arg_after("--output")
    profile = str(arg_after("--profile", "geometry"))
    if not input_arg:
        raise SystemExit("Missing required argument: --input")
    if profile not in {"geometry", "route", "appearance"}:
        raise SystemExit(f"Unsupported introspection profile: {profile}")

    try:
        load_scene(Path(input_arg))
        result = introspect(profile)
    except Exception as exc:  # noqa: BLE001 - report any failure as a clean result
        result = {
            "exec_ok": False,
            "exec_error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "blender_version": bpy.app.version_string,
            "objects": [],
            "materials": {},
        }

    payload = json.dumps(result, ensure_ascii=False)
    print(RESULT_START)
    print(payload)
    print(RESULT_END)

    if output_arg:
        out_path = Path(output_arg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
