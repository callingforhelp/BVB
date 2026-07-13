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
import sys
import traceback
from pathlib import Path

import bpy
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


def collect_materials() -> dict:
    materials = {}
    for material in bpy.data.materials:
        color = principled_base_color(material)
        if color is not None:
            materials[material.name] = color
    return materials


def introspect() -> dict:
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
            "material": material_name,
        }
        if obj.type == "LIGHT":
            entry["light_type"] = obj.data.type
            entry["light_energy"] = float(getattr(obj.data, "energy", 0.0))
        objects.append(entry)
    return {
        "exec_ok": True,
        "exec_error": None,
        "blender_version": bpy.app.version_string,
        "objects": objects,
        "materials": collect_materials(),
    }


def main() -> None:
    input_arg = arg_after("--input")
    output_arg = arg_after("--output")
    if not input_arg:
        raise SystemExit("Missing required argument: --input")

    try:
        load_scene(Path(input_arg))
        result = introspect()
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
