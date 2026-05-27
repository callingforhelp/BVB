"""Shared helpers for BVB evaluation scripts."""

from __future__ import annotations

import ast
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SceneObject:
    name: str
    kind: str
    group_key: str
    location: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None
    scale: tuple[float, float, float] | None = None
    material: str | None = None
    light_type: str | None = None
    light_energy: float | None = None


@dataclass
class SceneGroup:
    key: str
    component_count: int
    centroid: tuple[float, float, float] | None
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    object_names: list[str] = field(default_factory=list)


@dataclass
class SceneIndex:
    path: str
    parse_ok: bool
    parse_error: str | None
    objects: list[SceneObject]
    materials: dict[str, tuple[float, float, float]]

    @property
    def mesh_objects(self) -> list[SceneObject]:
        return [obj for obj in self.objects if obj.kind not in {"camera", "light"}]

    @property
    def lights(self) -> list[SceneObject]:
        return [obj for obj in self.objects if obj.kind == "light"]

    @property
    def cameras(self) -> list[SceneObject]:
        return [obj for obj in self.objects if obj.kind == "camera"]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def get_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    joined = ", ".join(keys)
    raise KeyError(f"Missing one of fields: {joined}")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def as_float_tuple(value: Any, length: int = 3) -> tuple[float, ...] | None:
    if not isinstance(value, (tuple, list)) or len(value) < length:
        return None
    try:
        return tuple(float(value[i]) for i in range(length))
    except (TypeError, ValueError):
        return None


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return call_name(node.value)
    return None


def bsdf_input_assignment_name(target: ast.AST) -> str | None:
    if not isinstance(target, ast.Attribute) or target.attr != "default_value":
        return None
    if not isinstance(target.value, ast.Subscript):
        return None
    if call_name(target.value.value) != "_bsdf.inputs":
        return None
    value = literal_value(target.value.slice)
    return str(value) if value is not None else None


def keyword_value(call: ast.Call, name: str) -> Any:
    for keyword in call.keywords:
        if keyword.arg == name:
            return literal_value(keyword.value)
    return None


def target_name(target: ast.AST) -> str | None:
    return call_name(target)


def append_material_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if call_name(node.func) != "obj.data.materials.append" or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Subscript) and call_name(arg.value) == "bpy.data.materials":
        return literal_value(arg.slice)
    return None


def primitive_kind(name: str) -> str | None:
    prefix = "bpy.ops.mesh.primitive_"
    suffix = "_add"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    if name == "bpy.ops.object.camera_add":
        return "camera"
    if name == "bpy.ops.object.light_add":
        return "light"
    return None


def extract_material_var_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if call_name(node.func) != "bpy.data.materials.new":
        return None
    value = keyword_value(node, "name")
    if isinstance(value, str):
        return value
    if node.args:
        value = literal_value(node.args[0])
        if isinstance(value, str):
            return value
    return None


def parse_bpy_scene(path: Path) -> SceneIndex:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return SceneIndex(
            path=str(path),
            parse_ok=False,
            parse_error=f"{exc.__class__.__name__}: {exc}",
            objects=[],
            materials={},
        )

    materials: dict[str, tuple[float, float, float]] = {}
    material_vars: dict[str, str] = {}
    active_bsdf_material: str | None = None
    objects: list[SceneObject] = []
    current: SceneObject | None = None

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and stmt.targets:
            first_target = stmt.targets[0]
            first_target_name = target_name(first_target)

            if isinstance(stmt.value, ast.Call):
                material_name = extract_material_var_name(stmt.value)
                if material_name and isinstance(first_target, ast.Name):
                    material_vars[first_target.id] = material_name
                    continue

                op_name = call_name(stmt.value.func)
                if first_target_name == "obj" and op_name == "bpy.context.active_object":
                    continue

            if first_target_name == "_bsdf":
                root = call_name(stmt.value)
                if root:
                    for var_name, material_name in material_vars.items():
                        if root.startswith(var_name + "."):
                            active_bsdf_material = material_name
                            break
                continue

            if first_target_name == "_bsdf.inputs":
                continue

            value = literal_value(stmt.value)
            input_name = bsdf_input_assignment_name(first_target)
            if (
                input_name == "Base Color"
                and active_bsdf_material
                and (color := as_float_tuple(value, 3)) is not None
            ):
                materials[active_bsdf_material] = color[:3]  # type: ignore[assignment]
                continue

            if current is None:
                continue

            if first_target_name == "obj.name" and isinstance(value, str):
                current.name = value
                current.group_key = value
            elif first_target_name == "obj.rotation_euler" and (rot := as_float_tuple(value, 3)):
                current.rotation = rot[:3]  # type: ignore[assignment]
            elif first_target_name == "obj.scale" and (scale := as_float_tuple(value, 3)):
                current.scale = scale[:3]  # type: ignore[assignment]
            elif first_target_name == "obj.data.energy":
                try:
                    current.light_energy = float(value)
                except (TypeError, ValueError):
                    pass

        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            op_name = call_name(stmt.value.func)
            kind = primitive_kind(op_name or "")
            if kind:
                location = as_float_tuple(keyword_value(stmt.value, "location"), 3)
                light_type = keyword_value(stmt.value, "type") if kind == "light" else None
                name = kind
                current = SceneObject(
                    name=name,
                    kind=kind,
                    group_key=name,
                    location=location[:3] if location else None,  # type: ignore[index]
                    light_type=str(light_type) if light_type else None,
                )
                objects.append(current)
                continue

            if current is not None:
                material_name = append_material_name(stmt.value)
                if material_name:
                    current.material = material_name

    return SceneIndex(path=str(path), parse_ok=True, parse_error=None, objects=objects, materials=materials)


def object_extent(obj: SceneObject) -> tuple[float, float, float]:
    if obj.scale is not None:
        return tuple(abs(value) for value in obj.scale)
    return (1.0, 1.0, 1.0)


def build_groups(scene_index: SceneIndex) -> dict[str, SceneGroup]:
    grouped: dict[str, list[SceneObject]] = {}
    for obj in scene_index.mesh_objects:
        grouped.setdefault(obj.group_key, []).append(obj)

    groups: dict[str, SceneGroup] = {}
    for key, objects in grouped.items():
        located = [obj for obj in objects if obj.location is not None]
        if located:
            xs, ys, zs = zip(*(obj.location for obj in located if obj.location is not None))
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
            mins = [math.inf, math.inf, math.inf]
            maxs = [-math.inf, -math.inf, -math.inf]
            for obj in located:
                assert obj.location is not None
                extent = object_extent(obj)
                for axis in range(3):
                    mins[axis] = min(mins[axis], obj.location[axis] - extent[axis] / 2)
                    maxs[axis] = max(maxs[axis], obj.location[axis] + extent[axis] / 2)
            bbox_min = tuple(mins)
            bbox_max = tuple(maxs)
        else:
            centroid = None
            bbox_min = None
            bbox_max = None

        groups[key] = SceneGroup(
            key=key,
            component_count=len(objects),
            centroid=centroid,
            bbox_min=bbox_min,  # type: ignore[arg-type]
            bbox_max=bbox_max,  # type: ignore[arg-type]
            object_names=[obj.name for obj in objects],
        )
    return groups


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Judge API request failed: {exc.code} {body}") from exc

    return data["choices"][0]["message"]["content"]
