#!/usr/bin/env python3
"""LLM-as-judge scaffold for code-level BVB evaluation.

The input JSONL should contain one record per scene:
  - id or scene_id
  - gt_bpy_path: path to the human/GT exported bpy script
  - pred_bpy_path or generated_bpy_path: path to the agent exported bpy script

The script calls an OpenAI-compatible Chat Completions API and writes JSONL
judgments. Configure the model with:
  - OPENAI_API_KEY
  - BVB_JUDGE_MODEL, or --model
  - OPENAI_BASE_URL, optional, default https://api.openai.com/v1
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = """You are a strict evaluator for Blender Python scene reconstruction for BVB.
Compare a ground-truth bpy scene summary and an agent-generated bpy scene summary.
Judge semantic scene similarity, not textual or naming similarity.

Prioritize QA-relevant scene facts:
- major object categories and counts
- room layout, walls, doors, windows, and connectivity
- spatial relationships, relative positions, scale, and orientation
- visible furniture and object groups
- material/color similarity when it affects recognition
- lights/camera only when they affect visibility or viewpoint

Return only valid JSON with:
{
  "semantic_score": number from 0 to 1,
  "major_differences": [short strings],
  "missing_objects": [short strings],
  "extra_objects": [short strings],
  "spatial_errors": [short strings],
  "execution_risk": "low" | "medium" | "high",
  "rationale": "short explanation"
}
"""

QA_SYSTEM_PROMPT = """You answer visual-spatial questions from a Blender Python scene reconstruction.
You are given the agent-generated scene as structured bpy-derived data and a list
of questions. Answer only from the reconstructed scene, not from prior knowledge.

Return only valid JSON with:
{
  "answers": [
    {
      "id": string or number matching the input question id,
      "answer": "short answer"
    }
  ]
}

Use concise answers. For counting questions, answer with a digit such as "3".
For numeric size/area/distance questions, answer with only the number and no unit.
For multiple-choice questions with options, answer with only the option letter
such as "A", "B", "C", or "D".
"""


CATEGORY_ALIASES = {
    "couch": "sofa",
    "loveseat": "sofa",
    "settee": "sofa",
    "desk": "table",
    "nightstand": "table",
    "sideboard": "cabinet",
    "wardrobe": "cabinet",
    "bookshelf": "shelf",
    "bookcase": "shelf",
    "carpet": "rug",
    "tv": "television",
}

CATEGORY_WORDS = {
    "armchair",
    "baseboard",
    "bed",
    "bench",
    "cabinet",
    "ceiling",
    "chair",
    "counter",
    "curtain",
    "desk",
    "door",
    "dresser",
    "floor",
    "frame",
    "lamp",
    "light",
    "mirror",
    "monitor",
    "ottoman",
    "picture",
    "plant",
    "rug",
    "shelf",
    "sofa",
    "stool",
    "table",
    "television",
    "tv",
    "wall",
    "window",
}

PART_WORDS = {
    "arm",
    "back",
    "base",
    "body",
    "bottom",
    "cushion",
    "front",
    "glass",
    "handle",
    "headboard",
    "leg",
    "left",
    "l",
    "panel",
    "right",
    "r",
    "seat",
    "shade",
    "shelf",
    "side",
    "slat",
    "top",
}


@dataclass
class SceneObject:
    name: str
    kind: str
    category: str
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
    category: str
    component_count: int
    centroid: tuple[float, float, float] | None
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    materials: list[str] = field(default_factory=list)
    material_colors: list[tuple[float, float, float]] = field(default_factory=list)
    object_names: list[str] = field(default_factory=list)


@dataclass
class SceneSummary:
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'")
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return number_words.get(text, text)


def extract_first_number(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def option_letter(value: Any) -> str | None:
    text = normalize_answer(value).upper()
    match = re.match(r"^([A-Z])(?:\b|[.)])", text)
    if match:
        return match.group(1)
    if len(text) == 1 and "A" <= text <= "Z":
        return text
    return None


def option_text_by_letter(options: Any) -> dict[str, str]:
    mapping = {}
    if not isinstance(options, list):
        return mapping
    for option in options:
        text = str(option).strip()
        match = re.match(r"^([A-Za-z])\s*[.)]\s*(.*)$", text)
        if match:
            mapping[match.group(1).upper()] = normalize_answer(match.group(2))
    return mapping


def numeric_tolerance(question_type: str | None, ground_truth: float) -> float | None:
    if question_type == "object_size_estimation":
        return max(10.0, 0.10 * abs(ground_truth))
    if question_type == "room_size_estimation":
        return max(1.0, 0.10 * abs(ground_truth))
    if question_type == "object_abs_distance":
        return max(0.2, 0.15 * abs(ground_truth))
    return None


def is_correct_answer(answer: Any, ground_truth: Any, question_type: str | None = None, options: Any = None) -> bool:
    gt_letter = option_letter(ground_truth)
    option_texts = option_text_by_letter(options)
    if gt_letter and option_texts:
        pred_letter = option_letter(answer)
        if pred_letter:
            return pred_letter == gt_letter
        return normalize_answer(answer) == option_texts.get(gt_letter)

    gt_num = extract_first_number(ground_truth)
    pred_num = extract_first_number(answer)
    tolerance = numeric_tolerance(question_type, gt_num) if gt_num is not None else None
    if gt_num is not None and pred_num is not None and tolerance is not None:
        return abs(pred_num - gt_num) <= tolerance

    if question_type == "object_counting" and gt_num is not None and pred_num is not None:
        return int(round(pred_num)) == int(round(gt_num))

    return normalize_answer(answer) == normalize_answer(ground_truth)


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


def split_name_tokens(name: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return [token for token in re.split(r"[^A-Za-z0-9]+", text.lower()) if token]


def normalize_category(token: str) -> str:
    return CATEGORY_ALIASES.get(token, token)


def category_and_group(name: str, kind: str) -> tuple[str, str]:
    if kind in {"camera", "light"}:
        return kind, kind

    tokens = split_name_tokens(name)
    category = "object"
    category_index = 0
    for index, token in enumerate(tokens):
        normalized = normalize_category(token)
        if normalized in CATEGORY_WORDS:
            category = normalized
            category_index = index
            break

    # Multi-part objects usually look like Chair_1_Back or Sofa_Seat.
    instance = None
    if category_index + 1 < len(tokens) and tokens[category_index + 1].isdigit():
        instance = tokens[category_index + 1]
    if instance is not None:
        group_key = f"{category}_{instance}"
    elif category != "object":
        useful_tokens = [
            token
            for token in tokens[category_index + 1 :]
            if token not in PART_WORDS and not token.isdigit()
        ]
        suffix = f"_{'_'.join(useful_tokens[:1])}" if useful_tokens else ""
        group_key = f"{category}{suffix}"
    else:
        group_key = "_".join(tokens[:2]) if tokens else "object"

    return category, group_key


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


def parse_bpy_scene(path: Path) -> SceneSummary:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return SceneSummary(
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
                category, group_key = category_and_group(value, current.kind)
                current.name = value
                current.category = category
                current.group_key = group_key
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
                category, group_key = category_and_group(name, kind)
                current = SceneObject(
                    name=name,
                    kind=kind,
                    category=category,
                    group_key=group_key,
                    location=location[:3] if location else None,  # type: ignore[index]
                    light_type=str(light_type) if light_type else None,
                )
                objects.append(current)
                continue

            if current is not None:
                material_name = append_material_name(stmt.value)
                if material_name:
                    current.material = material_name

    return SceneSummary(
        path=str(path),
        parse_ok=True,
        parse_error=None,
        objects=objects,
        materials=materials,
    )


def object_extent(obj: SceneObject) -> tuple[float, float, float]:
    if obj.scale is not None:
        return tuple(abs(value) for value in obj.scale)
    return (1.0, 1.0, 1.0)


def build_groups(summary: SceneSummary) -> dict[str, SceneGroup]:
    grouped: dict[str, list[SceneObject]] = {}
    for obj in summary.mesh_objects:
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

        materials = sorted({obj.material for obj in objects if obj.material})
        material_colors = [summary.materials[name] for name in materials if name in summary.materials]
        groups[key] = SceneGroup(
            key=key,
            category=objects[0].category,
            component_count=len(objects),
            centroid=centroid,
            bbox_min=bbox_min,  # type: ignore[arg-type]
            bbox_max=bbox_max,  # type: ignore[arg-type]
            materials=materials,  # type: ignore[arg-type]
            material_colors=material_colors,
            object_names=[obj.name for obj in objects],
        )
    return groups


def category_counts(groups: dict[str, SceneGroup]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups.values():
        counts[group.category] = counts.get(group.category, 0) + 1
    return counts


def scene_diagonal(groups: dict[str, SceneGroup]) -> float:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for group in groups.values():
        if group.bbox_min is None or group.bbox_max is None:
            continue
        for axis in range(3):
            mins[axis] = min(mins[axis], group.bbox_min[axis])
            maxs[axis] = max(maxs[axis], group.bbox_max[axis])
    if not all(math.isfinite(value) for value in mins + maxs):
        return 1.0
    return max(1.0, math.dist(tuple(mins), tuple(maxs)))


def count_similarity(gt_counts: dict[str, int], pred_counts: dict[str, int]) -> float:
    categories = set(gt_counts) | set(pred_counts)
    if not categories:
        return 1.0
    scores = []
    for category in categories:
        gt_count = gt_counts.get(category, 0)
        pred_count = pred_counts.get(category, 0)
        scores.append(safe_divide(min(gt_count, pred_count), max(gt_count, pred_count)))
    return sum(scores) / len(scores)


def color_similarity(
    gt_colors: list[tuple[float, float, float]], pred_colors: list[tuple[float, float, float]]
) -> float | None:
    if not gt_colors or not pred_colors:
        return None
    best_scores = []
    for gt_color in gt_colors:
        best = 0.0
        for pred_color in pred_colors:
            distance = math.dist(gt_color, pred_color)
            best = max(best, clamp01(1.0 - distance / math.sqrt(3)))
        best_scores.append(best)
    return sum(best_scores) / len(best_scores)


def bbox_size(group: SceneGroup) -> tuple[float, float, float] | None:
    if group.bbox_min is None or group.bbox_max is None:
        return None
    return tuple(max(0.0, group.bbox_max[i] - group.bbox_min[i]) for i in range(3))


def size_similarity(gt_group: SceneGroup, pred_group: SceneGroup) -> float | None:
    gt_size = bbox_size(gt_group)
    pred_size = bbox_size(pred_group)
    if gt_size is None or pred_size is None:
        return None
    axis_scores = []
    for gt_value, pred_value in zip(gt_size, pred_size):
        if gt_value == 0 and pred_value == 0:
            axis_scores.append(1.0)
        else:
            axis_scores.append(safe_divide(min(gt_value, pred_value), max(gt_value, pred_value)))
    return sum(axis_scores) / len(axis_scores)


def group_similarity(gt_group: SceneGroup, pred_group: SceneGroup, diagonal: float) -> dict[str, float]:
    if gt_group.centroid is None or pred_group.centroid is None:
        spatial = 0.0
    else:
        distance = math.dist(gt_group.centroid, pred_group.centroid)
        spatial = math.exp(-distance / max(diagonal * 0.25, 1e-6))

    size = size_similarity(gt_group, pred_group)
    color = color_similarity(gt_group.material_colors, pred_group.material_colors)
    component = safe_divide(
        min(gt_group.component_count, pred_group.component_count),
        max(gt_group.component_count, pred_group.component_count),
    )

    parts = [0.5 * spatial, 0.25 * (size if size is not None else 0.5), 0.15 * component]
    parts.append(0.10 * (color if color is not None else 0.5))
    return {
        "score": sum(parts),
        "spatial": spatial,
        "size": size if size is not None else 0.5,
        "color": color if color is not None else 0.5,
        "component": component,
    }


def match_groups(
    gt_groups: dict[str, SceneGroup], pred_groups: dict[str, SceneGroup]
) -> list[dict[str, Any]]:
    diagonal = scene_diagonal(gt_groups)
    candidates = []
    for gt_key, gt_group in gt_groups.items():
        for pred_key, pred_group in pred_groups.items():
            if gt_group.category != pred_group.category:
                continue
            metrics = group_similarity(gt_group, pred_group, diagonal)
            candidates.append((metrics["score"], gt_key, pred_key, metrics))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_gt: set[str] = set()
    used_pred: set[str] = set()
    matches = []
    for _, gt_key, pred_key, metrics in candidates:
        if gt_key in used_gt or pred_key in used_pred:
            continue
        used_gt.add(gt_key)
        used_pred.add(pred_key)
        matches.append(
            {
                "gt_key": gt_key,
                "pred_key": pred_key,
                "category": gt_groups[gt_key].category,
                "metrics": metrics,
            }
        )
    return matches


def compute_static_metrics(gt_summary: SceneSummary, pred_summary: SceneSummary) -> dict[str, Any]:
    gt_groups = build_groups(gt_summary)
    pred_groups = build_groups(pred_summary)
    gt_counts = category_counts(gt_groups)
    pred_counts = category_counts(pred_groups)
    matches = match_groups(gt_groups, pred_groups)

    matched_gt = {match["gt_key"] for match in matches}
    matched_pred = {match["pred_key"] for match in matches}
    missing_groups = [key for key in gt_groups if key not in matched_gt]
    extra_groups = [key for key in pred_groups if key not in matched_pred]

    category_score = count_similarity(gt_counts, pred_counts)
    object_match_score = safe_divide(sum(match["metrics"]["score"] for match in matches), len(gt_groups))
    spatial_score = safe_divide(sum(match["metrics"]["spatial"] for match in matches), len(matches))
    material_score = safe_divide(sum(match["metrics"]["color"] for match in matches), len(matches))
    camera_score = 1.0 if bool(gt_summary.cameras) == bool(pred_summary.cameras) else 0.5
    light_count_score = count_similarity({"light": len(gt_summary.lights)}, {"light": len(pred_summary.lights)})
    health_score = 1.0
    if not gt_summary.parse_ok or not pred_summary.parse_ok:
        health_score = 0.0
    elif not pred_summary.objects:
        health_score = 0.0

    static_score = (
        0.25 * category_score
        + 0.35 * object_match_score
        + 0.20 * spatial_score
        + 0.10 * material_score
        + 0.05 * light_count_score
        + 0.03 * camera_score
        + 0.02 * health_score
    )

    return {
        "static_score": clamp01(static_score),
        "category_count_score": category_score,
        "object_match_score": object_match_score,
        "spatial_score": spatial_score,
        "material_score": material_score,
        "light_count_score": light_count_score,
        "camera_score": camera_score,
        "health_score": health_score,
        "gt_counts": gt_counts,
        "pred_counts": pred_counts,
        "num_gt_groups": len(gt_groups),
        "num_pred_groups": len(pred_groups),
        "num_matched_groups": len(matches),
        "missing_groups": missing_groups,
        "extra_groups": extra_groups,
        "matches": matches[:50],
    }


def compact_scene_summary(summary: SceneSummary) -> dict[str, Any]:
    groups = build_groups(summary)
    return {
        "path": summary.path,
        "parse_ok": summary.parse_ok,
        "parse_error": summary.parse_error,
        "num_objects": len(summary.objects),
        "num_mesh_objects": len(summary.mesh_objects),
        "num_lights": len(summary.lights),
        "num_cameras": len(summary.cameras),
        "category_counts": category_counts(groups),
        "groups": [
            {
                "key": group.key,
                "category": group.category,
                "component_count": group.component_count,
                "centroid": group.centroid,
                "bbox_min": group.bbox_min,
                "bbox_max": group.bbox_max,
                "materials": group.materials[:5],
                "object_names": group.object_names[:8],
            }
            for group in sorted(groups.values(), key=lambda item: (item.category, item.key))
        ],
        "lights": [
            {
                "name": light.name,
                "type": light.light_type,
                "location": light.location,
                "energy": light.light_energy,
            }
            for light in summary.lights
        ],
        "cameras": [
            {
                "name": camera.name,
                "location": camera.location,
                "rotation": camera.rotation,
            }
            for camera in summary.cameras
        ],
    }


def load_qa_by_scene(path: Path) -> dict[str, list[dict[str, Any]]]:
    qa_by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in read_jsonl(path):
        scene_name = str(get_first(record, ("scene_name", "scene_id")))
        qa_by_scene.setdefault(scene_name, []).append(
            {
                "id": get_first(record, ("id", "qa_id")),
                "question_type": record.get("question_type"),
                "question": get_first(record, ("question",)),
                "ground_truth": get_first(record, ("ground_truth", "answer")),
                "options": record.get("options"),
            }
        )
    return qa_by_scene


def build_qa_prompt(scene_id: str, pred_summary: dict[str, Any], qa_items: list[dict[str, Any]]) -> str:
    questions = [
        {
            "id": item["id"],
            "question_type": item.get("question_type"),
            "question": item["question"],
            "options": item.get("options"),
        }
        for item in qa_items
    ]
    return f"""Scene ID: {scene_id}

Agent-generated structured scene summary:
```json
{json.dumps(pred_summary, indent=2, ensure_ascii=False)}
```

Questions to answer from this reconstructed scene:
```json
{json.dumps(questions, indent=2, ensure_ascii=False)}
```

Answer-format rules:
- Counting questions: return only a digit, e.g. "3".
- Size/area/distance questions: return only the numeric value, without units.
- Multiple-choice questions with options: return only the option letter, e.g. "B".

Return one answer for every input question id."""


def score_qa_answers(qa_items: list[dict[str, Any]], response: dict[str, Any]) -> dict[str, Any]:
    answers_by_id = {}
    for item in response.get("answers", []):
        if isinstance(item, dict) and "id" in item:
            answers_by_id[str(item["id"])] = item.get("answer", "")

    rows = []
    correct_count = 0
    for item in qa_items:
        qa_id = str(item["id"])
        predicted = answers_by_id.get(qa_id, "")
        correct = is_correct_answer(
            predicted,
            item["ground_truth"],
            str(item.get("question_type")) if item.get("question_type") is not None else None,
            item.get("options"),
        )
        correct_count += int(correct)
        rows.append(
            {
                "id": item["id"],
                "question_type": item.get("question_type"),
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "predicted_answer": predicted,
                "correct": correct,
            }
        )

    return {
        "num_questions": len(qa_items),
        "num_answered": sum(1 for row in rows if str(row["predicted_answer"]).strip()),
        "num_correct": correct_count,
        "code_qa_score": safe_divide(correct_count, len(qa_items)),
        "rows": rows,
    }


def read_text_limited(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n# ... middle truncated for judge context ...\n\n" + text[-half:]


def build_user_prompt(
    scene_id: str,
    gt_summary: dict[str, Any],
    pred_summary: dict[str, Any],
    static_metrics: dict[str, Any] | None,
    gt_code: str | None = None,
    pred_code: str | None = None,
) -> str:
    prompt = f"""Scene ID: {scene_id}

Ground-truth structured scene summary:
```json
{json.dumps(gt_summary, indent=2, ensure_ascii=False)}
```

Agent-generated structured scene summary:
```json
{json.dumps(pred_summary, indent=2, ensure_ascii=False)}
```"""

    if static_metrics is not None:
        prompt += f"""
Deterministic static metrics:
```json
{json.dumps(static_metrics, indent=2, ensure_ascii=False)}
```"""

    prompt += """
Evaluate the agent-generated scene against the ground truth.
Return semantic_score as the final score. Use a concise score from 0 to 1,
for example 0.62 rather than 0.6200. Judge semantic scene similarity from
the structured summaries, not textual similarity or exact object naming."""

    if gt_code is not None and pred_code is not None:
        prompt += f"""

Ground-truth bpy script:
```python
{gt_code}
```

Agent-generated bpy script:
```python
{pred_code}
```
"""
    return prompt


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


def resolve_record_path(raw_path: Any, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    based = base_dir / path
    if based.exists():
        return based
    return path


def summarize_scores(rows: list[dict[str, Any]], *, include_static: bool) -> dict[str, Any]:
    static_scores = [
        row["static_metrics"]["static_score"]
        for row in rows
        if isinstance(row.get("static_metrics"), dict)
    ]
    semantic_scores = []
    for row in rows:
        score = row.get("code_semantic_score")
        if not isinstance(score, (int, float)):
            judgment = row.get("semantic_judgment") or row.get("judgment")
            if isinstance(judgment, dict):
                score = judgment.get("semantic_score")
        if isinstance(score, (int, float)):
            semantic_scores.append(score)
    qa_scores = [
        row.get("code_qa_score")
        for row in rows
        if isinstance(row.get("code_qa_score"), (int, float))
    ]
    summary = {"num_scenes": len(rows)}
    if semantic_scores:
        summary["mean_code_semantic_score"] = safe_divide(sum(semantic_scores), len(semantic_scores))
    if qa_scores:
        summary["mean_code_qa_score"] = safe_divide(sum(qa_scores), len(qa_scores))
        summary["num_scenes_with_qa"] = len(qa_scores)
        summary["num_qa_questions"] = sum(
            row.get("qa_judgment", {}).get("num_questions", 0)
            for row in rows
            if isinstance(row.get("qa_judgment"), dict)
        )
        summary["num_qa_correct"] = sum(
            row.get("qa_judgment", {}).get("num_correct", 0)
            for row in rows
            if isinstance(row.get("qa_judgment"), dict)
        )
    if include_static:
        combined_scores = [row.get("combined_score") for row in rows if row.get("combined_score") is not None]
        summary["mean_combined_score"] = safe_divide(sum(combined_scores), len(combined_scores))
        summary["mean_static_score"] = safe_divide(sum(static_scores), len(static_scores))
    return summary


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for record in read_jsonl(path):
        if "id" in record:
            completed.add(str(record["id"]))
    return completed


def write_jsonl_row(out: Any, row: dict[str, Any]) -> None:
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    os.fsync(out.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code-level BVB metrics for exported bpy scripts.")
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL file with gt/pred bpy paths.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL judgments.")
    parser.add_argument("--summary-output", type=Path, help="Optional aggregate JSON output.")
    parser.add_argument("--model", default=os.getenv("BVB_JUDGE_MODEL"), help="Judge model name.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max-chars", type=int, default=20000, help="Max chars per script pair when sending code.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional judge temperature. Omit to use the model default.",
    )
    parser.add_argument("--static-only", action="store_true", help="Compute deterministic static metrics only.")
    parser.add_argument(
        "--metric",
        choices=("semantic", "qa", "both"),
        default="semantic",
        help="Which code-level metric to compute.",
    )
    parser.add_argument(
        "--qa-metadata",
        type=Path,
        default=Path(__file__).resolve().parent / "test.jsonl",
        help="QA metadata JSONL used for --metric qa or both.",
    )
    parser.add_argument("--resume", action="store_true", help="Append to output and skip already written ids.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on the first judge API error.")
    parser.add_argument(
        "--include-code",
        action="store_true",
        help="Also include truncated raw bpy scripts in the LLM prompt.",
    )
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.7,
        help="Weight of LLM semantic_score in combined_score.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling the API.")
    args = parser.parse_args()

    if not args.dry_run and not args.static_only:
        if not args.api_key:
            raise SystemExit("Missing OPENAI_API_KEY or --api-key.")
        if not args.model:
            raise SystemExit("Missing BVB_JUDGE_MODEL or --model.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    completed_ids = load_completed_ids(args.output) if args.resume else set()
    records = list(read_jsonl(args.pairs))
    mode = "a" if args.resume else "w"
    qa_by_scene = load_qa_by_scene(args.qa_metadata) if args.metric in {"qa", "both"} else {}

    if completed_ids:
        print(f"Resuming from {args.output}; skipping {len(completed_ids)} completed ids.", flush=True)

    with args.output.open(mode, encoding="utf-8") as out:
        for index, record in enumerate(records, start=1):
            scene_id = str(get_first(record, ("id", "scene_id")))
            if scene_id in completed_ids:
                continue

            print(f"[{index}/{len(records)}] Judging {scene_id}...", flush=True)
            gt_path = resolve_record_path(
                get_first(record, ("gt_bpy_path", "ground_truth_bpy_path")),
                args.pairs.parent,
            )
            pred_path = resolve_record_path(
                get_first(record, ("pred_bpy_path", "generated_bpy_path", "agent_bpy_path")),
                args.pairs.parent,
            )

            gt_summary = parse_bpy_scene(gt_path)
            pred_summary = parse_bpy_scene(pred_path)
            static_metrics = compute_static_metrics(gt_summary, pred_summary) if args.static_only else None

            compact_gt = compact_scene_summary(gt_summary)
            compact_pred = compact_scene_summary(pred_summary)
            gt_code = read_text_limited(gt_path, args.max_chars) if args.include_code else None
            pred_code = read_text_limited(pred_path, args.max_chars) if args.include_code else None
            row = {
                "id": scene_id,
                "gt_bpy_path": str(gt_path),
                "pred_bpy_path": str(pred_path),
            }
            combined_score = None if static_metrics is None else static_metrics["static_score"]
            if args.metric in {"semantic", "both"}:
                user_prompt = build_user_prompt(
                    scene_id,
                    compact_gt,
                    compact_pred,
                    static_metrics,
                    gt_code,
                    pred_code,
                )
                if args.dry_run or args.static_only:
                    semantic_judgment: dict[str, Any] = {
                        "dry_run": args.dry_run,
                        "static_only": args.static_only,
                        "prompt_chars": len(user_prompt),
                    }
                else:
                    try:
                        content = call_chat_completion(
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=user_prompt,
                            temperature=args.temperature,
                        )
                        semantic_judgment = json.loads(content)
                        semantic_score = semantic_judgment.get("semantic_score")
                        if isinstance(semantic_score, (int, float)):
                            normalized_score = clamp01(float(semantic_score))
                            row["code_semantic_score"] = normalized_score
                            if static_metrics is not None:
                                llm_weight = clamp01(args.llm_weight)
                                combined_score = clamp01(
                                    llm_weight * normalized_score
                                    + (1.0 - llm_weight) * static_metrics["static_score"]
                                )
                    except KeyboardInterrupt:
                        print(f"Interrupted while judging {scene_id}; no result was returned for this scene.", file=sys.stderr)
                        raise
                    except Exception as exc:
                        if args.stop_on_error:
                            raise
                        semantic_judgment = {
                            "error": str(exc),
                            "execution_risk": "high",
                            "semantic_score": None,
                        }
                row["semantic_judgment"] = semantic_judgment

            if args.metric in {"qa", "both"}:
                qa_items = qa_by_scene.get(scene_id, [])
                if not qa_items:
                    row["qa_judgment"] = {
                        "num_questions": 0,
                        "num_answered": 0,
                        "num_correct": 0,
                        "code_qa_score": None,
                        "rows": [],
                    }
                else:
                    qa_prompt = build_qa_prompt(scene_id, compact_pred, qa_items)
                    if args.dry_run or args.static_only:
                        qa_judgment = {
                            "dry_run": args.dry_run,
                            "prompt_chars": len(qa_prompt),
                            "num_questions": len(qa_items),
                        }
                    else:
                        try:
                            content = call_chat_completion(
                                base_url=args.base_url,
                                api_key=args.api_key,
                                model=args.model,
                                system_prompt=QA_SYSTEM_PROMPT,
                                user_prompt=qa_prompt,
                                temperature=args.temperature,
                            )
                            qa_response = json.loads(content)
                            qa_judgment = score_qa_answers(qa_items, qa_response)
                            row["code_qa_score"] = qa_judgment["code_qa_score"]
                        except KeyboardInterrupt:
                            print(f"Interrupted while QA judging {scene_id}; no QA result was returned for this scene.", file=sys.stderr)
                            raise
                        except Exception as exc:
                            if args.stop_on_error:
                                raise
                            qa_judgment = {
                                "error": str(exc),
                                "num_questions": len(qa_items),
                                "num_answered": 0,
                                "num_correct": 0,
                                "code_qa_score": None,
                                "rows": [],
                            }
                    row["qa_judgment"] = qa_judgment

            if static_metrics is not None:
                row["static_metrics"] = static_metrics
                row["combined_score"] = combined_score
            rows.append(row)
            write_jsonl_row(out, row)
            status_parts = []
            if isinstance(row.get("code_semantic_score"), (int, float)):
                status_parts.append(f"code_semantic_score={float(row['code_semantic_score']):g}")
            if isinstance(row.get("code_qa_score"), (int, float)):
                qa_judgment = row.get("qa_judgment", {})
                if isinstance(qa_judgment, dict):
                    correct = qa_judgment.get("num_correct", "?")
                    total = qa_judgment.get("num_questions", "?")
                    status_parts.append(f"code_qa_score={float(row['code_qa_score']):g} ({correct}/{total})")
                else:
                    status_parts.append(f"code_qa_score={float(row['code_qa_score']):g}")
            if not status_parts:
                status_parts.append("dry_run" if args.dry_run else "error")
            print(f"[{index}/{len(records)}] Wrote {scene_id}: {', '.join(status_parts)}", flush=True)

    summary_rows = list(read_jsonl(args.output))
    summary = summarize_scores(summary_rows, include_static=args.static_only)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
