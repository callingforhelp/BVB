"""Pure geometry helpers for deterministic BVB unit-test evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from eval_utils import SceneGroup


EPSILON = 1e-9

SEMANTIC_ALIASES = {
    "bathtub": ("bathtub", "bath tub", "tub"),
    "bed": ("bed", "mattress", "bed frame"),
    "bookshelf": ("bookshelf", "bookcase"),
    "cabinet": ("cabinet", "cupboard"),
    "chair": ("chair", "armchair", "task chair", "folding seat"),
    "computer mouse": ("computer mouse", "mouse"),
    "computer tower": ("computer tower", "pc tower", "desktop tower"),
    "couch": ("couch", "sofa", "sectional"),
    "cutting board": ("cutting board", "chopping board"),
    "doorframe": ("doorframe", "door frame"),
    "floor or room boundary": ("floor", "room boundary", "carpet floor", "ground"),
    "heater": ("heater", "radiator"),
    "monitor": ("monitor", "computer display", "monitor screen"),
    "nightstand": ("nightstand", "bedside table"),
    "pan": ("pan", "skillet", "frying pan"),
    "refrigerator": ("refrigerator", "fridge"),
    "shoe rack": ("shoe rack", "shoe shelf"),
    "sink": ("sink", "basin"),
    "sofa": ("sofa", "couch", "sectional"),
    "stove": ("stove", "cooktop", "range"),
    "television": ("television", "tv", "tv screen"),
    "toilet": ("toilet", "commode"),
    "trash can": ("trash can", "trash bin", "waste bin", "garbage can"),
    "wardrobe": ("wardrobe", "closet"),
}

SEMANTIC_STOP_WORDS = {
    "a",
    "about",
    "at",
    "beside",
    "by",
    "corner",
    "first",
    "in",
    "is",
    "left",
    "near",
    "of",
    "on",
    "other",
    "passing",
    "past",
    "right",
    "target",
    "the",
    "to",
    "until",
    "with",
    "your",
}

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "colorful",
    "dark",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}


def lookup_group(groups: dict[str, SceneGroup], key: Any) -> SceneGroup | None:
    if key is None:
        return None
    key_text = str(key)
    if key_text in groups:
        return groups[key_text]
    lowered = key_text.lower()
    for group in groups.values():
        if group.key.lower() == lowered:
            return group
        if any(name.lower() == lowered for name in group.object_names):
            return group
    return None


def matching_groups(groups: dict[str, SceneGroup], key: Any) -> list[SceneGroup]:
    if key is None:
        return []
    key_text = str(key)
    if key_text in groups:
        return [groups[key_text]]
    lowered = key_text.lower()
    return [
        group
        for group in groups.values()
        if group.key.lower() == lowered
        or any(name.lower() == lowered for name in group.object_names)
    ]


def _semantic_tokens(value: Any) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    text = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", text)
    text = text.lower().replace("_", " ")
    tokens = {
        token[:-1] if token.endswith("s") and len(token) > 3 else token
        for token in re.findall(r"[a-z0-9]+", text)
        if not token.isdigit()
    }
    return tokens - SEMANTIC_STOP_WORDS - COLOR_WORDS


def semantic_match(semantic_ref: Any, group: SceneGroup) -> bool:
    requested_text = str(semantic_ref).lower().replace("_", " ").strip()
    requested_tokens = _semantic_tokens(requested_text)
    candidate_tokens = _semantic_tokens(" ".join([group.key, *group.object_names]))
    if requested_tokens and requested_tokens.issubset(candidate_tokens):
        return True

    aliases = set()
    for canonical, values in SEMANTIC_ALIASES.items():
        canonical_tokens = _semantic_tokens(canonical)
        if canonical == requested_text or (
            canonical_tokens and canonical_tokens.issubset(requested_tokens)
        ):
            aliases.update(values)
    if not aliases:
        aliases.add(requested_text)
    return any(
        bool(alias_tokens := _semantic_tokens(alias))
        and alias_tokens.issubset(candidate_tokens)
        for alias in aliases
    )


def grounded_union(
    groups: dict[str, SceneGroup],
    grounded_keys: Any,
    semantic_ref: Any,
    *,
    allow_auto_match: bool = True,
) -> tuple[SceneGroup | None, dict[str, Any]]:
    raw_keys = grounded_keys if isinstance(grounded_keys, list) else [grounded_keys]
    accepted = []
    rejected = []
    for key in raw_keys:
        if key is None or str(key) not in groups:
            rejected.append({"key": key, "reason": "not_exact_group_key"})
            continue
        group = groups[str(key)]
        if not semantic_match(semantic_ref, group):
            rejected.append({"key": key, "reason": "semantic_mismatch"})
            continue
        accepted.append(str(key))
    auto_matched = False
    auto_match_reason = None
    if not accepted and allow_auto_match:
        candidates = [
            group for group in groups.values() if semantic_match(semantic_ref, group)
        ]
        if len(candidates) == 1:
            accepted = [candidates[0].key]
            auto_matched = True
            auto_match_reason = "unique_semantic_candidate"
    return union_groups(groups, accepted), {
        "semantic_ref": semantic_ref,
        "accepted_group_keys": accepted,
        "auto_matched": auto_matched,
        "auto_match_reason": auto_match_reason,
        "rejected_groundings": rejected,
    }


def union_groups(groups: dict[str, SceneGroup], keys: Any) -> SceneGroup | None:
    raw_keys = keys if isinstance(keys, list) else [keys]
    selected: list[SceneGroup] = []
    seen: set[str] = set()
    for key in raw_keys:
        for group in matching_groups(groups, key):
            if group.key not in seen:
                selected.append(group)
                seen.add(group.key)
    if not selected:
        return None

    bbox_groups = [group for group in selected if group.bbox_min is not None and group.bbox_max is not None]
    bbox_min = (
        tuple(min(group.bbox_min[axis] for group in bbox_groups) for axis in range(3))
        if bbox_groups
        else None
    )
    bbox_max = (
        tuple(max(group.bbox_max[axis] for group in bbox_groups) for axis in range(3))
        if bbox_groups
        else None
    )
    centroid = (
        tuple((bbox_min[axis] + bbox_max[axis]) / 2.0 for axis in range(3))
        if bbox_min is not None and bbox_max is not None
        else None
    )
    footprint_values = [
        group.footprint_area
        for group in selected
        if group.footprint_area is not None
    ]
    footprint_area = None
    if footprint_values:
        footprint_area = sum(footprint_values)
        rectangles = [
            (
                group.bbox_min[0],
                group.bbox_min[1],
                group.bbox_max[0],
                group.bbox_max[1],
            )
            for group in selected
            if group.bbox_min is not None
            and group.bbox_max is not None
            and group.footprint_area is not None
        ]
        if len(rectangles) == len(footprint_values):
            footprint_area = min(footprint_area, rectangle_union_area(rectangles))
    return SceneGroup(
        key="+".join(group.key for group in selected),
        component_count=sum(group.component_count for group in selected),
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        object_names=[name for group in selected for name in group.object_names],
        footprint_area=footprint_area,
        dimensions=selected[0].dimensions if len(selected) == 1 else None,
    )


def rectangle_union_area(
    rectangles: list[tuple[float, float, float, float]],
) -> float:
    if not rectangles:
        return 0.0
    xs = sorted({value for rectangle in rectangles for value in (rectangle[0], rectangle[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (bottom, top)
            for x_min, bottom, x_max, top in rectangles
            if x_min < right and x_max > left and top > bottom
        )
        covered = 0.0
        current_start = current_end = None
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start > current_end:
                covered += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        if current_start is not None:
            covered += current_end - current_start
        area += (right - left) * covered
    return area


def bbox_size(group: SceneGroup | None) -> tuple[float, float, float] | None:
    if group is None or group.bbox_min is None or group.bbox_max is None:
        return None
    return tuple(max(0.0, group.bbox_max[i] - group.bbox_min[i]) for i in range(3))


def infer_floor_group(groups: dict[str, SceneGroup]) -> SceneGroup | None:
    candidates = []
    for group in groups.values():
        size = bbox_size(group)
        if size is None:
            continue
        horizontal_extent = max(size[0], size[1])
        area = size[0] * size[1]
        if area < 1.0 or size[2] > max(0.25, horizontal_extent * 0.05):
            continue
        name_tokens = _semantic_tokens(" ".join([group.key, *group.object_names]))
        if not name_tokens.intersection({"floor", "ground", "plane", "carpet"}):
            continue
        candidates.append((area, group))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].key))
    return candidates[0][1]


def bbox_distance(group_a: SceneGroup | None, group_b: SceneGroup | None) -> float | None:
    if (
        group_a is None
        or group_b is None
        or group_a.bbox_min is None
        or group_a.bbox_max is None
        or group_b.bbox_min is None
        or group_b.bbox_max is None
    ):
        return None
    squared = 0.0
    for axis in range(3):
        if group_a.bbox_max[axis] < group_b.bbox_min[axis]:
            delta = group_b.bbox_min[axis] - group_a.bbox_max[axis]
        elif group_b.bbox_max[axis] < group_a.bbox_min[axis]:
            delta = group_a.bbox_min[axis] - group_b.bbox_max[axis]
        else:
            delta = 0.0
        squared += delta * delta
    return math.sqrt(squared)


def point_bbox_distance(point: tuple[float, float, float], group: SceneGroup) -> float | None:
    if group.bbox_min is None or group.bbox_max is None:
        return None
    squared = 0.0
    for axis in range(3):
        if point[axis] < group.bbox_min[axis]:
            delta = group.bbox_min[axis] - point[axis]
        elif point[axis] > group.bbox_max[axis]:
            delta = point[axis] - group.bbox_max[axis]
        else:
            delta = 0.0
        squared += delta * delta
    return math.sqrt(squared)


def _xy(group: SceneGroup | None) -> tuple[float, float] | None:
    if group is None or group.centroid is None:
        return None
    return float(group.centroid[0]), float(group.centroid[1])


def _normalize(vector: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(*vector)
    if length <= EPSILON:
        return None
    return vector[0] / length, vector[1] / length


def egocentric_frame(
    anchor: SceneGroup | None, facing: SceneGroup | None
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    origin = _xy(anchor)
    target = _xy(facing)
    if origin is None or target is None:
        return None
    forward = _normalize((target[0] - origin[0], target[1] - origin[1]))
    if forward is None:
        return None
    # Blender is Z-up. Rotating forward clockwise around +Z yields screen/right.
    right = (forward[1], -forward[0])
    return origin, forward, right


def relative_direction(
    *,
    anchor: SceneGroup | None,
    facing: SceneGroup | None,
    query: SceneGroup | None,
    difficulty: str,
    reference: SceneGroup | None = None,
) -> str | None:
    frame = egocentric_frame(anchor, facing)
    query_xy = _xy(query)
    if frame is None or query_xy is None:
        return None
    origin, forward, right = frame

    def components(point: tuple[float, float]) -> tuple[float, float]:
        vector = (point[0] - origin[0], point[1] - origin[1])
        return (
            vector[0] * forward[0] + vector[1] * forward[1],
            vector[0] * right[0] + vector[1] * right[1],
        )

    forward_offset, right_offset = components(query_xy)
    if difficulty == "easy":
        reference_xy = _xy(reference or facing)
        if reference_xy is None:
            return None
        _, reference_right = components(reference_xy)
        return "right" if right_offset - reference_right >= 0.0 else "left"
    if difficulty == "medium":
        vector = _normalize((query_xy[0] - origin[0], query_xy[1] - origin[1]))
        if vector is None:
            return None
        cosine = vector[0] * forward[0] + vector[1] * forward[1]
        if cosine <= math.cos(math.radians(135.0)):
            return "back"
        return "right" if right_offset >= 0.0 else "left"
    if difficulty == "hard":
        depth = "front" if forward_offset >= 0.0 else "back"
        side = "right" if right_offset >= 0.0 else "left"
        return f"{depth}-{side}"
    return None


def closest_candidate(
    anchor: SceneGroup | None, candidates: dict[str, SceneGroup | None], *, tie_tolerance: float = 1e-6
) -> tuple[str | None, dict[str, float | None]]:
    distances = {label: bbox_distance(anchor, group) for label, group in candidates.items()}
    valid = [(label, value) for label, value in distances.items() if value is not None]
    if not valid:
        return None, distances
    valid.sort(key=lambda item: (item[1], item[0]))
    if len(valid) > 1 and abs(valid[0][1] - valid[1][1]) <= tie_tolerance:
        return None, distances
    return valid[0][0], distances


def parse_turns(text: str) -> list[str]:
    turns = []
    for match in __import__("re").finditer(r"turn\s+(back|left|right)", text.lower()):
        turns.append(f"turn_{match.group(1)}")
    return turns


def required_route_turns(
    *,
    start: SceneGroup | None,
    facing: SceneGroup | None,
    waypoints: Iterable[SceneGroup | None],
    straight_threshold_degrees: float = 45.0,
    back_threshold_degrees: float = 135.0,
) -> list[str] | None:
    frame = egocentric_frame(start, facing)
    current = _xy(start)
    if frame is None or current is None:
        return None
    _, heading, _ = frame
    output: list[str] = []
    for waypoint in waypoints:
        target = _xy(waypoint)
        if target is None:
            return None
        desired = _normalize((target[0] - current[0], target[1] - current[1]))
        if desired is None:
            return None
        dot = max(-1.0, min(1.0, heading[0] * desired[0] + heading[1] * desired[1]))
        angle = math.degrees(math.acos(dot))
        cross = heading[0] * desired[1] - heading[1] * desired[0]
        if angle >= back_threshold_degrees:
            output.append("turn_back")
        elif angle > straight_threshold_degrees:
            output.append("turn_left" if cross > 0.0 else "turn_right")
        heading = desired
        current = target
    return output


def _turn_between(
    heading: tuple[float, float],
    desired: tuple[float, float],
    *,
    straight_threshold_degrees: float = 45.0,
    back_threshold_degrees: float = 135.0,
) -> str | None:
    dot = max(-1.0, min(1.0, heading[0] * desired[0] + heading[1] * desired[1]))
    angle = math.degrees(math.acos(dot))
    cross = heading[0] * desired[1] - heading[1] * desired[0]
    if angle >= back_threshold_degrees:
        return "turn_back"
    if angle > straight_threshold_degrees:
        return "turn_left" if cross > 0.0 else "turn_right"
    return None


def route_turns_from_camera(
    *,
    start: SceneGroup | None,
    facing: SceneGroup | None,
    route_steps: list[dict[str, Any]],
    temporal: dict[str, Any] | None,
) -> tuple[list[str] | None, dict[str, Any]]:
    if not temporal or not temporal.get("camera_motion"):
        return None, {"reason": "static_camera"}
    frames = temporal.get("frames")
    if not isinstance(frames, list) or not frames:
        return None, {"reason": "missing_camera_trajectory"}

    start_frame = frames[0]
    start_position_raw = start_frame.get("camera_position")
    forward_raw = start_frame.get("camera_forward")
    if not isinstance(start_position_raw, list) or len(start_position_raw) < 3:
        return None, {"reason": "missing_camera_position"}
    current_position = tuple(float(start_position_raw[i]) for i in range(3))
    heading = None
    if isinstance(forward_raw, list) and len(forward_raw) >= 2:
        heading = _normalize((float(forward_raw[0]), float(forward_raw[1])))
    if heading is None:
        frame = egocentric_frame(start, facing)
        heading = frame[1] if frame is not None else None
    if heading is None:
        return None, {"reason": "missing_initial_heading"}

    matched = []
    search_start = 0
    for step in route_steps:
        target = step.get("group")
        relation = str(step.get("relation") or "near").lower()
        if not isinstance(target, SceneGroup) or target.centroid is None:
            return None, {"reason": "missing_route_group", "matched_steps": matched}
        target_size = bbox_size(target)
        horizontal_extent = max(target_size[:2]) if target_size is not None else 0.0
        reach_threshold = max(1.5, min(4.0, horizontal_extent * 0.5 + 1.0))
        candidates = []
        for index in range(search_start, len(frames)):
            frame = frames[index]
            position_raw = frame.get("camera_position")
            matrix = frame.get("world_to_camera")
            if not isinstance(position_raw, list) or len(position_raw) < 3:
                continue
            if relation in {"left", "right"}:
                if not isinstance(matrix, list) or len(matrix) != 4:
                    continue
                local_x, _, local_z = _transform_point(matrix, target.centroid)
                if local_z >= 0.0:
                    continue
                if relation == "right" and local_x <= 0.0:
                    continue
                if relation == "left" and local_x >= 0.0:
                    continue
            position = tuple(float(position_raw[i]) for i in range(3))
            distance = point_bbox_distance(position, target)
            if distance is None or distance > reach_threshold:
                continue
            candidates.append((distance, index, position))
        if not candidates:
            return None, {
                "reason": f"route_landmark_not_reached:{relation}",
                "reach_threshold": reach_threshold,
                "matched_steps": matched,
            }
        distance, index, event_position = min(candidates, key=lambda item: (item[0], item[1]))
        matched.append(
            {
                "frame": int(frames[index].get("frame", index)),
                "relation": relation,
                "group": target.key,
                "distance": distance,
                "camera_position": list(event_position),
            }
        )
        search_start = index + 1
        if search_start >= len(frames) and len(matched) < len(route_steps):
            return None, {
                "reason": "route_step_after_trajectory_end",
                "matched_steps": matched,
            }

    turns = []
    for event in matched:
        event_position = tuple(float(value) for value in event["camera_position"])
        desired = _normalize(
            (
                event_position[0] - current_position[0],
                event_position[1] - current_position[1],
            )
        )
        if desired is None:
            continue
        turn = _turn_between(heading, desired)
        if turn is not None:
            turns.append(turn)
        heading = desired
        current_position = event_position
    return turns, {"matched_steps": matched}


def _bbox_corners(group: SceneGroup) -> list[tuple[float, float, float]]:
    if group.bbox_min is None or group.bbox_max is None:
        return []
    return [
        (x, y, z)
        for x in (group.bbox_min[0], group.bbox_max[0])
        for y in (group.bbox_min[1], group.bbox_max[1])
        for z in (group.bbox_min[2], group.bbox_max[2])
    ]


def _frame_group(group: SceneGroup, frame: dict[str, Any]) -> SceneGroup:
    animated = frame.get("animated_bboxes")
    if not isinstance(animated, dict):
        return group
    entries = [
        animated[name]
        for name in group.object_names
        if name in animated and isinstance(animated[name], dict)
    ]
    if not entries:
        return group
    mins = [entry.get("bbox_min") for entry in entries if entry.get("bbox_min") is not None]
    maxs = [entry.get("bbox_max") for entry in entries if entry.get("bbox_max") is not None]
    if len(entries) < group.component_count and group.bbox_min is not None and group.bbox_max is not None:
        mins.append(group.bbox_min)
        maxs.append(group.bbox_max)
    if not mins or not maxs:
        return group
    bbox_min = tuple(min(float(value[axis]) for value in mins) for axis in range(3))
    bbox_max = tuple(max(float(value[axis]) for value in maxs) for axis in range(3))
    centroid = tuple((bbox_min[axis] + bbox_max[axis]) / 2.0 for axis in range(3))
    return SceneGroup(
        key=group.key,
        component_count=group.component_count,
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        object_names=group.object_names,
        footprint_area=group.footprint_area,
        dimensions=group.dimensions,
    )


def _transform_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    vector = (point[0], point[1], point[2], 1.0)
    result = [sum(float(matrix[row][col]) * vector[col] for col in range(4)) for row in range(4)]
    return result[0], result[1], result[2]


def bbox_in_camera_frustum(group: SceneGroup, camera_frame: dict[str, Any], camera: dict[str, Any]) -> bool:
    matrix = camera_frame.get("world_to_camera")
    if not isinstance(matrix, list) or len(matrix) != 4:
        return False
    angle_x = float(camera.get("angle_x") or 0.0)
    angle_y = float(camera.get("angle_y") or 0.0)
    clip_start = float(camera.get("clip_start") or 0.01)
    clip_end = float(camera.get("clip_end") or 1e9)
    if angle_x <= 0.0 or angle_y <= 0.0:
        return False
    points = _bbox_corners(group)
    if group.centroid is not None:
        points.append(group.centroid)
    tan_x = math.tan(angle_x / 2.0)
    tan_y = math.tan(angle_y / 2.0)
    for point in points:
        x, y, z_local = _transform_point(matrix, point)
        depth = -z_local
        if clip_start <= depth <= clip_end and abs(x) <= depth * tan_x and abs(y) <= depth * tan_y:
            return True
    return False


def ray_aabb_distance(
    origin: tuple[float, float, float],
    target: tuple[float, float, float],
    group: SceneGroup,
) -> float | None:
    if group.bbox_min is None or group.bbox_max is None:
        return None
    direction = tuple(target[i] - origin[i] for i in range(3))
    length = math.sqrt(sum(value * value for value in direction))
    if length <= EPSILON:
        return None
    direction = tuple(value / length for value in direction)
    near, far = 0.0, length
    for axis in range(3):
        if abs(direction[axis]) <= EPSILON:
            if origin[axis] < group.bbox_min[axis] or origin[axis] > group.bbox_max[axis]:
                return None
            continue
        first = (group.bbox_min[axis] - origin[axis]) / direction[axis]
        second = (group.bbox_max[axis] - origin[axis]) / direction[axis]
        near = max(near, min(first, second))
        far = min(far, max(first, second))
        if near > far:
            return None
    return near if 0.0 <= near <= length else None


@dataclass
class AppearanceResult:
    order: list[str]
    first_visible_frames: dict[str, int | None]
    reason: str | None = None


def appearance_order(
    *,
    category_groups: dict[str, SceneGroup | None],
    all_groups: dict[str, SceneGroup],
    temporal: dict[str, Any] | None,
) -> AppearanceResult:
    if not temporal or (
        not temporal.get("camera_motion")
        and not temporal.get("animated_objects")
    ):
        return AppearanceResult([], {key: None for key in category_groups}, "static_camera")
    camera = temporal.get("camera")
    frames = temporal.get("frames")
    if not isinstance(camera, dict) or not isinstance(frames, list) or not frames:
        return AppearanceResult([], {key: None for key in category_groups}, "missing_camera_trajectory")

    first: dict[str, int | None] = {key: None for key in category_groups}
    category_rank = {category: index for index, category in enumerate(category_groups)}
    exact_visibility = temporal.get("first_visible_frames")
    if isinstance(exact_visibility, dict):
        for category, target in category_groups.items():
            if target is None:
                continue
            visible_frames = [
                int(exact_visibility[name])
                for name in target.object_names
                if name in exact_visibility and exact_visibility[name] is not None
            ]
            if visible_frames:
                first[category] = min(visible_frames)
        if any(value is None for value in first.values()):
            return AppearanceResult([], first, "category_never_visible")
        ordered = sorted(
            first,
            key=lambda category: (first[category], category_rank[category]),
        )
        return AppearanceResult(ordered, first)

    hidden = set(temporal.get("hidden_objects") or [])
    for frame in frames:
        position_raw = frame.get("camera_position")
        if not isinstance(position_raw, list) or len(position_raw) < 3:
            continue
        camera_position = tuple(float(position_raw[i]) for i in range(3))
        for category, target in category_groups.items():
            if first[category] is not None or target is None:
                continue
            if any(name in hidden for name in target.object_names):
                continue
            frame_target = _frame_group(target, frame)
            if not bbox_in_camera_frustum(frame_target, frame, camera):
                continue
            if frame_target.centroid is None:
                continue
            target_distance = math.dist(camera_position, frame_target.centroid)
            target_names = set(target.object_names)
            occluded = False
            for group in all_groups.values():
                if target_names.intersection(group.object_names):
                    continue
                frame_occluder = _frame_group(group, frame)
                distance = ray_aabb_distance(
                    camera_position,
                    frame_target.centroid,
                    frame_occluder,
                )
                if distance is not None and distance + 1e-4 < target_distance:
                    occluded = True
                    break
            if not occluded:
                first[category] = int(frame.get("frame", 0))
    if any(value is None for value in first.values()):
        return AppearanceResult([], first, "category_never_visible")
    ordered = sorted(
        first,
        key=lambda category: (first[category], category_rank[category]),
    )
    return AppearanceResult(ordered, first)
