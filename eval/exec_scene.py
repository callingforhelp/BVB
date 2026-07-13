"""Host-side bridge for execution-based scene evaluation.

Runs ``scene_introspect.py`` inside Blender as a subprocess to obtain the real
scene geometry of a submission (a ``.blend`` or a ``bpy`` ``.py`` file), then
builds ``SceneIndex`` / ``SceneGroup`` structures that are drop-in compatible
with the existing static pipeline in ``unit_test_metric.py``.

The difference from ``eval_utils.parse_bpy_scene`` is that bounding boxes here
are the *real* world-space AABBs read from Blender (matrix_world @ bound_box on
the evaluated object), not approximations inferred from source ``obj.scale``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from eval_utils import SceneGroup, SceneIndex, SceneObject


RESULT_START = "===BVB_INTROSPECT_START==="
RESULT_END = "===BVB_INTROSPECT_END==="
INTROSPECT_SCRIPT = Path(__file__).resolve().parent / "scene_introspect.py"
# Bump when scene_introspect.py output changes so stale caches are ignored.
INTROSPECT_VERSION = "1"


def resolve_blender_executable(value: str | None) -> str:
    candidates = []
    if value:
        candidates.append(value)
    candidates.extend(
        [
            "blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.3.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.2.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.1.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.0.app/Contents/MacOS/Blender",
        ]
    )
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        path = Path(candidate)
        if path.is_file():
            return str(path)
    raise SystemExit(
        "Could not find Blender. Pass --blender /path/to/Blender or set BLENDER_BIN."
    )


def _cache_key(input_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(INTROSPECT_VERSION.encode("utf-8"))
    digest.update(input_path.read_bytes())
    return digest.hexdigest()


def _extract_payload(stdout: str) -> dict[str, Any] | None:
    start = stdout.rfind(RESULT_START)
    end = stdout.rfind(RESULT_END)
    if start == -1 or end == -1 or end < start:
        return None
    block = stdout[start + len(RESULT_START) : end].strip()
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None


def introspect_scene(
    input_path: Path,
    *,
    blender_bin: str | None = None,
    timeout: float = 180.0,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the introspection payload for a submission file.

    Failures (missing file, Blender crash, timeout, bad output) are returned as
    a payload with ``exec_ok=False`` rather than raising, so the caller can mark
    affected tests as failed/errored without aborting the whole run.
    """
    if not input_path.exists():
        return {"exec_ok": False, "exec_error": f"missing file: {input_path}", "objects": [], "materials": {}}

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_cache_key(input_path)}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    blender = resolve_blender_executable(blender_bin)
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(INTROSPECT_SCRIPT),
        "--",
        "--input",
        str(input_path),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"exec_ok": False, "exec_error": f"timeout after {timeout}s", "objects": [], "materials": {}}

    payload = _extract_payload(completed.stdout or "")
    if payload is None:
        tail = (completed.stderr or completed.stdout or "")[-2000:]
        payload = {
            "exec_ok": False,
            "exec_error": f"no introspection output (returncode={completed.returncode})",
            "stderr_tail": tail,
            "objects": [],
            "materials": {},
        }

    if cache_path is not None:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _as_tuple(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def scene_index_from_payload(payload: dict[str, Any], path: Path) -> SceneIndex:
    objects: list[SceneObject] = []
    for entry in payload.get("objects", []):
        objects.append(
            SceneObject(
                name=str(entry.get("name")),
                kind=str(entry.get("kind", "mesh")),
                group_key=str(entry.get("name")),
                location=_as_tuple(entry.get("location")),
                rotation=_as_tuple(entry.get("rotation")),
                scale=_as_tuple(entry.get("scale")),
                material=entry.get("material"),
                light_type=entry.get("light_type"),
                light_energy=entry.get("light_energy"),
            )
        )
    materials = {
        name: tuple(color)
        for name, color in payload.get("materials", {}).items()
        if isinstance(color, (list, tuple)) and len(color) >= 3
    }
    return SceneIndex(
        path=str(path),
        parse_ok=bool(payload.get("exec_ok")),
        parse_error=payload.get("exec_error"),
        objects=objects,
        materials=materials,
    )


def groups_from_payload(payload: dict[str, Any]) -> dict[str, SceneGroup]:
    """Build one group per mesh object using the real world-space AABB.

    Granularity matches the static pipeline (one group keyed by object name), so
    the judge grounding for counting/size/distance keeps working unchanged. The
    object's collection name is added to ``object_names`` so a judge that refers
    to the collection still resolves via ``group_lookup``.
    """
    groups: dict[str, SceneGroup] = {}
    for entry in payload.get("objects", []):
        if entry.get("kind") in {"camera", "light"}:
            continue
        key = str(entry.get("name"))
        bbox_min = _as_tuple(entry.get("world_bbox_min"))
        bbox_max = _as_tuple(entry.get("world_bbox_max"))
        centroid = _as_tuple(entry.get("centroid"))
        names = [key]
        collection = entry.get("collection")
        if collection and collection != key:
            names.append(str(collection))
        groups[key] = SceneGroup(
            key=key,
            component_count=1,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            object_names=names,
        )
    return groups
