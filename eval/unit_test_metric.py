#!/usr/bin/env python3
"""Run BVB unit tests against a scene-code submission.

The input tests are materialized in ``unit_tests.jsonl``. This runner does not
derive tests from QA at evaluation time. It treats every scene source the same:
human or agent output is simply a submission (``.blend`` or ``.py``).

Evaluation policy:
- simple checks are judged by rules;
- geometry tests ask the judge model to identify the relevant object/group
  parameters, then deterministic code computes pass/fail;
- tests without sufficient deterministic evidence are marked unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from eval_utils import (
    SceneGroup,
    build_groups,
    call_chat_completion_with_usage,
    get_first,
    parse_bpy_scene,
    read_jsonl,
    safe_divide,
)
from exec_scene import (
    groups_from_payload,
    introspect_scene,
    scene_manifest,
    scene_index_from_payload,
    temporal_from_payload,
)
from scene_geometry import (
    appearance_order,
    bbox_distance,
    bbox_size,
    closest_candidate,
    grounded_union,
    infer_floor_group,
    relative_direction,
    route_turns_from_camera,
)


BATCH_SYSTEM_PROMPT = """You ground semantic object references for Blender unit tests.
You are given a compact scene manifest and tests containing semantic references.
Map each reference to exact group_key values from the manifest. Do not decide
pass/fail; deterministic host code does that.

Return only valid JSON:
{
  "results": [
    {
      "grounding_id": "same grounding_id as input",
      "grounded_params": {},
      "passed": null,
      "rationale": "at most 12 words"
    }
  ]
}

Fill grounded_params using these exact schemas:
- count_objects: {
    "object_instances": [["part_group_key", ...], ...]
  }
- longest_dimension: {"object_group": "group_key or list"}
- closest_distance: {
    "object_a_group": "group_key or list",
    "object_b_group": "group_key or list"
  }
- room_area: {"object_group": "group_key or list"}
- relative_direction: {
    "anchor_group": "group_key or list",
    "facing_group": "group_key or list",
    "query_group": "group_key or list"
  }
- closest_among: {
    "anchor_group": "group_key or list",
    "candidate_groups": {"semantic_label": "group_key or list"}
  }
- route_planning: {
    "start_group": "group_key or list",
    "facing_group": "group_key or list",
    "route_steps": [
      {"landmark_group": "group_key or list", "relation": "near|left|right|pass"}
    ]
  }
- appearance_order: {
    "category_groups": {"semantic_category": "group_key or list"}
  }
Return only exact group_key values from the first manifest column. Collection
names are context for finding related parts but are never valid return values.
Return a list of exact group keys when one physical object has multiple parts.

Use genuine synonyms and constituent parts: bed may map to mattress plus bed
frame, sofa may map to couch/sectional parts, and tv maps to television. Do not
substitute a different category merely because it is nearby or related: a
computer mouse is not a monitor, a cutting board is not a generic board joint,
and a kettle is not the entire kitchen. If the requested object is absent,
return null (or [] for plural fields). Keep rationale concise.
"""

JUDGE_PRICES_PER_MILLION = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-sol": (5.00, 30.00),
}

EVALUATOR_FILES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "scene_geometry.py",
    Path(__file__).resolve().parent / "exec_scene.py",
    Path(__file__).resolve().parent / "scene_introspect.py",
    Path(__file__).resolve().parent / "eval_utils.py",
)


class JudgeResponseError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def numeric_pass(actual: float | None, expected: float | None, tolerance: float | None) -> bool:
    if actual is None or expected is None or tolerance is None:
        return False
    return abs(actual - expected) <= tolerance


def resolve_record_path(raw_path: Any, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    based = base_dir / path
    if based.exists():
        return based
    return path


def load_tests(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    global_tests = []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for test in read_jsonl(path):
        scene_name = str(get_first(test, ("scene_name", "scene_id")))
        if scene_name == "*":
            global_tests.append(test)
        else:
            by_scene.setdefault(scene_name, []).append(test)
    return global_tests, by_scene


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for record in read_jsonl(path):
        if "id" in record:
            completed.add(str(record["id"]))
    return completed


def sha256_files(paths: list[Path] | tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_eval_config(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "evaluator_hash": sha256_files(EVALUATOR_FILES),
        "unit_tests_hash": sha256_files((args.unit_tests.resolve(),)),
        "model": args.model,
        "mode": args.mode,
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "test_types": sorted(args.test_types) if args.test_types else None,
        "scene_ids": [str(record.get("id")) for record in records],
        "input_price_per_million": args.input_price_per_million,
        "output_price_per_million": args.output_price_per_million,
        "allow_python_exec": args.allow_python_exec,
        "shard": args.shard,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def validate_resume_config(path: Path, current: dict[str, Any]) -> None:
    if not path.exists():
        raise SystemExit(f"Cannot resume without evaluation config: {path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != current:
        changed = sorted(
            key for key in set(stored) | set(current) if stored.get(key) != current.get(key)
        )
        raise SystemExit(
            "Refusing to resume with changed evaluation config: "
            + ", ".join(changed)
        )


def records_from_run(run_dir: Path) -> list[dict[str, Any]]:
    blends_dir = run_dir / "blends"
    if not blends_dir.is_dir():
        raise SystemExit(f"No blends directory: {blends_dir}")
    records = [
        {"id": blend.stem, "submission_path": str(blend.resolve())}
        for blend in sorted(blends_dir.glob("*.blend"))
    ]
    if not records:
        raise SystemExit(f"No .blend submissions found in {blends_dir}")
    return records


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    sample: int | None,
    seed: int,
    scene_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if scene_ids is not None:
        by_id = {str(record.get("id")): record for record in records}
        missing = [scene_id for scene_id in scene_ids if scene_id not in by_id]
        if missing:
            raise SystemExit(f"Missing requested submission ids: {', '.join(missing)}")
        return [by_id[scene_id] for scene_id in scene_ids]
    if sample is not None:
        population = sorted(records, key=lambda record: str(record.get("id")))
        count = min(sample, len(records))
        return sorted(
            random.Random(seed).sample(population, count),
            key=lambda record: str(record.get("id")),
        )
    return records[:limit] if limit is not None else records


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        index, count = (int(item) for item in value.split("/"))
    except ValueError as exc:
        raise SystemExit("--shard must look like i/N, e.g. 1/8") from exc
    if not (1 <= index <= count):
        raise SystemExit("--shard must satisfy 1 <= i <= N")
    return index, count


def write_jsonl_row(out: Any, row: dict[str, Any]) -> None:
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    os.fsync(out.fileno())


def judge_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> float:
    default_prices = JUDGE_PRICES_PER_MILLION.get(model)
    if default_prices is None and (
        input_price_per_million is None or output_price_per_million is None
    ):
        raise ValueError(
            f"Unknown judge pricing for {model}; pass both price overrides."
        )
    default_prices = default_prices or (0.0, 0.0)
    input_price = default_prices[0] if input_price_per_million is None else input_price_per_million
    output_price = default_prices[1] if output_price_per_million is None else output_price_per_million
    return round(
        prompt_tokens * input_price / 1_000_000
        + completion_tokens * output_price / 1_000_000,
        6,
    )


GROUNDING_PARAM_KEYS = {
    "count_objects": ("object_ref",),
    "longest_dimension": ("object_ref",),
    "closest_distance": ("object_refs",),
    "room_area": ("object_ref",),
    "relative_direction": ("anchor_ref", "facing_ref", "query_ref"),
    "closest_among": ("anchor_ref", "candidate_refs"),
    "route_planning": ("question", "start_ref", "facing_ref", "route_steps"),
    "appearance_order": ("category_refs",),
}


def grounding_test_payload(test: dict[str, Any]) -> dict[str, Any]:
    function = str(test.get("function"))
    params = test.get("params", {}) if isinstance(test.get("params"), dict) else {}
    keys = GROUNDING_PARAM_KEYS.get(function, tuple(params))
    return {
        "function": function,
        "semantic_params": {key: params.get(key) for key in keys},
    }


def prepare_grounding_requests(
    tests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    requests = []
    by_signature: dict[str, str] = {}
    test_to_grounding: dict[str, str] = {}
    for test in tests:
        payload = grounding_test_payload(test)
        signature = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        grounding_id = by_signature.get(signature)
        if grounding_id is None:
            grounding_id = f"ground_{len(requests) + 1:04d}"
            by_signature[signature] = grounding_id
            requests.append({"grounding_id": grounding_id, **payload})
        test_to_grounding[str(test.get("test_id"))] = grounding_id
    return requests, test_to_grounding


def build_batch_prompt(
    scene_id: str,
    manifest: dict[str, Any],
    grounding_requests: list[dict[str, Any]],
) -> str:
    return f"""Scene ID: {scene_id}

Scene manifest:
```json
{json.dumps(manifest, separators=(",", ":"), ensure_ascii=False)}
```

Grounding requests:
```json
{json.dumps(grounding_requests, separators=(",", ":"), ensure_ascii=False)}
```

Return one result for every input grounding_id."""


def call_json_judge(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
    max_completion_tokens: int,
    reasoning_effort: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion = call_chat_completion_with_usage(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    usage = {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "finish_reason": completion.finish_reason,
    }
    try:
        parsed = json.loads(completion.content)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(
            f"Judge returned invalid JSON (finish_reason={completion.finish_reason}): {exc}",
            usage,
        ) from exc
    return parsed, usage


def merge_raw_usage(*usages: dict[str, Any]) -> dict[str, Any]:
    reasons = [
        str(usage["finish_reason"])
        for usage in usages
        if usage.get("finish_reason") is not None
    ]
    return {
        "prompt_tokens": sum(int(usage.get("prompt_tokens") or 0) for usage in usages),
        "completion_tokens": sum(int(usage.get("completion_tokens") or 0) for usage in usages),
        "finish_reason": reasons[-1] if len(set(reasons)) == 1 else reasons,
    }


def validate_grounding_results(
    grounding_requests: list[dict[str, Any]],
    results: list[dict[str, Any]],
    usage: dict[str, Any],
) -> None:
    expected_ids = {str(item["grounding_id"]) for item in grounding_requests}
    returned_ids = [
        str(item.get("grounding_id"))
        for item in results
        if item.get("grounding_id") is not None
    ]
    counts = {grounding_id: returned_ids.count(grounding_id) for grounding_id in set(returned_ids)}
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected_ids - set(returned_ids))
    invalid_params = sorted(
        str(item.get("grounding_id"))
        for item in results
        if item.get("grounding_id") in expected_ids
        and not isinstance(item.get("grounded_params"), dict)
    )
    if missing or duplicates or invalid_params:
        raise JudgeResponseError(
            "Invalid grounding response: "
            f"missing={missing}, duplicates={duplicates}, invalid_params={invalid_params}",
            usage,
        )


def call_grounding_requests(
    *,
    scene_id: str,
    manifest: dict[str, Any],
    grounding_requests: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_batch_prompt(scene_id, manifest, grounding_requests)
    try:
        response, usage = call_json_judge(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            system_prompt=BATCH_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
        )
    except JudgeResponseError as exc:
        if len(grounding_requests) <= 1:
            raise
        midpoint = len(grounding_requests) // 2
        left, left_usage = call_grounding_requests(
            scene_id=scene_id,
            manifest=manifest,
            grounding_requests=grounding_requests[:midpoint],
            args=args,
        )
        right, right_usage = call_grounding_requests(
            scene_id=scene_id,
            manifest=manifest,
            grounding_requests=grounding_requests[midpoint:],
            args=args,
        )
        return left + right, merge_raw_usage(exc.usage, left_usage, right_usage)

    results = [
        item for item in response.get("results", []) if isinstance(item, dict)
    ]
    if usage.get("finish_reason") == "length":
        if len(grounding_requests) <= 1:
            raise JudgeResponseError("Judge response truncated for one grounding request", usage)
        midpoint = len(grounding_requests) // 2
        left, left_usage = call_grounding_requests(
            scene_id=scene_id,
            manifest=manifest,
            grounding_requests=grounding_requests[:midpoint],
            args=args,
        )
        right, right_usage = call_grounding_requests(
            scene_id=scene_id,
            manifest=manifest,
            grounding_requests=grounding_requests[midpoint:],
            args=args,
        )
        return left + right, merge_raw_usage(usage, left_usage, right_usage)
    validate_grounding_results(grounding_requests, results, usage)
    return results, usage


def finish_test(
    test: dict[str, Any],
    *,
    scene_id: str,
    status: str,
    actual: Any = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(test)
    result["scene_id"] = scene_id
    result["status"] = status
    result["actual"] = actual
    result["evidence"] = evidence or {}
    return result


def evaluate_rule_test(scene_id: str, pred_path: Path, scene_index: Any, test: dict[str, Any]) -> dict[str, Any]:
    params = test.get("params", {}) if isinstance(test.get("params"), dict) else {}
    check = params.get("check")
    if check == "parse_ok":
        return finish_test(
            test,
            scene_id=scene_id,
            status="pass" if scene_index.parse_ok else "fail",
            actual=scene_index.parse_ok,
            evidence={"path": str(pred_path), "parse_error": scene_index.parse_error},
        )
    if check == "has_mesh_objects":
        actual = len(scene_index.mesh_objects)
        return finish_test(test, scene_id=scene_id, status="pass" if actual > 0 else "fail", actual=actual)
    if check == "has_camera":
        actual = len(scene_index.cameras)
        return finish_test(test, scene_id=scene_id, status="pass" if actual > 0 else "fail", actual=actual)
    if check == "has_light":
        actual = len(scene_index.lights)
        return finish_test(test, scene_id=scene_id, status="pass" if actual > 0 else "fail", actual=actual)
    return finish_test(test, scene_id=scene_id, status="unsupported", evidence={"reason": f"unknown rule: {check}"})


def evaluate_function_test_with_params(
    scene_id: str,
    test: dict[str, Any],
    groups: dict[str, SceneGroup],
    params: dict[str, Any],
    *,
    temporal: dict[str, Any] | None,
    execution_mode: str,
) -> dict[str, Any]:
    function = test.get("function")
    expected = test.get("expected", {}) if isinstance(test.get("expected"), dict) else {}
    semantic_params = test.get("params", {}) if isinstance(test.get("params"), dict) else {}
    grounding_evidence: dict[str, Any] = {}
    if function == "count_objects":
        instances = params.get("object_instances")
        if not isinstance(instances, list):
            legacy = params.get("object_groups")
            instances = [[value] for value in legacy] if isinstance(legacy, list) else []
        valid_instances = []
        reports = []
        for instance in instances:
            grounded, report = grounded_union(
                groups,
                instance,
                semantic_params.get("object_ref"),
            )
            reports.append(report)
            if grounded is not None:
                valid_instances.append(grounded)
        grounding_evidence["instances"] = reports
        actual = len(valid_instances)
        passed = actual == expected.get("count")
    elif function == "longest_dimension":
        group, grounding_evidence = grounded_union(
            groups,
            params.get("object_group"),
            semantic_params.get("object_ref"),
        )
        size = group.dimensions if group and group.dimensions is not None else bbox_size(group)
        actual = max(size) * 100.0 if size else None
        grounding_evidence["dimension_method"] = (
            "evaluated_object_dimensions"
            if group is not None and group.dimensions is not None
            else "world_aabb"
        )
        passed = numeric_pass(actual, expected.get("value"), expected.get("tolerance"))
    elif function == "closest_distance":
        refs = semantic_params.get("object_refs") or [None, None]
        group_a, report_a = grounded_union(groups, params.get("object_a_group"), refs[0])
        group_b, report_b = grounded_union(groups, params.get("object_b_group"), refs[1])
        grounding_evidence = {"object_a": report_a, "object_b": report_b}
        actual = bbox_distance(group_a, group_b) if group_a and group_b else None
        passed = numeric_pass(actual, expected.get("value"), expected.get("tolerance"))
    elif function == "room_area":
        # Room area still needs a selected floor/room group; ask the model which
        # group to use so we do not hard-code naming conventions.
        group, grounding_evidence = grounded_union(
            groups,
            params.get("object_group"),
            semantic_params.get("object_ref"),
        )
        inferred_floor = infer_floor_group(groups)
        selected_size = bbox_size(group) if group else None
        selected_area = (
            group.footprint_area
            if group is not None and group.footprint_area is not None
            else selected_size[0] * selected_size[1] if selected_size else None
        )
        if (selected_area is None or selected_area < 1.0) and inferred_floor is not None:
            group = inferred_floor
            grounding_evidence["geometry_fallback"] = group.key
        size = bbox_size(group) if group else None
        actual = (
            group.footprint_area
            if group is not None and group.footprint_area is not None
            else size[0] * size[1] if size else None
        )
        grounding_evidence["area_method"] = (
            "mesh_projected_footprint"
            if group is not None and group.footprint_area is not None
            else "bbox_xy"
        )
        passed = numeric_pass(actual, expected.get("value"), expected.get("tolerance"))
    elif function == "relative_direction":
        if execution_mode != "execute":
            return finish_test(
                test,
                scene_id=scene_id,
                status="unsupported",
                evidence={"reason": "relative_direction_requires_execute_mode"},
            )
        anchor, anchor_report = grounded_union(
            groups, params.get("anchor_group"), semantic_params.get("anchor_ref")
        )
        facing, facing_report = grounded_union(
            groups, params.get("facing_group"), semantic_params.get("facing_ref")
        )
        query, query_report = grounded_union(
            groups, params.get("query_group"), semantic_params.get("query_ref")
        )
        grounding_evidence = {
            "anchor": anchor_report,
            "facing": facing_report,
            "query": query_report,
        }
        actual = relative_direction(
            anchor=anchor,
            facing=facing,
            query=query,
            reference=facing,
            difficulty=str(test.get("difficulty") or ""),
        )
        passed = actual is not None and actual == expected.get("answer")
    elif function == "closest_among":
        if execution_mode != "execute":
            return finish_test(
                test,
                scene_id=scene_id,
                status="unsupported",
                evidence={"reason": "closest_among_requires_execute_mode"},
            )
        anchor, anchor_report = grounded_union(
            groups, params.get("anchor_group"), semantic_params.get("anchor_ref")
        )
        semantic_refs = semantic_params.get("candidate_refs") or []
        grounded = params.get("candidate_groups")
        if not isinstance(grounded, dict):
            grounded = {}
        candidates = {}
        candidate_reports = {}
        for label in semantic_refs:
            candidate, report = grounded_union(
                groups,
                grounded.get(str(label)),
                label,
            )
            candidates[str(label)] = candidate
            candidate_reports[str(label)] = report
        actual, distances = closest_candidate(anchor, candidates)
        passed = actual is not None and actual == expected.get("answer")
        return finish_test(
            test,
            scene_id=scene_id,
            status="pass" if passed else "fail",
            actual=actual,
            evidence={
                "judge_params": params,
                "grounding_validation": {
                    "anchor": anchor_report,
                    "candidates": candidate_reports,
                },
                "distances": distances,
            },
        )
    elif function == "route_planning":
        if execution_mode != "execute":
            return finish_test(
                test,
                scene_id=scene_id,
                status="unsupported",
                evidence={"reason": "route_planning_requires_execute_mode"},
            )
        grounded_steps = params.get("route_steps")
        semantic_steps = semantic_params.get("route_steps") or []
        if not isinstance(grounded_steps, list):
            grounded_steps = []
        route_steps = []
        route_reports = []
        for index, step in enumerate(grounded_steps):
            if not isinstance(step, dict):
                continue
            semantic_relation = (
                semantic_steps[index].get("relation")
                if index < len(semantic_steps) and isinstance(semantic_steps[index], dict)
                else "near"
            )
            semantic_ref = (
                semantic_steps[index].get("ref")
                if index < len(semantic_steps) and isinstance(semantic_steps[index], dict)
                else None
            )
            semantic_text = str(semantic_ref or "").lower()
            allow_auto_match = not any(
                qualifier in semantic_text
                for qualifier in ("other", "first", "corner", "beside", "near")
            )
            route_group, route_report = grounded_union(
                groups,
                step.get("landmark_group"),
                semantic_ref,
                allow_auto_match=allow_auto_match,
            )
            route_reports.append(route_report)
            route_steps.append(
                {
                    "group": route_group,
                    "relation": step.get("relation") or semantic_relation,
                }
            )
        start_group, start_report = grounded_union(
            groups, params.get("start_group"), semantic_params.get("start_ref")
        )
        facing_group, facing_report = grounded_union(
            groups, params.get("facing_group"), semantic_params.get("facing_ref")
        )
        actual, route_evidence = route_turns_from_camera(
            start=start_group,
            facing=facing_group,
            route_steps=route_steps,
            temporal=temporal,
        )
        expected_turns = expected.get("turns")
        passed = actual is not None and actual == expected_turns
        return finish_test(
            test,
            scene_id=scene_id,
            status="pass" if passed else "fail",
            actual=actual,
            evidence={
                "judge_params": params,
                "grounding_validation": {
                    "start": start_report,
                    "facing": facing_report,
                    "route_steps": route_reports,
                },
                **route_evidence,
            },
        )
    elif function == "appearance_order":
        if execution_mode != "execute":
            return finish_test(
                test,
                scene_id=scene_id,
                status="unsupported",
                evidence={"reason": "appearance_order_requires_execute_mode"},
            )
        semantic_refs = semantic_params.get("category_refs") or []
        grounded = params.get("category_groups")
        if not isinstance(grounded, dict):
            grounded = {}
        category_groups = {}
        category_reports = {}
        for category in semantic_refs:
            category_group, report = grounded_union(
                groups,
                grounded.get(str(category)),
                category,
            )
            category_groups[str(category)] = category_group
            category_reports[str(category)] = report
        result = appearance_order(
            category_groups=category_groups,
            all_groups=groups,
            temporal=temporal,
        )
        actual = result.order
        passed = actual == expected.get("order")
        return finish_test(
            test,
            scene_id=scene_id,
            status="pass" if passed else "fail",
            actual=actual,
            evidence={
                "reason": result.reason,
                "first_visible_frames": result.first_visible_frames,
                "judge_params": params,
                "grounding_validation": category_reports,
            },
        )
    else:
        return finish_test(test, scene_id=scene_id, status="unsupported", evidence={"reason": f"unknown function: {function}"})

    return finish_test(
        test,
        scene_id=scene_id,
        status="pass" if passed else "fail",
        actual=actual,
        evidence={"judge_params": params, "grounding_validation": grounding_evidence},
    )


def run_judged_tests_batch(
    scene_id: str,
    tests: list[dict[str, Any]],
    groups: dict[str, SceneGroup],
    manifest: dict[str, Any],
    temporal: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grounding_requests, test_to_grounding = prepare_grounding_requests(tests)
    results, usage = call_grounding_requests(
        scene_id=scene_id,
        manifest=manifest,
        grounding_requests=grounding_requests,
        args=args,
    )
    by_id = {
        str(item.get("grounding_id")): item
        for item in results
        if isinstance(item, dict) and item.get("grounding_id") is not None
    }
    outputs = []
    for test in tests:
        result = by_id.get(test_to_grounding.get(str(test.get("test_id")), ""), {})
        if test.get("evaluator") == "function":
            params = result.get("grounded_params", {})
            if not isinstance(params, dict):
                params = {}
            evaluated = evaluate_function_test_with_params(
                scene_id,
                test,
                groups,
                params,
                temporal=temporal,
                execution_mode=args.mode,
            )
            evidence = dict(evaluated.get("evidence", {}))
            evidence["judge_rationale"] = result.get("rationale")
            evaluated["evidence"] = evidence
            outputs.append(evaluated)
        else:
            outputs.append(
                finish_test(
                    test,
                    scene_id=scene_id,
                    status="unsupported",
                    evidence={"reason": "direct_proposition_judging_disabled"},
                )
            )
    return outputs, usage


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tests = [test for row in rows for test in row["unit_tests"]]
    evaluated = [test for test in tests if test["status"] in {"pass", "fail"}]
    passed = [test for test in tests if test["status"] == "pass"]
    non_basic_tests = [test for test in tests if test.get("test_type") != "basic_validity"]
    non_basic_evaluated = [test for test in non_basic_tests if test["status"] in {"pass", "fail"}]
    non_basic_passed = [test for test in non_basic_tests if test["status"] == "pass"]
    by_type: dict[str, dict[str, Any]] = {}
    for test in tests:
        item = by_type.setdefault(
            str(test.get("test_type")),
            {"num_tests": 0, "num_evaluated": 0, "num_passed": 0, "num_unsupported": 0, "num_error": 0},
        )
        item["num_tests"] += 1
        if test["status"] in {"pass", "fail"}:
            item["num_evaluated"] += 1
        if test["status"] == "pass":
            item["num_passed"] += 1
        if test["status"] == "unsupported":
            item["num_unsupported"] += 1
        if test["status"] == "error":
            item["num_error"] += 1
    for item in by_type.values():
        item["pass_rate"] = safe_divide(item["num_passed"], item["num_evaluated"])

    judge_prompt_tokens = sum(int(row.get("judge_usage", {}).get("prompt_tokens") or 0) for row in rows)
    judge_completion_tokens = sum(int(row.get("judge_usage", {}).get("completion_tokens") or 0) for row in rows)
    judge_cost_usd = sum(float(row.get("judge_usage", {}).get("cost_usd") or 0.0) for row in rows)
    judge_models = sorted({str(row["judge_model"]) for row in rows if row.get("judge_model")})
    summary = {
        "num_scenes": len(rows),
        "num_tests": len(tests),
        "num_evaluated": len(evaluated),
        "num_passed": len(passed),
        "num_unsupported": sum(1 for test in tests if test["status"] == "unsupported"),
        "num_error": sum(1 for test in tests if test["status"] == "error"),
        "unit_test_pass_rate": safe_divide(len(passed), len(evaluated)),
        "unit_test_pass_rate_without_basic_validity": safe_divide(
            len(non_basic_passed),
            len(non_basic_evaluated),
        ),
        "num_evaluated_without_basic_validity": len(non_basic_evaluated),
        "num_passed_without_basic_validity": len(non_basic_passed),
        "judge_usage": {
            "model": judge_models[0] if len(judge_models) == 1 else judge_models,
            "prompt_tokens": judge_prompt_tokens,
            "completion_tokens": judge_completion_tokens,
            "cost_usd": round(judge_cost_usd, 6),
        },
        "by_test_type": by_type,
    }
    appearance_tests = [
        test for test in tests if test.get("test_type") == "obj_appearance_order"
    ]
    if appearance_tests:
        total_categories = 0
        visible_categories = 0
        complete_tests = 0
        complete_passed = 0
        for test in appearance_tests:
            frames = test.get("evidence", {}).get("first_visible_frames", {})
            if not isinstance(frames, dict):
                frames = {}
            total_categories += len(frames)
            visible_categories += sum(value is not None for value in frames.values())
            complete = bool(frames) and all(value is not None for value in frames.values())
            complete_tests += complete
            complete_passed += complete and test.get("status") == "pass"
        summary["appearance_diagnostics"] = {
            "num_tests": len(appearance_tests),
            "visible_category_slots": visible_categories,
            "total_category_slots": total_categories,
            "category_visibility_rate": safe_divide(
                visible_categories,
                total_categories,
            ),
            "num_complete_order_tests": complete_tests,
            "num_complete_order_passed": complete_passed,
            "pass_rate_given_complete_order": safe_divide(
                complete_passed,
                complete_tests,
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BVB materialized unit tests.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run",
        type=Path,
        help="Stage-1 run directory containing blends/. Outputs default into this directory.",
    )
    source.add_argument(
        "--submissions",
        type=Path,
        help="JSONL manifest with id and submission_path for non-standard inputs.",
    )
    parser.add_argument("--unit-tests", type=Path, default=Path(__file__).resolve().parent / "unit_tests.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--model", default=os.getenv("BVB_JUDGE_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
        help="Judge reasoning effort; semantic grounding normally needs none.",
    )
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--test-types", nargs="*")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Evaluate only the first N submissions.")
    selection.add_argument("--sample", type=int, help="Evaluate a deterministic random sample.")
    selection.add_argument("--scene-ids", nargs="+", help="Evaluate these exact scene ids.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by --sample.")
    parser.add_argument("--shard", help="Evaluate shard i/N into independent output files.")
    parser.add_argument(
        "--mode",
        choices=("static", "execute"),
        default="static",
        help="static: AST-parse the bpy source (default). execute: run the "
        "submission in Blender and read real geometry from the depsgraph.",
    )
    parser.add_argument(
        "--blender",
        default=os.getenv("BLENDER_BIN"),
        help="Blender executable for --mode execute (or set BLENDER_BIN).",
    )
    parser.add_argument(
        "--exec-timeout",
        type=float,
        default=180.0,
        help="Per-scene Blender timeout in seconds for --mode execute.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional cache directory for introspection JSON (--mode execute).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-python-exec",
        action="store_true",
        help="Allow trusted .py submissions to execute in host Blender.",
    )
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed scene ids.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if (
        not args.dry_run
        and args.model not in JUDGE_PRICES_PER_MILLION
        and (
            args.input_price_per_million is None
            or args.output_price_per_million is None
        )
    ):
        parser.error(
            f"Unknown pricing for --model {args.model}; pass "
            "--input-price-per-million and --output-price-per-million."
        )

    global_tests, tests_by_scene = load_tests(args.unit_tests)
    shard = parse_shard(args.shard)
    selected_types = set(args.test_types) if args.test_types else None
    rows = []
    if args.run is not None:
        records = records_from_run(args.run)
        record_base_dir = args.run
        if shard and args.output is None:
            output_path = args.run / f"unit_tests.shard-{shard[0]}-of-{shard[1]}.jsonl"
        else:
            output_path = args.output or (args.run / "unit_tests.jsonl")
        if shard and args.summary_output is None:
            summary_output = args.run / f"summary.shard-{shard[0]}-of-{shard[1]}.json"
        else:
            summary_output = args.summary_output or (args.run / "summary.json")
    else:
        assert args.submissions is not None
        records = list(read_jsonl(args.submissions))
        record_base_dir = args.submissions.parent
        if args.output is None:
            parser.error("--output is required with --submissions")
        output_path = args.output
        summary_output = args.summary_output
    records = [
        record for record in records if str(record.get("id")) in tests_by_scene
    ]
    records = select_records(
        records,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
        scene_ids=args.scene_ids,
    )
    if shard:
        records = records[shard[0] - 1 :: shard[1]]
    eval_config = build_eval_config(args, records)
    config_path = output_path.with_suffix(output_path.suffix + ".config.json")
    if args.resume:
        validate_resume_config(config_path, eval_config)
    else:
        write_json_atomic(config_path, eval_config)
    completed_ids = load_completed_ids(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    if completed_ids:
        print(f"Resuming from {output_path}; skipping {len(completed_ids)} completed scene(s).", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8") as out:
        for index, record in enumerate(records, start=1):
            scene_id = str(get_first(record, ("id", "scene_id")))
            if scene_id in completed_ids:
                continue
            print(f"[{index}/{len(records)}] Evaluating {scene_id}...", flush=True)
            pred_path = resolve_record_path(
                get_first(record, ("submission_path",)),
                record_base_dir,
            )
            if (
                args.mode == "execute"
                and pred_path.suffix.lower() != ".blend"
                and not args.allow_python_exec
            ):
                raise SystemExit(
                    f"Refusing to execute non-.blend submission without "
                    f"--allow-python-exec: {pred_path}"
                )
            candidate_tests = [
                test
                for test in global_tests + tests_by_scene.get(scene_id, [])
                if not selected_types or test.get("test_type") in selected_types
            ]
            if not candidate_tests:
                print(
                    f"[{index}/{len(records)}] {scene_id}: no selected tests; skipping.",
                    flush=True,
                )
                continue
            test_types = {str(test.get("test_type")) for test in candidate_tests}
            if "obj_appearance_order" in test_types:
                introspection_profile = "appearance"
            elif "route_planning" in test_types:
                introspection_profile = "route"
            else:
                introspection_profile = "geometry"
            payload = None
            if args.mode == "execute":
                payload = introspect_scene(
                    pred_path,
                    blender_bin=args.blender,
                    timeout=args.exec_timeout,
                    cache_dir=args.cache_dir,
                    profile=introspection_profile,
                )
                scene_index = scene_index_from_payload(payload, pred_path)
                groups = groups_from_payload(payload)
            else:
                scene_index = parse_bpy_scene(pred_path)
                groups = build_groups(scene_index)
            manifest = scene_manifest(scene_index, groups, payload)
            temporal = temporal_from_payload(payload or {})
            unit_tests = []
            judged_tests = []
            for test in candidate_tests:
                evaluator = test.get("evaluator")
                if evaluator == "rule":
                    unit_tests.append(evaluate_rule_test(scene_id, pred_path, scene_index, test))
                elif evaluator == "function":
                    judged_tests.append(test)
                elif evaluator == "proposition":
                    unit_tests.append(
                        finish_test(
                            test,
                            scene_id=scene_id,
                            status="unsupported",
                            evidence={"reason": "direct_proposition_judging_disabled"},
                        )
                    )
                else:
                    unit_tests.append(
                        finish_test(
                            test,
                            scene_id=scene_id,
                            status="unsupported",
                            evidence={
                                "reason": test.get("params", {}).get(
                                    "unsupported_reason",
                                    f"unsupported evaluator: {evaluator}",
                                )
                            },
                        )
                    )

            judge_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            if (
                judged_tests
                and args.mode == "execute"
                and isinstance(payload, dict)
                and not payload.get("exec_ok")
            ):
                exec_error = payload.get("exec_error") or "unknown Blender introspection failure"
                unit_tests.extend(
                    finish_test(
                        test,
                        scene_id=scene_id,
                        status="error",
                        evidence={"reason": "introspection_failed", "exec_error": exec_error},
                    )
                    for test in judged_tests
                )
                judged_tests = []
            if judged_tests:
                print(
                    f"[{index}/{len(records)}] {scene_id}: rule_tests={len(unit_tests)}, judged_tests={len(judged_tests)}",
                    flush=True,
                )
                if args.dry_run:
                    unit_tests.extend(finish_test(test, scene_id=scene_id, status="dry_run") for test in judged_tests)
                else:
                    try:
                        if not args.api_key or not args.model:
                            raise RuntimeError("Missing OPENAI_API_KEY or BVB_JUDGE_MODEL for judged unit tests.")
                        judged_outputs, usage = run_judged_tests_batch(
                            scene_id,
                            judged_tests,
                            groups,
                            manifest,
                            temporal,
                            args,
                        )
                        unit_tests.extend(judged_outputs)
                        judge_usage = {
                            **usage,
                            "cost_usd": judge_cost(
                                args.model,
                                usage["prompt_tokens"],
                                usage["completion_tokens"],
                                input_price_per_million=args.input_price_per_million,
                                output_price_per_million=args.output_price_per_million,
                            ),
                        }
                    except Exception as exc:
                        if args.stop_on_error:
                            raise
                        unit_tests.extend(
                            finish_test(test, scene_id=scene_id, status="error", evidence={"error": str(exc)})
                            for test in judged_tests
                        )
            row = {
                "id": scene_id,
                "submission_path": str(pred_path),
                "unit_tests": unit_tests,
                "judge_model": args.model,
                "judge_usage": judge_usage,
                "num_tests": len(unit_tests),
                "num_passed": sum(1 for test in unit_tests if test["status"] == "pass"),
                "num_evaluated": sum(1 for test in unit_tests if test["status"] in {"pass", "fail"}),
            }
            row["unit_test_pass_rate"] = safe_divide(row["num_passed"], row["num_evaluated"])
            rows.append(row)
            write_jsonl_row(out, row)
            print(
                f"[{index}/{len(records)}] Wrote {scene_id}: "
                f"passed={row['num_passed']}/{row['num_evaluated']} "
                f"tests={row['num_tests']}",
                flush=True,
            )

    aggregate_rows = list(read_jsonl(output_path))
    aggregate = summarize(aggregate_rows)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
