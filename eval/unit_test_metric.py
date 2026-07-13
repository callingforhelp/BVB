#!/usr/bin/env python3
"""Run BVB unit tests against a scene-code submission.

The input tests are materialized in ``unit_tests.jsonl``. This runner does not
derive tests from QA at evaluation time. It treats every scene source the same
way: Human, Claude, or any other agent is just a submission with a bpy file.

Evaluation policy:
- simple checks are judged by rules;
- geometry tests ask the judge model to identify the relevant object/group
  parameters, then deterministic code computes pass/fail;
- tests that cannot be computed are judged directly as pass/fail statements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from eval_utils import (
    SceneGroup,
    build_groups,
    call_chat_completion,
    get_first,
    parse_bpy_scene,
    read_jsonl,
    safe_divide,
)
from exec_scene import (
    groups_from_payload,
    introspect_scene,
    scene_index_from_payload,
)


BATCH_SYSTEM_PROMPT = """You help run unit tests for a Blender scene.
You are given one Blender Python scene file and a batch of unit tests.

Return only valid JSON:
{
  "results": [
    {
      "test_id": "same test_id as input",
      "grounded_params": {},
      "passed": true or false or null,
      "rationale": "short explanation"
    }
  ]
}

For tests with evaluator="function", do NOT decide pass/fail yourself. Fill
grounded_params so the script can call the deterministic function:
- count_objects: {"object_groups": ["group_key", ...]}
- longest_dimension: {"object_group": "group_key"}
- closest_distance: {"object_a_group": "group_key", "object_b_group": "group_key"}
- room_area: {"object_group": "group_key"}
Set passed to null for function tests.

For tests with evaluator="proposition", judge whether the statement is true for
the scene. Set passed to true or false. grounded_params can be empty.
Use object, collection, or variable names that appear in the Blender Python code
when filling grounded_params.
"""


def bbox_size(group: SceneGroup) -> tuple[float, float, float] | None:
    if group.bbox_min is None or group.bbox_max is None:
        return None
    return tuple(max(0.0, group.bbox_max[i] - group.bbox_min[i]) for i in range(3))


def bbox_distance(group_a: SceneGroup, group_b: SceneGroup) -> float | None:
    if (
        group_a.bbox_min is None
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


def write_jsonl_row(out: Any, row: dict[str, Any]) -> None:
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    os.fsync(out.fileno())


def group_lookup(groups: dict[str, SceneGroup], key: Any) -> SceneGroup | None:
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


def read_text_limited(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n# ... middle truncated for judge context ...\n\n" + text[-half:]


def build_batch_prompt(scene_id: str, bpy_code: str, tests: list[dict[str, Any]]) -> str:
    return f"""Scene ID: {scene_id}

Blender Python scene code:
```python
{bpy_code}
```

Unit tests:
```json
{json.dumps(tests, indent=2, ensure_ascii=False)}
```

Return one result for every input test_id."""


def call_json_judge(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
) -> dict[str, Any]:
    content = call_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    return json.loads(content)


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
) -> dict[str, Any]:
    function = test.get("function")
    expected = test.get("expected", {}) if isinstance(test.get("expected"), dict) else {}
    if function == "count_objects":
        object_groups = params.get("object_groups", [])
        actual = len(object_groups) if isinstance(object_groups, list) else None
        passed = actual == expected.get("count")
    elif function == "longest_dimension":
        group = group_lookup(groups, params.get("object_group"))
        size = bbox_size(group) if group else None
        actual = max(size) * 100.0 if size else None
        passed = numeric_pass(actual, expected.get("value"), expected.get("tolerance"))
    elif function == "closest_distance":
        group_a = group_lookup(groups, params.get("object_a_group"))
        group_b = group_lookup(groups, params.get("object_b_group"))
        actual = bbox_distance(group_a, group_b) if group_a and group_b else None
        passed = numeric_pass(actual, expected.get("value"), expected.get("tolerance"))
    elif function == "room_area":
        # Room area still needs a selected floor/room group; ask the model which
        # group to use so we do not hard-code naming conventions.
        group = group_lookup(groups, params.get("object_group"))
        size = bbox_size(group) if group else None
        actual = size[0] * size[1] if size else None
        passed = numeric_pass(actual, expected.get("value"), expected.get("tolerance"))
    else:
        return finish_test(test, scene_id=scene_id, status="unsupported", evidence={"reason": f"unknown function: {function}"})

    return finish_test(
        test,
        scene_id=scene_id,
        status="pass" if passed else "fail",
        actual=actual,
        evidence={"judge_params": params},
    )


def evaluate_proposition_test_with_result(scene_id: str, test: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    passed = bool(result.get("passed"))
    return finish_test(
        test,
        scene_id=scene_id,
        status="pass" if passed else "fail",
        actual=passed,
        evidence={"judge_response": result},
    )


def run_judged_tests_batch(
    scene_id: str,
    tests: list[dict[str, Any]],
    groups: dict[str, SceneGroup],
    bpy_code: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompt = build_batch_prompt(scene_id, bpy_code, tests)
    response = call_json_judge(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        system_prompt=BATCH_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=args.temperature,
    )
    by_id = {
        str(item.get("test_id")): item
        for item in response.get("results", [])
        if isinstance(item, dict) and item.get("test_id") is not None
    }
    outputs = []
    for test in tests:
        result = by_id.get(str(test.get("test_id")), {})
        if test.get("evaluator") == "function":
            params = result.get("grounded_params", {})
            if not isinstance(params, dict):
                params = {}
            evaluated = evaluate_function_test_with_params(scene_id, test, groups, params)
            evidence = dict(evaluated.get("evidence", {}))
            evidence["judge_rationale"] = result.get("rationale")
            evaluated["evidence"] = evidence
            outputs.append(evaluated)
        elif test.get("evaluator") == "proposition":
            outputs.append(evaluate_proposition_test_with_result(scene_id, test, result))
        else:
            outputs.append(finish_test(test, scene_id=scene_id, status="unsupported"))
    return outputs


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

    return {
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
        "by_test_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BVB materialized unit tests.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--unit-tests", type=Path, default=Path(__file__).resolve().parent / "unit_tests.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--model", default=os.getenv("BVB_JUDGE_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-chars", type=int, default=60000, help="Max chars of bpy code sent per scene.")
    parser.add_argument("--test-types", nargs="*")
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
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed scene ids.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    global_tests, tests_by_scene = load_tests(args.unit_tests)
    selected_types = set(args.test_types) if args.test_types else None
    rows = []
    records = list(read_jsonl(args.pairs))
    completed_ids = load_completed_ids(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"

    if completed_ids:
        print(f"Resuming from {args.output}; skipping {len(completed_ids)} completed scene(s).", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, encoding="utf-8") as out:
        for index, record in enumerate(records, start=1):
            scene_id = str(get_first(record, ("id", "scene_id")))
            if scene_id in completed_ids:
                continue
            print(f"[{index}/{len(records)}] Evaluating {scene_id}...", flush=True)
            pred_path = resolve_record_path(
                get_first(record, ("pred_bpy_path", "generated_bpy_path", "agent_bpy_path", "gt_bpy_path")),
                args.pairs.parent,
            )
            if args.mode == "execute":
                payload = introspect_scene(
                    pred_path,
                    blender_bin=args.blender,
                    timeout=args.exec_timeout,
                    cache_dir=args.cache_dir,
                )
                scene_index = scene_index_from_payload(payload, pred_path)
                groups = groups_from_payload(payload)
            else:
                scene_index = parse_bpy_scene(pred_path)
                groups = build_groups(scene_index)
            bpy_code = read_text_limited(pred_path, args.max_chars)
            candidate_tests = global_tests + tests_by_scene.get(scene_id, [])
            unit_tests = []
            judged_tests = []
            for test in candidate_tests:
                if selected_types and test.get("test_type") not in selected_types:
                    continue
                evaluator = test.get("evaluator")
                if evaluator == "rule":
                    unit_tests.append(evaluate_rule_test(scene_id, pred_path, scene_index, test))
                elif evaluator in {"function", "proposition"}:
                    judged_tests.append(test)
                else:
                    unit_tests.append(finish_test(test, scene_id=scene_id, status="unsupported"))

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
                        unit_tests.extend(run_judged_tests_batch(scene_id, judged_tests, groups, bpy_code, args))
                    except Exception as exc:
                        if args.stop_on_error:
                            raise
                        unit_tests.extend(
                            finish_test(test, scene_id=scene_id, status="error", evidence={"error": str(exc)})
                            for test in judged_tests
                        )
            row = {
                "id": scene_id,
                "pred_bpy_path": str(pred_path),
                "unit_tests": unit_tests,
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

    aggregate_rows = list(read_jsonl(args.output))
    aggregate = summarize(aggregate_rows)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
