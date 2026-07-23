#!/usr/bin/env python3
"""Materialize BVB unit tests from VSI-Bench QA metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from eval_utils import call_chat_completion, get_first, read_jsonl


STATEMENT_SYSTEM_PROMPT = """You rewrite BVB unit-test statements.
You are given JSON unit tests generated from QA metadata. Rewrite only the
"statement" field so it reads like a natural, precise pass/fail scene test.

Rules:
- Return only valid JSON: {"statements": [{"test_id": "...", "statement": "..."}]}.
- Preserve every input test_id exactly.
- Do not change expected values, units, tolerances, evaluator, params, or source_qa_id.
- Do not mention answer options unless the option text is the actual condition.
- Keep statements concise and declarative.
- Avoid awkward template wording like "object(s)".
"""


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def first_number(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def object_size_tolerance(expected_cm: float) -> float:
    return max(10.0, abs(expected_cm) * 0.10)


def room_area_tolerance(expected_m2: float) -> float:
    return max(1.0, abs(expected_m2) * 0.10)


def distance_tolerance(expected_m: float) -> float:
    return max(0.2, abs(expected_m) * 0.15)


OBJECT_ALIASES = {
    "tv": "television",
}


def canonical_object_name(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"\(s\)", "", text)
    text = re.sub(r"[^a-z0-9_ -]+", "", text)
    text = text.replace(" ", "_")
    return OBJECT_ALIASES.get(text, text)


def object_from_count_question(question: str) -> str | None:
    match = re.search(r"how many\s+(.+?)\s+are in", question, flags=re.I)
    if not match:
        match = re.search(r"how many\s+(.+?)\s+is in", question, flags=re.I)
    if not match:
        return None
    return canonical_object_name(match.group(1))


def object_from_size_question(question: str) -> str | None:
    match = re.search(r"of the\s+(.+?),\s+measured", question, flags=re.I)
    if not match:
        return None
    # The question also contains "length of the longest dimension ..."; keep
    # only the object phrase after the final "of the".
    raw = re.split(r"\bof the\b", match.group(1), flags=re.I)[-1]
    return canonical_object_name(raw)


def objects_from_distance_question(question: str) -> tuple[str, str] | None:
    match = re.search(r"between the\s+(.+?)\s+and the\s+(.+?)\s+\(", question, flags=re.I)
    if not match:
        match = re.search(r"between the\s+(.+?)\s+and the\s+(.+?)$", question, flags=re.I)
    if not match:
        return None
    return canonical_object_name(match.group(1)), canonical_object_name(match.group(2))


def option_text(options: Any, letter: Any) -> str | None:
    if not isinstance(options, list):
        return None
    target = str(letter).strip().upper()
    for option in options:
        match = re.match(r"^([A-Za-z])\s*[.)]\s*(.*)$", str(option).strip())
        if match and match.group(1).upper() == target:
            return match.group(2).strip()
    return None


def direction_refs(question: str) -> tuple[str, str, str] | None:
    match = re.search(
        r"standing by the\s+(.+?)\s+and facing the\s+(.+?),\s+is the\s+(.+?)\s+to",
        question,
        flags=re.I | re.S,
    )
    if not match:
        return None
    return tuple(canonical_object_name(match.group(index)) for index in range(1, 4))  # type: ignore[return-value]


def relative_distance_refs(question: str) -> tuple[str, list[str]] | None:
    match = re.search(
        r"which of these objects\s*\((.+?)\)\s+is the closest to the\s+(.+?)\?",
        question,
        flags=re.I | re.S,
    )
    if not match:
        return None
    candidates = [canonical_object_name(item) for item in match.group(1).split(",")]
    return canonical_object_name(match.group(2)), candidates


def appearance_categories(question: str) -> list[str] | None:
    match = re.search(r"following categories in the video:\s*(.+?)\?", question, flags=re.I | re.S)
    if not match:
        return None
    return [canonical_object_name(item) for item in match.group(1).split(",")]


def appearance_answer_order(answer: str | None) -> list[str] | None:
    if not answer:
        return None
    normalized = re.sub(r"\bthen\b", ",", answer, flags=re.I)
    values = [canonical_object_name(item) for item in normalized.split(",") if item.strip()]
    return values or None


def route_spec(question: str) -> dict[str, Any] | None:
    start_match = re.search(
        r"beginning at the\s+(.+?)\s+(?:and\s+)?facing the\s+(.+?)\.",
        question,
        flags=re.I | re.S,
    )
    route_steps = []
    for match in re.finditer(
        r"Go\s+f(?:or|o)ward\s+(until|passing|past)\s+(.*?)(?=\s+\d+\.|\s+You have reached|$)",
        question,
        flags=re.I | re.S,
    ):
        movement = match.group(1).lower()
        raw = match.group(2).strip().rstrip(".")
        raw = re.sub(r"^(the|a)\s+", "", raw, flags=re.I)
        if re.search(r"\b(?:on|to)\s+your\s+right\b|\bon\s+the\s+right\b", raw, flags=re.I):
            relation = "right"
        elif re.search(r"\b(?:on|to)\s+your\s+left\b|\bon\s+the\s+left\b", raw, flags=re.I):
            relation = "left"
        elif movement in {"passing", "past"} or re.search(
            r"\bpassing\b|\bpassed\b|\bpast\b",
            raw,
            flags=re.I,
        ):
            relation = "pass"
        else:
            relation = "near"
        route_steps.append({"ref": raw.lower(), "relation": relation})
    if not route_steps:
        return None
    return {
        "start_ref": canonical_object_name(start_match.group(1)) if start_match else None,
        "facing_ref": canonical_object_name(start_match.group(2)) if start_match else None,
        "route_steps": route_steps,
    }


def turn_sequence(answer: str | None) -> list[str] | None:
    if not answer:
        return None
    turns = [f"turn_{value}" for value in re.findall(r"turn\s+(back|left|right)", answer.lower())]
    return turns or None


def make_case(
    *,
    scene_name: str,
    test_type: str,
    statement: str,
    evaluator: str,
    source_qa_id: Any = None,
    function: str | None = None,
    expected: Any = None,
    params: dict[str, Any] | None = None,
    difficulty: str | None = None,
) -> dict[str, Any]:
    test = {
        "test_id": None,
        "scene_name": scene_name,
        "test_type": test_type,
        "statement": statement,
        "source_qa_id": source_qa_id,
        "evaluator": evaluator,
        "expected": expected,
        "params": params or {},
    }
    if difficulty:
        test["difficulty"] = difficulty
    if function:
        test["function"] = function
    return test


def basic_validity_tests() -> list[dict[str, Any]]:
    return [
        make_case(
            scene_name="*",
            test_type="basic_validity",
            statement="The Blender submission loads successfully.",
            evaluator="rule",
            params={"check": "parse_ok"},
        ),
        make_case(
            scene_name="*",
            test_type="basic_validity",
            statement="The scene contains at least one mesh object.",
            evaluator="rule",
            params={"check": "has_mesh_objects"},
        ),
        make_case(
            scene_name="*",
            test_type="basic_validity",
            statement="The scene contains at least one camera.",
            evaluator="rule",
            params={"check": "has_camera"},
        ),
        make_case(
            scene_name="*",
            test_type="basic_validity",
            statement="The scene contains at least one light.",
            evaluator="rule",
            params={"check": "has_light"},
        ),
    ]


def qa_to_unit_test(record: dict[str, Any]) -> dict[str, Any]:
    scene_name = str(get_first(record, ("scene_name", "scene_id")))
    qa_id = get_first(record, ("id", "qa_id"))
    question_type = str(record.get("question_type", ""))
    test_type = question_type
    difficulty = None
    direction_match = re.match(r"^(object_rel_direction)_(easy|medium|hard)$", question_type)
    if direction_match:
        test_type = direction_match.group(1)
        difficulty = direction_match.group(2)
    question = str(get_first(record, ("question",)))
    ground_truth = record.get("ground_truth")
    expected_number = first_number(ground_truth)

    if question_type == "object_counting":
        category = object_from_count_question(question)
        expected_count = int(expected_number) if expected_number is not None else None
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=f"The scene contains {expected_count} {category} object(s).",
            evaluator="function",
            function="count_objects",
            source_qa_id=qa_id,
            expected={"count": expected_count},
            params={"object_ref": category},
        )

    if question_type == "object_size_estimation":
        category = object_from_size_question(question)
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=f"The {category} has a longest dimension of approximately {expected_number} cm.",
            evaluator="function",
            function="longest_dimension",
            source_qa_id=qa_id,
            expected={"value": expected_number, "unit": "cm", "tolerance": object_size_tolerance(expected_number or 0.0)},
            params={"object_ref": category},
        )

    if question_type == "room_size_estimation":
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=f"The room area is approximately {expected_number} square meters.",
            evaluator="function",
            function="room_area",
            source_qa_id=qa_id,
            expected={"value": expected_number, "unit": "m2", "tolerance": room_area_tolerance(expected_number or 0.0)},
            params={"object_ref": "floor or room boundary"},
        )

    if question_type == "object_abs_distance":
        objects = objects_from_distance_question(question)
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=(
                f"The closest-point distance between {objects[0]} and {objects[1]} is approximately {expected_number} meters."
                if objects
                else question
            ),
            evaluator="function",
            function="closest_distance",
            source_qa_id=qa_id,
            expected={"value": expected_number, "unit": "m", "tolerance": distance_tolerance(expected_number or 0.0)},
            params={"object_refs": list(objects) if objects else None},
        )

    correct_option = option_text(record.get("options"), ground_truth)
    if direction_match:
        refs = direction_refs(question)
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=correct_option or question,
            evaluator="function",
            function="relative_direction",
            source_qa_id=qa_id,
            expected={
                "ground_truth": ground_truth,
                "answer": correct_option.lower() if correct_option else None,
            },
            params={
                "anchor_ref": refs[0] if refs else None,
                "facing_ref": refs[1] if refs else None,
                "query_ref": refs[2] if refs else None,
            },
            difficulty=difficulty,
        )

    if question_type == "object_rel_distance":
        refs = relative_distance_refs(question)
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=correct_option or question,
            evaluator="function",
            function="closest_among",
            source_qa_id=qa_id,
            expected={
                "ground_truth": ground_truth,
                "answer": canonical_object_name(correct_option) if correct_option else None,
            },
            params={
                "anchor_ref": refs[0] if refs else None,
                "candidate_refs": refs[1] if refs else None,
            },
        )

    if question_type == "route_planning":
        spec = route_spec(question)
        turns = turn_sequence(correct_option)
        supported = bool(spec and turns)
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=correct_option or question,
            evaluator="function" if supported else "unsupported",
            function="route_planning" if supported else None,
            source_qa_id=qa_id,
            expected={"ground_truth": ground_truth, "turns": turns},
            params={
                "question": question,
                "start_ref": spec.get("start_ref") if spec else None,
                "facing_ref": spec.get("facing_ref") if spec else None,
                "route_steps": spec.get("route_steps") if spec else None,
                "unsupported_reason": None if supported else "unparseable_route_instruction",
            },
        )

    if question_type == "obj_appearance_order":
        return make_case(
            scene_name=scene_name,
            test_type=test_type,
            statement=correct_option or question,
            evaluator="function",
            function="appearance_order",
            source_qa_id=qa_id,
            expected={
                "ground_truth": ground_truth,
                "order": appearance_answer_order(correct_option),
            },
            params={
                "category_refs": appearance_categories(question),
                "question": question,
            },
        )

    statement = correct_option if correct_option else f"For the question '{question}', the correct answer is '{ground_truth}'."
    return make_case(
        scene_name=scene_name,
        test_type=test_type,
        statement=statement,
        evaluator="proposition",
        source_qa_id=qa_id,
        expected={"ground_truth": ground_truth},
        params={"question": question, "options": record.get("options")},
        difficulty=difficulty,
    )


def rewrite_statements_with_llm(
    tests: list[dict[str, Any]],
    *,
    statement_output: Path,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float | None,
    batch_size: int,
) -> dict[str, str]:
    statement_output.parent.mkdir(parents=True, exist_ok=True)
    if statement_output.exists():
        statement_output.unlink()
    all_statements: dict[str, str] = {}
    for start in range(0, len(tests), batch_size):
        batch = tests[start : start + batch_size]
        end = start + len(batch)
        print(f"[{start + 1}-{end}/{len(tests)}] Rewriting statements...", flush=True)
        prompt_items = [
            {
                "test_id": test["test_id"],
                "scene_name": test["scene_name"],
                "test_type": test["test_type"],
                "statement": test["statement"],
                "source_qa_id": test.get("source_qa_id"),
                "evaluator": test["evaluator"],
                "function": test.get("function"),
                "expected": test.get("expected"),
                "params": test.get("params"),
            }
            for test in batch
        ]
        user_prompt = f"""Rewrite the statements for these unit tests:
```json
{json.dumps(prompt_items, indent=2, ensure_ascii=False)}
```"""
        content = call_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=STATEMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        response = json.loads(content)
        by_id = {
            str(item.get("test_id")): str(item.get("statement", "")).strip()
            for item in response.get("statements", [])
            if isinstance(item, dict) and item.get("test_id") is not None
        }
        all_statements.update({key: value for key, value in by_id.items() if value})
        with statement_output.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "batch_start": start + 1,
                        "batch_end": end,
                        "statements": [
                            {"test_id": key, "statement": value}
                            for key, value in sorted(by_id.items())
                            if value
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            os.fsync(f.fileno())
        print(f"[{start + 1}-{end}/{len(tests)}] Wrote {statement_output}", flush=True)
    return all_statements


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate materialized BVB unit tests.")
    parser.add_argument("--metadata", type=Path, default=Path(__file__).resolve().parent / "test.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "unit_tests.jsonl")
    parser.add_argument("--no-basic-validity", action="store_true")
    parser.add_argument("--llm-statements", action="store_true", help="Rewrite statements with an LLM.")
    parser.add_argument("--statement-output", type=Path, help="Intermediate JSONL for LLM statement rewrites.")
    parser.add_argument("--statement-batch-size", type=int, default=50)
    parser.add_argument("--model", default=os.getenv("BVB_JUDGE_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    tests = []
    if not args.no_basic_validity:
        tests.extend(basic_validity_tests())
    tests.extend(qa_to_unit_test(record) for record in read_jsonl(args.metadata))
    for index, test in enumerate(tests, start=1):
        test["test_id"] = f"ut_{index:06d}"

    if args.llm_statements:
        if not args.api_key:
            raise SystemExit("Missing OPENAI_API_KEY or --api-key for --llm-statements.")
        if not args.model:
            raise SystemExit("Missing BVB_JUDGE_MODEL or --model for --llm-statements.")
        statement_output = args.statement_output
        if statement_output is None:
            statement_output = args.output.with_name(args.output.stem + ".statements.jsonl")
        rewritten = rewrite_statements_with_llm(
            tests,
            statement_output=statement_output,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            batch_size=args.statement_batch_size,
        )
        for test in tests:
            statement = rewritten.get(str(test["test_id"]))
            if statement:
                test["statement"] = statement

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as f:
        for test in tests:
            f.write(json.dumps(test, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_output.replace(args.output)

    print(json.dumps({"output": str(args.output), "num_tests": len(tests)}, indent=2))


if __name__ == "__main__":
    main()
