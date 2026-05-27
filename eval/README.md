# BVB Evaluation Scripts

This folder contains lightweight evaluation utilities for BVB.

The current evaluation has two levels:

1. Vision-level VQA evaluation: ask the same QA item on the original video and the rendered reconstruction, then compare both answers to the ground truth.
2. Code-level unit-test evaluation: treat each exported `bpy` scene as a testable scene program.

## Vision-Level Metrics

Use `videoqa_metric.py` after you have VQA answers for both the original video and the rendered video.

Input prediction JSONL format:

```json
{"id": 0, "original_answer": "4", "rendered_answer": "3"}
{"id": 1, "original_answer": "2", "rendered_answer": "2"}
```

The `id` field should match `test.jsonl`. The script joins each prediction with the QA metadata and ground truth.

Run:

```bash
python videoqa_metric.py \
  --metadata test.jsonl \
  --predictions predictions.jsonl \
  --output results/vqa_metrics.json
```

Reported metrics:

- `original_accuracy`: accuracy of the VQA model on the source videos.
- `rendered_accuracy`: accuracy of the same VQA model on rendered reconstruction videos.
- `delta_accuracy`: `rendered_accuracy - original_accuracy`.
- `retention_rate`: `#(original correct and rendered correct) / #(original correct)`.

Hallucination rate is intentionally not included yet. We should refine its definition before treating it as an official metric.

## Batch Export Blend Results To BPY

Use `batch_export_blend_to_bpy.py` when an agent run produced `.blend` files and you want code-level evaluation.

The default result layout is:

```text
results/claude-sonnet-4.6/
  blend/   # input .blend files
  bpy/     # exported .py files
```

From this `eval/` directory, run:

```bash
python batch_export_blend_to_bpy.py \
  --overwrite \
  --manifest results/claude-sonnet-4.6/export_manifest.jsonl
```

This exports:

```text
results/claude-sonnet-4.6/blend/09c1414f1b.blend
-> results/claude-sonnet-4.6/bpy/09c1414f1b.py
```

Useful options:

- `--input-dir`: directory containing `.blend` files.
- `--output-dir`: directory where exported `.py` files should be written.
- `--blender`: path to the Blender executable. If omitted, the script tries `blender` on `PATH` and common macOS app locations.
- `--limit N`: export only the first `N` files for a smoke test.
- `--dry-run`: print planned exports without launching Blender.
- `--overwrite`: regenerate existing `.py` files.
- `--manifest`: write one JSONL row per attempted export with the input path, output path, return code, and success flag.

Smoke test:

```bash
python batch_export_blend_to_bpy.py --limit 3 --dry-run
python batch_export_blend_to_bpy.py --limit 1 --overwrite
```

## Code Unit Tests

Use `generate_unit_tests.py` and `unit_test_metric.py` after scenes have been exported to `bpy` Python scripts. This is the current code-level evaluation path. Instead of asking the judge model to answer QA freely, scene requirements are materialized once as pass/fail unit tests, then every submission is evaluated against the same test file.

First generate the stable input test suite:

```bash
python generate_unit_tests.py \
  --metadata test.jsonl \
  --output unit_tests.jsonl \
  --statement-output unit_tests.statements.jsonl \
  --llm-statements \
  --statement-batch-size 50
```

`unit_tests.jsonl` is the input test set. It should be versioned/reviewed like any other benchmark artifact. The runner does not convert from `test.jsonl` at evaluation time.

Then evaluate any submission against that same test set. `human`, `claude-sonnet-4.6`, or any other result directory is just a submission with scene code. For example, to evaluate the 30 refined scenes for Claude and human:

```bash
python unit_test_metric.py \
  --pairs results/claude-sonnet-4.6/code_pairs_refined30.jsonl \
  --unit-tests unit_tests.jsonl \
  --output results/claude-sonnet-4.6/unit_tests_refined30.jsonl \
  --summary-output results/claude-sonnet-4.6/unit_tests_summary_refined30.json \
  --resume

python unit_test_metric.py \
  --pairs results/gt/code_pairs_refined30.jsonl \
  --unit-tests unit_tests.jsonl \
  --output results/gt/unit_tests_refined30.jsonl \
  --summary-output results/gt/unit_tests_summary_refined30.json \
  --resume
```

The generated test suite currently contains:

- `basic_validity`: new tests that do not come from QA, such as parse success, non-empty scene, camera exists, and light exists.
- QA-derived tests: tests converted from `test.jsonl`, keeping the original VSI-Bench `question_type` as the unit-test type.

QA-derived tests use one of three evaluator modes:

- `rule`: direct checks that need no model, currently used for `basic_validity`.
- `function`: the judge model first finds the relevant scene object/group parameters, then the script calls a deterministic function to compute pass/fail. This is used for `object_counting`, `object_size_estimation`, `room_size_estimation`, and `object_abs_distance`.
- `proposition`: the judge model directly decides whether the test statement is true for the scene. This is used for QA types that are not yet connected to a deterministic function, such as `route_planning`, `object_rel_direction`, and `object_rel_distance`.

Configure an OpenAI-compatible API before running tests with `function` or `proposition` evaluators:

```bash
export OPENAI_API_KEY="..."
export BVB_JUDGE_MODEL="gpt-5.5"
```

Use `--dry-run` to verify test loading without making model calls. In dry-run mode, only `rule` tests are evaluated; tests that require the judge model are marked as `dry_run`.

Each input test case has a structure like:

```json
{
  "test_id": "ut_000123",
  "scene_name": "scene0461_00",
  "test_type": "object_counting",
  "source_qa_id": 123,
  "evaluator": "function",
  "function": "count_objects",
  "expected": {"count": 4},
  "params": {"object_ref": "chair"}
}
```

New tests such as `basic_validity` have `source_qa_id: null`.

The summary reports:

```json
{
  "num_scenes": 30,
  "num_tests": 289,
  "num_evaluated": 289,
  "num_passed": 180,
  "num_unsupported": 0,
  "unit_test_pass_rate": 0.6228373702422145,
  "by_test_type": {
    "basic_validity": {"pass_rate": 0.9916666666666667},
    "object_counting": {"pass_rate": 0.32432432432432434}
  }
}
```

All tests are preserved, including tests where `human` fails. Low `human` scores are diagnostic signals for improving test extraction, object matching, unit calibration, or the scene itself; they are not a reason to remove tests.

This unit-test metric is experimental. It is meant to support the "scene code as a testable artifact" direction. In particular, object matching and unit calibration still need improvement.
