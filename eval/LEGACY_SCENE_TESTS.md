# Experimental Scene Test and Blender export

These utilities support the earlier code-level evaluation and scene diagnostics.
**Scene Test is not an axis in the current BVB leaderboard or Overall score.**
Use [the current evaluation guide](README.md) for Dual VQA and Latent Similarity.
The historical implementation and examples below are retained for reproducibility.
Run their commands from the `eval/` directory.

## Optional: Batch Export Blend Results To BPY

The unit-test evaluator reads `.blend` files directly; BPY export is not
required. Use `batch_export_blend_to_bpy.py` only when you explicitly need
portable source code for inspection or another tool.

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

Use `generate_unit_tests.py` and `unit_test_metric.py` to evaluate either a
`.blend` or `.py` submission. In execute mode Blender produces a structured
scene manifest (names, collections, materials, world-space AABBs, and camera
trajectory). The judge only grounds semantic references to manifest group keys;
host-side deterministic code computes pass/fail.

### How evaluation works

```text
Stage-1 run/blends/*.blend
          |
          v
unit_test_metric.py --run <run>
          |
          +-- select global + scene-specific tests from unit_tests.jsonl
          |
          +-- launch Blender with the minimum required introspection profile
          |     geometry   -> objects, evaluated dimensions/AABBs, floor footprint
          |     route      -> geometry + camera trajectory
          |     appearance -> route profile + per-frame ray-cast visibility
          |
          +-- build a compact semantic manifest
          |     [group_key, collection, material, animated]
          |
          +-- gpt-5.4-mini grounds semantic refs to exact group keys
          |     "chair" -> ["ChairSeat", "ChairBack"]
          |
          +-- host validates grounding with aliases/exact-key checks
          |
          +-- deterministic Python computes pass/fail
                count, size, distance, direction, room area,
                route turns, appearance order
```

The model that created a submission and the judge are different roles. For
example, `mini-harness-gpt-5.6-sol-...` means GPT-5.6 Sol created the `.blend`;
`--model gpt-5.4-mini` means Mini only grounds names during evaluation. Mini
does not generate or modify the scene and does not decide pass/fail.

One full 288-scene run makes at most one grounding batch per scene (split only
when an API response is truncated or incomplete). Rule tests never call an API.
Every output row records judge tokens, estimated cost, grounding evidence,
deterministic actual values, and pass/fail status.

### Test branches

- `basic_validity`: Blender load success, non-empty mesh scene, camera, light.
- `object_counting`: judge returns physical instances and their part group keys;
  host counts validated instances.
- `object_size_estimation`: evaluated object dimensions for one mesh; multi-part
  objects use union geometry.
- `object_abs_distance` / `object_rel_distance`: world-AABB surface separation.
- `object_rel_direction`: deterministic egocentric XY geometry.
- `room_size_estimation`: projected horizontal mesh footprint; bbox fallback
  only when no footprint is available.
- `route_planning`: ordered camera-trajectory landmark events and turn sequence.
- `obj_appearance_order`: first visible frame from camera projection plus
  Blender ray casts; transparent materials do not block the ray.

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

Then evaluate any submission against that same test set. `human`,
`claude-sonnet-4.6`, or any agent result is simply another submission:

```bash
python unit_test_metric.py \
  --run ../sandbox/results/mini-harness-claude-sonnet-4-6-run01 \
  --unit-tests unit_tests.jsonl \
  --mode execute \
  --cache-dir ../sandbox/results/mini-harness-claude-sonnet-4-6-run01/introspection-cache \
  --resume
```

`--run` scans `blends/` directly and writes `unit_tests.jsonl` plus
`summary.json` into the same run. For submissions outside the Stage-1 directory
layout, pass `--submissions manifest.jsonl`; each row must contain `id` and
`submission_path`. Execute mode accepts `.blend` by default. Trusted `.py`
submissions require explicit `--allow-python-exec`; never enable it for
untrusted code outside an isolated container.

The generated test suite currently contains:

- `basic_validity`: new tests that do not come from QA, such as parse success, non-empty scene, camera exists, and light exists.
- QA-derived tests: tests converted from `test.jsonl`, keeping the original VSI-Bench `question_type` as the unit-test type.

QA-derived tests use these evaluator modes:

- `rule`: direct checks that need no model, currently used for `basic_validity`.
- `function`: the judge maps semantic references to exact scene group keys;
  deterministic code evaluates counting, size, absolute/relative distance,
  egocentric direction, route turns, and appearance order.
- `unsupported`: the selected evaluator mode cannot provide required evidence
  (for example running exact spatial functions in `--mode static`). In execute
  mode a submission with no camera motion fails temporal tests rather than
  being excluded. Direct LLM proposition guessing is disabled.

Configure an OpenAI-compatible API before running function tests. The recommended
judge is the inexpensive `gpt-5.4-mini`:

```bash
export OPENAI_API_KEY="..."
```

Pass the non-secret model name explicitly for reproducible commands:

```bash
python unit_test_metric.py \
  --run ../sandbox/results/mini-harness-claude-sonnet-4-6-run01 \
  --mode execute \
  --model gpt-5.4-mini \
  --limit 10
```

Use `--dry-run --limit 10` to validate loading and Blender introspection without
model calls, then remove `--dry-run` for a paid smoke test. Each output row and
the summary record judge prompt/completion tokens and estimated USD cost.

For comparable random smoke tests across multiple agent runs, use one seed:

```bash
python smoke_matrix.py \
  --runs \
    ../sandbox/results/mini-harness-gpt-5.6-sol-reasoning-high-run01 \
    ../sandbox/results/mini-harness-minimax-m3-run01 \
    ../sandbox/results/mini-harness-seed-2.0-lite-run01 \
  --model gpt-5.4-mini \
  --sample 10 \
  --seed 42 \
  --stop-on-error
```

Every run receives the same sorted scene-ID sample. Add `--dry-run` to validate
the matrix without API calls.

Use exact scene IDs and selected test types for targeted evaluator debugging:

```bash
python smoke_matrix.py \
  --runs \
    ../sandbox/results/mini-harness-gpt-5.6-sol-reasoning-high-run01 \
    ../sandbox/results/mini-harness-minimax-m3-run01 \
  --model gpt-5.4-mini \
  --scene-ids 09c1414f1b 0d2ee665be \
  --test-types obj_appearance_order route_planning
```

Compare the resulting score, token, and cost summaries:

```bash
python compare_smoke.py \
  --runs \
    ../sandbox/results/mini-harness-gpt-5.6-sol-reasoning-high-run01 \
    ../sandbox/results/mini-harness-minimax-m3-run01 \
    ../sandbox/results/mini-harness-seed-2.0-lite-run01 \
  --model gpt-5.4-mini \
  --sample 10 \
  --seed 42
```

For full-scale evaluation, run independent scene shards (each command writes
separate output files):

```bash
for i in $(seq 1 8); do
  python unit_test_metric.py \
    --run ../sandbox/results/mini-harness-gpt-5.6-sol-reasoning-high-run01 \
    --mode execute \
    --model gpt-5.4-mini \
    --shard "$i/8" &
done
wait

# Merge into unit_tests.jsonl + summary.json, then delete shard intermediates.
python merge_eval_shards.py \
  --run ../sandbox/results/mini-harness-gpt-5.6-sol-reasoning-high-run01 \
  --cleanup
```

To evaluate many completed Stage-1 runs in one go (skips runs that already
have `summary.json`, auto-merges, and cleans shard intermediates):

```bash
cd ../sandbox
caffeinate -i ./run_eval_batch.sh
# or preview the queue first:
./run_eval_batch.sh --dry-run
```

If a shard is interrupted, rerun that same shard command with `--resume`.
`run_eval_batch.sh` resumes a shard only when that shard's `.config.json`
already exists, so fresh runs and interrupted runs both work.

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
  "num_tests": 149,
  "num_evaluated": 149,
  "num_passed": 80,
  "num_unsupported": 0,
  "unit_test_pass_rate": 0.5369127516778524,
  "basic_validity_aggregation": "scene_all_or_nothing",
  "judge_usage": {
    "prompt_tokens": 12345,
    "completion_tokens": 678,
    "cost_usd": 0.01231
  },
  "by_test_type": {
    "basic_validity": {
      "aggregation": "scene_all_or_nothing",
      "pass_rate": 0.9666666666666667,
      "atomic_pass_rate": 0.9916666666666667
    },
    "object_counting": {"pass_rate": 0.32432432432432434}
  }
}
```

`basic_validity` is aggregated **per scene**: the four atomic checks (parse / mesh / camera / light) contribute a single credit that passes only when every evaluated check for that scene passes. This avoids saturating the overall micro-average with many easy basic tests. Atomic rates remain available under `atomic_pass_rate`.

All tests are preserved, including tests where `human` fails. Low `human` scores are diagnostic signals for improving test extraction, object matching, unit calibration, or the scene itself; they are not a reason to remove tests.

### Current geometric approximations

- Object dimensions use evaluated Blender dimensions for single meshes; a
  multi-part semantic object falls back to its union world AABB.
- Closest surface distance currently uses world AABB separation, which can
  underestimate distance for rotated or concave meshes.
- Room area uses the evaluated mesh's projected horizontal footprint and avoids
  double-counting overlapping floor parts; bbox area is a last-resort fallback.
- Appearance visibility uses camera projection plus Blender ray casts. Materials
  with low alpha or Principled transmission are treated as transparent.
- Camera trajectories are sampled at up to 600 coarse frames, then each detected
  first-visible interval is refined frame-by-frame.

This unit-test metric is experimental. It is meant to support the "scene code as a testable artifact" direction. In particular, object matching and unit calibration still need improvement.
