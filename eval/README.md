# BVB Evaluation Scripts

This folder contains lightweight evaluation utilities for BVB.

The current evaluation has two levels:

1. Vision-level VQA evaluation: ask the same QA item on the original video and the rendered reconstruction, then compare both answers to the ground truth.
2. Code-level LLM judge evaluation: compare exported GT `bpy` code against exported agent `bpy` code.

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

## Code-Level LLM Judge

Use `llm_judge_metric.py` after you have exported both ground-truth and agent-generated scenes to `bpy` Python scripts.

The script supports two code-level scores:

- `code_semantic_score`: compare GT and agent `bpy` scene summaries directly with an LLM judge.
- `code_qa_score`: ask an LLM to answer all QA pairs for a scene from the agent `bpy` scene summary, then compare those answers to `test.jsonl` ground truth.

Use `--metric semantic`, `--metric qa`, or `--metric both` to choose which score to compute.

### Code Semantic Score

For `code_semantic_score`, the script parses both `bpy` files into structured scene summaries, sends those summaries to an OpenAI-compatible judge model, and asks for one score from 0 to 1:

```json
{"semantic_score": 0.63}
```

The score is a semantic scene similarity score from 0 to 1. It should be reported concisely, for example `0.63`, rather than padded with fake precision like `0.6300`.

The judge is instructed to compare:

- major object categories and counts
- room layout, walls, doors, windows, and connectivity
- spatial relationships, relative positions, scale, and orientation
- visible furniture and object groups
- material/color similarity when it affects recognition
- lights/camera only when they affect visibility or viewpoint

The judge should compare scene semantics, not raw code text or exact object naming.

### Code QA Score

For `code_qa_score`, the script uses `test.jsonl` as QA metadata. For each scene ID in the pair file, it finds all QA rows whose `scene_name` matches the scene ID, sends the agent-generated scene summary and those questions to the judge model, and asks the model to answer the questions from the reconstructed scene.

QA is batched by scene: one API call contains all questions for that scene. The scene summary is not resent once per question.

The returned answers are compared to `ground_truth` from `test.jsonl` with type-aware matching:

- `object_counting`: numeric exact match after extracting the first number.
- `object_size_estimation`: numeric match with tolerance `max(10 cm, 10% of GT)`.
- `room_size_estimation`: numeric match with tolerance `max(1.0 m^2, 10% of GT)`.
- `object_abs_distance`: numeric match with tolerance `max(0.2 m, 15% of GT)`.
- multiple-choice questions such as direction and route planning: accepts either the option letter (`A`, `B`, `C`, `D`) or the text of the correct option.
- other questions: normalized exact string match.

The scene-level score is:

```text
code_qa_score = number of correctly answered QA pairs / number of QA pairs for that scene
```

The run-level score is the arithmetic mean of scene-level `code_qa_score` values:

```text
mean_code_qa_score = average(code_qa_score over scenes with QA)
```

Input pairs JSONL format:

```json
{"id": "09c1414f1b", "gt_bpy_path": "gt/09c1414f1b.py", "pred_bpy_path": "results/claude-sonnet-4.6/bpy/09c1414f1b.py"}
```

Run semantic score only:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/claude-sonnet-4.6/code_semantic_judge.jsonl \
  --summary-output results/claude-sonnet-4.6/code_semantic_summary.json \
  --metric semantic
```

Run QA score only:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/claude-sonnet-4.6/code_qa_judge.jsonl \
  --summary-output results/claude-sonnet-4.6/code_qa_summary.json \
  --metric qa
```

Run both scores:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/claude-sonnet-4.6/code_both_judge.jsonl \
  --summary-output results/claude-sonnet-4.6/code_both_summary.json \
  --metric both
```

For the current 30 manually refined scenes:

```bash
python llm_judge_metric.py \
  --pairs results/claude-sonnet-4.6/code_pairs_refined30.jsonl \
  --output results/claude-sonnet-4.6/code_both_refined30.jsonl \
  --summary-output results/claude-sonnet-4.6/code_both_summary_refined30.json \
  --metric both \
  --resume
```

Use `--resume` for long runs. The script writes one JSONL row immediately after each scene returns, flushes it to disk, and skips completed IDs on the next run. Do not delete the output JSONL while a run is active; `--resume` uses that file as the checkpoint.

Configure an OpenAI-compatible API:

```bash
export OPENAI_API_KEY="..."
export BVB_JUDGE_MODEL="gpt-5.5"
# Optional, for OpenAI-compatible providers:
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Before making API calls, validate the input file with:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/claude-sonnet-4.6/code_both_judge.dry_run.jsonl \
  --metric both \
  --dry-run
```

Per-scene output includes:

- `id`: scene ID.
- `gt_bpy_path`: ground-truth exported `bpy` file.
- `pred_bpy_path`: agent exported `bpy` file.
- `code_semantic_score`: scene-level semantic score from 0 to 1, when `--metric semantic` or `--metric both` is used.
- `semantic_judgment`: LLM semantic judge JSON, including `semantic_score`, differences, spatial errors, execution risk, and rationale.
- `code_qa_score`: scene-level QA accuracy from 0 to 1, when `--metric qa` or `--metric both` is used.
- `qa_judgment`: QA scoring details, including `num_questions`, `num_answered`, `num_correct`, and per-question rows.

Example output row:

```json
{"id":"scene0461_00","gt_bpy_path":".../BVB/BVB/bpy/scene0461_00.py","pred_bpy_path":".../results/claude-sonnet-4.6/bpy/scene0461_00.py","code_semantic_score":0.62,"semantic_judgment":{"semantic_score":0.62,"major_differences":["..."],"missing_objects":[],"extra_objects":[],"spatial_errors":["..."],"execution_risk":"low","rationale":"..."},"code_qa_score":0.5,"qa_judgment":{"num_questions":2,"num_answered":2,"num_correct":1,"code_qa_score":0.5,"rows":[{"id":123,"question":"How many chair(s) are in this room?","ground_truth":"4","predicted_answer":"3","correct":false}]}}
```

The summary file reports the arithmetic mean over all completed scene scores:

```json
{
  "num_scenes": 30,
  "mean_code_semantic_score": 0.6346666666666665,
  "mean_code_qa_score": 0.5,
  "num_scenes_with_qa": 30,
  "num_qa_questions": 80,
  "num_qa_correct": 40
}
```

Use `mean_code_semantic_score` and/or `mean_code_qa_score` depending on which metric was requested.
