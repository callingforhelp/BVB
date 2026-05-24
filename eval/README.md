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

The script first parses both `bpy` files into structured scene summaries and computes deterministic static metrics. It can run in two modes:

1. Static-only mode: reproducible, cheap, no API required.
2. LLM judge mode: sends the structured summaries and static metrics to an OpenAI-compatible judge model for a semantic score.

Input pairs JSONL format:

```json
{"id": "09c1414f1b", "gt_bpy_path": "gt/09c1414f1b.py", "pred_bpy_path": "results/claude-sonnet-4.6/bpy/09c1414f1b.py"}
```

Static-only run:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/claude-sonnet-4.6/static_judge.jsonl \
  --summary-output results/claude-sonnet-4.6/static_summary.json \
  --static-only
```

Configure an OpenAI-compatible API:

```bash
export OPENAI_API_KEY="..."
export BVB_JUDGE_MODEL="gpt-4.1"
# Optional, for OpenAI-compatible providers:
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Run:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/claude-sonnet-4.6/llm_judge.jsonl \
  --summary-output results/claude-sonnet-4.6/llm_judge_summary.json
```

Before making API calls, validate the input file with:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/claude-sonnet-4.6/llm_judge.dry_run.jsonl \
  --dry-run
```

Per-scene output includes:

- `static_metrics`: deterministic category/count, object-match, spatial, material, light, camera, and health scores.
- `judgment`: LLM judge JSON when LLM mode is enabled; otherwise a dry-run/static-only marker.
- `combined_score`: static score in static-only mode, or a weighted blend of `semantic_score` and `static_score` in LLM mode.

The LLM judge returns a JSON object with a semantic score, major differences, missing/extra objects, spatial errors, execution risk, and rationale.
