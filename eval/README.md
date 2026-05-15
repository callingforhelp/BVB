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

## Code-Level LLM Judge

Use `llm_judge_metric.py` after you have exported both ground-truth and agent-generated scenes to `bpy` Python scripts.

Input pairs JSONL format:

```json
{"id": "09c1414f1b", "gt_bpy_path": "gt/09c1414f1b.py", "pred_bpy_path": "pred/09c1414f1b.py"}
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
  --output results/llm_judge.jsonl
```

Before making API calls, validate the input file with:

```bash
python llm_judge_metric.py \
  --pairs code_pairs.jsonl \
  --output results/llm_judge.dry_run.jsonl \
  --dry-run
```

The judge returns a JSON object with a semantic score, major differences, missing/extra objects, spatial errors, execution risk, and rationale.
