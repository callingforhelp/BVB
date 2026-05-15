#!/usr/bin/env python3
"""LLM-as-judge scaffold for code-level BVB evaluation.

The input JSONL should contain one record per scene:
  - id or scene_id
  - gt_bpy_path: path to the human/GT exported bpy script
  - pred_bpy_path or generated_bpy_path: path to the agent exported bpy script

The script calls an OpenAI-compatible Chat Completions API and writes JSONL
judgments. Configure the model with:
  - OPENAI_API_KEY
  - BVB_JUDGE_MODEL, or --model
  - OPENAI_BASE_URL, optional, default https://api.openai.com/v1
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = """You are a strict evaluator for Blender Python scene reconstruction.
Compare a ground-truth bpy script and an agent-generated bpy script.
Judge semantic scene similarity, not textual similarity.
Focus on QA-relevant geometry, object counts, object categories, spatial layout,
visibility, scale, and orientation. Penalize code that would fail to execute.
Return only valid JSON with:
{
  "semantic_score": number from 0 to 1,
  "major_differences": [short strings],
  "missing_objects": [short strings],
  "extra_objects": [short strings],
  "spatial_errors": [short strings],
  "execution_risk": "low" | "medium" | "high",
  "rationale": "short explanation"
}
"""


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def get_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    joined = ", ".join(keys)
    raise KeyError(f"Missing one of fields: {joined}")


def read_text_limited(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n# ... middle truncated for judge context ...\n\n" + text[-half:]


def build_user_prompt(scene_id: str, gt_code: str, pred_code: str) -> str:
    return f"""Scene ID: {scene_id}

Ground-truth bpy script:
```python
{gt_code}
```

Agent-generated bpy script:
```python
{pred_code}
```

Evaluate the agent-generated script against the ground truth."""


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Judge API request failed: {exc.code} {body}") from exc

    return data["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge for exported bpy scripts.")
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL file with gt/pred bpy paths.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL judgments.")
    parser.add_argument("--model", default=os.getenv("BVB_JUDGE_MODEL"), help="Judge model name.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max-chars", type=int, default=60000, help="Max chars per script pair.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling the API.")
    args = parser.parse_args()

    if not args.dry_run:
        if not args.api_key:
            raise SystemExit("Missing OPENAI_API_KEY or --api-key.")
        if not args.model:
            raise SystemExit("Missing BVB_JUDGE_MODEL or --model.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for record in read_jsonl(args.pairs):
            scene_id = str(get_first(record, ("id", "scene_id")))
            gt_path = Path(get_first(record, ("gt_bpy_path", "ground_truth_bpy_path")))
            pred_path = Path(get_first(record, ("pred_bpy_path", "generated_bpy_path", "agent_bpy_path")))

            gt_code = read_text_limited(gt_path, args.max_chars)
            pred_code = read_text_limited(pred_path, args.max_chars)
            user_prompt = build_user_prompt(scene_id, gt_code, pred_code)

            if args.dry_run:
                judgment: dict[str, Any] = {"dry_run": True, "prompt_chars": len(user_prompt)}
            else:
                content = call_chat_completion(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                )
                judgment = json.loads(content)

            out.write(
                json.dumps(
                    {
                        "id": scene_id,
                        "gt_bpy_path": str(gt_path),
                        "pred_bpy_path": str(pred_path),
                        "judgment": judgment,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
