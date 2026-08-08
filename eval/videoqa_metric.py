#!/usr/bin/env python3
"""DEPRECATED: offline aggregator over legacy Dual VQA prediction JSONL.

The live Dual VQA metric lives in ``dual_vqa_metric.py`` (writes
``dual_vqa.jsonl`` / ``dual_vqa_summary.json``). Video Similarity uses
``vjepa_sim_metric.py``. This script is kept only for reproducing older
prediction files that already store paired original/rendered answers.
"""

Expected prediction JSONL fields:
  - id or qa_id: QA id matching metadata
  - original_answer or orig_answer: answer on the original video
  - rendered_answer or render_answer: answer on the rendered video

The script joins predictions with metadata ground truth and reports:
  - original accuracy
  - rendered accuracy
  - delta accuracy = rendered accuracy - original accuracy
  - retention rate = #(original correct and rendered correct) / #(original correct)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class QaItem:
    qa_id: str
    scene_name: str
    question: str
    ground_truth: str


@dataclass(frozen=True)
class Prediction:
    qa_id: str
    original_answer: str
    rendered_answer: str


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


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'")

    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return number_words.get(text, text)


def is_correct(answer: Any, ground_truth: Any) -> bool:
    return normalize_answer(answer) == normalize_answer(ground_truth)


def load_metadata(path: Path) -> dict[str, QaItem]:
    items: dict[str, QaItem] = {}
    for record in read_jsonl(path):
        qa_id = str(get_first(record, ("id", "qa_id")))
        items[qa_id] = QaItem(
            qa_id=qa_id,
            scene_name=str(record.get("scene_name", "")),
            question=str(record.get("question", "")),
            ground_truth=str(get_first(record, ("ground_truth", "answer"))),
        )
    return items


def load_predictions(path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    for record in read_jsonl(path):
        qa_id = str(get_first(record, ("id", "qa_id")))
        predictions.append(
            Prediction(
                qa_id=qa_id,
                original_answer=str(get_first(record, ("original_answer", "orig_answer"))),
                rendered_answer=str(
                    get_first(record, ("rendered_answer", "render_answer", "reconstructed_answer"))
                ),
            )
        )
    return predictions


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(metadata: dict[str, QaItem], predictions: list[Prediction]) -> dict[str, Any]:
    rows = []
    missing_ids = []

    for prediction in predictions:
        item = metadata.get(prediction.qa_id)
        if item is None:
            missing_ids.append(prediction.qa_id)
            continue

        original_correct = is_correct(prediction.original_answer, item.ground_truth)
        rendered_correct = is_correct(prediction.rendered_answer, item.ground_truth)
        rows.append(
            {
                "id": item.qa_id,
                "scene_name": item.scene_name,
                "question": item.question,
                "ground_truth": item.ground_truth,
                "original_answer": prediction.original_answer,
                "rendered_answer": prediction.rendered_answer,
                "original_correct": original_correct,
                "rendered_correct": rendered_correct,
            }
        )

    total = len(rows)
    original_correct_count = sum(row["original_correct"] for row in rows)
    rendered_correct_count = sum(row["rendered_correct"] for row in rows)
    retained_count = sum(row["original_correct"] and row["rendered_correct"] for row in rows)

    original_accuracy = safe_divide(original_correct_count, total)
    rendered_accuracy = safe_divide(rendered_correct_count, total)

    return {
        "num_predictions": len(predictions),
        "num_evaluated": total,
        "num_missing_metadata": len(missing_ids),
        "missing_metadata_ids": missing_ids,
        "original_accuracy": original_accuracy,
        "rendered_accuracy": rendered_accuracy,
        "delta_accuracy": rendered_accuracy - original_accuracy,
        "retention_rate": safe_divide(retained_count, original_correct_count),
        "counts": {
            "original_correct": original_correct_count,
            "rendered_correct": rendered_correct_count,
            "retained": retained_count,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BVB vision-level VQA metrics.")
    parser.add_argument("--metadata", type=Path, default=Path("test.jsonl"), help="QA metadata JSONL.")
    parser.add_argument("--predictions", type=Path, required=True, help="VQA prediction JSONL.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path with per-question rows.")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    predictions = load_predictions(args.predictions)
    result = compute_metrics(metadata, predictions)

    summary = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
