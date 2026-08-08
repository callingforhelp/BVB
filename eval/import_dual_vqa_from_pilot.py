#!/usr/bin/env python3
"""Import misc/videoqa_pilot api_log into official Dual VQA run artifacts.

Writes:
  sandbox/results/_dual_vqa_shared/original_answers.jsonl
  sandbox/results/_dual_vqa_shared/text_only_answers.jsonl   (optional)
  sandbox/results/<run>/dual_vqa.jsonl
  sandbox/results/<run>/dual_vqa_summary.json

Does not call any API. Reuses already-paid pilot answers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dual_vqa_scoring import (
    DUAL_VQA_JSONL,
    DUAL_VQA_SUMMARY,
    ORIGINAL_ANSWERS,
    TEXT_ONLY_ANSWERS,
    load_qa_metadata,
    make_answer_row,
    make_dual_vqa_row,
    read_jsonl,
    semantic_correct,
    shared_dir,
    summarize_dual_vqa_rows,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "sandbox" / "results"
DEFAULT_META = Path(__file__).resolve().parent / "test.jsonl"
DEFAULT_PILOT_LOG = (
    REPO_ROOT / "misc" / "videoqa_pilot" / "out_gpt54mini_all" / "api_log.jsonl"
)
DEFAULT_TEXT_LOG = (
    REPO_ROOT / "misc" / "videoqa_pilot" / "out_gpt54mini_textonly" / "api_log.jsonl"
)
ORIG_LABEL = "__original__"


def _score_bank(
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    *,
    view: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        qa_id = str(record["qa_id"])
        qa = metadata.get(qa_id)
        if qa is None:
            continue
        answer = str(record.get("answer", ""))
        model = str(record.get("model") or "unknown")
        n_frames = record.get("n_frames")
        rows.append(
            make_answer_row(
                qa_id=qa_id,
                qa=qa,
                answer=answer,
                view=view,
                model=model,
                n_frames=int(n_frames) if n_frames is not None else None,
                correct=semantic_correct(answer, qa),
            )
        )
    rows.sort(key=lambda row: int(row["qa_id"]))
    return rows


def import_pilot(
    *,
    pilot_log: Path,
    text_only_log: Path | None,
    results_dir: Path,
    metadata_path: Path,
    runs: set[str] | None,
    force: bool,
) -> dict[str, Any]:
    metadata = load_qa_metadata(metadata_path)
    originals_raw: dict[str, dict[str, Any]] = {}
    rendered_raw: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for record in read_jsonl(pilot_log):
        qa_id = str(record["qa_id"])
        view = str(record.get("view") or "")
        run_label = str(record.get("run_label") or "")
        if view == "original" or run_label == ORIG_LABEL:
            # Last write wins if duplicates appear.
            originals_raw[qa_id] = record
            continue
        if view != "rendered":
            continue
        if runs is not None and run_label not in runs:
            continue
        rendered_raw[run_label][qa_id] = record

    shared = shared_dir(results_dir)
    shared.mkdir(parents=True, exist_ok=True)
    original_rows = _score_bank(
        list(originals_raw.values()), metadata, view="original"
    )
    write_jsonl(shared / ORIGINAL_ANSWERS, original_rows)
    original_correct = {
        row["qa_id"]: bool(row["correct"]) for row in original_rows
    }

    text_rows: list[dict[str, Any]] = []
    if text_only_log is not None and text_only_log.is_file():
        text_raw = list(read_jsonl(text_only_log))
        text_rows = _score_bank(text_raw, metadata, view="text_only")
        write_jsonl(shared / TEXT_ONLY_ANSWERS, text_rows)

    imported_runs: list[str] = []
    skipped_runs: list[str] = []
    for run_name in sorted(rendered_raw):
        run_dir = results_dir / run_name
        out_jsonl = run_dir / DUAL_VQA_JSONL
        out_summary = run_dir / DUAL_VQA_SUMMARY
        if not force and out_jsonl.is_file() and out_summary.is_file():
            skipped_runs.append(run_name)
            continue
        if not run_dir.is_dir():
            # Still write artifacts so sync can pick them up even if local
            # blends/renders were pruned; parent must exist as the run folder.
            run_dir.mkdir(parents=True, exist_ok=True)

        dual_rows: list[dict[str, Any]] = []
        judge_models: set[str] = set()
        n_frames_vals: set[int] = set()
        for qa_id, record in sorted(
            rendered_raw[run_name].items(), key=lambda item: int(item[0])
        ):
            qa = metadata.get(qa_id)
            if qa is None:
                continue
            answer = str(record.get("answer", ""))
            judge = str(record.get("model") or "unknown")
            n_frames = record.get("n_frames")
            judge_models.add(judge)
            if n_frames is not None:
                n_frames_vals.add(int(n_frames))
            rendered_correct = semantic_correct(answer, qa)
            dual_rows.append(
                make_dual_vqa_row(
                    qa_id=qa_id,
                    qa=qa,
                    rendered_answer=answer,
                    rendered_correct=rendered_correct,
                    original_correct=original_correct.get(qa_id),
                    judge_model=judge,
                    n_frames=int(n_frames) if n_frames is not None else None,
                )
            )
        write_jsonl(out_jsonl, dual_rows)
        summary = summarize_dual_vqa_rows(
            dual_rows,
            judge_model=(
                next(iter(judge_models))
                if len(judge_models) == 1
                else sorted(judge_models)
            ),
            n_frames=(
                next(iter(n_frames_vals))
                if len(n_frames_vals) == 1
                else sorted(n_frames_vals)
            ),
            source=f"import:{pilot_log}",
        )
        out_summary.write_text(json.dumps(summary, indent=2) + "\n")
        imported_runs.append(run_name)

    report = {
        "pilot_log": str(pilot_log),
        "text_only_log": None if text_only_log is None else str(text_only_log),
        "results_dir": str(results_dir),
        "n_original_answers": len(original_rows),
        "n_text_only_answers": len(text_rows),
        "n_imported_runs": len(imported_runs),
        "n_skipped_runs": len(skipped_runs),
        "imported_runs": imported_runs,
        "skipped_runs": skipped_runs,
    }
    (shared / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-log", type=Path, default=DEFAULT_PILOT_LOG)
    parser.add_argument("--text-only-log", type=Path, default=DEFAULT_TEXT_LOG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--qa-metadata", type=Path, default=DEFAULT_META)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional run directory name; repeatable. Default: all runs in log.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dual_vqa.jsonl / dual_vqa_summary.json.",
    )
    parser.add_argument(
        "--skip-text-only",
        action="store_true",
        help="Do not import the text-only answer bank.",
    )
    args = parser.parse_args()

    if not args.pilot_log.is_file():
        raise SystemExit(f"missing pilot log: {args.pilot_log}")

    text_log = None if args.skip_text_only else args.text_only_log
    report = import_pilot(
        pilot_log=args.pilot_log,
        text_only_log=text_log,
        results_dir=args.results_dir,
        metadata_path=args.qa_metadata,
        runs=set(args.run) if args.run else None,
        force=args.force,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
