from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_s1_export import export
from flinter_mvp.frame_prompt import (
    frame_id,
    max_frame_reference,
    normalize_prompt_state,
    permuted_prompt_state,
    render_prompt,
    shift_prompt_state,
    validate_prompt_state,
)


def observable(frames: list[int]) -> dict:
    return {
        "candidates": [
            {"frame": frame, "frame_id": f"frame_{frame}",
             "gate": "pixel_gate", "revealed": False}
            for frame in frames
        ],
        "choices": [{"frame_id": f"frame_{frame}"} for frame in frames],
        "free_signals": {},
    }


def action(index: int, frame: int, *, det: bool = True,
           typed: bool = True, judges: int = 2) -> dict:
    return {
        "index": index,
        "frame": frame,
        "clean_fp": False,
        "det_correct": det,
        "type_correct": typed,
        "total_judges": judges,
        "first_new_discontinuity_call": 1,
    }


def state(source: str, step: int, frames: list[int], actions: list[dict]) -> dict:
    return {
        "clip": f"{source}__clip", "source": source, "step": step,
        "observable": observable(frames), "actions": actions,
    }


def fixture_benchmark(path: Path) -> Path:
    unique_train = state("source_a", 0, [10, 20, 30], [
        action(0, 10, det=False),
        action(1, 20, judges=1),
        action(2, 30, typed=False),
    ])
    uniform = state("source_a", 1, [40, 50], [
        action(0, 40), action(1, 50),
    ])
    partial_tie = state("source_a", 2, [60, 70, 80], [
        action(0, 60, judges=1), action(1, 70, judges=1),
        action(2, 80, det=False),
    ])
    single = state("source_a", 3, [90], [action(0, 90)])
    unique_val = state("source_b", 0, [100, 110], [
        action(0, 100, det=False), action(1, 110, judges=1),
    ])
    path.write_text(json.dumps({
        "states": [unique_train, uniform, partial_tie, single, unique_val],
    }))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_frame_prompt_is_strict_and_permutation_keeps_physical_choice() -> None:
    obs = normalize_prompt_state(observable([10, 20, 30]))
    validate_prompt_state(obs)
    permuted = permuted_prompt_state(obs, [2, 0, 1])
    assert [choice["frame_id"] for choice in permuted["choices"]] == [
        frame_id(30), frame_id(10), frame_id(20),
    ]
    assert frame_id(20) in render_prompt(permuted)
    shifted = shift_prompt_state(permuted, 1000)
    assert shifted["candidates"][0]["frame"] == 1030
    assert shifted["choices"][0]["frame_id"] == frame_id(1030)
    assert max_frame_reference(shifted) == 1030

    leaky = normalize_prompt_state(observable([10, 20]))
    leaky["candidates"][0]["expected"] = "room_swap"
    with pytest.raises(ValueError):
        validate_prompt_state(leaky)


def test_export_filters_ties_splits_before_augmentation_and_is_deterministic(
    tmp_path: Path,
) -> None:
    benchmark = fixture_benchmark(tmp_path / "benchmark.json")
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"
    first = export(benchmark, out_a, ["source_b"], augment=4)
    second = export(benchmark, out_b, ["source_b"], augment=4)

    assert first["outputs"]["train.jsonl"]["rows"] == 4
    assert first["outputs"]["val.jsonl"]["rows"] == 4
    assert first["dataset_ids"] == second["dataset_ids"]
    assert (out_a / "train.jsonl").read_bytes() == (out_b / "train.jsonl").read_bytes()
    assert (out_a / "val.jsonl").read_bytes() == (out_b / "val.jsonl").read_bytes()

    train = read_jsonl(out_a / "train.jsonl")
    val = read_jsonl(out_a / "val.jsonl")
    assert train[0]["messages"][-1]["content"] == frame_id(20)
    assert len({row["messages"][-1]["content"] for row in train}) == 4
    assert val[0]["messages"][-1]["content"] == frame_id(110)
    assert len({row["messages"][-1]["content"] for row in val}) == 4
    assert len({row["messages"][0]["content"] for row in train}) == 4
    assert all("det_correct" not in row["messages"][0]["content"] for row in train)

    report = json.loads((out_a / "quality_report.json").read_text())
    assert report["passed"] is True
    assert report["excluded"] == {
        "non_decision": 1, "partial_tie": 1, "uniform": 1,
    }
    assert report["shortcut_baseline"]["frequency_prior_accuracy"] == 0
    assert report["shortcut_baseline"]["passed"] is True
    assert {row["canonical_label"] for row in report["rows"]["train"]} == {
        frame_id(20),
    }
    assert {row["source"] for row in report["rows"]["train"]} == {"source_a"}
    assert {row["source"] for row in report["rows"]["val"]} == {"source_b"}


def test_export_rejects_missing_or_empty_validation_source(tmp_path: Path) -> None:
    benchmark = fixture_benchmark(tmp_path / "benchmark.json")
    with pytest.raises(SystemExit, match="absent from benchmark"):
        export(benchmark, tmp_path / "missing", ["source_z"])
    payload = json.loads(benchmark.read_text())
    payload["states"].append(state("source_c", 0, [120, 130], [
        action(0, 120), action(1, 130),
    ]))
    benchmark.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="no eligible rows"):
        export(benchmark, tmp_path / "empty", ["source_c"], augment=2)


def test_export_rejects_validation_split_solved_by_frame_frequency(
    tmp_path: Path,
) -> None:
    benchmark = fixture_benchmark(tmp_path / "benchmark.json")
    payload = json.loads(benchmark.read_text())
    payload["states"][-1] = state("source_b", 0, [20, 110], [
        action(0, 20, judges=1), action(1, 110, det=False),
    ])
    benchmark.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="frequency_prior_accuracy"):
        export(benchmark, tmp_path / "shortcut", ["source_b"], augment=4,
               val_augment=1)
