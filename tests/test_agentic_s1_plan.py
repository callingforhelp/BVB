from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_s1_export import export
from agentic_s1_plan import (
    build_plan,
    remote_inventory,
    validate_export,
    write_plan,
)
from flinter_mvp.frame_prompt import frame_id


def _observable(frames: list[int]) -> dict:
    return {
        "candidates": [
            {"frame": frame, "frame_id": frame_id(frame),
             "gate": "pixel_gate", "revealed": False}
            for frame in frames
        ],
        "choices": [{"frame_id": frame_id(frame)} for frame in frames],
        "free_signals": {},
    }


def _action(index: int, frame: int, *, det: bool = True) -> dict:
    return {
        "index": index, "frame": frame, "clean_fp": False,
        "det_correct": det, "type_correct": True, "total_judges": 1,
        "first_new_discontinuity_call": 1,
    }


def fixture_benchmark(path: Path) -> Path:
    def state(source: str, frames: list[int], actions: list[dict]) -> dict:
        return {
            "clip": f"{source}__clip", "source": source, "step": 0,
            "observable": _observable(frames), "actions": actions,
        }

    path.write_text(json.dumps({"states": [
        state("source_a", [10, 20, 30], [
            _action(0, 10, det=False), _action(1, 20),
            _action(2, 30, det=False),
        ]),
        state("source_b", [100, 110], [
            _action(0, 100, det=False), _action(1, 110),
        ]),
    ]}))
    return path


class FakeTokenizer:
    name_or_path = "fake/qwen"
    init_kwargs = {"_commit_hash": "tokenizer-commit"}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is False
        assert all("weight" not in message for message in messages)
        return list(range(sum(len(message["content"].split()) + 3
                              for message in messages)))

    def encode(self, value, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 2] if value.startswith("frame_") else [1]


class FakeGetClient:
    def __init__(self):
        self.paths = []

    def get(self, path: str) -> dict:
        self.paths.append(path)
        responses = {
            "/accounts": {"accounts": [{"name": "accounts/test-account"}]},
            "/accounts/test-account/datasets": {
                "datasets": [{"name": "accounts/test-account/datasets/other"}],
            },
            "/accounts/test-account/supervisedFineTuningJobs": {
                "supervisedFineTuningJobs": [],
            },
            "/accounts/test-account/models": {"models": []},
            "/accounts/fireworks/models/qwen3p6-35b-a3b": {
                "model": {
                    "name": "accounts/fireworks/models/qwen3p6-35b-a3b",
                    "state": "READY",
                    "supportsLora": True,
                    "supervisedLoraTunable": True,
                },
            },
        }
        return responses[path]


def exported_manifest(tmp_path: Path) -> Path:
    benchmark = fixture_benchmark(tmp_path / "benchmark.json")
    out = tmp_path / "export"
    export(benchmark, out, ["source_b"], augment=4)
    return out / "manifest.json"


def test_plan_exact_token_cost_ids_and_hash_are_deterministic(tmp_path: Path) -> None:
    manifest = exported_manifest(tmp_path)
    kwargs = dict(
        epochs=4,
        lora_rank=32,
        batch_size_samples=128,
        rate_per_million=3.0,
        pricing_fetched_at_utc="2026-09-22T12:00:00Z",
        tokenizer=FakeTokenizer(),
    )
    first = build_plan(manifest, **kwargs)
    second = build_plan(manifest, **kwargs)

    assert first == second
    assert first["approval_hash"] == second["approval_hash"]
    assert first["gates"]["exact_tokenization"] is True
    assert first["token_audit"]["train"]["answer"]["histogram"] == {"2": 4}
    expected = (first["token_audit"]["train"]["rendered"]["total"]
                * 4 / 1_000_000 * 3.0)
    assert first["cost"]["point_estimate_usd"] == pytest.approx(expected)
    assert first["resources"]["train_dataset_id"].startswith("jev-act-v1-")
    assert first["resources"]["output_model_id"].startswith("jev-agentic-s1-")


def test_plan_rejects_tampered_export(tmp_path: Path) -> None:
    manifest = exported_manifest(tmp_path)
    train = manifest.parent / "train.jsonl"
    train.write_text(train.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_export(manifest)


def test_remote_inventory_is_get_only_and_reports_collision_state(tmp_path: Path) -> None:
    plan = build_plan(
        exported_manifest(tmp_path),
        epochs=1, lora_rank=8, batch_size_samples=4,
        rate_per_million=3.0,
        pricing_fetched_at_utc="2026-09-22T12:00:00Z",
        tokenizer=FakeTokenizer(),
    )
    client = FakeGetClient()
    inventory = remote_inventory(client, plan)
    assert inventory["collision_free"] is True
    assert inventory["counts"] == {"datasets": 1, "jobs": 0, "models": 0}
    assert client.paths == [
        "/accounts",
        "/accounts/test-account/datasets",
        "/accounts/test-account/supervisedFineTuningJobs",
        "/accounts/test-account/models",
        "/accounts/fireworks/models/qwen3p6-35b-a3b",
    ]


def test_override_is_marked_non_exact_and_write_refuses_drift(tmp_path: Path) -> None:
    plan = build_plan(
        exported_manifest(tmp_path),
        epochs=1, lora_rank=8, batch_size_samples=4,
        rate_per_million=3.0,
        pricing_fetched_at_utc="2026-09-22T12:00:00Z",
        token_count_override=100,
        val_token_count_override=20,
    )
    assert plan["gates"]["exact_tokenization"] is False
    output = tmp_path / "plan.json"
    write_plan(output, plan)
    write_plan(output, plan)
    drifted = json.loads(json.dumps(plan))
    drifted["training"]["epochs"] = 2
    with pytest.raises(SystemExit, match="pass --force"):
        write_plan(output, drifted)
