from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_s1_execute import approved_plan, dataset_specs, job_body
from agentic_s1_plan import refresh_approval_hash


def plan(tmp_path: Path) -> tuple[Path, dict]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "train.jsonl").write_text("{}\n")
    (artifact / "val.jsonl").write_text("{}\n")
    value = {
        "schema": "agentic-s1-fireworks-plan/v1",
        "method": "managed_lora_sft",
        "workflow_path": "managed_rest",
        "target_account_id": "test-account",
        "base_model": "accounts/fireworks/models/qwen3p6-35b-a3b",
        "tokenizer": {},
        "answer_contract": "stable_physical_frame_id",
        "prompt_version": "agentic-s1-frame-v3",
        "data": {
            "train": {"rows": 1, "sha256": "train"},
            "validation": {"rows": 1, "sha256": "val"},
        },
        "training": {
            "epochs": 2, "lora_rank": 8, "batch_size_samples": 16,
            "purpose": "omitted; platform default",
        },
        "resources": {
            "train_dataset_id": "train-id",
            "validation_dataset_id": "val-id",
            "job_id": "job-id",
            "output_model_id": "model-id",
        },
        "token_audit": {},
        "cost": {},
        "gates": {
            "local_quality_passed": True,
            "exact_tokenization": True,
            "remote_inventory_checked": True,
            "paid_mutation_approved": False,
        },
    }
    value = refresh_approval_hash(value)
    path = artifact / "fireworks_plan.json"
    path.write_text(json.dumps(value))
    return path, value


def test_approval_hash_and_specs_are_locked(tmp_path: Path) -> None:
    path, value = plan(tmp_path)
    loaded = approved_plan(path, value["approval_hash"])
    specs = dataset_specs(loaded, path)
    assert [item["id"] for item in specs] == ["train-id", "val-id"]
    with pytest.raises(SystemExit, match="supplied approval hash"):
        approved_plan(path, "wrong")
    value["training"]["epochs"] = 3
    path.write_text(json.dumps(value))
    with pytest.raises(SystemExit, match="current contents"):
        approved_plan(path, value["approval_hash"])


def test_job_payload_matches_plan_and_omits_purpose(tmp_path: Path) -> None:
    path, value = plan(tmp_path)
    loaded = approved_plan(path, value["approval_hash"])
    payload = job_body(loaded, "test-account")
    assert payload == {
        "displayName": "job-id",
        "baseModel": "accounts/fireworks/models/qwen3p6-35b-a3b",
        "dataset": "accounts/test-account/datasets/train-id",
        "evaluationDataset": "accounts/test-account/datasets/val-id",
        "outputModel": "accounts/test-account/models/model-id",
        "epochs": 2,
        "loraRank": 8,
        "batchSizeSamples": 16,
        "evalAutoCarveout": False,
        "wandbConfig": {"enabled": False},
    }
    assert "purpose" not in payload
