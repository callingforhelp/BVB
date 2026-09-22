"""Approval-hash-locked Fireworks executor for Agentic S1 Managed SFT.

The executor has three explicit phases: ``preflight`` performs GET-only
collision checks, ``upload`` creates and validates the two immutable datasets,
and ``train`` creates or reconciles one caller-named SFT job.  It refuses any
plan whose approval material no longer hashes to the supplied approval hash.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentic_s1_plan import (
    API,
    _credential,
    load_json,
    refresh_approval_hash,
    remote_inventory,
)

TERMINAL_DATASET_STATES = {"READY"}
TERMINAL_JOB_STATES = {
    "JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
    "COMPLETED", "FAILED", "CANCELLED",
}


class FireworksHttpError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, detail: str):
        super().__init__(f"{method} {path} -> {status}: {detail}")
        self.status = status


class FireworksClient:
    def __init__(self, api_key: str, *, api_base: str = API,
                 session_id: str | None = None):
        if not api_key:
            raise ValueError("Fireworks API key is missing")
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._session_id = session_id

    def _request(self, method: str, path: str, *, body: dict | None = None,
                 raw: bytes | None = None, content_type: str | None = None,
                 allow_not_found: bool = False) -> dict[str, Any] | None:
        url = path if path.startswith("http") else f"{self._api_base}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._session_id:
            headers["X-Fireworks-Session-Id"] = self._session_id
        data = raw
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        elif content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            if allow_not_found and exc.code == 404:
                return None
            raise FireworksHttpError(method, path, exc.code, detail) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"{method} {path} returned a non-object response")
        return value

    def get(self, path: str) -> dict[str, Any]:
        value = self._request("GET", path)
        assert value is not None
        return value

    def get_optional(self, path: str) -> dict[str, Any] | None:
        return self._request("GET", path, allow_not_found=True)

    def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        value = self._request("POST", path, body=body)
        assert value is not None
        return value

    def post_file(self, path: str, file_path: Path) -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{file_path.name}"\r\n'
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode()
        payload = prefix + file_path.read_bytes() + (
            f"\r\n--{boundary}--\r\n").encode()
        value = self._request(
            "POST", path, raw=payload,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        assert value is not None
        return value


def approved_plan(plan_path: Path, approval_hash: str) -> dict[str, Any]:
    plan = load_json(plan_path)
    actual = plan.get("approval_hash")
    recomputed = refresh_approval_hash(plan).get("approval_hash")
    if actual != recomputed:
        raise SystemExit("plan approval hash does not match its current contents")
    if approval_hash != actual:
        raise SystemExit(
            f"supplied approval hash does not match plan: {approval_hash}")
    gates = plan.get("gates", {})
    for gate in ("local_quality_passed", "exact_tokenization",
                 "remote_inventory_checked"):
        if gates.get(gate) is not True:
            raise SystemExit(f"required plan gate is not true: {gate}")
    if plan.get("training", {}).get("purpose") != "omitted; platform default":
        raise SystemExit("the approved plan must omit the Fireworks purpose field")
    return plan


def dataset_specs(plan: Mapping[str, Any], plan_path: Path) -> list[dict[str, Any]]:
    root = plan_path.parent
    resources = plan["resources"]
    data = plan["data"]
    return [
        {
            "id": resources["train_dataset_id"],
            "path": root / "train.jsonl",
            "rows": int(data["train"]["rows"]),
            "sha256": data["train"]["sha256"],
        },
        {
            "id": resources["validation_dataset_id"],
            "path": root / "val.jsonl",
            "rows": int(data["validation"]["rows"]),
            "sha256": data["validation"]["sha256"],
        },
    ]


def job_body(plan: Mapping[str, Any], account_id: str) -> dict[str, Any]:
    resources = plan["resources"]
    training = plan["training"]
    prefix = f"accounts/{account_id}"
    return {
        "displayName": resources["job_id"],
        "baseModel": plan["base_model"],
        "dataset": f"{prefix}/datasets/{resources['train_dataset_id']}",
        "evaluationDataset": (
            f"{prefix}/datasets/{resources['validation_dataset_id']}"),
        "outputModel": f"{prefix}/models/{resources['output_model_id']}",
        "epochs": training["epochs"],
        "loraRank": training["lora_rank"],
        "batchSizeSamples": training["batch_size_samples"],
        "evalAutoCarveout": False,
        "wandbConfig": {"enabled": False},
    }


def _unwrap(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, payload)
    if not isinstance(value, dict):
        raise RuntimeError(f"response field {key!r} is not an object")
    return dict(value)


def _verify_existing_dataset(dataset: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    count = dataset.get("exampleCount")
    if count is not None and int(count) != spec["rows"]:
        raise RuntimeError(
            f"existing dataset {spec['id']} has {count} rows; expected {spec['rows']}")
    fmt = dataset.get("format")
    if fmt not in (None, "CHAT"):
        raise RuntimeError(
            f"existing dataset {spec['id']} has format {fmt!r}; expected CHAT")


def wait_dataset(client: FireworksClient, account_id: str,
                 spec: Mapping[str, Any], *, timeout: int = 600,
                 interval: int = 5) -> dict[str, Any]:
    path = f"/accounts/{account_id}/datasets/{spec['id']}"
    deadline = time.monotonic() + timeout
    while True:
        dataset = _unwrap(client.get(path), "dataset")
        _verify_existing_dataset(dataset, spec)
        state = dataset.get("state")
        if state in TERMINAL_DATASET_STATES:
            return dataset
        status = dataset.get("status") or {}
        if isinstance(status, dict) and status.get("code") not in (None, "OK"):
            raise RuntimeError(
                f"dataset {spec['id']} failed validation: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"dataset {spec['id']} did not reach READY within {timeout}s")
        time.sleep(interval)


def upload_datasets(client: FireworksClient, plan: Mapping[str, Any],
                    plan_path: Path) -> dict[str, Any]:
    account_id = str(plan["target_account_id"])
    results: list[dict[str, Any]] = []
    for spec in dataset_specs(plan, plan_path):
        resource_path = f"/accounts/{account_id}/datasets/{spec['id']}"
        existing_payload = client.get_optional(resource_path)
        if existing_payload is not None:
            existing = _unwrap(existing_payload, "dataset")
            _verify_existing_dataset(existing, spec)
            if existing.get("state") != "READY":
                existing = wait_dataset(client, account_id, spec)
            results.append({
                "id": spec["id"], "action": "reused",
                "state": existing.get("state"), "rows": spec["rows"],
            })
            continue
        client.post_json(
            f"/accounts/{account_id}/datasets",
            {
                "datasetId": spec["id"],
                "dataset": {
                    "displayName": spec["id"],
                    "exampleCount": str(spec["rows"]),
                    "userUploaded": {},
                    "format": "CHAT",
                },
            },
        )
        client.post_file(f"{resource_path}:upload", spec["path"])
        ready = wait_dataset(client, account_id, spec)
        results.append({
            "id": spec["id"], "action": "created",
            "state": ready.get("state"), "rows": spec["rows"],
        })
    return {"account_id": account_id, "datasets": results}


def _money(estimated: Mapping[str, Any] | None) -> float | None:
    if not isinstance(estimated, Mapping):
        return None
    units = int(estimated.get("units") or 0)
    nanos = int(estimated.get("nanos") or 0)
    return units + nanos / 1_000_000_000


def _verify_existing_job(job: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    fields = (
        "baseModel", "dataset", "evaluationDataset", "outputModel", "epochs",
        "loraRank", "batchSizeSamples", "evalAutoCarveout",
    )
    mismatches = {
        key: {"actual": job.get(key), "expected": expected.get(key)}
        for key in fields if job.get(key) != expected.get(key)
    }
    if mismatches:
        raise RuntimeError(f"existing job does not match approved plan: {mismatches}")


def create_job(client: FireworksClient, plan: Mapping[str, Any]) -> dict[str, Any]:
    account_id = str(plan["target_account_id"])
    resources = plan["resources"]
    job_id = resources["job_id"]
    path = f"/accounts/{account_id}/supervisedFineTuningJobs/{job_id}"
    expected = job_body(plan, account_id)
    existing_payload = client.get_optional(path)
    if existing_payload is not None:
        job = _unwrap(existing_payload, "supervisedFineTuningJob")
        _verify_existing_job(job, expected)
        action = "reused"
    else:
        query_id = urllib.parse.quote(job_id, safe="")
        created = client.post_json(
            f"/accounts/{account_id}/supervisedFineTuningJobs"
            f"?supervisedFineTuningJobId={query_id}",
            expected,
        )
        job = _unwrap(created, "supervisedFineTuningJob")
        _verify_existing_job(job, expected)
        action = "created"
    return {
        "account_id": account_id,
        "action": action,
        "job_id": job_id,
        "name": job.get("name"),
        "state": job.get("state"),
        "estimated_cost_usd": _money(job.get("estimatedCost")),
        "job_progress": job.get("jobProgress"),
    }


def job_status(client: FireworksClient, plan: Mapping[str, Any]) -> dict[str, Any]:
    account_id = str(plan["target_account_id"])
    job_id = plan["resources"]["job_id"]
    payload = client.get(
        f"/accounts/{account_id}/supervisedFineTuningJobs/{job_id}")
    job = _unwrap(payload, "supervisedFineTuningJob")
    return {
        "account_id": account_id,
        "job_id": job_id,
        "name": job.get("name"),
        "state": job.get("state"),
        "status": job.get("status"),
        "estimated_cost_usd": _money(job.get("estimatedCost")),
        "job_progress": job.get("jobProgress"),
        "create_time": job.get("createTime"),
        "completed_time": job.get("completedTime"),
        "output_model": job.get("outputModel"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("preflight", "upload", "train", "status"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approval-hash", required=True)
    parser.add_argument("--api-base", default=API)
    parser.add_argument("--session-id")
    args = parser.parse_args(argv)

    plan = approved_plan(args.plan, args.approval_hash)
    key = _credential("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY is unavailable")
    client = FireworksClient(
        key, api_base=args.api_base, session_id=args.session_id)
    if args.command == "preflight":
        value = remote_inventory(client, plan)
    elif args.command == "upload":
        value = upload_datasets(client, plan, args.plan)
    elif args.command == "train":
        value = create_job(client, plan)
    else:
        value = job_status(client, plan)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
