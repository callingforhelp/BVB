"""Read-only Fireworks plan and exact token/cost audit for Agentic S1.

This module cannot create, upload, update, or delete Fireworks resources.  Its
remote surface exposes GET only and is used solely to detect collisions with
the content-addressed IDs proposed by the local export manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

API = "https://api.fireworks.ai/v1"
PLAN_SCHEMA = "agentic-s1-fireworks-plan/v1"
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p6-35b-a3b"
DEFAULT_TOKENIZER = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_EPOCHS = 2
DEFAULT_LORA_RANK = 8
DEFAULT_BATCH_SIZE = 16
DEFAULT_PRICING_SOURCE = "https://fireworks.ai/pricing"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: row must be an object")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"{path}:{lineno}: expected exactly two messages")
        if messages[0].get("role") != "user" or messages[0].get("weight") != 0:
            raise ValueError(f"{path}:{lineno}: user message must have weight 0")
        if messages[1].get("role") != "assistant":
            raise ValueError(f"{path}:{lineno}: final message must be assistant")
        answer = messages[1].get("content")
        if not isinstance(answer, str) or not answer.startswith("frame_"):
            raise ValueError(f"{path}:{lineno}: invalid physical frame answer")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def validate_export(manifest_path: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "agentic-s1-export/v1":
        raise ValueError("unsupported export manifest schema")
    if manifest.get("answer_contract") != "stable_physical_frame_id":
        raise ValueError("manifest does not use stable physical frame IDs")
    if manifest.get("quality_passed") is not True:
        raise ValueError("export quality gate did not pass")
    root = manifest_path.parent
    paths = {name: root / name for name in ("train.jsonl", "val.jsonl")}
    for name, path in paths.items():
        expected = manifest.get("outputs", {}).get(name, {})
        if not path.is_file():
            raise ValueError(f"missing required export file: {path}")
        data = path.read_bytes()
        if sha256_bytes(data) != expected.get("sha256"):
            raise ValueError(f"hash mismatch for {path}")
        if len(data.splitlines()) != expected.get("rows"):
            raise ValueError(f"row-count mismatch for {path}")
    return manifest, paths["train.jsonl"], paths["val.jsonl"]


def clean_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Remove Fireworks-only loss weights before HF chat-template rendering."""
    return [
        {key: value for key, value in message.items() if key != "weight"}
        for message in row["messages"]
    ]


def _token_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("unexpected batched tokenizer result")
        rendered = rendered[0]
    return [int(token) for token in rendered]


def _answer_ids(tokenizer: Any, answer: str) -> list[int]:
    encoded = tokenizer.encode(answer, add_special_tokens=False)
    return [int(token) for token in encoded]


def _stats(lengths: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def audit_tokens(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    rendered_lengths: list[int] = []
    answer_lengths: list[int] = []
    candidate_lengths: list[int] = []
    mixed_candidate_length_rows = 0
    for row in rows:
        messages = clean_messages(row)
        rendered_lengths.append(len(_token_ids(tokenizer, messages)))
        answer_lengths.append(len(_answer_ids(tokenizer, messages[-1]["content"])))
        prompt = messages[0]["content"]
        json_start = prompt.find("{")
        if json_start < 0:
            raise ValueError("frame prompt has no JSON state")
        prompt_state = json.loads(prompt[json_start:])
        choices = [choice["frame_id"] for choice in prompt_state["choices"]]
        if messages[-1]["content"] not in choices:
            raise ValueError("assistant frame ID is absent from prompt choices")
        row_lengths = [len(_answer_ids(tokenizer, choice)) for choice in choices]
        candidate_lengths.extend(row_lengths)
        if len(set(row_lengths)) > 1:
            mixed_candidate_length_rows += 1
    return {
        "rendered": _stats(rendered_lengths),
        "answer": {
            **_stats(answer_lengths),
            "histogram": {
                str(length): count
                for length, count in sorted(Counter(answer_lengths).items())
            },
        },
        "candidate_sequences": {
            **_stats(candidate_lengths),
            "histogram": {
                str(length): count
                for length, count in sorted(Counter(candidate_lengths).items())
            },
            "rows_with_mixed_token_lengths": mixed_candidate_length_rows,
            "raw_sequence_logprob_length_comparable": (
                mixed_candidate_length_rows == 0),
        },
    }


def _credential(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    path = Path.home() / ".dsh" / ".credentials.yaml"
    if not path.is_file():
        return None
    try:
        import yaml
        value = yaml.safe_load(path.read_text()) or {}
        token = value.get("refs", {}).get(name)
        return token if isinstance(token, str) and token else None
    except Exception:
        return None


def resolve_tokenizer_revision(tokenizer_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi(token=_credential("HF_TOKEN")).model_info(tokenizer_id)
    if not isinstance(info.sha, str) or not info.sha:
        raise RuntimeError(f"could not resolve a revision for {tokenizer_id}")
    return info.sha


def load_tokenizer(tokenizer_id: str, revision: str) -> Any:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"revision": revision}
    token = _credential("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)


def tokenizer_identity(tokenizer: Any, requested_id: str,
                       resolved_revision: str | None = None) -> dict[str, Any]:
    init = getattr(tokenizer, "init_kwargs", {}) or {}
    return {
        "requested": requested_id,
        "resolved_name_or_path": getattr(tokenizer, "name_or_path", requested_id),
        "revision": (resolved_revision or init.get("revision")
                     or init.get("_commit_hash")),
        "class": type(tokenizer).__name__,
    }


def resource_ids(
    manifest: Mapping[str, Any], base_model: str, lora_rank: int, epochs: int,
    batch_size_samples: int,
) -> dict[str, Any]:
    outputs = manifest["outputs"]
    run_input = {
        "train_sha256": outputs["train.jsonl"]["sha256"],
        "val_sha256": outputs["val.jsonl"]["sha256"],
        "base_model": base_model,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "batch_size_samples": batch_size_samples,
    }
    suffix = sha256_bytes(canonical_bytes(run_input))[:12]
    return {
        "train_dataset_id": manifest["dataset_ids"]["train"],
        "validation_dataset_id": manifest["dataset_ids"]["val"],
        "job_id": f"jev-agentic-s1-{suffix}",
        "output_model_id": f"jev-agentic-s1-{suffix}",
        "content_suffix": suffix,
    }


def build_plan(
    manifest_path: Path,
    *,
    epochs: int,
    lora_rank: int,
    batch_size_samples: int,
    rate_per_million: float,
    pricing_fetched_at_utc: str,
    tokenizer_id: str = DEFAULT_TOKENIZER,
    tokenizer_revision: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    tokenizer: Any | None = None,
    token_count_override: int | None = None,
    val_token_count_override: int | None = None,
) -> dict[str, Any]:
    if epochs < 1 or lora_rank < 1 or batch_size_samples < 1:
        raise ValueError("epochs, rank, and batch size must be positive")
    if rate_per_million <= 0:
        raise ValueError("rate_per_million must be positive")
    manifest, train_path, val_path = validate_export(manifest_path)
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)

    exact = token_count_override is None
    if exact:
        if tokenizer is None:
            tokenizer_revision = (
                tokenizer_revision or resolve_tokenizer_revision(tokenizer_id))
            tokenizer = load_tokenizer(tokenizer_id, tokenizer_revision)
        train_audit = audit_tokens(train_rows, tokenizer)
        val_audit = audit_tokens(val_rows, tokenizer)
        tokenizer_info = tokenizer_identity(
            tokenizer, tokenizer_id, tokenizer_revision)
    else:
        if token_count_override is None or token_count_override < 1:
            raise ValueError("token_count_override must be positive")
        if val_token_count_override is None or val_token_count_override < 1:
            raise ValueError("val_token_count_override must be positive")
        train_audit = {"rendered": {"count": len(train_rows),
                                    "total": token_count_override},
                       "answer": {"status": "not measured under override"}}
        val_audit = {"rendered": {"count": len(val_rows),
                                  "total": val_token_count_override},
                     "answer": {"status": "not measured under override"}}
        tokenizer_info = {"requested": tokenizer_id,
                          "status": "not loaded; test override"}

    train_tokens = int(train_audit["rendered"]["total"])
    billable_tokens = train_tokens * epochs
    point_cost = billable_tokens / 1_000_000 * rate_per_million
    ids = resource_ids(
        manifest, base_model, lora_rank, epochs, batch_size_samples)
    core = {
        "schema": PLAN_SCHEMA,
        "method": "managed_lora_sft",
        "workflow_path": "managed_rest",
        "target_account_id": None,
        "base_model": base_model,
        "tokenizer": tokenizer_info,
        "answer_contract": manifest["answer_contract"],
        "prompt_version": manifest["prompt_version"],
        "data": {
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "train": {**manifest["outputs"]["train.jsonl"],
                      "rendered_tokens": train_tokens},
            "validation": {**manifest["outputs"]["val.jsonl"],
                           "rendered_tokens": int(val_audit["rendered"]["total"])},
            "split": manifest["split"],
            "tie_policy": manifest["tie_policy"],
        },
        "training": {
            "epochs": epochs,
            "lora_rank": lora_rank,
            "batch_size_samples": batch_size_samples,
            "estimated_optimizer_steps": (
                (len(train_rows) + batch_size_samples - 1)
                // batch_size_samples * epochs),
            "evaluation_dataset_required": True,
            "eval_auto_carveout": False,
            "purpose": "PURPOSE_PILOT",
            "learning_rate": "model-selected platform default",
            "learning_rate_scheduler": "constant after warmup (API default)",
            "learning_rate_warmup_steps": (
                "platform-resolved, unknown before create"),
            "optimizer": "platform-resolved, unknown before create",
            "optimizer_weight_decay": (
                "platform-resolved, unknown before create"),
            "max_context_length": "platform default",
            "observed_max_rendered_tokens": max(
                train_audit["rendered"].get("max", 0),
                val_audit["rendered"].get("max", 0)),
            "wandb": "disabled",
            "reservation_placement": (
                "REST default: try account reservation, then shared fallback"),
        },
        "resources": ids,
        "token_audit": {
            "exact_hf_chat_template": exact,
            "train": train_audit,
            "validation": val_audit,
        },
        "cost": {
            "route": "managed-sft",
            "estimate_type": "planning_range",
            "rate_usd_per_million_tokens": rate_per_million,
            "rate_certainty": "published",
            "usage_certainty": "dataset-derived" if exact else "test-override",
            "pricing_source": DEFAULT_PRICING_SOURCE,
            "pricing_fetched_at_utc": pricing_fetched_at_utc,
            "billable_train_tokens": billable_tokens,
            "point_estimate_usd": round(point_cost, 6),
            "planning_low_usd": round(point_cost * 0.7, 6),
            "planning_high_usd": round(point_cost * 1.3, 6),
            "excludes": [
                "paid evaluation inference",
                "post-training deployment uptime",
                "Hugging Face or GMI hosting",
            ],
        },
        "gates": {
            "local_quality_passed": True,
            "exact_tokenization": exact,
            "remote_inventory_checked": False,
            "paid_mutation_approved": False,
        },
    }
    return refresh_approval_hash(core)


def approval_material(plan: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema", "method", "workflow_path", "target_account_id",
        "base_model", "tokenizer", "answer_contract", "prompt_version",
        "data", "training", "resources", "token_audit", "cost",
    )
    return {field: plan[field] for field in fields}


def refresh_approval_hash(plan: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(plan)
    result["approval_hash"] = sha256_bytes(
        canonical_bytes(approval_material(result)))
    return result


class GetClient(Protocol):
    def get(self, path: str) -> dict[str, Any]: ...


class FireworksReadOnlyClient:
    """Deliberately exposes no mutating HTTP method."""

    def __init__(self, api_key: str, api_base: str = API):
        if not api_key:
            raise ValueError("Fireworks API key is missing")
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")

    def get(self, path: str) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self._api_base}{path}"
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"GET {path} -> {exc.code}: {detail}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"GET {path} returned a non-object response")
        return value


def _collection(client: GetClient, path: str, key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token: str | None = None
    for _ in range(100):
        page_path = path
        if page_token:
            separator = "&" if "?" in path else "?"
            page_path = f"{path}{separator}pageToken={urllib.parse.quote(page_token)}"
        payload = client.get(page_path)
        page = payload.get(key, [])
        if not isinstance(page, list):
            raise RuntimeError(f"GET {page_path} missing list field {key!r}")
        items.extend(item for item in page if isinstance(item, dict))
        token = payload.get("nextPageToken") or payload.get("next_page_token")
        if not token:
            return items
        page_token = str(token)
    raise RuntimeError(f"pagination exceeded 100 pages for {path}")


def _safe_item(item: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "name", "datasetId", "modelId", "state", "displayName", "outputModel",
        "baseModel", "dataset", "evaluationDataset", "loraRank", "epochs",
        "exampleCount", "createTime", "supportsLora", "supervisedLoraTunable",
        "contextLength",
    )
    return {key: item[key] for key in allowed if key in item}


def _matches(items: Sequence[Mapping[str, Any]], identifier: str | None) -> list[dict]:
    if not identifier:
        return []
    matches = []
    for item in items:
        values = [item.get("name"), item.get("datasetId"), item.get("modelId"),
                  item.get("outputModel")]
        if any(value == identifier or (
                isinstance(value, str) and value.rsplit("/", 1)[-1] == identifier)
               for value in values):
            matches.append(_safe_item(item))
    return matches


def remote_inventory(client: GetClient, plan: Mapping[str, Any]) -> dict[str, Any]:
    accounts = _collection(client, "/accounts", "accounts")
    if not accounts:
        raise RuntimeError("no Fireworks account visible to the configured key")
    account_name = accounts[0].get("name")
    if not isinstance(account_name, str) or not account_name:
        raise RuntimeError("Fireworks account response has no name")
    account_id = account_name.rsplit("/", 1)[-1]
    prefix = f"/accounts/{account_id}"
    datasets = _collection(client, f"{prefix}/datasets", "datasets")
    jobs = _collection(
        client, f"{prefix}/supervisedFineTuningJobs",
        "supervisedFineTuningJobs",
    )
    models = _collection(client, f"{prefix}/models", "models")
    base_model = client.get(f"/{plan['base_model']}")
    ids = plan["resources"]
    collisions = {
        "train_dataset": _matches(datasets, ids["train_dataset_id"]),
        "validation_dataset": _matches(
            datasets, ids["validation_dataset_id"]),
        "job": (_matches(jobs, ids["job_id"])
                + _matches(jobs, ids["output_model_id"])),
        "output_model": _matches(models, ids["output_model_id"]),
    }
    return {
        "checked_with": "GET-only Fireworks inventory",
        "account_id": account_id,
        "counts": {
            "datasets": len(datasets), "jobs": len(jobs), "models": len(models),
        },
        "base_model": _safe_item(
            base_model.get("model", base_model)
            if isinstance(base_model.get("model", base_model), Mapping)
            else {}),
        "collisions": collisions,
        "collision_free": not any(collisions.values()),
    }


def write_plan(path: Path, plan: Mapping[str, Any], force: bool = False) -> None:
    data = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() != data and not force:
        raise SystemExit(f"{path} exists with different content; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--batch-size-samples", type=int,
                        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision",
                        help="pin HF tokenizer commit; auto-resolved if omitted")
    parser.add_argument("--rate-per-million", type=float, required=True)
    parser.add_argument("--pricing-fetched-at-utc", required=True)
    parser.add_argument("--remote-inventory", action="store_true")
    parser.add_argument("--api-base", default=API)
    parser.add_argument("--token-count", type=int,
                        help="test-only train-token override; not approval-ready")
    parser.add_argument("--val-token-count", type=int,
                        help="test-only validation-token override")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(
        args.manifest,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
        batch_size_samples=args.batch_size_samples,
        rate_per_million=args.rate_per_million,
        pricing_fetched_at_utc=args.pricing_fetched_at_utc,
        tokenizer_id=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        base_model=args.base_model,
        token_count_override=args.token_count,
        val_token_count_override=args.val_token_count,
    )
    if args.remote_inventory:
        api_key = _credential("FIREWORKS_API_KEY")
        if not api_key:
            raise SystemExit("FIREWORKS_API_KEY is unavailable")
        inventory = remote_inventory(
            FireworksReadOnlyClient(api_key, args.api_base), plan)
        plan["remote_inventory"] = inventory
        plan["target_account_id"] = inventory["account_id"]
        plan["gates"]["remote_inventory_checked"] = True
        plan = refresh_approval_hash(plan)
    write_plan(args.output, plan, args.force)
    print(json.dumps({
        "approval_hash": plan["approval_hash"],
        "output": str(args.output),
        "resources": plan["resources"],
        "cost": plan["cost"],
        "gates": plan["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
