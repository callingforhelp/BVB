"""Fireworks SFT driver — pure REST, no firectl signin needed.

API: https://api.fireworks.ai  (Bearer FIREWORKS_API_KEY, or
~/.dsh/.credentials.yaml refs.FIREWORKS_API_KEY). The account id is
auto-discovered via GET /v1/accounts.

    python3 ft_fireworks.py upload                  # create + upload all datasets
    python3 ft_fireworks.py train --set all --name jev-s1-all
    python3 ft_fireworks.py train --set loso_09c1414f1b --name jev-s1-loso09
    python3 ft_fireworks.py status <job-id> [--watch]
    python3 ft_fireworks.py jobs

Datasets live in results/ft_fw/*.jsonl (ft_fw_dataset.py). Base model:
accounts/fireworks/models/qwen3p6-35b-a3b (managed LoRA SFT, 262k ctx).
Hyperparams mirror the Tinker runs: loraRank 32, batchSizeSamples 128,
1 epoch. Job-create responses include estimatedCost — printed for review.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "ft_fw"
API = "https://api.fireworks.ai/v1"
BASE_MODEL = "accounts/fireworks/models/qwen3p6-35b-a3b"
LORA_RANK = 32
BATCH_SAMPLES = 128
EPOCHS = 1


def _key() -> str:
    if os.environ.get("FIREWORKS_API_KEY"):
        return os.environ["FIREWORKS_API_KEY"]
    import yaml
    creds = yaml.safe_load(open(Path.home() / ".dsh" / ".credentials.yaml"))
    return creds["refs"]["FIREWORKS_API_KEY"]


def _req(method: str, path: str, body: dict | None = None,
         raw: bytes | None = None, ctype: str | None = None) -> dict:
    url = path if path.startswith("http") else f"{API}{path}"
    headers = {"Authorization": f"Bearer {_key()}"}
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        raise SystemExit(f"{method} {path} -> {e.code}: {detail}")


def account() -> str:
    accts = _req("GET", "/accounts").get("accounts", [])
    if not accts:
        raise SystemExit("no accounts visible to this key")
    return accts[0]["name"].split("/")[-1]


def ds_id(name: str) -> str:
    return "jev-s1-" + name.replace("_", "-")


def upload() -> None:
    acct = account()
    for f in sorted(DATA.glob("*.jsonl")):
        did = ds_id(f.stem)
        n = sum(1 for _ in open(f))
        r = _req("POST", f"/accounts/{acct}/datasets",
                 {"datasetId": did,
                  "dataset": {"userUploaded": {}, "format": "CHAT",
                              "exampleCount": str(n)}})
        state = r.get("state")
        if state not in (None, "READY", "UPLOADING"):
            print(f"  {did}: create returned state={state}")
        # multipart upload
        boundary = uuid.uuid4().hex
        payload = (f"--{boundary}\r\n"
                   f'Content-Disposition: form-data; name="file"; '
                   f'filename="{f.name}"\r\n'
                   f"Content-Type: application/jsonl\r\n\r\n").encode() \
            + f.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        _req("POST", f"/accounts/{acct}/datasets/{did}:upload",
             raw=payload, ctype=f"multipart/form-data; boundary={boundary}")
        print(f"  {did}: uploaded {n} rows")


def train(set_name: str, model_name: str, val: bool = True) -> None:
    acct = account()
    stem = set_name if (DATA / f"{set_name}.jsonl").exists() \
        else f"{set_name}_train"
    job = {
        "baseModel": BASE_MODEL,
        "dataset": f"accounts/{acct}/datasets/{ds_id(stem)}",
        "outputModel": f"accounts/{acct}/models/{model_name}",
        "loraRank": LORA_RANK,
        "batchSizeSamples": BATCH_SAMPLES,
        "epochs": EPOCHS,
    }
    # every round gets an eval set: fold jobs use the true held-out val;
    # 'all' has no fold val so auto-carve a slice out of training data.
    if val and (DATA / f"{set_name}_val.jsonl").exists():
        job["evaluationDataset"] = \
            f"accounts/{acct}/datasets/{ds_id(set_name + '_val')}"
        job["evalAutoCarveout"] = False
    else:
        job["evalAutoCarveout"] = True
    r = _req("POST", f"/accounts/{acct}/supervisedFineTuningJobs", job)
    print(json.dumps(r, indent=1)[:2000])
    est = r.get("supervisedFineTuningJob", {}).get("estimatedCost") \
        or r.get("estimatedCost")
    if est:
        print("estimatedCost:", est)


def status(acct: str | None, job_id: str, watch: bool) -> None:
    acct = acct or account()
    while True:
        r = _req("GET",
                 f"/accounts/{acct}/supervisedFineTuningJobs/{job_id}")
        job = r.get("supervisedFineTuningJob", r)
        st = job.get("state")
        prog = job.get("jobProgress", {})
        print(st, json.dumps(prog)[:300])
        if not watch or st in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                               "SUCCEEDED", "FAILED", "CANCELLED"):
            return
        time.sleep(60)


def jobs() -> None:
    acct = account()
    r = _req("GET", f"/accounts/{acct}/supervisedFineTuningJobs")
    for j in r.get("supervisedFineTuningJobs", []):
        print(j.get("name"), j.get("state"),
              j.get("outputModel", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload")
    tr = sub.add_parser("train")
    tr.add_argument("--set", required=True,
                    help="all | loso_<src> (uses _train/_val files)")
    tr.add_argument("--name", required=True, help="output model id")
    tr.add_argument("--no-val", action="store_true")
    st = sub.add_parser("status"); st.add_argument("job_id")
    st.add_argument("--watch", action="store_true")
    sub.add_parser("jobs")
    a = ap.parse_args()
    if a.cmd == "upload":
        upload()
    elif a.cmd == "train":
        train(a.set, a.name, val=not a.no_val)
    elif a.cmd == "status":
        status(None, a.job_id, a.watch)
    elif a.cmd == "jobs":
        jobs()


if __name__ == "__main__":
    sys.exit(main())
