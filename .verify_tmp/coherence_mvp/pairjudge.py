#!/usr/bin/env python3
"""Pairwise comparative judging (experiment: idea #2).

Pipeline per corpus_v3 clip:
  1. Cheap gate nominates candidate seams from signals_v3 curves:
     union of top-3 photo_jump peaks + top-3 orb-drop peaks,
     non-max suppressed at +-5 frames, capped at 5, plus one
     uniformly-random control transition -> <=6 candidates/clip.
  2. Extract 4-frame strip around each candidate seam
     (i-1, i | i+1, i+2 — the seam sits between i and i+1).
  3. VLM binary pair-verdict via s1-spike provider chain
     (ark-plan -> oc-vision-exp -> zenmux), JSON out:
     {"continuous": bool, "same_scene": bool,
      "break_kind": str, "confidence": float}
  4. Save per-clip verdicts to results/pairjudge/<clip>.json.

The decisive measurement: does the VLM say "continuous" on the gate's
false alarms (clean clips) while confirming real breaks?

Run from s1-spike venv:
  ~/s1-spike/.venv/bin/python pairjudge.py [--workers 10] [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SPIKE = Path.home() / "s1-spike"
sys.path.insert(0, str(SPIKE))

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus_v3"
SIGNALS = ROOT / "results" / "signals_v3"
OUT_DIR = ROOT / "results" / "pairjudge"

FPS = 30.0
TOPK = 3
NMS_HALF = 5
MAX_CANDIDATES = 5


# ------------------------------------------------------------- gate

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 0 else 1.0
    return (x - med) / scale


def nominate(clip_id: str) -> list[int]:
    """Candidate seam indices i (transition between frame i and i+1)."""
    d = np.load(SIGNALS / f"{clip_id}.npz")
    cands: set[int] = set()
    for key, sign in (("photo_jump", 1.0), ("orb_inliers", -1.0)):
        x = sign * d[key].astype(np.float64)      # orb DROP = evidence
        z = robust_z(x)
        order = np.argsort(-z)
        picked = 0
        for i in order:
            if all(abs(int(i) - c) > NMS_HALF for c in cands):
                cands.add(int(i))
                picked += 1
            if picked >= TOPK:
                break
    cands = sorted(cands)[:MAX_CANDIDATES]
    rng = np.random.RandomState(hash(clip_id) & 0xFFFF)
    ctrl = int(rng.randint(30, 870))
    if all(abs(ctrl - c) > NMS_HALF for c in cands):
        cands.append(ctrl)
    return sorted(cands)


# ------------------------------------------------------------- frames

def extract_strip(mp4: Path, i: int) -> list[tuple[str, str]]:
    """4 frames around seam i|i+1 -> [(label,b64)..] 480w JPEG."""
    idxs = [i - 1, i, i + 1, i + 2]
    idxs = [max(0, min(899, t)) for t in idxs]
    sel = "+".join(f"eq(n,{t})" for t in idxs)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-vf", f"select='{sel}',scale=480:-1",
         "-f", "image2pipe", "-vcodec", "mjpeg",
         "-q:v", "5", "-"],
        capture_output=True, check=True)
    blobs = split_jpegs(p.stdout)
    labels = ["A1", "A2", "B1", "B2"]
    return [(l, base64.b64encode(b).decode())
            for l, b in zip(labels, blobs)][:4]


def split_jpegs(buf: bytes) -> list[bytes]:
    out, start = [], None
    i = 0
    while i < len(buf) - 1:
        if buf[i] == 0xFF and buf[i + 1] == 0xD8:
            start = i
        elif buf[i] == 0xFF and buf[i + 1] == 0xD9 and start is not None:
            out.append(buf[start:i + 2])
            start = None
        i += 1
    return out


# ------------------------------------------------------------- judge

PROMPT = """You are shown 4 frames from one video, in order A1, A2, B1, B2.
A1 and A2 are consecutive frames immediately BEFORE a candidate point.
B1 and B2 are consecutive frames immediately AFTER it.

Question: is the A2->B1 transition a normal continuation of the same
continuous footage, or a discontinuity?

A discontinuity = scene/room change, content jump-cut, inserted foreign
footage, sudden time skip, or replayed/duplicated material restarting.
Natural camera motion, object motion, and fast pans are CONTINUOUS.

Reply with ONLY a JSON object:
{"continuous": true/false, "same_scene": true/false,
 "break_kind": "none"|"scene_change"|"temporal_jump"|"replay"|"other",
 "confidence": 0.0-1.0}"""


def judge_one(clip_id: str, cand_i: int, mp4: Path) -> dict:
    from routes import ROUTES, OC_SESSION
    from seg_spike import LabeledZenmuxProvider, FailoverVisionProvider
    from system_one_adapter.providers.base import Message

    strip = extract_strip(mp4, cand_i)
    if len(strip) < 4:
        return {"frame": cand_i, "error": f"strip={len(strip)}"}
    providers = []
    # pair-judge is a bounded binary task: fast routes first
    # (ark-plan measured ~200s/call; zenmux 1.8s, oc-vision-exp 2.6s)
    order = ["zenmux", "oc-vision-exp", "ark-plan"]
    for name in order:
        cfg = ROUTES[name]
        if not cfg.get("vision"):
            continue
        providers.append(LabeledZenmuxProvider(
            api_key=cfg["key"](), labeled_frames=strip,
            model_name=cfg["model"], base_url=cfg["url"],
            send_thinking=cfg["thinking"],
            extra_headers=cfg["headers"]() if cfg.get("headers") else None))
    provider = FailoverVisionProvider(providers)
    t0 = time.time()
    try:
        res = provider.request(
            [Message(role="user", content=PROMPT)],
            schema={}, structured=False)
        dt = time.time() - t0
        m = re.search(r"\{[^{}]*\}", res.text, re.S)
        verdict = json.loads(m.group(0)) if m else {"parse_error": res.text[:200]}
        return {"frame": cand_i, "verdict": verdict,
                "route": provider.model_name, "secs": round(dt, 1)}
    except Exception as e:  # all routes failed
        return {"frame": cand_i, "error": f"{type(e).__name__}: {e}",
                "secs": round(time.time() - t0, 1)}


# ------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    clips = manifest["clips"]
    if args.only:
        clips = [c for c in clips if args.only in c["id"]]
    if args.limit:
        clips = clips[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for c in clips:
        out = OUT_DIR / f"{c['id']}.json"
        if out.exists():
            continue
        for i in nominate(c["id"]):
            jobs.append((c["id"], i, c["breaks_s"], CORPUS / c["path"]))
    print(f"{len(clips)} clips, {len(jobs)} candidate judgments")

    results: dict[str, list] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, cid, i, mp4): (cid, i, brk)
                for cid, i, brk, mp4 in jobs}
        for fut in as_completed(futs):
            cid, i, brk = futs[fut]
            r = fut.result()
            results.setdefault(cid, []).append(r)
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
            # incremental flush
            (OUT_DIR / f"{cid}.json").write_text(json.dumps(
                {"clip_id": cid, "breaks_s": brk,
                 "candidates": sorted(results[cid],
                                      key=lambda r: r["frame"])},
                indent=2))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
