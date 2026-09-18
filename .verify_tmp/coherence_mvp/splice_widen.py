#!/usr/bin/env python3
"""Splice gate-widening: top-8 peaks (was top-3+3 capped 5) on splice
clips only; judges only candidates not already covered by a verdict.

Judge is 0-FP, so recall-optimized gating is free — each extra
candidate costs ~2s and cannot add false breaks.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus_v3"
SIGNALS = ROOT / "results" / "signals_v3"
PJ = ROOT / "results" / "pairjudge"
OUT_DIR = ROOT / "results" / "pairjudge_wide"

sys.path.insert(0, str(Path.home() / "s1-spike"))
sys.path.insert(0, str(ROOT))
from pairjudge import judge_one, robust_z  # noqa: E402

TOPK, NMS, CAP = 8, 5, 10


def nominate_wide(clip_id: str) -> list[int]:
    d = np.load(SIGNALS / f"{clip_id}.npz")
    cands: set[int] = set()
    for key, sign in (("photo_jump", 1.0), ("orb_inliers", -1.0)):
        z = robust_z(sign * d[key].astype(np.float64))
        picked = 0
        for i in np.argsort(-z):
            if all(abs(int(i) - c) > NMS for c in cands):
                cands.add(int(i)); picked += 1
            if picked >= TOPK:
                break
    return sorted(cands)[:CAP]


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    clips = [c for c in manifest["clips"] if c["operator"] == "splice"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in clips:
        out = OUT_DIR / f"{c['id']}.json"
        if out.exists():
            continue
        covered = {x["frame"] for x in
                   json.loads((PJ / f"{c['id']}.json").read_text())
                   ["candidates"]}
        for i in nominate_wide(c["id"]):
            if i not in covered:
                jobs.append((c["id"], i, c["breaks_s"], CORPUS / c["path"]))
    print(f"{len(clips)} splice clips, {len(jobs)} new judgments")
    results: dict[str, list] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(judge_one, cid, i, mp4): (cid, i, brk)
                for cid, i, brk, mp4 in jobs}
        for fut in as_completed(futs):
            cid, i, brk = futs[fut]
            results.setdefault(cid, []).append(fut.result())
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
            (OUT_DIR / f"{cid}.json").write_text(json.dumps(
                {"clip_id": cid, "breaks_s": brk,
                 "candidates": sorted(results[cid],
                                      key=lambda r: r["frame"])},
                indent=2))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
