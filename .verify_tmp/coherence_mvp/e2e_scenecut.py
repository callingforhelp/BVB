#!/usr/bin/env python3
"""End-to-end arm: judge scenecut-gate candidates on corpus_v3.

For each clip, non-keyint scenecut I-frames (from codec_stats npz)
become pair-judge candidates. Positions already covered by an existing
pairjudge verdict (+-3 frames) reuse it — no duplicate calls.

Output: results/pairjudge_scenecut/<clip>.json
Run: ~/s1-spike/.venv/bin/python e2e_scenecut.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus_v3"
CODEC = ROOT / "results" / "codec_stats"
PJ = ROOT / "results" / "pairjudge"
OUT_DIR = ROOT / "results" / "pairjudge_scenecut"

sys.path.insert(0, str(Path.home() / "s1-spike"))
sys.path.insert(0, str(ROOT))
from pairjudge import judge_one  # noqa: E402

KEYINT = 250 / 30.0


def scenecut_candidates(clip_id: str) -> list[int]:
    d = np.load(CODEC / f"{clip_id}.npz")
    pts, key = d["pts"], d["is_iframe"]
    iframes = np.where(key[1:] == 1)[0] + 1
    out = []
    for i in iframes:
        phase = pts[i] % KEYINT
        if 0.15 * KEYINT < phase < 0.85 * KEYINT:
            out.append(int(i))
    return out


def already_judged(clip_id: str) -> list[int]:
    f = PJ / f"{clip_id}.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    return [c["frame"] for c in d["candidates"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for c in manifest["clips"]:
        out = OUT_DIR / f"{c['id']}.json"
        if out.exists():
            continue
        covered = already_judged(c["id"])
        for i in scenecut_candidates(c["id"]):
            if all(abs(i - j) > 3 for j in covered):
                jobs.append((c["id"], i, c["breaks_s"], CORPUS / c["path"]))
    print(f"{len(jobs)} new scenecut-gate judgments")

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
            (OUT_DIR / f"{cid}.json").write_text(json.dumps(
                {"clip_id": cid, "breaks_s": brk,
                 "candidates": sorted(results[cid],
                                      key=lambda r: r["frame"])},
                indent=2))
    # write empty results for clips with no new candidates
    for c in manifest["clips"]:
        f = OUT_DIR / f"{c['id']}.json"
        if not f.exists():
            f.write_text(json.dumps(
                {"clip_id": c["id"], "breaks_s": c["breaks_s"],
                 "candidates": []}))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
