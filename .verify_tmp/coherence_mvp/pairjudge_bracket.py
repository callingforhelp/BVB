#!/usr/bin/env python3
"""Signal-bracket pairjudge pass (Fix B): nominate candidates at
signal-predicted seam positions that no existing pass covered, and judge
them with a dense strip (TowerH annotation_boundary_review pattern:
+/-1 s of context instead of 4 adjacent frames = 0.13 s).

Signal positions: scenecut I-frames, dup_dense_last_frame, echo_best_pair
endpoints (the last-original / first-resumed frames of a swapped segment
echo each other — for several room_swaps this pair IS the seam pair).

Output: results/pairjudge_bracket/<cid>.json — same schema as pairjudge.

  ~/s1-spike/.venv/bin/python pairjudge_bracket.py [--workers 8] [--only X]
"""
import argparse
import base64
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from jev_loop import _seam_positions
from pairjudge import judge_one, split_jpegs

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_v3"
RES = ROOT / "results"
OUT_DIR = RES / "pairjudge_bracket"
COVER_TOL = 10       # a signal position is covered if a candidate is within
MAX_TARGETS = 4      # bounded nominations per clip

# dense strip: 12 frames spanning +/-1 s around the seam (seam between
# the -1 and +1 offset frames)
OFFSETS = [-30, -24, -18, -12, -6, -1, 1, 6, 12, 18, 24, 30]
LABELS = [f"t{o/30:+.2f}s" for o in OFFSETS]

PROMPT = """You are shown 12 frames from one video in time order, spanning
+/-1 second around a candidate boundary point. Labels give each frame's
time offset relative to the candidate boundary (negative = before,
positive = after). The boundary itself lies between t-0.03s and t+0.03s.

Question: is there a discontinuity at the boundary — scene/room change,
content jump-cut, inserted foreign footage, sudden time skip, or
replayed/duplicated material restarting?

Judge with the FULL 2 seconds of context: a real scene change is visible
across the span even when the two frames nearest the boundary happen to
look similar (e.g. the camera returned to a similar-looking view).
Natural camera motion, object motion, and fast pans are CONTINUOUS.

Reply with ONLY a JSON object:
{"continuous": true/false, "same_scene": true/false,
 "break_kind": "none"|"scene_change"|"temporal_jump"|"replay"|"other",
 "confidence": 0.0-1.0}"""


def free_signals(cid: str) -> dict:
    sig = np.load(RES / "signals_v3" / f"{cid}.npz")
    d = np.asarray(np.load(RES / "dupfrac" / f"{cid}.npz")["dup"],
                   dtype=np.float64)
    dense = np.convolve(d, np.ones(60), "same") / 60 >= 0.2
    codec = np.load(RES / "codec_stats" / f"{cid}.npz")
    pts, key = codec["pts"], codec["is_iframe"]
    from jev_loop import KEYINT
    scenecut = [int(i) for i in np.where(key[1:] == 1)[0] + 1
                if 0.15 * KEYINT < pts[i] % KEYINT < 0.85 * KEYINT]
    echo = json.loads((RES / "echo" / f"{cid}.json").read_text())
    return {"scenecut_iframe_frames": scenecut,
            "dup_dense_last_frame":
                int(np.where(dense)[0].max()) if dense.any() else None,
            "echo_best_pair": echo.get("best_pair")}


def covered_frames(cid: str) -> set:
    fr = set()
    for d in ("pairjudge", "pairjudge_scenecut", "pairjudge_wide"):
        f = RES / d / f"{cid}.json"
        if f.exists():
            fr.update(c["frame"] for c in
                      json.loads(f.read_text()).get("candidates", []))
    return fr


def targets_for(cid: str) -> list[int]:
    pos = _seam_positions(free_signals(cid))
    cov = covered_frames(cid)
    out = [p for p in sorted(pos)
           if 4 < p < 895 and
           all(abs(p - c) > COVER_TOL for c in cov)]
    return out[:MAX_TARGETS]


def extract_dense(mp4: Path, i: int) -> list[tuple[str, str]]:
    """12 frames at OFFSETS around seam i -> [(label,b64)..] 480w JPEG."""
    idxs = [max(0, min(899, i + o)) for o in OFFSETS]
    sel = "+".join(f"eq(n,{t})" for t in idxs)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-vf", f"select='{sel}',scale=480:-1",
         "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "-"],
        capture_output=True, check=True)
    blobs = split_jpegs(p.stdout)
    return [(l, base64.b64encode(b).decode())
            for l, b in zip(LABELS, blobs)][:len(OFFSETS)]


def judge_dense(clip_id: str, cand_i: int, mp4: Path) -> dict:
    # NOTE: a 12-frame dense strip was tried and REJECTED — the same model
    # that calls scene_change conf 1.0 on a focused 4-frame strip answers
    # "continuous" on the 12-frame version (attention dilutes across the
    # span; the one frame-pair that matters averages out). Reverted to the
    # proven 4-frame strip; the value of this pass is the signal-predicted
    # POSITIONS, not a wider view.
    r = judge_one(clip_id, cand_i, mp4)
    r["gate"], r["strip"] = "signal_seam", "std4"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    clips = manifest["clips"]
    if args.only:
        clips = [c for c in clips if args.only in c["id"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for c in clips:
        out = OUT_DIR / f"{c['id']}.json"
        if out.exists():
            continue
        tgts = targets_for(c["id"])
        if not tgts:
            out.write_text(json.dumps(
                {"clip_id": c["id"], "breaks_s": c["breaks_s"],
                 "candidates": []}, indent=2))
            continue
        for i in tgts:
            jobs.append((c["id"], i, c["breaks_s"], CORPUS / c["path"]))
    print(f"{len(clips)} clips, {len(jobs)} bracket judgments")
    if args.dry_run:
        from collections import Counter
        for cid, i, _, _ in jobs:
            print(f"  {cid}: frame {i}")
        return 0

    results: dict[str, list] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_dense, cid, i, mp4): (cid, i, brk)
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
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
