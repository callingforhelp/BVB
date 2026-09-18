#!/usr/bin/env python3
"""Real-video clean-arm check: codec-I gate + photo_jump gate -> pair-judge.

For each input video (natural continuous recording):
  1. ffprobe packet stats -> non-keyint I-frames (encoder scenecut detections)
  2. cv2 photo_jump -> top-3 NMS peaks (cheap fallback gate)
  3. pair-judge every candidate -> all should be "continuous" on real footage;
     any "discontinuous" is either a real cut or an FP -> inspect.

Run: ~/s1-spike/.venv/bin/python realvideo_check.py video1.mp4 [video2.mp4 ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / "s1-spike"))
sys.path.insert(0, str(Path(__file__).parent))
from pairjudge import judge_one, robust_z  # noqa: E402

OUT = Path(__file__).parent / "results" / "realvideo"
OUT.mkdir(exist_ok=True)
NMS = 5
TOPK = 3


def packet_stream(mp4: Path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,size,flags",
         "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    pkts = json.loads(p.stdout)["packets"]
    pkts.sort(key=lambda x: float(x["pts_time"]))
    pts = np.array([float(x["pts_time"]) for x in pkts])
    key = np.array([1 if "K" in x["flags"] else 0 for x in pkts])
    return pts, key


def scenecut_iframes(pts, key):
    """Mid-GOP I-frames that aren't on the encoder's regular keyint."""
    iframes = list(np.where(key[1:] == 1)[0] + 1)
    if len(iframes) < 2:
        return [int(i) for i in iframes]  # any mid I-frame is scenecut
    gaps = np.diff(pts[iframes])
    keyint = float(np.median(gaps))       # regular cadence estimate
    out = []
    for i in iframes:
        # off-cadence: not near a multiple of keyint from the first I-frame
        phase = (pts[i] - pts[iframes[0]]) % keyint
        if 0.15 * keyint < phase < 0.85 * keyint:
            out.append(int(i))
    return out


def photo_jump_candidates(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    prev = None
    diffs = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 120))
        diffs.append(0.0 if prev is None else float(np.abs(g.astype(float) - prev.astype(float)).mean()))
        prev = g
    cap.release()
    diffs = np.array(diffs[1:])
    z = robust_z(diffs)
    order = np.argsort(-z)
    picked = []
    for i in order:
        if all(abs(int(i) - c) > NMS for c in picked):
            picked.append(int(i))
        if len(picked) >= TOPK:
            break
    return picked, z


def main() -> int:
    videos = [Path(a) for a in sys.argv[1:]]
    for v in videos:
        pts, key = packet_stream(v)
        sc = scenecut_iframes(pts, key)
        pj, z = photo_jump_candidates(v)
        cand = sorted(set(sc) | set(pj))
        verdicts = []
        for i in cand:
            r = judge_one(v.stem, i, v)
            verdicts.append(r)
            vj = r.get("verdict", {})
            print(f"{v.name[:46]:46s} f{i:5d} "
                  f"{'I' if i in sc else '-'}{'P' if i in pj else '-'} "
                  f"{str(vj)[:110]}", flush=True)
        (OUT / f"{v.stem}.json").write_text(json.dumps(
            {"video": str(v), "candidates": cand, "verdicts": verdicts},
            indent=2, ensure_ascii=False))
        print(f"-> {OUT / (v.stem + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
