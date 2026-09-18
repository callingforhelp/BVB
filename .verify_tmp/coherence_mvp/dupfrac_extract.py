#!/usr/bin/env python3
"""Extract adjacent-frame near-duplicate indicator per corpus_v3 clip.

dup[i] = 1 if mean-abs-diff between frame i and i+1 < EPS.
Detects slowdown-by-frame-duplication (timewarp slow arm): the region
carries ~50% adjacent dups; clean footage <=1%.

Output: results/dupfrac/<clip>.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus_v3"
OUT_DIR = ROOT / "results" / "dupfrac"
EPS = 1.0   # mean-abs-diff on 160x120 gray, matches probe calibration


def curve(mp4: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    prev = None
    d = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),
                       (160, 120)).astype(np.float32)
        d.append(0.0 if prev is None else
                 float(np.abs(g - prev).mean() < EPS))
        prev = g
    cap.release()
    return np.array(d[1:], dtype=np.float32)


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(manifest["clips"]):
        out = OUT_DIR / f"{c['id']}.npz"
        if out.exists():
            continue
        np.savez(out, dup=curve(CORPUS / c["path"]))
        if i % 15 == 0:
            print(i, c["id"], flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
