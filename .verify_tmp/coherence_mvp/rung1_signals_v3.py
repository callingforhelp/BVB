#!/usr/bin/env python3
"""One-pass extraction of all Rung-1 v2 signal curves on corpus_v3 -> npz."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_v3"
OUT = ROOT / "results" / "signals_v3"
SMALL = (160, 120)
LAGS = list(range(60, 301, 30))
WIN = 30


def frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(f, SMALL), cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def photo_jump(fs: list[np.ndarray]) -> np.ndarray:
    a = np.stack([f.astype(np.float32) for f in fs])
    return np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))


def orb_inliers(fs: list[np.ndarray]) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    kps, dess = [], []
    for f in fs:
        k, d = orb.detectAndCompute(f, None)
        kps.append(k)
        dess.append(d)
    out = np.zeros(len(fs) - 1, dtype=np.float64)
    for t in range(1, len(fs)):
        k0, k1, d0, d1 = kps[t - 1], kps[t], dess[t - 1], dess[t]
        if d0 is None or d1 is None or len(d0) < 8 or len(d1) < 8:
            continue
        good = [m for m, n in bf.knnMatch(d0, d1, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        out[t - 1] = float(mask.sum()) if mask is not None else 0.0
    return out


def dense_flow(fs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    meds, mags = [], []
    for t in range(1, len(fs)):
        flow = cv2.calcOpticalFlowFarneback(fs[t - 1], fs[t], None,
                                          0.5, 3, 15, 3, 5, 1.2, 0)
        meds.append(np.median(flow.reshape(-1, 2), axis=0))
        mags.append(np.median(np.linalg.norm(flow, axis=2)))
    return np.array(meds), np.array(mags)


def flow_flip(meds: np.ndarray) -> np.ndarray:
    n = len(meds)
    out = np.zeros(max(0, n - 2 * WIN), dtype=np.float64)
    for i, t in enumerate(range(WIN, n - WIN)):
        a = np.median(meds[t - WIN:t], axis=0)
        b = np.median(meds[t:t + WIN], axis=0)
        den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        out[i] = 1.0 - float(np.dot(a, b)) / den
    return out


def motion_step(mags: np.ndarray) -> np.ndarray:
    n = len(mags)
    out = np.zeros(max(0, n - 2 * WIN), dtype=np.float64)
    for i, t in enumerate(range(WIN, n - WIN)):
        out[i] = abs(float(np.median(mags[t:t + WIN])) - float(np.median(mags[t - WIN:t])))
    return out


def recur_frac(fs: list[np.ndarray]) -> np.ndarray:
    a = np.stack([f.astype(np.int16) for f in fs])
    n = len(fs)
    out = np.zeros(n, dtype=np.float64)
    for lag in LAGS:
        frac = (np.abs(a[lag:] - a[:-lag]) < 8).mean(axis=(1, 2))
        out[lag:] = np.maximum(out[lag:], frac)
    return out


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for clip in manifest["clips"]:
        name = clip["id"]
        if only and only not in name:
            continue
        dst = OUT / f"{name}.npz"
        if dst.exists():
            continue
        fs = frames(CORPUS / clip["path"])
        if len(fs) != 900:
            raise RuntimeError(f"{name}: decoded {len(fs)} frames")
        meds, mags = dense_flow(fs)
        np.savez_compressed(
            dst,
            photo_jump=photo_jump(fs),
            orb_inliers=orb_inliers(fs),
            flow_flip=flow_flip(meds),
            motion_step=motion_step(mags),
            recur_frac=recur_frac(fs),
        )
        print(f"{name}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
