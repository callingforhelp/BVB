#!/usr/bin/env python3
"""Temporal-echo detector for reversed segments.

A reversed segment [pre, rev, post] makes the two seams echo:
  output[s1]   = base[b]   vs output[s2]   = base[b+1]  -> near-identical
  output[s1-1] = base[a-1] vs output[s2-1] = base[a]    -> near-identical

So MSE(f[s1],f[s2]) AND MSE(f[s1-1],f[s2-1]) both collapse to
adjacent-frame level while the seams themselves are discontinuities.

echo(s1,s2) = (T / mse(f[s1],f[s2])) * (T / mse(f[s1-1],f[s2-1])),
  T = 3 * median adjacent-frame MSE of THIS clip (self-anchored).

True pair -> each factor ~3x. False pairs -> at least one factor ~0.
Multiplicative AND-gate per signal rules; median+MAD scale.

No VLM calls. Output: results/echo/<clip>.json + echo_summary.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus_v3"
OUT_DIR = ROOT / "results" / "echo"

MIN_GAP, MAX_GAP = 20, 600   # seams at least 20f apart
T_MULT = 3.0


def gray_frames(mp4: Path) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-vf", "scale=160:120", "-f", "rawvideo",
         "-pix_fmt", "gray", "-"],
        capture_output=True, check=True)
    n = len(p.stdout) // (160 * 120)
    return np.frombuffer(p.stdout, np.uint8,
                         n * 160 * 120).reshape(n, 120, 160)


def all_pairs_mse(F: np.ndarray) -> np.ndarray:
    """F: (N,h,w) -> MSE (N,N) via |a|^2+|b|^2-2ab."""
    X = F.reshape(len(F), -1).astype(np.float32)
    sq = (X * X).sum(axis=1)
    G = X @ X.T
    mse = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0) / X.shape[1]
    return mse


def detect(mp4: Path):
    F = gray_frames(mp4)
    N = len(F)
    M = all_pairs_mse(F)
    adj = np.array([M[i, i + 1] for i in range(N - 1)])
    T = T_MULT * np.median(adj)
    T = max(T, 1e-6)
    # 4-factor AND-gate: the echo pair must ALSO sit on discontinuities.
    #   seam evidence:  M[i-1,i]/T  and  M[j-1,j]/T   (high = real jump)
    #   echo evidence:  T/M[i,j]    and  T/M[i-1,j-1] (high = repeated)
    best = (0.0, -1, -1)
    scores = []
    for i in range(1, N - 1):
        j0, j1 = i + MIN_GAP, min(N - 1, i + MAX_GAP)
        if j0 >= j1:
            continue
        jj = np.arange(j0, j1)
        s = ((M[i - 1, i] / T) * (M[jj - 1, jj] / T)
             * (T / np.maximum(M[i, jj], 1e-6))
             * (T / np.maximum(M[i - 1, jj - 1], 1e-6)))
        k = int(s.argmax())
        scores.append(float(s[k]))
        if s[k] > best[0]:
            best = (float(s[k]), i, int(j0 + k))
    return {"n_frames": N, "T": round(float(T), 2),
            "best_score": round(best[0], 3),
            "best_pair": [best[1], best[2]],
            "median_echo": round(float(np.median(scores)), 3),
            "max_echo": round(float(max(scores)), 3)}


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for i, c in enumerate(manifest["clips"]):
        out = OUT_DIR / f"{c['id']}.json"
        if not out.exists():
            r = detect(CORPUS / c["path"])
            r["breaks_s"] = c["breaks_s"]
            r["operator"] = c["operator"]
            out.write_text(json.dumps(r, indent=2))
        else:
            r = json.loads(out.read_text())
        summary[c["id"]] = r
        if i % 10 == 0:
            print(i, c["id"], flush=True)
    (ROOT / "results" / "echo_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("done -> results/echo_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
