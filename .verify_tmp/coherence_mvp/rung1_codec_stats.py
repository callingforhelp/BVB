#!/usr/bin/env python3
"""Codec-residual stream (experiment: idea #3 prediction-error signal).

For each corpus_v3 clip, dump per-video-packet {pts_time, size, keyframe}
via ffprobe (no decode). Produces:
  bits[t]      packet size in bits for frame at pts index t
  is_iframe[t] 1 if keyframe
  bits_z[t]    robust z of bits within clip over non-I frames
               (median + MAD, per signal rules)

A break = where the codec's forward predictor fails: P/B frames suddenly
cost many bits, or x264 scene-cut inserts a mid-GOP I-frame.

Output: results/codec_stats/<clip_id>.npz + codec_stats.json summary
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus_v3"
OUT_DIR = ROOT / "results" / "codec_stats"


def packet_stream(mp4: Path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,size,flags",
         "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    pkts = json.loads(p.stdout)["packets"]
    pkts.sort(key=lambda x: float(x["pts_time"]))
    pts = np.array([float(x["pts_time"]) for x in pkts])
    bits = np.array([int(x["size"]) * 8 for x in pkts], dtype=np.float64)
    key = np.array([1 if "K" in x["flags"] else 0 for x in pkts],
                   dtype=np.int8)
    return pts, bits, key


def robust_z(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """z of x computed on subset idx (non-I frames), applied to all."""
    v = x[idx]
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scale = 1.4826 * mad if mad > 0 else 1.0
    z = np.zeros_like(x)
    z[idx] = (x[idx] - med) / scale
    return z


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    clips = manifest["clips"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for i, c in enumerate(clips):
        mp4 = CORPUS / c["path"]
        pts, bits, key = packet_stream(mp4)
        non_i = key == 0
        z = robust_z(bits, non_i)
        np.savez(OUT_DIR / f"{c['id']}.npz",
                 pts=pts, bits=bits, is_iframe=key, bits_z=z)
        mid_iframes = np.where(key[1:] == 1)[0] + 1
        summary[c["id"]] = {
            "operator": c["operator"],
            "breaks_s": c["breaks_s"],
            "n_packets": int(len(bits)),
            "n_mid_iframes": int(len(mid_iframes)),
            "mid_iframe_pts": [round(float(pts[t]), 4) for t in mid_iframes],
            "bits_z_max": round(float(z.max()), 2),
            "bits_z_argmax_pts": round(float(pts[int(z.argmax())]), 4),
        }
        if i % 15 == 0:
            print(f"{i}/{len(clips)} {c['id']}", flush=True)
    (ROOT / "results" / "codec_stats.json").write_text(
        json.dumps(summary, indent=2))
    print(f"wrote {len(summary)} -> results/codec_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
