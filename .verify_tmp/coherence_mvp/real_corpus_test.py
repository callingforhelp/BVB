#!/usr/bin/env python3
"""Real-footage planted-operator test.

Builds 18s@25fps (450f) clips from Li Zong egocentric videos with the
same corruption operators as corpus_v3, then runs the full cascade:
dupfrac + echo + recur + scenecut/photo_jump gate -> pairjudge.

Output: results/realplanted/<clip>.mp4, results/realplanted_report.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "realplanted"
SCRATCH = OUT / "_scratch"
FPS, FRAMES, W = 25, 450, 18.0

sys.path.insert(0, str(Path.home() / "s1-spike"))
sys.path.insert(0, str(ROOT))
from pairjudge import judge_one, robust_z  # noqa: E402
from echo_detect import detect as echo_detect_fn  # noqa: E402
from rung1_codec_stats import packet_stream  # noqa: E402

FFMPEG = "/opt/homebrew/bin/ffmpeg"

SRC = {
    "beverage": ("/Volumes/Timemachine1/Videos/Li Zong/lightwheel标注sample/"
                 "Beverage Sorting and Arrangement/"
                 "060eacbb-50c1-478c-8fe5-e0ec4988ce5b/head_left_camera/"
                 "head_left_camera_raw.mp4"),
    "desk": ("/Volumes/Timemachine1/Videos/Li Zong/lightwheel标注sample/"
             "clear desk trash/039c0668-2428-4a1e-852e-90f1561dc220/"
             "head_left_camera/head_left_camera_raw.mp4"),
}

ENC = ["-vf", "scale=640:480,fps=25,setpts=N/(25*TB)", "-frames:v",
       str(FRAMES), "-c:v", "libx264", "-preset", "veryfast", "-crf",
       "20", "-pix_fmt", "yuv420p", "-an"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])


def cut(src, dst, start, seconds, extra_vf=None, frames=None):
    vf = ["scale=640:480", "fps=25"]
    if extra_vf:
        vf.append(extra_vf)
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", str(start), "-t",
           str(seconds), "-i", src, "-vf", ",".join(vf)]
    if frames:
        cmd += ["-frames:v", str(frames)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an", str(dst)]
    run(cmd)


def concat(parts, dst, tag):
    lst = SCRATCH / f"l_{tag}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), *ENC, str(dst)])


def build():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    clips = []
    for name, src in SRC.items():
        s = lambda t: SCRATCH / f"{name}_{t}.mp4"
        base = s("base"); cut(src, base, 0, W, frames=FRAMES)
        spec = {}
        spec[f"{name}__clean"] = (None, [])
        # splice: jump +4s inside same source at 9s
        a, b = s("sp_a"), s("sp_b")
        cut(base, a, 0, 9); cut(base, b, 13, 9)
        spec[f"{name}__splice"] = ((a, b), [9.0])
        # reverse: [0:7][7:12 rev][12:18]
        a, b, c = s("rv_a"), s("rv_b"), s("rv_c")
        cut(base, a, 0, 7); cut(base, b, 7, 5, "reverse,fps=25")
        cut(base, c, 12, 6)
        spec[f"{name}__reverse"] = ((a, b, c), [7.0, 12.0])
        # loop: [0:7]+[0:7]+[7:11]
        a, b = s("lp_a"), s("lp_b")
        cut(base, a, 0, 7); cut(base, b, 7, 4)
        spec[f"{name}__loop"] = ((a, a, b), [7.0])
        # timewarp: [0:7 @0.5x->14s][7:11 @2x->4s]
        a, b = s("tw_a"), s("tw_b")
        cut(base, a, 0, 7, "setpts=2.0*PTS")
        cut(base, b, 7, 4, "setpts=0.5*PTS")
        spec[f"{name}__timewarp"] = ((a, b), [14.0])
        for cid, (parts, brk) in spec.items():
            dst = OUT / f"{cid}.mp4"
            if parts is None:
                subprocess.run(["cp", str(base), str(dst)], check=True)
            else:
                concat(list(parts), dst, cid)
            clips.append({"id": cid, "src": name, "breaks_s": brk,
                          "path": str(dst)})
    # room_swap: cross-source swap [A 0:7][B 7:12][A 12:18]
    for name, other in (("beverage", "desk"), ("desk", "beverage")):
        s = lambda t: SCRATCH / f"rs_{name}_{t}.mp4"
        a, c = s("a"), s("c")
        baseA = SCRATCH / f"{name}_base.mp4"
        baseB = SCRATCH / f"{other}_base.mp4"
        cut(baseA, a, 0, 7); cut(baseA, c, 12, 6)
        mid = s("m"); cut(baseB, mid, 7, 5)
        dst = OUT / f"{name}__room_swap.mp4"
        concat([a, mid, c], dst, f"rs_{name}")
        clips.append({"id": f"{name}__room_swap", "src": name,
                      "breaks_s": [7.0, 12.0], "path": str(dst)})
    return clips


# ------------------------------------------------- detectors

def dupfrac_curve(mp4, eps=1.0):
    import cv2
    cap = cv2.VideoCapture(str(mp4)); prev = None; d = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),
                       (160, 120)).astype(np.float32)
        d.append(0 if prev is None else float(np.abs(g - prev).mean() < eps))
        prev = g
    cap.release()
    return np.array(d[1:])


def dup_bounds(d, W_=60, T_=0.2):
    if len(d) < W_:
        return []
    hot = np.convolve(d, np.ones(W_) / W_, "valid") >= T_
    edges = np.where(np.diff(np.concatenate(([0], hot.astype(int),
                                             [0]))))[0]
    out = []
    for j0, j1 in zip(edges[::2], edges[1::2]):
        dups = np.where(d[j0:j1 + W_] == 1)[0]
        if len(dups):
            out.append(int(j0 + dups[-1]) + 1 - 3)
    return out


def recur_onsets(mp4, lag_lo=60, lag_hi=400, eps=2.0, min_run=60):
    """exact-replay: a lag-L run where consecutive frames repeat."""
    import cv2
    cap = cv2.VideoCapture(str(mp4)); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                             (160, 120)).astype(np.float32))
    cap.release()
    X = np.stack(fr)
    n = len(X)
    # find lag with longest run of near-identical f[i]==f[i+L]
    best = (0, 0, 0)
    for L in range(lag_lo, min(lag_hi, n - 1)):
        run = 0; on0 = 0
        for i in range(n - L):
            m = float(np.abs(X[i] - X[i + L]).mean())
            if m < eps:
                run += 1
                if run == 1:
                    on0 = i
                if run > best[0]:
                    best = (run, on0, L)
            else:
                run = 0
    if best[0] >= min_run:
        return [best[1] + best[2]]  # onset of replay = start + lag
    return []


def gate_candidates(mp4, pts, keyint_pts):
    sc = [i for i, p in enumerate(pts)
          if 0.15 * 8 < (p % 8) < 7.8 and i > 0]  # rough off-cadence
    import cv2
    cap = cv2.VideoCapture(str(mp4)); prev = None; diffs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120))
        diffs.append(0.0 if prev is None else
                     float(np.abs(g.astype(float) - prev.astype(float)).mean()))
        prev = g
    cap.release()
    z = robust_z(np.array(diffs[1:]))
    pj = [int(i) for i in np.argsort(-z)[:3]]
    return sorted(set(sc[:4]) | set(pj))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    clips = build()
    print(f"built {len(clips)} planted clips")
    results = {}
    jobs = []
    for c in clips:
        mp4 = Path(c["path"])
        pts, bits, key = packet_stream(mp4)
        cand = gate_candidates(mp4, pts, [])
        d = dupfrac_curve(mp4)
        det = {"dup": dup_bounds(d), "recur": recur_onsets(mp4),
               "echo": echo_detect_fn(mp4)}
        for i in cand:
            jobs.append((c["id"], i, mp4))
        results[c["id"]] = {"breaks_s": c["breaks_s"], "cand": cand,
                            "det": det}
    print(f"{len(jobs)} judge calls")
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(judge_one, cid, i, mp4): (cid, i)
                for cid, i, mp4 in jobs}
        for fut in as_completed(futs):
            cid, i = futs[fut]
            results[cid].setdefault("judge", []).append(fut.result())
    (ROOT / "results" / "realplanted_report.json").write_text(
        json.dumps(results, indent=2, default=float))
    print("-> results/realplanted_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
