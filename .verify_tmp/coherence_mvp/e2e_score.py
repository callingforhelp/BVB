#!/usr/bin/env python3
"""End-to-end cascade scoring on corpus_v3.

Detection arms (union):
  A. pairjudge verdicts on photo_jump+orb candidates   (results/pairjudge)
  B. pairjudge verdicts on scenecut-I candidates        (results/pairjudge_scenecut)
  C. recur_frac specialist: raw >= 0.9 = exact replay   (signals_v3)

Rules: candidate frame i = seam between i and i+1; a break is detected
if >=1 discontinuous verdict lands within +-15 frames (0.5s). A
discontinuous verdict not near a break = false positive. No fitted
thresholds — the judge is the decision boundary.

Output: results/e2e_cascade.json + printed table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
FPS = 30.0
TOL = 15
RECUR_T = 0.9   # raw semantic-duplicate threshold (v4 protocol)


def load_verdicts(clip_id: str):
    """[(frame, verdict_dict, gate)] merged across judge files."""
    out = []
    for sub, gate in (("pairjudge", "pix"), ("pairjudge_scenecut", "sc")):
        f = ROOT / "results" / sub / f"{clip_id}.json"
        if not f.exists():
            continue
        for c in json.loads(f.read_text())["candidates"]:
            if "verdict" in c and "continuous" in c["verdict"]:
                out.append((c["frame"], c["verdict"], gate))
    return out


def recur_onsets(clip_id: str) -> list[int]:
    d = np.load(ROOT / "results" / "signals_v3" / f"{clip_id}.npz")
    x = d["recur_frac"].astype(np.float64)
    hot = x >= RECUR_T
    onsets, run = [], 0
    for i, h in enumerate(hot):
        if h and run == 0:
            onsets.append(i)
        run = run + 1 if h else 0
    return onsets


DUP_W, DUP_T = 60, 0.2   # trailing window, dense-dup fraction
ECHO_T = 10.0            # 4-factor echo score; reverse min 21.8, clean max 2.2


def echo_boundaries(clip_id: str) -> list[int]:
    """Reverse arm: 4-factor temporal-echo pair -> both seam positions."""
    f = ROOT / "results" / "echo" / f"{clip_id}.json"
    if not f.exists():
        return []
    r = json.loads(f.read_text())
    if r["best_score"] >= ECHO_T:
        return [int(x) for x in r["best_pair"]]
    return []


def dup_boundaries(clip_id: str) -> list[int]:
    """Timewarp arm: boundary = last frame of a dense-dup region."""
    f = ROOT / "results" / "dupfrac" / f"{clip_id}.npz"
    if not f.exists():
        return []
    d = np.load(f)["dup"].astype(np.float64)
    if len(d) < DUP_W:
        return []
    hot = np.convolve(d, np.ones(DUP_W) / DUP_W, "valid") >= DUP_T
    if not hot.any():
        return []
    # contiguous hot runs: index j covers frames [j, j+DUP_W);
    # boundary = last dup frame inside the run's span (+1, -3 calib.)
    edges = np.where(np.diff(np.concatenate(([0], hot.view(np.int8),
                                             [0]))))[0]
    out = []
    for j0, j1 in zip(edges[::2], edges[1::2]):
        span = d[j0: j1 + DUP_W]
        dups = np.where(span == 1)[0]
        if len(dups):
            out.append(int(j0 + dups[-1]) + 1 - 3)
    return out


def main() -> int:
    manifest = json.loads((ROOT / "corpus_v3" / "manifest.json").read_text())
    table = {}
    detail = {}
    for c in manifest["clips"]:
        cid, op = c["id"], c["operator"]
        bfr = [b * FPS for b in c["breaks_s"]]
        verdicts = load_verdicts(cid)
        disc = [(f + 0.5, v, g) for f, v, g in verdicts
                if v["continuous"] is False]
        onsets = recur_onsets(cid)
        dups = dup_boundaries(cid)
        echos = echo_boundaries(cid)
        # detection per break
        hits = set()
        kinds = []
        for b in bfr:
            near = [(s, v) for s, v, g in disc if abs(s - b) <= TOL]
            if near:
                hits.add(b)
                kinds.append(near[0][1].get("break_kind", "?"))
            elif any(abs(o - b) <= TOL for o in onsets):
                hits.add(b)
                kinds.append("replay(recur)")
            elif any(abs(o - b) <= TOL for o in dups):
                hits.add(b)
                kinds.append("timewarp(dup)")
            elif any(abs(o - b) <= TOL for o in echos):
                hits.add(b)
                kinds.append("reverse(echo)")
        # false positives: detections far from any planted break.
        # echo is a PAIR hypothesis: count once if neither endpoint hits.
        fp_j = sum(1 for s, v, g in disc
                   if all(abs(s - b) > TOL for b in bfr))
        fp_r = sum(1 for o in onsets + dups
                   if all(abs(o - b) > TOL for b in bfr))
        if echos and all(all(abs(o - b) > TOL for b in bfr)
                         for o in echos):
            fp_r += 1
        d = table.setdefault(op, {"breaks": 0, "hit": 0, "fp": 0,
                                  "clips": 0, "clip_flagged": 0,
                                  "kinds": {}})
        d["clips"] += 1
        d["breaks"] += len(bfr)
        d["hit"] += len(hits)
        d["fp"] += fp_j + fp_r
        if hits or fp_j or fp_r:
            d["clip_flagged"] += 1
        for k in kinds:
            d["kinds"][k] = d["kinds"].get(k, 0) + 1
        detail[cid] = {"detected": sorted(hits), "fp": fp_j + fp_r,
                       "n_cand": len(verdicts), "n_disc": len(disc),
                       "recur_onsets": onsets}

    print(f"{'operator':16s} {'recall':>10s} {'FP':>5s} {'clips flagged':>14s}  kinds")
    for op, d in sorted(table.items()):
        print(f"{op:16s} {d['hit']:>4d}/{d['breaks']:<5d} {d['fp']:>5d} "
              f"{d['clip_flagged']:>8d}/{d['clips']:<5d}  {d['kinds']}")
    (ROOT / "results" / "e2e_cascade.json").write_text(json.dumps(
        {"table": table, "detail": detail}, indent=2))
    print("-> results/e2e_cascade.json")
    return 0


if __name__ == "__main__":
    main()
