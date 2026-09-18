#!/usr/bin/env python3
"""Leave-one-source-out Rung-1 evaluation, protocol v3 (frozen)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_v3"
SIG = ROOT / "results" / "signals_v3"
REPORT = ROOT / "results" / "rung1_v3.json"
FPS = 30
TOL_S = 0.5
T_S = 30.0
PAIRS = {
    "splice": "orb_inliers",
    "room_swap": "orb_inliers",
    "reverse_segment": "flow_flip",
    "loop": "recur_frac",
    "timewarp": "motion_step",
}
LOW_IS_BAD = {"orb_inliers"}
OFFSET = {"orb_inliers": 1, "recur_frac": 0, "flow_flip": 30, "motion_step": 30}
ALL_SIGNALS = sorted(set(PAIRS.values()))


def robust_z(x: np.ndarray, low_is_bad: bool) -> np.ndarray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    z = (x - med) / max(mad, 1e-9)
    return -z if low_is_bad else z


def fired_runs(z: np.ndarray, tau: float) -> list[tuple[int, float]]:
    above = z > tau
    runs, t = [], 0
    while t < len(z):
        if not above[t]:
            t += 1
            continue
        end = t
        while end + 1 < len(z) and above[end + 1]:
            end += 1
        seg = z[t:end + 1]
        runs.append((t + int(np.argmax(seg)), float(seg.max())))
        t = end + 1
    return runs


def match(detections: list[int], breaks_f: list[float], off: int) -> tuple[int, int]:
    remaining = sorted(breaks_f)
    hits = 0
    for det_frame, _ in detections:
        det_s = (det_frame + off) / FPS
        best = None
        for b in remaining:
            if abs(det_s - b) <= TOL_S and (best is None or abs(det_s - b) < abs(det_s - best)):
                best = b
        if best is not None:
            hits += 1
            remaining.remove(best)
    return hits, len(detections)


def chance_hit(k: int, b: int) -> float:
    if k == 0 or b == 0:
        return 0.0
    w = 2 * TOL_S / T_S
    return 1.0 - (1.0 - w) ** (k * b)


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    sources = sorted({c["source"] for c in manifest["clips"]})
    zcache: dict[str, dict[str, np.ndarray]] = {}
    for c in manifest["clips"]:
        npz = np.load(SIG / f"{c['id']}.npz")
        zcache[c["id"]] = {n: robust_z(npz[n], n in LOW_IS_BAD) for n in ALL_SIGNALS}

    folds = []
    for held in sources:
        cal = [c["id"] for c in manifest["clips"]
               if c["operator"] == "none" and c["source"] != held]
        assert len(cal) == 12, f"calibration n={len(cal)} != 12"
        taus = {n: max(float(zcache[c][n].max()) for c in cal) for n in ALL_SIGNALS}
        fold = {"held_out": held, "taus": taus, "clips": []}
        for c in manifest["clips"]:
            if c["source"] != held:
                continue
            op = c["operator"]
            entry = {"id": c["id"], "operator": op, "breaks_s": c["breaks_s"]}
            if op in PAIRS:
                sig = PAIRS[op]
                z, tau = zcache[c["id"]][sig], taus[sig]
                runs = fired_runs(z, tau)
                hits, ndet = match(runs, c["breaks_s"], OFFSET[sig])
                entry.update({
                    "check": sig, "tau": tau, "z_max": float(z.max()),
                    "margin": float(z.max() / max(tau, 1e-9)),
                    "n_detections": ndet, "hits": hits,
                    "detections_s": [round((f + OFFSET[sig]) / FPS, 3) for f, _ in runs],
                    "chance_hit": chance_hit(ndet, len(c["breaks_s"])),
                })
            else:
                entry["specificity"] = {
                    n: {"z_max": float(zcache[c["id"]][n].max()), "tau": taus[n],
                        "flagged": bool(zcache[c["id"]][n].max() > taus[n])}
                    for n in ALL_SIGNALS}
            fold["clips"].append(entry)
        folds.append(fold)

    summary = {}
    for op, sig in PAIRS.items():
        sens, margins, hits_t, dets_t, chance, clean_flags = [], [], 0, 0, [], []
        for fold in folds:
            for e in fold["clips"]:
                if e["operator"] == op:
                    sens.append(e["hits"] > 0)
                    if e["hits"] > 0:
                        margins.append(e["margin"])
                    hits_t += e["hits"]
                    dets_t += e["n_detections"]
                    chance.append(e["chance_hit"])
                elif e["operator"] == "none":
                    clean_flags.append(not e["specificity"][sig]["flagged"])
        n_sens = sum(sens)
        summary[f"{sig}__{op}"] = {
            "held_out_clean_specificity": f"{sum(clean_flags)}/15",
            "sensitivity": f"{n_sens}/15",
            "precision": round(hits_t / dets_t, 3) if dets_t else None,
            "recall_breaks": hits_t,
            "n_breaks": sum(len(e["breaks_s"]) for f in folds for e in f["clips"]
                            if e["operator"] == op),
            "median_margin": round(float(np.median(margins)), 3) if margins else None,
            "mean_chance_hit": round(float(np.mean(chance)), 3),
        }

    verdict = {}
    for key, s in summary.items():
        n_sens = int(s["sensitivity"].split("/")[0])
        crit = {"specificity_15of15": s["held_out_clean_specificity"] == "15/15",
                "sensitivity_ge12": n_sens >= 12,
                "precision_ge_0.5": (s["precision"] or 0) >= 0.5,
                "above_chance": (n_sens / 15) > s["mean_chance_hit"],
                "margin_ge_1.5": (s["median_margin"] or 0) >= 1.5}
        verdict[key] = {"pass": all(crit.values()), "criteria": crit}

    REPORT.write_text(json.dumps(
        {"protocol": "RUNG1_PROTOCOL_V3.md", "label": "VALIDATION-HELDOUT",
         "summary": summary, "verdict": verdict, "folds": folds}, indent=2))
    print(json.dumps({"summary": summary, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
