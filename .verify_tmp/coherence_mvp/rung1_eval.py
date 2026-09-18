#!/usr/bin/env python3
"""Leave-one-source-out Rung-1 evaluation per RUNG1_PROTOCOL.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_v2"
SIGNALS = ROOT / "results" / "signals"
REPORT = ROOT / "results" / "rung1_v1.json"
FPS = 30
TOL_S = 0.5
T_S = 30.0
PAIRS = {
    "splice": "orb_inliers",
    "room_swap": "orb_inliers",
    "reverse_segment": "flow_reversal",
    "loop": "recurrence",
    "timewarp": "motion_scale",
}
LOW_IS_BAD = {"orb_inliers"}  # signals where small values are anomalous


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


def match(detections: list[int], breaks_f: list[float]) -> tuple[int, int]:
    """Greedy one-to-one matching within TOL_S. Returns (hits, n_dets)."""
    remaining = sorted(breaks_f)
    hits = 0
    for det_frame, _ in detections:
        det_s = det_frame / FPS
        best = None
        for b in remaining:
            if abs(det_s - b) <= TOL_S and (best is None or abs(det_s - b) < abs(det_s - best)):
                best = b
        if best is not None:
            hits += 1
            remaining.remove(best)
    return hits, len(detections)


def chance_hit(k: int, b: int) -> float:
    """P(at least one hit) if K detections placed uniformly at random."""
    if k == 0 or b == 0:
        return 0.0
    w = 2 * TOL_S / T_S
    return 1.0 - (1.0 - w) ** (k * b)


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    clips = {c["id"]: c for c in manifest["clips"]}
    sources = sorted({c["source"] for c in manifest["clips"]})
    zcache: dict[str, dict[str, np.ndarray]] = {}
    for cid, c in clips.items():
        npz = np.load(SIGNALS / f"{c['id']}.npz")
        zcache[cid] = {name: robust_z(npz[name], name in LOW_IS_BAD)
                       for name in npz.files}

    folds = []
    for held in sources:
        cal_ids = [c["id"] for c in manifest["clips"]
                   if c["operator"] == "none" and c["source"] != held]
        taus = {name: max(float(zcache[c][name].max()) for c in cal_ids)
                for name in PAIRS.values()}
        fold = {"held_out": held, "taus": taus, "clips": []}
        for c in manifest["clips"]:
            if c["source"] != held:
                continue
            op = c["operator"]
            signal = PAIRS.get(op)
            entry = {"id": c["id"], "operator": op, "breaks_s": c["breaks_s"]}
            if signal is not None:
                z = zcache[c["id"]][signal]
                tau = taus[signal]
                runs = fired_runs(z, tau)
                hits, ndet = match(runs, c["breaks_s"])
                entry.update({
                    "check": signal, "tau": tau, "z_max": float(z.max()),
                    "margin": float(z.max() / max(tau, 1e-9)),
                    "n_detections": ndet, "hits": hits,
                    "detections_s": [round(f / FPS, 3) for f, _ in runs],
                    "chance_hit": chance_hit(ndet, len(c["breaks_s"])),
                })
            else:
                # clean clip: evaluated against every check for specificity
                spec = {}
                for name in PAIRS.values():
                    z = zcache[c["id"]][name]
                    spec[name] = {"z_max": float(z.max()), "tau": taus[name],
                                  "flagged": bool(z.max() > taus[name])}
                entry["specificity"] = spec
            fold["clips"].append(entry)
        folds.append(fold)

    summary = {}
    for op, signal in PAIRS.items():
        sens, margins, hits_t, dets_t, chance = [], [], 0, 0, []
        clean_flags = []
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
                    clean_flags.append(not e["specificity"][signal]["flagged"])
        summary[f"{signal}__{op}"] = {
            "held_out_clean_specificity": f"{sum(clean_flags)}/5",
            "sensitivity": f"{sum(sens)}/5",
            "precision": round(hits_t / dets_t, 3) if dets_t else None,
            "recall_breaks": hits_t,
            "n_breaks": sum(1 for f in folds for e in f["clips"]
                            if e["operator"] == op for _ in e["breaks_s"]),
            "median_margin": round(float(np.median(margins)), 3) if margins else None,
            "mean_chance_hit": round(float(np.mean(chance)), 3),
        }

    verdict = {}
    for key, s in summary.items():
        spec_ok = s["held_out_clean_specificity"] == "5/5"
        sens_ok = int(s["sensitivity"].split("/")[0]) >= 4
        prec_ok = (s["precision"] or 0) >= 0.5
        chance_ok = (s["sensitivity"] != "0/5") and (
            sum(int(s["sensitivity"].split("/")[0]) for _ in [0]) / 5) > s["mean_chance_hit"]
        margin_ok = (s["median_margin"] or 0) >= 1.5
        verdict[key] = {"pass": spec_ok and sens_ok and prec_ok and chance_ok and margin_ok,
                        "criteria": {"specificity_5of5": spec_ok, "sensitivity_ge4": sens_ok,
                                     "precision_ge_0.5": prec_ok, "above_chance": chance_ok,
                                     "margin_ge_1.5": margin_ok}}

    REPORT.write_text(json.dumps(
        {"protocol": "RUNG1_PROTOCOL.md v1", "label": "VALIDATION-HELDOUT",
         "summary": summary, "verdict": verdict, "folds": folds}, indent=2))
    print(json.dumps({"summary": summary, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
