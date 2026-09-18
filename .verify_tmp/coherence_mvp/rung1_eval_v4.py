#!/usr/bin/env python3
"""v4 evaluation: loop pair on raw recur_frac with run-onset detection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_v3"
SIG = ROOT / "results" / "signals_v3"
REPORT = ROOT / "results" / "rung1_v4.json"
FPS = 30
TOL_S = 0.5
T_S = 30.0


def runs_above(x: np.ndarray, tau: float) -> list[tuple[int, int, float]]:
    above = x > tau
    out, t = [], 0
    while t < len(x):
        if not above[t]:
            t += 1
            continue
        end = t
        while end + 1 < len(x) and above[end + 1]:
            end += 1
        out.append((t, end, float(x[t:end + 1].max())))
        t = end + 1
    return out


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    sources = sorted({c["source"] for c in manifest["clips"]})
    curves = {c["id"]: np.load(SIG / f"{c['id']}.npz")["recur_frac"]
              for c in manifest["clips"]}

    folds = []
    for held in sources:
        cal = [c["id"] for c in manifest["clips"]
               if c["operator"] == "none" and c["source"] != held]
        assert len(cal) == 12
        tau = max(float(curves[c].max()) for c in cal)
        fold = {"held_out": held, "tau_raw": tau, "clips": []}
        for c in manifest["clips"]:
            if c["source"] != held or c["operator"] not in ("none", "loop"):
                continue
            x = curves[c["id"]]
            runs = runs_above(x, tau)
            onsets_s = [r[0] / FPS for r in runs]
            entry = {"id": c["id"], "operator": c["operator"],
                     "breaks_s": c["breaks_s"], "max_frac": float(x.max()),
                     "n_runs": len(runs), "onsets_s": [round(s, 3) for s in onsets_s]}
            if c["operator"] == "loop":
                hits = sum(1 for b in c["breaks_s"]
                           if any(abs(s - b) <= TOL_S for s in onsets_s))
                entry.update({"hits": hits,
                              "margin": float(x.max() / max(tau, 1e-9)),
                              "chance_hit": 1 - (1 - 2 * TOL_S / T_S) ** len(runs)})
            else:
                entry["flagged"] = bool(x.max() > tau)
            fold["clips"].append(entry)
        folds.append(fold)

    sens, margins, hits_t, dets_t = [], [], 0, 0
    clean_flags, chance = [], []
    for fold in folds:
        for e in fold["clips"]:
            if e["operator"] == "loop":
                sens.append(e["hits"] > 0)
                margins.append(e["margin"])
                hits_t += e["hits"]
                dets_t += e["n_runs"]
                chance.append(e["chance_hit"])
            else:
                clean_flags.append(not e["flagged"])
    n_sens = sum(sens)
    summary = {
        "held_out_clean_specificity": f"{sum(clean_flags)}/15",
        "sensitivity": f"{n_sens}/15",
        "precision": round(hits_t / dets_t, 3) if dets_t else None,
        "recall_breaks": hits_t, "n_breaks": 15,
        "median_margin": round(float(np.median(margins)), 3),
        "mean_chance_hit": round(float(np.mean(chance)), 3),
    }
    crit = {"specificity_15of15": summary["held_out_clean_specificity"] == "15/15",
            "sensitivity_ge12": n_sens >= 12,
            "precision_ge_0.5": (summary["precision"] or 0) >= 0.5,
            "above_chance": (n_sens / 15) > summary["mean_chance_hit"],
            "margin_ge_1.5": summary["median_margin"] >= 1.5}
    verdict = {"recur_frac__loop": {"pass": all(crit.values()), "criteria": crit}}
    REPORT.write_text(json.dumps(
        {"protocol": "RUNG1_PROTOCOL_V4.md", "label": "VALIDATION-HELDOUT",
         "summary": {"recur_frac__loop": summary}, "verdict": verdict,
         "folds": folds}, indent=2))
    print(json.dumps({"summary": summary, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
