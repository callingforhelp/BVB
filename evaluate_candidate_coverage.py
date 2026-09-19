#!/usr/bin/env python3
"""Evaluate Flinter's cheap candidate proposal stage on the existing corpus.

This is intentionally a proposal-coverage experiment, not a Jev or VLM result.
It answers: do cheap signals put a candidate close to the known seam?  The
manifest is used only for evaluation; it is never supplied to the proposer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from flinter_mvp import generate_candidates

SIGNAL_DIR = "results/signals_v3"
MANIFEST = "corpus_v3/manifest.json"


def robust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values, [5, 95])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def local_peaks(values: np.ndarray, higher_is_better: bool) -> list[tuple[float, int]]:
    score = robust(values if higher_is_better else -values)
    peaks: list[tuple[float, int]] = []
    for i in range(1, len(score) - 1):
        if score[i] >= score[i - 1] and score[i] >= score[i + 1] and score[i] > 0.55:
            peaks.append((float(score[i]), i))
    return sorted(peaks, reverse=True)


def propose(npz_path: Path, max_candidates: int) -> list[float]:
    data = np.load(npz_path)
    peaks: dict[int, float] = {}
    specs = {
        "photo_jump": True,
        "orb_inliers": False,
        "flow_flip": True,
        "motion_step": True,
        "recur_frac": True,
    }
    for name, high in specs.items():
        if name not in data:
            continue
        for score, idx in local_peaks(data[name], high)[:max_candidates * 2]:
            peaks[idx] = max(peaks.get(idx, 0.0), score)
    # Existing signals are sampled at 30fps.  Keep this conversion local to the
    # proposal adapter; the core only operates on seconds.
    ranked = sorted(peaks.items(), key=lambda pair: (-pair[1], pair[0]))
    return [idx / 30.0 for idx, _ in ranked[:max_candidates]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=2 / 30)
    parser.add_argument("--output", type=Path, default=Path("results/flinter_candidate_coverage.json"))
    args = parser.parse_args()

    manifest = json.loads((args.root / MANIFEST).read_text())
    rows = []
    for clip in manifest["clips"]:
        npz = args.root / SIGNAL_DIR / f"{clip['id']}.npz"
        if not npz.exists():
            continue
        points = propose(npz, args.max_candidates)
        breaks = clip.get("breaks_s", [])
        covered = [min((abs(point - truth) for point in points), default=float("inf")) <= args.tolerance for truth in breaks]
        rows.append({
            "clip_id": clip["id"],
            "operator": clip["operator"],
            "breaks_s": breaks,
            "proposals_s": points,
            "covered": covered,
            "coverage": all(covered) if covered else None,
        })

    changed = [row for row in rows if row["operator"] != "none"]
    clean = [row for row in rows if row["operator"] == "none"]
    summary = {
        "method": "cheap-signal-local-peak-proposals",
        "n_evaluated": len(rows),
        "n_changed": len(changed),
        "n_clean": len(clean),
        "changed_full_break_coverage": sum(bool(row["coverage"]) for row in changed),
        "changed_full_break_coverage_rate": (sum(bool(row["coverage"]) for row in changed) / len(changed)) if changed else None,
        "clean_with_any_proposal": sum(bool(row["proposals_s"]) for row in clean),
        "tolerance_s": args.tolerance,
        "max_candidates": args.max_candidates,
        "rows": rows,
    }
    destination = args.root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
