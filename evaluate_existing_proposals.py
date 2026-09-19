#!/usr/bin/env python3
"""Measure candidate coverage using the existing coherence MVP proposals.

Unlike evaluate_candidate_coverage.py, this adapter uses the exact cached
candidate union produced by the existing jev_loop gates.  It does not use the
manifest's breaks when proposing candidates; breaks are used only for scoring
coverage after proposal generation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, "/Users/oldap/s1-spike")


def load_jev_loop(path: Path):
    spec = importlib.util.spec_from_file_location("existing_jev_loop", path / "jev_loop.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load existing jev_loop")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tolerance-frames", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loop = load_jev_loop(args.root)
    manifest = json.loads((args.root / "corpus_v3/manifest.json").read_text())
    by_id = {row["id"]: row for row in manifest["clips"]}
    rows = []
    for clip_id, truth in by_id.items():
        state = loop.load_clip_state(clip_id)
        frames = [int(candidate["frame"]) for candidate in state["candidates"]]
        breaks = [round(float(seconds) * 30) for seconds in truth.get("breaks_s", [])]
        covered = [min((abs(frame - target) for frame in frames), default=10**9) <= args.tolerance_frames for target in breaks]
        rows.append({
            "clip_id": clip_id,
            "operator": truth["operator"],
            "break_frames": breaks,
            "candidate_frames": frames,
            "candidate_count": len(frames),
            "covered": covered,
            "coverage": all(covered) if covered else None,
        })
    changed = [row for row in rows if row["operator"] != "none"]
    clean = [row for row in rows if row["operator"] == "none"]
    summary = {
        "method": "existing-jev-loop-candidate-union",
        "n_evaluated": len(rows),
        "n_changed": len(changed),
        "n_clean": len(clean),
        "changed_full_break_coverage": sum(bool(row["coverage"]) for row in changed),
        "changed_full_break_coverage_rate": sum(bool(row["coverage"]) for row in changed) / len(changed),
        "clean_candidate_count_mean": sum(row["candidate_count"] for row in clean) / len(clean),
        "tolerance_frames": args.tolerance_frames,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
