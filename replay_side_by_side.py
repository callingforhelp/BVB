#!/usr/bin/env python3
"""Side-by-side offline replay: existing jev_loop policy vs bounded MVP policy.

For every manifest clip this driver

1. builds the existing gated candidate union via `load_clip_state` (cached
   signals + cached pair-judge verdicts -- identical input for both arms),
2. replays the recorded jevloop run's reveal order for the `existing` arm
   and re-arbitrates with its recorded Jev answers (no new API calls),
3. runs the bounded `choose_next_request` policy for the `bounded` arm on
   the same state (it chooses which cached verdict to reveal next),
4. applies the SAME deterministic arbitration to both arms,
5. scores both against the manifest (labels are used only for scoring and
   are never present in either policy's state).

Output: one JSON report -- per-arm detection/type accuracy, clean false
positives, abstentions, judge calls, candidate coverage, and an error
decomposition (missing proposals vs acquisition misses vs invisible
evidence vs typing/arbitration errors).

Run (s1 venv provides numpy + routes.cred for jev_loop import):
    ~/s1-spike/.venv/bin/python replay_side_by_side.py \
        --root ../coherence_mvp \
        --existing-dir ../coherence_mvp/results/jevloop_v9 \
        --output results/replay_side_by_side.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/Users/oldap/s1-spike")

from flinter_mvp.jev_replay import replay_existing, run_bounded  # noqa: E402

NORM = {"reverse_segment": "reverse"}
JUDGE_DIRS = ("pairjudge", "pairjudge_scenecut", "pairjudge_wide", "pairjudge_bracket")


def load_jev_loop(root: Path):
    # Replay only needs the parent's pure state/arbitration helpers. Avoid
    # importing its live API adapter and credentials, which are not required
    # for cached-replay evaluation and may not be installed in this worktree.
    import types
    if "routes" not in sys.modules:
        routes = types.ModuleType("routes")
        routes.cred = lambda _name: ""
        sys.modules["routes"] = routes
    spec = importlib.util.spec_from_file_location("existing_jev_loop", root / "jev_loop.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load existing jev_loop")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truth_op(clip: dict) -> str:
    return NORM.get(clip["operator"], clip["operator"])


def nearest_candidate(break_frame: int, frames: list[int]) -> tuple[int, int]:
    """(nearest candidate frame, distance) for a true break frame."""
    best, dist = -1, 10**9
    for f in frames:
        if abs(f - break_frame) < dist:
            best, dist = f, abs(f - break_frame)
    return best, dist


def coverage_rows(clip: dict, candidate_frames: list[int], tol: int) -> dict:
    breaks = [round(float(s) * 30) for s in clip.get("breaks_s", [])]
    rows = []
    for b in breaks:
        frame, dist = nearest_candidate(b, candidate_frames)
        rows.append({"break_frame": b, "nearest_frame": frame,
                     "distance": dist, "covered": dist <= tol})
    return {"breaks": rows,
            "covered": all(r["covered"] for r in rows) if rows else None}


def classify(exp: str, got: str, arm: dict, cov: dict, st: dict) -> "str | None":
    """Error decomposition: why did this arm miss (or abstain) on this clip?

    Kinds: missing_proposal | seam_not_judged | evidence_invisible |
    abstained | second_seam_not_found | typed_wrong | clean_fp.
    Returns None when the arm is fully correct and did not abstain.
    """
    judged = {int(c["frame"]) for c in st["candidates"] if c.get("revealed")}
    discont = set(arm["discontinuous_at"])
    abstained = arm["abstained"]

    if exp == "none":
        if got != "none":
            return "clean_fp"
        return "abstained" if abstained else None

    if got == "none" or got is None:
        if any(not r["covered"] for r in cov["breaks"]):
            return "missing_proposal"
        if any(r["nearest_frame"] not in judged for r in cov["breaks"]):
            return "seam_not_judged"
        return "evidence_invisible"

    if got != exp:
        if abstained:
            return "abstained"
        if exp == "room_swap" and len(discont) < 2:
            unresolved = [
                r for r in cov["breaks"] if r["nearest_frame"] not in discont
            ]
            if any(r["nearest_frame"] not in judged for r in unresolved):
                return "second_seam_not_judged"
            return "second_seam_invisible"
        return "typed_wrong"
    return "abstained" if abstained else None


def score_arm(per_clip: list[dict], key: str) -> dict:
    rows = []
    for c in per_clip:
        arm = c[key]
        if arm is None:
            continue
        exp, got = c["expected"], arm["composite_type"]
        rows.append({
            "clip_id": c["clip_id"], "op": exp, "got": got,
            "det_ok": (exp != "none") == (got != "none"),
            "type_ok": got == exp,
            "clean_fp": exp == "none" and got != "none",
            "abstained": arm["abstained"],
            "judge_calls": arm["judge_calls"],
            "n_candidates": arm["n_candidates"],
        })
    per_op: dict = {}
    for r in rows:
        st = per_op.setdefault(r["op"], [0, 0, 0])
        st[2] += 1
        st[0] += r["det_ok"]
        st[1] += r["type_ok"]
    return {
        "n": len(rows),
        "detection": sum(r["det_ok"] for r in rows),
        "typing": sum(r["type_ok"] for r in rows),
        "clean_fps": sum(r["clean_fp"] for r in rows),
        "abstained": sum(r["abstained"] for r in rows),
        "judge_calls": sum(r["judge_calls"] for r in rows),
        "candidates": sum(r["n_candidates"] for r in rows),
        "per_op": {k: {"det": v[0], "type": v[1], "n": v[2]}
                   for k, v in sorted(per_op.items())},
        "failures": [
            {"clip_id": r["clip_id"], "expected": r["op"], "got": r["got"],
             "kind": "detection" if not r["det_ok"] else "typing",
             "abstained": r["abstained"]}
            for r in rows if not r["type_ok"]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="existing coherence_mvp worktree (read-only)")
    ap.add_argument("--existing-dir", type=Path, required=True,
                    help="recorded jevloop result dir for the existing arm")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tolerance-frames", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--trace-dir", type=Path, default=None,
                    help="optional dir for bounded-arm JSONL decision traces")
    args = ap.parse_args()

    loop = load_jev_loop(args.root)
    manifest = json.loads((args.root / "corpus_v3" / "manifest.json").read_text())
    clips = {c["id"]: c for c in manifest["clips"]}
    cids = args.only or sorted(clips)

    per_clip: list[dict] = []
    skipped_existing: list[str] = []
    for cid in cids:
        clip = clips[cid]
        exp = truth_op(clip)
        # identical candidate union + cached judgments for both arms
        st_existing = loop.load_clip_state(cid)
        st_bounded = loop.load_clip_state(cid)
        frames = [int(c["frame"]) for c in st_bounded["candidates"]]
        cov = coverage_rows(clip, frames, args.tolerance_frames)

        record_path = args.existing_dir / f"{cid}.json"
        existing = None
        if record_path.exists():
            existing = replay_existing(st_existing, json.loads(record_path.read_text()))
        else:
            skipped_existing.append(cid)

        trace = None
        if args.trace_dir is not None:
            trace = str(args.trace_dir / f"{cid}.jsonl")
        bounded = run_bounded(st_bounded, video_id=cid, trace_path=trace)

        row = {
            "clip_id": cid,
            "expected": exp,
            "source": clip.get("source"),
            "candidate_frames": frames,
            "coverage": cov,
            "existing": existing,
            "bounded": bounded,
        }
        row["error_class"] = {
            "existing": classify(exp, existing["composite_type"], existing, cov, st_existing)
            if existing else "no_record",
            "bounded": classify(exp, bounded["composite_type"], bounded, cov, st_bounded),
        }
        per_clip.append(row)

    changed = [c for c in per_clip if c["expected"] != "none"]
    clean = [c for c in per_clip if c["expected"] == "none"]

    # determinism check: rerun the bounded arm on fresh states, compare
    mismatched = []
    for c in per_clip:
        again = run_bounded(loop.load_clip_state(c["clip_id"]), video_id=c["clip_id"])
        if again != c["bounded"]:
            mismatched.append(c["clip_id"])

    decomposition: dict = {}
    for arm in ("existing", "bounded"):
        decomposition[arm] = {}
        for c in per_clip:
            k = c["error_class"][arm]
            if k:
                decomposition[arm].setdefault(k, []).append(c["clip_id"])

    union_blob = json.dumps(
        {c["clip_id"]: c["candidate_frames"] for c in per_clip}, sort_keys=True
    )
    report = {
        "meta": {
            "root": str(args.root),
            "existing_dir": str(args.existing_dir),
            "n_clips": len(per_clip),
            "skipped_existing": skipped_existing,
            "same_candidate_union": all(
                c["existing"] is None
                or c["existing"]["recorded_n_candidates"] in (None, len(c["candidate_frames"]))
                for c in per_clip
            ),
            "candidate_union_sha1": hashlib.sha1(union_blob.encode()).hexdigest()[:12],
            "cached_judgment_sources": JUDGE_DIRS,
            "bounded_arm_deterministic": not mismatched,
            "determinism_mismatches": mismatched,
            "existing_replay_fidelity_mismatches": [
                c["clip_id"] for c in per_clip
                if c["existing"] and not c["existing"]["replay_matches_record"]
            ],
            "tolerance_frames": args.tolerance_frames,
        },
        "arms": {
            "existing": score_arm(per_clip, "existing"),
            "bounded": score_arm(per_clip, "bounded"),
        },
        "candidate_coverage": {
            "n_changed": len(changed),
            "n_clean": len(clean),
            "changed_full_break_coverage": sum(
                bool(c["coverage"]["covered"]) for c in changed
            ),
            "uncovered_breaks": [
                {"clip_id": c["clip_id"],
                 "break_frame": r["break_frame"], "nearest": r["nearest_frame"]}
                for c in changed for r in c["coverage"]["breaks"] if not r["covered"]
            ],
        },
        "error_decomposition": decomposition,
        "per_clip": per_clip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")

    for arm in ("existing", "bounded"):
        s = report["arms"][arm]
        print(f"{arm:8s} n={s['n']} det={s['detection']}/{s['n']} "
              f"type={s['typing']}/{s['n']} clean_fp={s['clean_fps']} "
              f"abstained={s['abstained']} judge_calls={s['judge_calls']}")
        for op, v in s["per_op"].items():
            print(f"           {op:12s} det={v['det']}/{v['n']} type={v['type']}/{v['n']}")
    print(f"coverage: {report['candidate_coverage']['changed_full_break_coverage']}"
          f"/{len(changed)} changed fully covered")
    print(f"fidelity mismatches: {report['meta']['existing_replay_fidelity_mismatches']}")
    print(f"bounded deterministic: {report['meta']['bounded_arm_deterministic']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
