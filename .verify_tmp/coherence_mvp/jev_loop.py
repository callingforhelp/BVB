"""Jev-as-loop-manager spike on corpus_v3 (reveal-mode agentic cascade).

Simulates the streaming decision layer on batch data: free deterministic
signals are revealed up front; Jev then iterates a bounded loop, choosing
per step whether to spend a VLM pair-judge call on a gate candidate or to
conclude. Judge calls replay cached verdicts (pairjudge +
pairjudge_scenecut + pairjudge_wide) so the spike costs only Jev calls.

Measured against the fixed pipeline (which judged EVERY candidate):
  - verdict accuracy (corrupted? + break_type)
  - judge calls used per clip  <- the cost claim
  - localization coverage

Run: ~/s1-spike/.venv/bin/python jev_loop.py --workers 8
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SPIKE = Path.home() / "s1-spike"
sys.path.insert(0, str(SPIKE))
from routes import cred  # noqa: E402

JEV_API = "https://api.typesafe.ai/v1/systemone"
KEY = cred("TYPESAFE_API_KEY")
OUT = ROOT / "results" / "jevloop"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 30.0
KEYINT = 250 / FPS
MAX_ITERS = 8
TYPES = ["none", "loop", "reverse", "room_swap", "splice", "timewarp"]

manifest = json.loads((ROOT / "corpus_v3" / "manifest.json").read_text())
CLIPS = {c["id"]: c for c in manifest["clips"]}


# ---------------------------------------------------------------- evidence

def load_clip_state(cid: str) -> dict:
    """Free-signal evidence + judgeable candidates (cached verdicts)."""
    sig = np.load(ROOT / "results" / "signals_v3" / f"{cid}.npz")
    recur = sig["recur_frac"].astype(np.float64)
    pj = sig["photo_jump"].astype(np.float64)

    dup = np.load(ROOT / "results" / "dupfrac" / f"{cid}.npz")["dup"]
    d = np.asarray(dup, dtype=np.float64)
    W, thr = 60, 0.2
    dense = np.convolve(d, np.ones(W), "same") / W >= thr
    dup_last = int(np.where(dense)[0].max()) if dense.any() else None

    echo = json.loads((ROOT / "results" / "echo" / f"{cid}.json").read_text())

    codec = np.load(ROOT / "results" / "codec_stats" / f"{cid}.npz")
    pts, key = codec["pts"], codec["is_iframe"]
    scenecut = [int(i) for i in np.where(key[1:] == 1)[0] + 1
                if 0.15 * KEYINT < pts[i] % KEYINT < 0.85 * KEYINT]

    order = np.argsort(pj)[::-1][:3]
    pj_top = [{"frame": int(f + 1), "z": round(float(pj[f]), 1)} for f in order]

    # candidates = union of all cached verdict frames, tagged by gate
    cands = {}
    for dname, gate in (("pairjudge", "pixel_gate"),
                        ("pairjudge_scenecut", "scenecut"),
                        ("pairjudge_wide", "wide_gate")):
        f = ROOT / "results" / dname / f"{cid}.json"
        if f.exists():
            for c in json.loads(f.read_text()).get("candidates", []):
                if c.get("verdict") is None:
                    continue
                cands.setdefault(c["frame"], {"frame": c["frame"],
                                             "gate": gate, "verdict": c["verdict"]})
    candidates = [dict(v, judged=None) for _, v in sorted(cands.items())]
    for c in candidates:
        c["revealed"] = None

    return {
        "free_signals": {
            "recur_frac_max": round(float(recur.max()), 3),
            "recur_argmax_frame": int(recur.argmax()),
            "dup_dense_last_frame": dup_last,
            "echo_best_score": round(float(echo["best_score"]), 2),
            "echo_best_pair": echo["best_pair"],
            "scenecut_iframe_frames": scenecut,
            "photo_jump_top3": pj_top,
        },
        "candidates": candidates,
    }


# ---------------------------------------------------------------- jev call

def jev(state: dict, questions: dict) -> dict:
    body = json.dumps({"state": json.dumps(state), "model": "jev-latest",
                       "questions": questions}).encode()
    req = urllib.request.Request(
        JEV_API, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["answers"]


def state_view(st: dict) -> dict:
    """Serialize evidence Jev may see (signals + revealed verdicts only)."""
    fs = st["free_signals"]
    return {
        "task": ("Decide whether this 30s video clip was tampered with, and "
                 "which candidates (if any) need a VLM continuity check."),
        "clip": {"duration_s": 30, "fps": 30},
        "free_signal_evidence": {
            "recur_frac_max": {
                "value": fs["recur_frac_max"], "frame": fs["recur_argmax_frame"],
                "meaning": ("fraction of frames that are exact pixel replays of an earlier frame; >=0.9 = "
                            "clip replays itself (loop edit). Values 0.5-0.8 are ambiguous: splices of "
                            "the same scene can also reach ~0.6 — do NOT call loop below 0.9.")},
            "dup_dense_last_frame": {
                "value": fs["dup_dense_last_frame"],
                "meaning": "last frame of a dense run of adjacent-duplicate frames (setpts slowdown). Non-null strongly indicates timewarp; null on clean/other."},
            "echo_best_score": {
                "value": fs["echo_best_score"], "pair": fs["echo_best_pair"],
                "meaning": ("self-anchored 4-factor score for a reversed segment: high only when TWO "
                            "discontinuities echo each other (content after seam A matches content after "
                            "seam B). >10 = strong reverse evidence; clean and other edits typically <3. "
                            "NOTE: best_pair is always the argmax — a pair existing with score<10 is NOT "
                            "an echo. Two discontinuities WITHOUT high echo => room_swap or splice, not reverse.")},
            "scenecut_iframe_frames": {
                "value": fs["scenecut_iframe_frames"],
                "meaning": "frames where the video encoder inserted a scene-cut I-frame; flags abrupt visual discontinuity. BASE RATE: clean unedited clips average ~2-3 of these per 30s from natural motion, so a few are normal."},
            "photo_jump_top3": {
                "value": fs["photo_jump_top3"],
                "meaning": ("top frame-difference peaks. IMPORTANT: z is normalized WITHIN this clip, so "
                            "the largest peak always scores high even on clean clips — read peak POSITIONS "
                            "and co-occurrence with other signals, not z magnitude.")},
            "typing_guide": ("How to type — count judge-confirmed discontinuities first: "
                            "recur_frac>=0.9 => loop. dup_dense_last_frame non-null => timewarp. "
                            "echo_score>10 => reverse. TWO confirmed discontinuities separated by seconds "
                            "+ echo<10 => room_swap. ONE confirmed discontinuity => splice (even if its "
                            "break_kind says scene_change). No confirmed discontinuities + quiet signals => none."),
        },
        "candidates": [
            {"id": i, "frame": c["frame"], "gate": c["gate"],
             "vlm_verdict": c["revealed"] if c["revealed"] else "not_judged"}
            for i, c in enumerate(st["candidates"])],
        "candidate_note": ("Each candidate is a suspicious transition the "
                           "gates flagged. BASE RATE: the gates are permissive "
                           "and nominate ~4-8 candidates per clip INCLUDING on "
                           "clean unedited clips, so a candidate existing is "
                           "weak evidence — only its VLM verdict is strong "
                           "evidence. A VLM check costs ~2s and is "
                           "near-perfect precision. Judge candidates whose "
                           "verdict would change your conclusion. "
                           "IMPORTANT: splice edits leave NO free-signal trace "
                           "— quiet signals do not rule them out. Before "
                           "concluding 'none', check at least the 2-3 "
                           "strongest candidates unless a decisive free "
                           "signal already explains the clip."),
    }


def questions_for(st: dict) -> dict:
    crit = {}
    for i, c in enumerate(st["candidates"]):
        if not c["revealed"]:
            crit[f"judge_{i}"] = (f"run the VLM continuity check on candidate "
                                  f"{i} at frame {c['frame']} (~{c['frame']/30:.1f}s)")
    crit["conclude"] = "stop gathering evidence and report the final verdict"
    return {
        "action": {"type": "choice", "instructions": "What is the best next step?", "criteria": crit},
        "corrupted": {"type": "noul", "instructions": "This clip contains an artificial temporal discontinuity (an edit: replayed footage, reversed segment, swapped scene, cut/splice, or retimed segment)."},
        "break_type": {"type": "choice", "instructions": "Which edit type best fits the evidence?", "criteria": {
            "none": "no artificial edit",
            "loop": "footage repeats itself (replayed segment)",
            "reverse": "a segment plays backward (reversed)",
            "room_swap": "the scene/location changes then returns",
            "splice": "a hard cut joins non-contiguous footage of the same scene",
            "timewarp": "part of the clip plays at altered speed"}},
        "sufficient": {"type": "noul", "instructions": "The current evidence is sufficient to conclude with confidence; more VLM checks would not change the verdict."},
    }


# ---------------------------------------------------------------- loop

def run_clip(cid: str) -> dict:
    st = load_clip_state(cid)
    transcript, jev_calls, jev_errors = [], 0, 0
    verdict = {}

    for it in range(MAX_ITERS):
        unjudged = [i for i, c in enumerate(st["candidates"]) if not c["revealed"]]
        if not unjudged and it == 0:
            break  # nothing to judge — conclude immediately
        try:
            ans = jev(state_view(st), questions_for(st))
            jev_calls += 1
        except Exception as e:
            jev_errors += 1
            transcript.append({"iter": it, "error": str(e)})
            break
        verdict = {k: v for k, v in ans.items()}
        act = ans.get("action", {}).get("choice", "conclude")
        transcript.append({"iter": it, "action": act,
                           "corrupted": ans.get("corrupted", {}).get("noul"),
                           "break_type": ans.get("break_type", {}).get("choice"),
                           "sufficient": ans.get("sufficient", {}).get("noul")})
        if act == "conclude" or not unjudged:
            break
        if act.startswith("judge_"):
            i = int(act.split("_")[1])
            if 0 <= i < len(st["candidates"]) and not st["candidates"][i]["revealed"]:
                st["candidates"][i]["revealed"] = st["candidates"][i]["verdict"]
                continue
        # invalid/duplicate action -> reveal next unjudged deterministically
        jev_errors += 1
        st["candidates"][unjudged[0]]["revealed"] = st["candidates"][unjudged[0]]["verdict"]

    judged = [c for c in st["candidates"] if c["revealed"]]
    discont = [c["frame"] for c in judged
               if c["revealed"].get("continuous") is False]
    out = {
        "clip_id": cid,
        "operator": CLIPS[cid]["operator"],
        "breaks_s": CLIPS[cid]["breaks_s"],
        "final": verdict,
        "n_candidates": len(st["candidates"]),
        "judge_calls": len(judged),
        "discontinuous_at": discont,
        "jev_calls": jev_calls,
        "jev_errors": jev_errors,
        "transcript": transcript,
    }
    (OUT / f"{cid}.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    cids = args.only or sorted(CLIPS)
    todo = [c for c in cids if not (OUT / f"{c}.json").exists()]
    print(f"{len(todo)} clips to run ({len(cids)-len(todo)} cached)")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_clip, c): c for c in todo}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
                bt = r["final"].get("break_type", {})
                bt = bt.get("choice", "?") if isinstance(bt, dict) else "?"
                print(f"[{i+1}/{len(todo)}] {r['clip_id']:42s} "
                      f"type={bt:10s} judges={r['judge_calls']}/"
                      f"{r['n_candidates']} jev={r['jev_calls']} "
                      f"err={r['jev_errors']}")
            except Exception as e:
                print(f"[{i+1}/{len(todo)}] {futs[f]} FAILED: {e}")


if __name__ == "__main__":
    main()
