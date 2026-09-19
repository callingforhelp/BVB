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
SUFF_TAU = 0.5
FAN_MARGIN = 0.2
MIN_JUDGED = 2
FLAT_EPS = 0.1     # gate v2: |d corrupted| + L1(d fan) below this = no info gain
TYPES = ["none", "loop", "reverse", "room_swap", "splice", "timewarp"]

manifest = json.loads((ROOT / "corpus_v3" / "manifest.json").read_text())
CLIPS = {c["id"]: c for c in manifest["clips"]}


# ---------------------------------------------------------------- prompt pack
# All mutable prompt text lives in a versioned pack (prompt_pack_vN.json) so
# evolve_loop.py can propose/apply edits as data diffs. DEFAULT_PACK mirrors
# v1 so this script still runs standalone without a pack file.

DEFAULT_PACK = {
    "version": 0,
    "task": ("Decide whether this 30s video clip was tampered with, and "
             "which candidates (if any) need a VLM continuity check."),
    "conclude_crit": "stop gathering evidence and report the final verdict",
    "judge_crit_template": ("run the VLM continuity check on candidate "
                            "{i} at frame {frame} (~{secs:.1f}s)"),
    "signals": {
        "recur_frac_max": ("fraction of frames that are near-exact replays of an earlier frame (lag 2-10s); "
                           ">=0.9 usually = loop edit. CAVEAT: a static motionless segment (e.g. a still "
                           "room swapped in) also scores ~1.0 — if dup_dense also fires and the repeated "
                           "region is contiguous, suspect STATIC segment => room_swap, not loop. "
                           "Values 0.5-0.8 ambiguous (splices of same scene reach ~0.6)."),
        "dup_dense_last_frame": "last frame of a dense run of adjacent-duplicate frames (setpts slowdown). Non-null strongly indicates timewarp; null on clean/other.",
        "echo_best_score": ("self-anchored 4-factor score for a reversed segment: high only when TWO "
                            "discontinuities echo each other (content after seam A matches content after "
                            "seam B). >10 = strong reverse evidence; clean and other edits typically <3. "
                            "NOTE: best_pair is always the argmax — a pair existing with score<10 is NOT "
                            "an echo. Two discontinuities WITHOUT high echo => room_swap or splice, not reverse."),
        "scenecut_iframe_frames": "frames where the video encoder inserted a scene-cut I-frame; flags abrupt visual discontinuity. BASE RATE: clean unedited clips average ~2-3 of these per 30s from natural motion, so a few are normal.",
        "photo_jump_top3": ("top frame-difference peaks. IMPORTANT: z is normalized WITHIN this clip, so "
                            "the largest peak always scores high even on clean clips — read peak POSITIONS "
                            "and co-occurrence with other signals, not z magnitude."),
    },
    "typing_guide": ("How to type — count judge-confirmed discontinuities first: "
                     "recur_frac>=0.9 => loop. dup_dense_last_frame non-null => timewarp. "
                     "echo_score>10 => reverse. TWO confirmed discontinuities separated by seconds "
                     "+ echo<10 => room_swap. ONE confirmed discontinuity => splice (even if its "
                     "break_kind says scene_change). No confirmed discontinuities + quiet signals => none."),
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
    "questions": {
        "action": "What is the best next step?",
        "corrupted": "This clip contains an artificial temporal discontinuity (an edit: replayed footage, reversed segment, swapped scene, cut/splice, or retimed segment).",
        "break_type": "Which edit type best fits the evidence?",
        "sufficient": "The current evidence is sufficient to conclude with confidence; more VLM checks would not change the verdict.",
        "is_type_template": "The evidence indicates this clip was edited with a {type} corruption ({desc}).",
    },
    "type_desc": {
        "loop": "footage repeats itself — a segment is replayed",
        "reverse": "a segment plays backward in time",
        "room_swap": "the scene/location changes to a different place then returns",
        "splice": "a hard cut joins non-contiguous footage of the same scene",
        "timewarp": "part of the clip plays at altered speed (slowed or sped up)",
    },
    "none_desc": "no artificial edit",
    "precedents_note": ("Each candidate may carry a 'precedent' field: an "
                        "outcome summary of the most visually similar "
                        "boundaries seen in OTHER videos (retrieved by "
                        "embedding, never from this clip). Treat it as a "
                        "prior, not a verdict — strong when it agrees with "
                        "your read of the signals, weak when it conflicts."),
}


def load_pack(path) -> dict:
    """Load a prompt pack JSON; falls back to DEFAULT_PACK when absent."""
    if path is None:
        return dict(DEFAULT_PACK)
    pack = json.loads(Path(path).read_text())
    merged = dict(DEFAULT_PACK)
    merged.update(pack)
    merged["signals"] = {**DEFAULT_PACK["signals"], **pack.get("signals", {})}
    merged["questions"] = {**DEFAULT_PACK["questions"], **pack.get("questions", {})}
    merged["type_desc"] = {**DEFAULT_PACK["type_desc"], **pack.get("type_desc", {})}
    return merged


def pack_hash(pack: dict) -> str:
    import hashlib
    blob = json.dumps({k: v for k, v in pack.items()
                       if k not in ("version", "parent", "notes")},
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- evidence

def load_clip_state(cid: str, bank=None) -> dict:
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

    # optional reasoning bank: per-candidate strip descriptors for retrieval
    if bank is not None:
        import strip_bank as sb
        mp4 = ROOT / "corpus_v3" / CLIPS[cid]["path"]
        for c in candidates:
            try:
                frames = sb.extract_strip_frames(mp4, c["frame"])
                ctx = sb.signal_context_for(cid, c["frame"], c["gate"])
                c["vec"] = sb.strip_descriptor(frames, ctx)
            except Exception:
                c["vec"] = None

    return {
        "source": CLIPS[cid]["source"],
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


def state_view(st: dict, pack: dict, bank=None) -> dict:
    """Serialize evidence Jev may see (signals + revealed verdicts only)."""
    fs = st["free_signals"]
    sig = pack["signals"]
    return {
        "task": pack["task"],
        "clip": {"duration_s": 30, "fps": 30},
        "free_signal_evidence": {
            "recur_frac_max": {
                "value": fs["recur_frac_max"], "frame": fs["recur_argmax_frame"],
                "meaning": sig["recur_frac_max"]},
            "dup_dense_last_frame": {
                "value": fs["dup_dense_last_frame"],
                "meaning": sig["dup_dense_last_frame"]},
            "echo_best_score": {
                "value": fs["echo_best_score"], "pair": fs["echo_best_pair"],
                "meaning": sig["echo_best_score"]},
            "scenecut_iframe_frames": {
                "value": fs["scenecut_iframe_frames"],
                "meaning": sig["scenecut_iframe_frames"]},
            "photo_jump_top3": {
                "value": fs["photo_jump_top3"],
                "meaning": sig["photo_jump_top3"]},
            "typing_guide": pack["typing_guide"],
        },
        "candidates": [
            {**({"precedent": bank.precedent_summary(
                     c["vec"], exclude_source=st["source"])["summary"]}
                 if bank is not None and c.get("vec") is not None else {}),
             "id": i, "frame": c["frame"], "gate": c["gate"],
             "vlm_verdict": c["revealed"] if c["revealed"] else "not_judged"}
            for i, c in enumerate(st["candidates"])],
        "candidate_note": pack["candidate_note"],
        **({"precedents_note": pack.get("precedents_note")}
            if bank is not None else {}),
    }


def questions_for(st: dict, pack: dict) -> dict:
    crit = {}
    for i, c in enumerate(st["candidates"]):
        if not c["revealed"]:
            crit[f"judge_{i}"] = pack["judge_crit_template"].format(
                i=i, frame=c["frame"], secs=c["frame"] / 30)
    crit["conclude"] = pack["conclude_crit"]
    pq = pack["questions"]
    q = {
        "action": {"type": "choice", "instructions": pq["action"], "criteria": crit},
        "corrupted": {"type": "noul", "instructions": pq["corrupted"]},
        "break_type": {"type": "choice", "instructions": pq["break_type"], "criteria": {
            "none": pack["none_desc"], **pack["type_desc"]}},
        "sufficient": {"type": "noul", "instructions": pq["sufficient"]},
    }
    # speculative fan-out: per-type nouls — parallel, no added latency
    for t, desc in pack["type_desc"].items():
        q[f"is_{t}"] = {"type": "noul",
                        "instructions": pq["is_type_template"].format(type=t, desc=desc)}
    return q


# ---------------------------------------------------------------- loop

def run_clip(cid: str, pack: dict, out_dir: Path, bank=None) -> dict:
    st = load_clip_state(cid, bank=bank)
    transcript, jev_calls, jev_errors = [], 0, 0
    verdict = {}
    abstained = False
    reveal_deltas: list[float] = []   # info gain produced by each reveal
    prev_belief = None                # (corrupted, fan) at last jev call

    for it in range(MAX_ITERS):
        unjudged = [i for i, c in enumerate(st["candidates"]) if not c["revealed"]]
        if not unjudged and it == 0:
            break  # nothing to judge — conclude immediately
        try:
            ans = jev(state_view(st, pack, bank), questions_for(st, pack))
            jev_calls += 1
        except Exception as e:
            jev_errors += 1
            transcript.append({"iter": it, "error": str(e)})
            break
        verdict = {k: v for k, v in ans.items()}
        act = ans.get("action", {}).get("choice", "conclude")
        suff = ans.get("sufficient", {}).get("noul", 0)
        fan = {t: ans.get(f"is_{t}", {}).get("noul") for t in pack["type_desc"]}
        belief = (ans.get("corrupted", {}).get("noul", 0), fan)
        # info gain from the previous iter's reveal (if any)
        if prev_belief is not None and transcript \
                and transcript[-1].get("_revealed") is not None:
            d = abs(belief[0] - prev_belief[0]) + sum(
                abs((belief[1].get(t) or 0) - (prev_belief[1].get(t) or 0))
                for t in pack["type_desc"])
            reveal_deltas.append(d)
        prev_belief = belief
        transcript.append({"iter": it, "action": act,
                           "corrupted": ans.get("corrupted", {}).get("noul"),
                           "break_type": ans.get("break_type", {}).get("choice"),
                           "sufficient": suff, "fan": fan})
        if act == "conclude":
            vals = sorted((v for v in fan.values() if v is not None), reverse=True)
            margin = (vals[0] - vals[1]) if len(vals) > 1 else 1.0
            corrupted_p = ans.get("corrupted", {}).get("noul", 0)
            judged_n = len([c for c in st["candidates"] if c["revealed"]])
            # conclude honored iff evidence floor met: clearly clean,
            # >=MIN_JUDGED verdicts seen, or one type dominates the fan.
            # (sufficient noul is logged but NOT gated on: Jev's suff
            #  calibration is pessimistic (median ~0.3) and its max came
            #  from a confidently-wrong case.)
            ok = (corrupted_p < 0.5 or judged_n >= MIN_JUDGED
                  or margin >= FAN_MARGIN)
            if ok:
                break
            if not unjudged:
                transcript[-1]["forced_end"] = True  # out of options, still uncertain
                break
            # gate v2 (ProgressGate addendum): conclude is blocked AND the
            # last 2 reveals moved the belief by ~nothing -> the remaining
            # forced judges would burn calls without changing the verdict.
            # Abstain 'corrupt_untyped' instead of forcing. (Blocked path
            # already implies corrupted>=0.5, so the verdict is consistent.)
            if len(reveal_deltas) >= 2 and \
                    all(d < FLAT_EPS for d in reveal_deltas[-2:]):
                abstained = True
                transcript[-1]["abstained"] = True
                break
            # confidence-gated: wants to stop but evidence thin/ambiguous -> force a check
            st["candidates"][unjudged[0]]["revealed"] = st["candidates"][unjudged[0]]["verdict"]
            transcript[-1]["forced_judge"] = unjudged[0]
            transcript[-1]["_revealed"] = unjudged[0]
            continue
        if not unjudged:
            break
        if act.startswith("judge_"):
            i = int(act.split("_")[1])
            if 0 <= i < len(st["candidates"]) and not st["candidates"][i]["revealed"]:
                st["candidates"][i]["revealed"] = st["candidates"][i]["verdict"]
                transcript[-1]["_revealed"] = i
                continue
        # invalid/duplicate action -> reveal next unjudged deterministically
        jev_errors += 1
        st["candidates"][unjudged[0]]["revealed"] = st["candidates"][unjudged[0]]["verdict"]
        transcript[-1]["_revealed"] = unjudged[0]

    judged = [c for c in st["candidates"] if c["revealed"]]
    discont = [c["frame"] for c in judged
               if c["revealed"].get("continuous") is False]
    last_fan = next((t["fan"] for t in reversed(transcript) if t.get("fan")), {})
    fan_type = max(last_fan, key=lambda k: last_fan.get(k) or 0) if last_fan else None
    ch = verdict.get("break_type", {}).get("choice")
    cor = verdict.get("corrupted", {}).get("noul", 0)
    composite = fan_type if (ch == "none" and cor > 0.5 and fan_type) else ch
    # seam-count arbitration: room_swap = TWO confirmed discontinuities,
    # splice = ONE. Jev gathers the evidence; code arbitrates the type.
    ndis = len(discont)
    if composite == "splice" and ndis >= 2:
        composite = "room_swap"
    elif composite == "room_swap" and ndis < 2:
        composite = "splice"
    if abstained:
        composite = "corrupt_untyped"   # refusal class: corrupt, untyped
    out = {
        "clip_id": cid,
        "operator": CLIPS[cid]["operator"],
        "breaks_s": CLIPS[cid]["breaks_s"],
        "final": verdict,
        "fan_type": fan_type,
        "fan_probs": last_fan,
        "composite_type": composite,
        "abstained": abstained,
        "pack_version": pack.get("version"),
        "pack_hash": pack_hash(pack),
        "n_candidates": len(st["candidates"]),
        "judge_calls": len(judged),
        "discontinuous_at": discont,
        "jev_calls": jev_calls,
        "jev_errors": jev_errors,
        "transcript": transcript,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cid}.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--pack", default=None,
                    help="prompt pack JSON (default: built-in v1 strings)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: results/jevloop)")
    ap.add_argument("--bank", default=None,
                    help="strip-bank dir (episodes.jsonl+embeddings.npy) "
                         "to inject per-candidate precedents")
    ap.add_argument("--jev", default="typesafe",
                    help="jev backend: 'typesafe' (default TypeSafe/OpenJev "
                         "endpoint), 's1:base' (Tinker base model), "
                         "'s1:tinker://<sampler-path>' (Tinker ckpt), or "
                         "'fw:accounts/<acct>/models/<id>' (Fireworks "
                         "deployment). s1 backends need tinker (s1-spike venv).")
    args = ap.parse_args()

    global jev
    if args.jev.startswith("fw:"):
        import s1_fw_serve
        backend = s1_fw_serve.FWJev(args.jev.removeprefix("fw:"))
        jev = backend.answers
        print(f"jev backend: FWJev({args.jev[3:]})")
    elif args.jev != "typesafe":
        import s1_serve
        path = args.jev.removeprefix("s1:")
        backend = s1_serve.S1Jev(model_path=None if path == "base" else path)
        jev = backend.answers
        print(f"jev backend: S1Jev({path or 'base model'})")

    pack = load_pack(args.pack)
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = None
    if args.bank:
        import strip_bank as sb
        bank = sb.StripBank.load(args.bank)
        print(f"bank: {len(bank._ids)} episodes from {args.bank}")

    cids = args.only or sorted(CLIPS)
    todo = [c for c in cids if not (out_dir / f"{c}.json").exists()]
    print(f"pack v{pack.get('version')} ({pack_hash(pack)}): "
          f"{len(todo)} clips to run ({len(cids)-len(todo)} cached)")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_clip, c, pack, out_dir, bank): c for c in todo}
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
