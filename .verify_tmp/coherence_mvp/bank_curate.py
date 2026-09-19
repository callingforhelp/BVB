#!/usr/bin/env python3
"""bank_curate.py — the self-improvement controller the bank was missing.

Loop: ARK (glm-5.3-flash, plan endpoint) reads failure episodes and proposes
dsh-pes-reasoning.v1 lesson entries -> we replay each candidate against the
affected clips through the live TypeSafe Jev endpoint -> keep only lessons
that measurably move typing toward truth without firing on controls.

ARK proposes, code arbitrates. No training, no blocked billing — the whole
round runs on free endpoints.

Usage:
  python3 bank_curate.py propose            # ARK distills lessons -> candidates.jsonl
  python3 bank_curate.py validate           # replay candidates -> verdicts.jsonl + report
  python3 bank_curate.py deposit            # copy PASS lessons into .codex bank
"""
import copy
import json
import os
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parent.parent
BANK_DIR = REPO / ".codex" / "reasoning-bank"
OUT = HERE / "results" / "bank_curate"
JEVLOOP = HERE / "results" / "jevloop"
CREDS = yaml.safe_load(open(os.path.expanduser("~/.dsh/.credentials.yaml")))["refs"]

ARK_URL = "https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions"
ARK_MODEL = "glm-5.3-flash"
JEV_API = "https://api.typesafe.ai/v1/systemone"

OP2T = {"reverse_segment": "reverse"}
TYPE_DESC = {
    "loop": "same temporal segment replays verbatim",
    "reverse": "a segment plays backwards",
    "room_swap": "a segment from a different room/scene is inserted",
    "splice": "two different segments are joined by a cut",
    "timewarp": "playback speed changes within the clip",
}

sys.path.insert(0, str(HERE))
import jev_loop as JL  # noqa: E402

PACK = JL.load_pack(str(HERE / "prompt_pack_v2.json"))


# ---------------------------------------------------------------- ark call

def ark(prompt: str, max_tokens: int = 8192) -> str:
    body = json.dumps({
        "model": ARK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
    }).encode()
    req = urllib.request.Request(
        ARK_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {CREDS['ARK_PLAN_FLASH_API_KEY']}",
                 "Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return d["choices"][0]["message"]["content"]


def jev(state: dict, questions: dict) -> dict:
    return JL.jev(state, questions)


# ---------------------------------------------------------------- evidence

def residual_clips(run_dir: Path = JEVLOOP) -> list[dict]:
    """Recompute the miss set from a jevloop transcript dir."""
    misses = []
    for f in sorted(run_dir.glob("*.json")):
        x = json.load(open(f))
        op = x.get("operator")
        if not op or op == "clean":
            continue
        tgt = OP2T.get(op, op)
        ct = x.get("composite_type") or (x.get("final") or {}).get(
            "break_type", {}).get("choice")
        if ct != tgt:
            misses.append({
                "clip_id": x["clip_id"], "true_type": tgt, "typed_as": ct,
                "breaks_s": x.get("breaks_s"), "judge_calls": x.get("judge_calls"),
                "trajectory": [
                    {"iter": t["iter"], "action": t.get("action"),
                     "corrupted": round(t.get("corrupted", 0), 2),
                     "break_type": t.get("break_type"),
                     "fan": {k: round(v, 2) for k, v in
                             (t.get("fan") or {}).items() if v > 0.05}}
                    for t in x["transcript"]],
            })
    return misses


def clip_signals(cid: str) -> dict:
    st = JL.load_clip_state(cid)
    return {k: v.get("value") for k, v in
            JL.state_view(st, PACK)["free_signal_evidence"].items()
            if isinstance(v, dict)}


def episode_card(m: dict) -> dict:
    st = JL.load_clip_state(m["clip_id"])
    sv = JL.state_view(st, PACK)
    judged = [{"frame": c["frame"], "gate": c["gate"],
               "vlm_verdict": c["verdict"]}
              for c in st["candidates"] if c.get("verdict")]
    return {**m, "free_signals": clip_signals(m["clip_id"]),
            "n_candidates": len(st["candidates"]),
            "judgeable_verdicts": judged[:8]}


# ---------------------------------------------------------------- propose

def propose(run_dir: Path = JEVLOOP, active_lessons: list[dict] | None = None,
            tag: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    misses = residual_clips(run_dir)
    print(f"{len(misses)} residual misses in {run_dir.name}")
    cards = [episode_card(m) for m in misses]

    active_block = ""
    if active_lessons:
        active_block = f"""
ALREADY-ACTIVE LESSONS (injected into every state this round — refine or
replace, do not repeat):
{json.dumps([{k: l[k] for k in ('cue', 'lesson')} for l in active_lessons],
            indent=1)}
"""

    prompt = f"""You are the reasoning-bank curator for a temporal-edit detector.
The detector asks a Jev policy to classify video corruption into exactly one of:
{json.dumps(TYPE_DESC, indent=1)}

Mechanics the lessons can influence:
- The policy iterates: free signals are shown up front; it may call
  judge_i (reveals that candidate's VLM verdict) or conclude.
- break_type is its typed answer; a code arbitrator then applies:
  composite=splice if the model typed room_swap but fewer than TWO judged
  seams are discontinuous; composite=room_swap if it typed splice but two
  or more seams are discontinuous. So TYPING the right class is not
  enough — the model must JUDGE BOTH seams of a swapped segment for the
  room_swap verdict to survive arbitration.
{active_block}
Below are ALL clips where the policy typed wrong (or missed) THIS round,
with the clip's free signals, judgeable VLM verdicts, and the policy's
decision trajectory (action per iter + type-fan probabilities).

FAILURE EPISODES:
{json.dumps(cards, indent=1)[:14000]}

Distill at most 6 dsh-pes-reasoning.v1 lesson entries that would prevent these
failure classes. Rules:
- cue: WHEN the lesson applies, phrased over observable evidence (signal
  values, seam counts, verdict patterns) — not clip IDs.
- lesson: the discriminating rule in 1-2 sentences, decisive not vague.
  Lessons may steer ACTION selection (which seams to judge) as well as
  typing — say so explicitly when they do.
- evidenceIds: the clip_ids that ground it.
- A lesson that fixes room_swap must not flip true splices/loops — keep
  the discriminating feature tight (seam COUNT + recurrence, not just
  "scene_change exists").
- Lessons must NOT encourage typing on clean footage.
- Prefer lessons that leverage patterns visible ACROSS multiple failures.

Return ONLY a JSON array of objects with keys: cue, lesson, evidenceIds."""

    text = ark(prompt)
    print("--- curator raw ---\n", text[:2000])
    i, j = text.find("["), text.rfind("]")
    cands = json.loads(text[i:j + 1])
    out = OUT / f"candidates{tag}.jsonl"
    with open(out, "w") as fh:
        for k, c in enumerate(cands):
            fh.write(json.dumps({
                "schemaVersion": "dsh-pes-reasoning.v1",
                "id": f"curate{tag}_{k:02d}", "traceKind": "curated_lesson",
                "producerSha": f"ark:{ARK_MODEL}", **c}) + "\n")
    print(f"\n{len(cands)} candidate lessons -> {out}")


# ---------------------------------------------------------------- validate

LESSON_MATCH = {
    "room_swap": ["room_swap", "swap", "scene", "segment", "two seam", "bracket"],
    "splice": ["splice", "cut", "join"],
    "loop": ["loop", "repeat", "replay"],
}


def lesson_applies(lesson: dict, miss: dict) -> bool:
    """Does this lesson plausibly target this miss? Match lesson text to the
    true type or the confusion pair."""
    text = (lesson["cue"] + " " + lesson["lesson"]).lower()
    if miss["clip_id"] in (lesson.get("evidenceIds") or []):
        return True
    keys = LESSON_MATCH.get(miss["true_type"], [])
    return any(k in text for k in keys)


def replay_with(cid: str, revealed_idx: list[int], lesson: dict | None,
                base: list[dict] | None = None) -> dict:
    st = JL.load_clip_state(cid)
    for i in revealed_idx:
        st["candidates"][i]["revealed"] = st["candidates"][i]["verdict"]
    sv = JL.state_view(st, PACK)
    lessons = [{"cue": l["cue"], "lesson": l["lesson"],
                "prior_outcome": "bank-curated"}
               for l in (base or [])]
    if lesson:
        lessons.append({"cue": lesson["cue"], "lesson": lesson["lesson"],
                        "prior_outcome": "bank-curated"})
    if lessons:
        sv["precedent_lessons"] = lessons
    return jev(sv, JL.questions_for(st, PACK))


def confusable_controls(miss: dict, run_dir: Path, k: int = 4) -> list[dict]:
    """Currently-CORRECT clips whose true type is confusable with the miss —
    the negative arm: a good lesson must not flip these."""
    bad_types = {t for t in TYPE_DESC if t != miss["true_type"]}
    out = []
    for f in sorted(run_dir.glob("*.json")):
        x = json.load(open(f))
        op = x.get("operator")
        if not op or op == "clean":
            continue
        tgt = OP2T.get(op, op)
        if tgt not in bad_types:
            continue
        ct = x.get("composite_type") or (x.get("final") or {}).get(
            "break_type", {}).get("choice")
        if ct != tgt:
            continue  # already a miss — belongs to the positive arm
        judged = [int(t["action"].split("_")[1]) for t in x["transcript"]
                  if str(t.get("action", "")).startswith("judge_")]
        out.append({"clip_id": x["clip_id"], "true_type": tgt,
                    "judged": judged})
        if len(out) >= k:
            break
    return out


def validate(run_dir: Path = JEVLOOP, cand_file: str = "candidates.jsonl",
             verdict_file: str = "verdicts.jsonl",
             base_lessons: list[dict] | None = None):
    cands = [json.loads(l) for l in open(OUT / cand_file)]
    misses = residual_clips(run_dir)
    verdicts = []
    for ci, les in enumerate(cands):
        targets = [m for m in misses if lesson_applies(les, m)]
        print(f"\n=== lesson {ci}: {les['lesson'][:90]}")
        print(f"    targets: {[m['clip_id'] for m in targets]}")
        deltas, flips, neg_flips = [], 0, 0
        # negative arm: correct clips of confusable types under the same lesson
        controls, seen = [], set()
        for m in targets:
            for c in confusable_controls(m, run_dir):
                if c["clip_id"] not in seen:
                    seen.add(c["clip_id"])
                    controls.append(c)
        print(f"    controls: {[c['clip_id'] for c in controls]}")
        ctrl_breaks = 0
        for c in controls:
            try:
                plus = replay_with(c["clip_id"], c["judged"], les,
                                   base=base_lessons)
            except Exception as e:
                print(f"    ctrl {c['clip_id']}: FAIL {e}")
                continue
            c1 = plus["break_type"]["choice"]
            broke = c1 != c["true_type"] and c1 != "none"
            ctrl_breaks += int(broke)
            if broke:
                print(f"    ctrl {c['clip_id']}: {c['true_type']}->{c1} BROKE")
        for m in targets:
            judged = [int(t["action"].split("_")[1]) for t in m["trajectory"]
                      if str(t.get("action", "")).startswith("judge_")]
            try:
                base = replay_with(m["clip_id"], judged, None,
                                   base=base_lessons)
                plus = replay_with(m["clip_id"], judged, les,
                                   base=base_lessons)
            except Exception as e:
                print(f"    {m['clip_id']}: REPLAY FAIL {e}")
                continue
            p0 = base["break_type"]["probabilities"].get(m["true_type"], 0)
            p1 = plus["break_type"]["probabilities"].get(m["true_type"], 0)
            c0 = base["break_type"]["choice"]
            c1 = plus["break_type"]["choice"]
            deltas.append(p1 - p0)
            flips += int(c0 != m["true_type"] and c1 == m["true_type"])
            neg_flips += int(c0 == m["true_type"] and c1 != m["true_type"])
            print(f"    {m['clip_id']}: {c0}->{c1} "
                  f"P(truth) {p0:.2f}->{p1:.2f}")
        mean_d = sum(deltas) / len(deltas) if deltas else 0
        # both arms: wins must not flip targets backward NOR break controls
        verdict = ("PASS" if (flips > 0 or mean_d >= 0.10)
                   and neg_flips == 0 and ctrl_breaks == 0
                   else "MIXED" if flips > neg_flips else
                   "WEAK" if mean_d > 0.02 else "FAIL")
        verdicts.append({**les, "verdict": verdict, "flips": flips,
                         "neg_flips": neg_flips, "ctrl_breaks": ctrl_breaks,
                         "mean_delta": round(mean_d, 3),
                         "n_targets": len(targets),
                         "n_controls": len(controls)})
        print(f"    -> {verdict} (flips={flips}, neg={neg_flips}, "
              f"ctrl_breaks={ctrl_breaks}, mean ΔP={mean_d:+.3f})")

    with open(OUT / verdict_file, "w") as fh:
        for v in verdicts:
            fh.write(json.dumps(v) + "\n")
    npass = sum(v["verdict"] == "PASS" for v in verdicts)
    print(f"\n{npass}/{len(verdicts)} PASS -> {OUT/verdict_file}")


# ---------------------------------------------------------------- deposit

def deposit():
    eps = BANK_DIR / "episodes.jsonl"
    existing = {json.loads(l)["id"] for l in open(eps)}
    added = 0
    with open(eps, "a") as fh:
        for v in map(json.loads, open(OUT / "verdicts.jsonl")):
            if v["verdict"] != "PASS" or v["id"] in existing:
                continue
            fh.write(json.dumps({
                "schemaVersion": "dsh-pes-reasoning.v1",
                "id": v["id"], "traceKind": "curated_lesson",
                "producerSha": v["producerSha"],
                "summary": f"curated lesson: {v['lesson'][:120]}",
                "cue": v["cue"], "lesson": v["lesson"],
                "evidenceIds": v.get("evidenceIds", []),
                "validated": {"flips": v["flips"],
                              "mean_delta": v["mean_delta"],
                              "n_targets": v["n_targets"]},
            }) + "\n")
            added += 1
    print(f"deposited {added} validated lessons -> {eps}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "propose":
        run_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else JEVLOOP
        active = ([json.loads(l) for l in open(sys.argv[3])]
                  if len(sys.argv) > 3 else None)
        propose(run_dir, active, tag="_r2" if active else "")
    elif cmd == "validate":
        run_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else JEVLOOP
        cand = sys.argv[3] if len(sys.argv) > 3 else "candidates.jsonl"
        verd = cand.replace("candidates", "verdicts")
        base = ([json.loads(l) for l in open(sys.argv[4])]
                if len(sys.argv) > 4 else None)
        validate(run_dir, cand, verd, base)
    elif cmd == "deposit":
        deposit()
