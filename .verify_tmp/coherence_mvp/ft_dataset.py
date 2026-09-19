"""Phase 3a — fine-tune dataset emitter.

Replays jev_loop transcripts to reconstruct the exact wire-format states
Jev faced, then labels each state with an ORACLE policy + manifest truth —
NOT Jev's own answers (that would clone its errors, incl. the demoted
`sufficient` mush).

Output rows (Fireworks-compatible JSONL):
  {"messages":[{"role":"user","content":<{"state":..., "questions":...}>},
               {"role":"assistant","content":<{"answers": {...}}>}]}

Oracle policy (HANDOFF locked + ProgressGate addendum):
  - edited clip + decisive free signal consistent w/ truth -> conclude
  - edited clip -> judge unjudged candidate NEAREST a true seam, until
    all seam-adjacent (+-8f) candidates are resolved; then conclude
  - edited clip + seams exhausted + no discontinuity + last-2 reveals
    continuous -> ABSTAIN 'corrupt-but-untyped' (flatlined info gain —
    better than conclude-none on sub-perceptual/coverage-gap clips)
  - clean clip -> judge the MIN_JUDGED strongest candidates (photo_jump z)
    then conclude; <2 candidates -> conclude immediately
  - loop/reverse/timewarp with matching decisive signal -> conclude at
    iter 0; withOUT it -> treated like splice/swap (seam-adjacent path)
  'abstain' is a standing criterion in every row's action question.

Verdict labels are EVIDENCE-derived, not raw truth: a quiet-evidence
splice and a clean clip are state-identical, so labeling corrupted=high
on one and low on the other is unlearnable and teaches hallucinated
certainty (breaks near-zero-FP). corrupted = calibrated belief from
what the state shows; break_type = the typing_guide rule applied to
revealed evidence (same rule code arbitration applies downstream).

Variants:
  plain  - no retrieval precedents (replays jevloop/evolve as-is)
  rag    - per-candidate precedent summaries from loso_<source> banks
           (source-excluded -> safe in BOTH train and val of every fold)
Augmentation: for rows with >=2 revealed candidates, one extra row with
a deterministic random subset of reveals masked (partial observation).

LOSO: results/ft_dataset/{variant}/loso_<src>/{train,val}.jsonl — train
excludes held-out source; val is only that source. Plus all.jsonl.

Usage:
  python3 ft_dataset.py [--out results/ft_dataset] [--workers 8] [--no-aug]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import jev_loop as J          # noqa: E402  (load_clip_state/state_view/questions_for)
from jev_score import NORM    # noqa: E402  (truth op normalization)

CLIPS = J.CLIPS
MIN_JUDGED = J.MIN_JUDGED
SEAM_TOL = 8                  # frames — same as strip_bank.make_entry

# ---------------------------------------------------------------- pack registry
# evolve result files record pack_hash computed BEFORE precedents_note was
# added to DEFAULT_PACK (Phase 2). Map recorded hash -> proposal file.
PROPOSAL_BY_OLD_HASH = {
    "4e09a698": "results/evolve/proposals/round2_v3.json",
    "5b726464": "results/evolve/proposals/round3_v3.json",
    "4a46d483": "results/evolve/proposals/run1_noguardrails/round1_v2.json",
    "42fcf468": "results/evolve/proposals/run1_noguardrails/round2_v2.json",
    "84dde051": "results/evolve/proposals/run1_noguardrails/round3_v2.json",
}
_pack_cache: dict[str, dict] = {}


def pack_for(result: dict) -> dict:
    """load_pack-equivalent merge, resolved by recorded pack_hash.
    Benign race: threads may compute the same pack; values are equal."""
    h = result.get("pack_hash")
    if h in _pack_cache:
        return _pack_cache[h]
    if h == "16a2260f":
        raw = json.loads((ROOT / "prompt_pack_v2.json").read_text())
    elif h in PROPOSAL_BY_OLD_HASH:
        raw = json.loads((ROOT / PROPOSAL_BY_OLD_HASH[h])
                         .read_text())["pack"]
    else:                      # None / unknown -> v1 baseline strings
        raw = json.loads((ROOT / "prompt_pack_v1.json").read_text())
    pack = dict(J.DEFAULT_PACK)
    pack.update(raw)
    for k in ("signals", "questions", "type_desc"):
        pack[k] = {**J.DEFAULT_PACK[k], **raw.get(k, {})}
    _pack_cache[h] = pack
    return pack


# ---------------------------------------------------------------- truth + signals

def truth_op(cid: str) -> str:
    return NORM.get(CLIPS[cid]["operator"], CLIPS[cid]["operator"])


def seam_frames(cid: str) -> list[int]:
    return [int(round(s * 30)) for s in CLIPS[cid].get("breaks_s", [])]


def signal_claim(fs: dict) -> str | None:
    """What the free signals claim, in typing_guide order.

    Measured on corpus_v3: echo>10 fires on EVERY loop and swap too (any
    doubled discontinuity echoes) — it only isolates reverse when recur
    is quiet. recur>=0.9 + dup_dense is the static-segment signature
    (guide caveat: suspect room_swap, not loop); it appears only on the
    09c1414f1b room_swaps, never on clean footage."""
    if fs["recur_frac_max"] >= 0.9:
        if fs["dup_dense_last_frame"] is not None:
            return "room_swap"          # static-segment caveat
        return "loop"
    if fs["dup_dense_last_frame"] is not None:
        return "timewarp"
    if fs["echo_best_score"] > 10:
        return "reverse"
    return None


# ---------------------------------------------------------------- oracle

ABSTAIN_CRIT = ("stop and report 'corrupt-but-untyped' for human review "
                "— use only when further checks won't change the picture "
                "(recent checks returned no new information: evidence has "
                "flatlined)")


def oracle_action(cid: str, st: dict, hist: list[int]) -> tuple[str, str]:
    """-> (action, reason). Uses truth for POLICY only (which evidence to
    gather / when to stop), never for verdict labels.

    hist = reveal-order list of candidate indices (time order, not index
    order) — flatline is measured on the LAST 2 reveals."""
    fs = st["free_signals"]
    cands = st["candidates"]
    unjudged = [i for i, c in enumerate(cands) if not c["revealed"]]
    judged_n = len(cands) - len(unjudged)
    op = truth_op(cid)
    seams = seam_frames(cid)

    # decisive free signal — except the room_swap caveat, which is a
    # suspicion the guide itself says to verify at the seams first
    if op != "none" and op != "room_swap" and signal_claim(fs) == op:
        return "conclude", "decisive_signal"

    if op == "none":
        if judged_n >= MIN_JUDGED or not unjudged:
            return ("conclude", "clean_floor_met" if judged_n >= MIN_JUDGED
                    else "no_candidates_left")
        # strongest remaining = highest photo_jump z near the candidate
        pj = {p["frame"]: p["z"] for p in fs["photo_jump_top3"]}
        i = max(unjudged, key=lambda i: pj.get(cands[i]["frame"], 0.0))
        return f"judge_{i}", "clean_probe"

    # edited: judge unjudged candidate nearest a true seam
    seam_adj = [i for i in unjudged
                if any(abs(cands[i]["frame"] - s) <= SEAM_TOL for s in seams)]
    if seam_adj:
        i = min(seam_adj,
                key=lambda i: min(abs(cands[i]["frame"] - s) for s in seams))
        return f"judge_{i}", "seam_adjacent"
    ndis = sum(1 for c in cands
               if c["revealed"] and c["revealed"].get("continuous") is False)
    if ndis:
        return "conclude", "seams_resolved"
    # stagnation check (ProgressGate addendum): no discontinuity found and
    # the last 2 reveals were both continuous -> blind probing gains
    # nothing -> abstain 'corrupt-but-untyped' instead of burning calls
    if len(hist) >= 2 and all(cands[i]["revealed"].get("continuous")
                            for i in hist[-2:]):
        return "abstain", "flatlined"
    if not unjudged:
        return "abstain", "no_evidence"
    i = min(unjudged,
            key=lambda i: min(abs(cands[i]["frame"] - s) for s in seams))
    return f"judge_{i}", "probe_blind"


def guide_type(st: dict) -> str:
    """typing_guide applied to REVEALED evidence (matches code arbitration)."""
    sig = signal_claim(st["free_signals"])
    if sig:
        return sig
    ndis = sum(1 for c in st["candidates"]
               if c["revealed"] and c["revealed"].get("continuous") is False)
    if ndis >= 2:
        return "room_swap"
    if ndis == 1:
        return "splice"
    return "none"


def gold_answers(cid: str, st: dict, pack: dict,
                 hist: list[int]) -> tuple[dict, str]:
    """Evidence-calibrated labels + oracle action. Returns (answers, reason)."""
    action, why = oracle_action(cid, st, hist)
    cands = st["candidates"]
    judged = [c for c in cands if c["revealed"]]
    unjudged_n = len(cands) - len(judged)
    ndis = sum(1 for c in judged
               if c["revealed"].get("continuous") is False)
    sig = signal_claim(st["free_signals"])
    gt = guide_type(st)

    # corrupted: belief calibrated to what the state actually shows
    if ndis > 0 or sig:
        corrupted = 0.9
    elif not judged:
        corrupted = 0.45 if cands else 0.1
    elif unjudged_n:
        corrupted = 0.25
    else:
        corrupted = 0.08

    # break_type distribution
    if ndis > 0 or sig:
        probs = {t: round(0.15 / 5, 3) for t in pack["type_desc"]}
        probs[gt] = 0.85
        conf = 0.85
    elif unjudged_n:               # provisional read, candidates pending
        probs = {"none": 0.5, "splice": 0.2, "room_swap": 0.12,
                 "loop": 0.06, "reverse": 0.06, "timewarp": 0.06}
        conf = 0.5
    else:                          # everything judged, quiet
        probs = {t: round(0.1 / 5, 3) for t in pack["type_desc"]}
        probs["none"] = 0.9
        conf = 0.9

    crit = [f"judge_{i}" for i, c in enumerate(cands) if not c["revealed"]]
    crit += ["conclude", "abstain"]   # abstain = standing option (gate v2)
    act_probs = {c: round(0.1 / max(len(crit) - 1, 1), 4) for c in crit}
    act_probs[action] = 0.9

    answers = {
        "action": {"type": "choice", "choice": action,
                   "confidence": 0.9, "probabilities": act_probs},
        "corrupted": {"type": "noul", "noul": corrupted},
        "break_type": {"type": "choice", "choice": gt,
                       "confidence": conf, "probabilities": probs},
        "sufficient": {"type": "noul",
                       "noul": 0.85 if action in ("conclude", "abstain")
                       else 0.15},
    }
    for t in pack["type_desc"]:
        answers[f"is_{t}"] = {"type": "noul",
                              "noul": round(probs.get(t, 0.02), 3)}
    return answers, why


# ---------------------------------------------------------------- replay + aug

def mask_row(row: dict, pack: dict, cid: str, bank, rng: random.Random,
             st_template: dict, hist: list[int]) -> dict | None:
    """Evidence-subset augmentation: re-seal a subset of revealed verdicts,
    recompute oracle gold on the masked state."""
    revealed = [i for i, c in enumerate(st_template["candidates"])
                if c["revealed"]]
    if len(revealed) < 2:
        return None
    keep = set(rng.sample(revealed, rng.randint(1, len(revealed) - 1)))
    st = copy.deepcopy(st_template)
    for i in revealed:
        if i not in keep:
            st["candidates"][i]["revealed"] = None
    answers, why = gold_answers(cid, st, pack,
                                [i for i in hist if i in keep])
    ques = J.questions_for(st, pack)
    ques["action"]["criteria"]["abstain"] = ABSTAIN_CRIT
    return {"state": J.state_view(st, pack, bank),
            "questions": ques,
            "answers": answers, "iter": row["iter"], "oracle": "aug:" + why}


# ---------------------------------------------------------------- emit

def row_json(r: dict) -> str:
    user = json.dumps({"state": json.dumps(r["state"]),
                       "questions": r["questions"]})
    asst = json.dumps({"answers": r["answers"]})
    return json.dumps({"messages": [{"role": "user", "content": user},
                                    {"role": "assistant", "content": asst}]})


def process_file(path: Path, variant: str, banks: dict, do_aug: bool,
                 rng: random.Random) -> list[dict]:
    result = json.loads(path.read_text())
    cid = result["clip_id"]
    if cid not in CLIPS:
        return []
    pack = pack_for(result)
    src = CLIPS[cid]["source"]
    bank = banks.get(src) if variant == "rag" else None
    out = []
    st = J.load_clip_state(cid, bank=bank)   # replay + aug snapshot
    hist: list[int] = []                     # reveal order (time, not index)
    for entry in result.get("transcript", []):
        state = J.state_view(st, pack, bank)
        ques = J.questions_for(st, pack)
        ques["action"]["criteria"]["abstain"] = ABSTAIN_CRIT
        answers, why = gold_answers(cid, st, pack, hist)
        row = {"state": state, "questions": ques, "answers": answers,
               "iter": entry["iter"], "oracle": why,
               "clip_id": cid, "source": src,
               "pack_hash": result.get("pack_hash")}
        out.append(row)
        if do_aug:
            aug = mask_row(row, pack, cid, bank, rng, st, hist)
            if aug:
                aug.update({"clip_id": cid, "source": src,
                            "pack_hash": result.get("pack_hash")})
                out.append(aug)
        if "error" in entry:
            break
        # advance state exactly like run_clip
        unjudged = [i for i, c in enumerate(st["candidates"])
                    if not c["revealed"]]
        act = entry.get("action", "conclude")
        if act == "conclude":
            fan = entry.get("fan") or {}
            vals = sorted((v for v in fan.values() if v is not None),
                          reverse=True)
            margin = vals[0] - vals[1] if len(vals) > 1 else 1.0
            ok = ((entry.get("corrupted") or 0) < 0.5
                  or len(st["candidates"]) - len(unjudged) >= MIN_JUDGED
                  or margin >= J.FAN_MARGIN)
            if ok or not unjudged:
                break
            st["candidates"][unjudged[0]]["revealed"] = \
                st["candidates"][unjudged[0]]["verdict"]
            hist.append(unjudged[0])
            continue
        if act.startswith("judge_"):
            i = int(act.split("_")[1])
            if 0 <= i < len(st["candidates"]) \
                    and not st["candidates"][i]["revealed"]:
                st["candidates"][i]["revealed"] = \
                    st["candidates"][i]["verdict"]
                hist.append(i)
                continue
        if not unjudged:
            break
        st["candidates"][unjudged[0]]["revealed"] = \
            st["candidates"][unjudged[0]]["verdict"]
        hist.append(unjudged[0])
    return out


def transcript_files() -> list[Path]:
    files = sorted((ROOT / "results" / "jevloop").glob("*.json"))
    for d in sorted((ROOT / "results" / "evolve").iterdir()):
        if d.is_dir():
            files += sorted(d.glob("*.json"))
    for d in sorted((ROOT / "results" / "banks").glob("eval_*")):
        files += sorted(d.glob("*.json"))
    return [f for f in files
            if json.loads(f.read_text()).get("clip_id") in CLIPS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ft_dataset"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["plain", "rag"])
    args = ap.parse_args()
    out_root = Path(args.out)
    rng = random.Random(20260918)

    banks = {}
    if "rag" in args.variants:
        import strip_bank as _sb
        for src in {c["source"] for c in CLIPS.values()}:
            d = ROOT / "results" / "banks" / f"loso_{src}"
            banks[src] = _sb.StripBank.load(d)
        print(f"loaded {len(banks)} loso banks")

    files = transcript_files()
    print(f"{len(files)} transcript files")

    for variant in args.variants:
        rows_by_clip: dict[str, list[dict]] = defaultdict(list)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_file, f, variant, banks,
                              not args.no_aug,
                              random.Random(rng.randint(0, 1 << 30))): f
                    for f in files}
            for i, fut in enumerate(as_completed(futs)):
                try:
                    for r in fut.result():
                        rows_by_clip[r["clip_id"]].append(r)
                except Exception as e:
                    print(f"  !! {futs[fut].name}: {e}")
        # dedup identical (user,assistant) content
        seen, rows = set(), []
        for cid, rs in rows_by_clip.items():
            for r in rs:
                line = row_json(r)
                h = hashlib.sha1(line.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                rows.append((line, r))
        vdir = out_root / variant
        vdir.mkdir(parents=True, exist_ok=True)
        meta = [{k: r[k] for k in ("clip_id", "source", "iter", "oracle",
                                  "pack_hash")} for _, r in rows]
        (vdir / "all.jsonl").write_text(
            "\n".join(l for l, _ in rows) + "\n")
        (vdir / "all.meta.jsonl").write_text(
            "\n".join(json.dumps(m) for m in meta) + "\n")
        sources = sorted({c["source"] for c in CLIPS.values()})
        for held in sources:
            fdir = vdir / f"loso_{held}"
            fdir.mkdir(exist_ok=True)
            tr = [(l, r) for l, r in rows if r["source"] != held]
            va = [(l, r) for l, r in rows if r["source"] == held]
            (fdir / "train.jsonl").write_text(
                "\n".join(l for l, _ in tr) + "\n")
            (fdir / "val.jsonl").write_text(
                "\n".join(l for l, _ in va) + "\n")
            (fdir / "train.meta.jsonl").write_text(
                "\n".join(json.dumps({k: r[k] for k in
                                      ("clip_id", "source", "iter",
                                       "oracle", "pack_hash")})
                          for _, r in tr) + "\n")
            (fdir / "val.meta.jsonl").write_text(
                "\n".join(json.dumps({k: r[k] for k in
                                      ("clip_id", "source", "iter",
                                       "oracle", "pack_hash")})
                          for _, r in va) + "\n")
        stats = Counter(r["oracle"] for _, r in rows)
        acts = Counter(r["answers"]["action"]["choice"].split("_")[0]
                       for _, r in rows)
        print(f"[{variant}] {len(rows)} rows "
              f"({sum(1 for _, r in rows if r['oracle'].startswith('aug'))} aug) "
              f"| actions: {dict(acts)} | oracle: {dict(stats)}")


if __name__ == "__main__":
    main()
