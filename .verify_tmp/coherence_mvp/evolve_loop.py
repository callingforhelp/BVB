"""Evolve-style prompt self-improvement loop over jev_loop's prompt pack.

Each round:
  1. score current pack (results dir)
  2. proposer LLM reads pack + failure classes + sample transcripts
     -> emits a FULL new pack JSON (text edits only)
  3. subset eval: all failures + all cleans + regression sample
     -> proceed only if it fixes more than it breaks, and never newly
        flags a clean clip (clean-arm veto)
  4. full-corpus confirm -> ACCEPT iff detection >= baseline AND
     typing strictly improves (ties -> fewer judge calls).
     If the accepted margin is <= 2 clips, re-run confirm once and
     require BOTH runs to beat baseline (Jev is non-deterministic).
  5. accepted pack -> prompt_pack_v{N}.json; every round writes a
     proposal record (diff fields, rationale, scores, decision).

The corpus is the objective; results are CALIBRATION on fitted data —
held-out source evals remain the validation gate downstream.

Run: ~/s1-spike/.venv/bin/python evolve_loop.py --base prompt_pack_v1.json --rounds 3
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "s1-spike"))

import jev_loop  # noqa: E402
import jev_score  # noqa: E402
from routes import ROUTES, cred  # noqa: E402

EVOLVE = ROOT / "results" / "evolve"
PROPOSALS = EVOLVE / "proposals"
EVOLVE.mkdir(parents=True, exist_ok=True)
PROPOSALS.mkdir(parents=True, exist_ok=True)

REGRESS_PER_OP = 3          # correct clips sampled per op for subset eval
TEXT_ROUTES = ["ark-plan", "opencode-go", "zenmux"]


# ---------------------------------------------------------------- eval

def eval_pack(pack: dict, tag: str, cids: list[str],
              workers: int = 8) -> dict:
    """Run jev_loop on cids into results/evolve/<tag>/ and score it."""
    out_dir = EVOLVE / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cids if not (out_dir / f"{c}.json").exists()]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(jev_loop.run_clip, c, pack, out_dir): c
                    for c in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    print(f"    eval FAIL {futs[f]}: {e}")
    return jev_score.score_dir(out_dir)


def subset_cids(base_score: dict) -> list[str]:
    """All failures + all cleans + REGRESS_PER_OP correct clips per op."""
    fail = {f["clip_id"] for f in base_score["failures"]}
    cleans = {r["clip_id"] for r in base_score["rows"] if r["op"] == "none"}
    picked = set()
    per_op = {}
    for r in sorted(base_score["rows"], key=lambda x: x["clip_id"]):
        if r["op"] == "none" or r["clip_id"] in fail or not r["type_ok"]:
            continue
        if per_op.get(r["op"], 0) >= REGRESS_PER_OP:
            continue
        per_op[r["op"]] = per_op.get(r["op"], 0) + 1
        picked.add(r["clip_id"])
    return sorted(fail | cleans | picked)


# ---------------------------------------------------------------- proposer

PROPOSER_SYS = (
    "You are improving the prompt/state descriptions fed to Jev, a "
    "probabilistic decision model that reads serialized video-forensic "
    "evidence and decides (a) which candidate boundaries to send to a VLM "
    "continuity judge, (b) whether the clip is corrupted, and (c) the "
    "corruption type. You will receive the current prompt pack JSON and a "
    "failure report. Rewrite the pack text to fix the failure classes "
    "WITHOUT breaking cases that work.\n"
    "RULES:\n"
    "- Return ONLY a JSON object {\"pack\": <full new pack>, \"rationale\": "
    "\"<2-4 sentences>\"}. No markdown fences, no prose outside JSON.\n"
    "- You may ONLY edit string VALUES: task, signals.*, typing_guide, "
    "candidate_note, questions.*, type_desc.*, none_desc, conclude_crit, "
    "judge_crit_template. Do not add/remove keys; keep {i},{frame},{secs},"
    "{type},{desc} placeholders where present.\n"
    "- Prefer calibrated knowledge (base rates, disambiguators, what a "
    "signal CANNOT distinguish) over stronger assertions.\n"
    "- Keep each field concise — long essays dilute the decision signal.\n"
    "- Do not claim thresholds you cannot justify from the failure data.\n"
    "- HARD VETO (auto-reject, verified by eval): any edit that raises "
    "false positives on CLEAN clips, or breaks clips that were already "
    "correct. Broadening what counts as corruption evidence WILL flag "
    "clean clips — every clip contains some repeated/static frames.\n"
    "- Prefer NARROW additive edits: typing disambiguation, precedence "
    "rules, 'if X then suspect Y not Z'. Do not rewrite a signal's core "
    "meaning to fix a typing error; the same text also drives detection.\n"
    "- recur_frac, dup_dense, echo are the PRIMARY detectors for loop, "
    "timewarp, reverse — weakening their decisiveness breaks those ops.")


def failure_report(base_score: dict, results_dir: Path) -> dict:
    """Compact failure classes + evidence context for the proposer."""
    classes = {}
    for f in base_score["failures"]:
        key = f"{f['expected']}->{f['got']}"
        classes.setdefault(key, []).append(f["clip_id"])
    examples = []
    for f in base_score["failures"][:8]:
        cid = f["clip_id"]
        st = jev_loop.load_clip_state(cid)
        r = json.loads((results_dir / f"{cid}.json").read_text())
        examples.append({
            "clip_id": cid, "truth": f["expected"], "predicted": f["got"],
            "kind": f["kind"],
            "free_signals": st["free_signals"],
            "discontinuous_at": r.get("discontinuous_at"),
            "transcript": r.get("transcript", [])[:6],
        })
    # clean-arm context: what quiet/correct decisions look like, so the
    # proposer can see the false-positive side of any broadening edit.
    clean_examples = []
    for r in base_score["rows"]:
        if r["op"] != "none":
            continue
        cid = r["clip_id"]
        st = jev_loop.load_clip_state(cid)
        clean_examples.append({"clip_id": cid,
                               "free_signals": st["free_signals"]})
        if len(clean_examples) == 3:
            break
    # correct loop exemplar so recur-heavy edits can be sanity-checked
    loop_example = None
    for r in base_score["rows"]:
        if r["op"] == "loop" and r["type_ok"]:
            st = jev_loop.load_clip_state(r["clip_id"])
            loop_example = {"clip_id": r["clip_id"],
                            "free_signals": st["free_signals"]}
            break
    return {"classes": {k: v for k, v in classes.items()},
            "examples": examples,
            "clean_examples": clean_examples,
            "loop_example": loop_example}


def propose(pack: dict, report: dict,
            prior_rejections: list[dict] | None = None) -> tuple[dict, str]:
    user = json.dumps({
        "current_pack": pack,
        "failure_report": report,
        "prior_rejected_proposals": prior_rejections or [],
        "current_score_context": (
            "baseline detection=88/90 typing=82/90 at 34% judge calls; "
            "residual errors shown in failure_report; prior_rejected_"
            "proposals shows edits already tried and vetoed — do NOT "
            "repeat their approach")}, indent=1)
    body_msgs = [{"role": "system", "content": PROPOSER_SYS},
                 {"role": "user", "content": user}]
    errs = []
    for name in TEXT_ROUTES:
        cfg = ROUTES[name]
        try:
            key = cfg["key"]()
        except Exception as e:
            errs.append(f"{name}: key {e}")
            continue
        body = {"model": cfg["model"], "messages": body_msgs,
                "temperature": 0.2, "max_completion_tokens": 16384}
        if cfg.get("thinking"):
            body["thinking"] = {"type": "enabled"}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {key}"}
        if cfg.get("headers"):
            headers.update(cfg["headers"]())
        req = urllib.request.Request(f"{cfg['url']}/chat/completions",
                                     data=json.dumps(body).encode(),
                                     method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read().decode())
            text = (data.get("choices") or [{}])[0].get(
                "message", {}).get("content") or ""
            obj = _extract_json(text)
            new_pack = _validate_pack(obj["pack"])
            return new_pack, obj.get("rationale", ""), name
        except Exception as e:
            errs.append(f"{name}: {str(e)[:160]}")
    raise RuntimeError("all proposer routes failed | " + " | ".join(errs))


def _extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    start = t.find("{")
    return json.loads(t[start:]) if start >= 0 else json.loads(t)


def _validate_pack(p: dict) -> dict:
    req_top = {"task", "conclude_crit", "judge_crit_template", "signals",
               "typing_guide", "candidate_note", "questions", "type_desc",
               "none_desc"}
    missing = req_top - set(p)
    assert not missing, f"pack missing keys: {missing}"
    for k in ("recur_frac_max", "dup_dense_last_frame", "echo_best_score",
              "scenecut_iframe_frames", "photo_jump_top3"):
        assert k in p["signals"], f"signals.{k} missing"
    for k in ("action", "corrupted", "break_type", "sufficient",
              "is_type_template"):
        assert k in p["questions"], f"questions.{k} missing"
    for t in ("loop", "reverse", "room_swap", "splice", "timewarp"):
        assert t in p["type_desc"], f"type_desc.{t} missing"
    for k, v in p.items():
        if k in ("version", "parent", "notes"):
            continue
        assert k in req_top, f"unexpected pack key: {k}"
        if isinstance(v, dict):
            assert all(isinstance(x, str) for x in v.values()), \
                f"pack.{k} has non-string leaf"
        else:
            assert isinstance(v, str), f"pack.{k} not a string"
    return p


def diff_fields(a: dict, b: dict) -> list[str]:
    out = []
    for k in a:
        if a[k] != b.get(k):
            out.append(k)
    return out


# ---------------------------------------------------------------- gate

def gate_subset(base_sub: dict, prop_sub: dict) -> tuple[bool, str]:
    """Proceed to full confirm iff fixes > breaks and no clean regression."""
    base_wrong = {f["clip_id"] for f in base_sub["failures"]}
    prop_wrong = {f["clip_id"] for f in prop_sub["failures"]}
    fixed = base_wrong - prop_wrong
    broken = prop_wrong - base_wrong
    clean_regress = [f["clip_id"] for f in prop_sub["failures"]
                     if f["expected"] == "none"]
    ok = len(fixed) > len(broken) and not clean_regress
    why = (f"fixed={sorted(fixed)} broken={sorted(broken)} "
           f"clean_regress={clean_regress}")
    return ok, why


def gate_confirm(base: dict, prop: dict) -> tuple[bool, str]:
    det_ok = prop["detection"] >= base["detection"]
    type_better = prop["typing"] > base["typing"]
    det_better = prop["detection"] > base["detection"]
    accept = (det_ok and type_better) or det_better
    margin = (prop["detection"] - base["detection"]
              + prop["typing"] - base["typing"])
    why = (f"det {base['detection']}->{prop['detection']} "
           f"type {base['typing']}->{prop['typing']} "
           f"judges {base['judge_calls']}->{prop['judge_calls']}")
    return accept, why, margin


# ---------------------------------------------------------------- loop

def next_version(pack: dict) -> int:
    return int(pack.get("version") or 0) + 1


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "prompt_pack_v1.json"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--propose-only", action="store_true",
                    help="emit one proposal record without evaluating")
    ap.add_argument("--base-results", default=str(ROOT / "results" / "jevloop"),
                    help="precomputed results dir for the base pack")
    args = ap.parse_args()

    pack = jev_loop.load_pack(args.base)
    all_cids = sorted(jev_loop.CLIPS)
    base_score = jev_score.score_dir(args.base_results)
    print(f"BASE v{pack.get('version')} ({jev_loop.pack_hash(pack)}): "
          f"det={base_score['detection']}/{base_score['n']} "
          f"type={base_score['typing']}/{base_score['n']} "
          f"judges={base_score['judge_calls']}")

    rejections = []
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        ver = next_version(pack)
        report = failure_report(base_score, Path(args.base_results))
        print(f"\n=== round {rnd}: proposing v{ver} "
          f"({sum(len(v) for v in report['classes'].values())} failures, "
          f"{len(report['classes'])} classes) ===")
        try:
            prop_pack, rationale, route = propose(pack, report, rejections)
        except Exception as e:
            print(f"propose failed: {e}")
            break
        prop_pack["version"] = ver
        prop_pack["parent"] = pack.get("version")
        prop_pack["notes"] = f"evolve round {rnd} via {route}"
        tag = f"v{ver}_r{rnd}_{jev_loop.pack_hash(prop_pack)[:6]}"
        rec = {"round": rnd, "parent": pack.get("version"),
               "version": ver, "route": route, "rationale": rationale,
               "diff_fields": diff_fields(pack, prop_pack),
               "pack": prop_pack, "decision": None,
               "scores": {}, "ts": time.time()}
        print(f"proposed ({route}): fields={rec['diff_fields']}")
        print(f"  rationale: {rationale[:300]}")

        if args.propose_only:
            rec["decision"] = "propose_only"
            (PROPOSALS / f"round{rnd}_v{ver}.json").write_text(
                json.dumps(rec, indent=1))
            print(f"propose-only: wrote proposals/round{rnd}_v{ver}.json")
            break

        # stage 1: subset eval
        sub = subset_cids(base_score)
        prop_sub = eval_pack(prop_pack, f"{tag}_sub", sub, args.workers)
        base_sub = {"failures": [f for f in base_score["failures"]
                                 if f["clip_id"] in set(sub)]}
        ok, why = gate_subset(base_sub, prop_sub)
        rec["scores"]["subset"] = {"n": prop_sub["n"],
                                   "detection": prop_sub["detection"],
                                   "typing": prop_sub["typing"],
                                   "judge_calls": prop_sub["judge_calls"],
                                   "gate": why}
        print(f"subset ({prop_sub['n']} clips): det={prop_sub['detection']} "
              f"type={prop_sub['typing']} | {why}")
        if not ok:
            rec["decision"] = "rejected_subset"
            (PROPOSALS / f"round{rnd}_v{ver}.json").write_text(
                json.dumps(rec, indent=1))
            rejections.append({"diff_fields": rec["diff_fields"],
                               "rationale": rationale[:400],
                               "veto": why})
            print(f"REJECTED at subset gate ({time.time()-t0:.0f}s)")
            continue

        # stage 2: full-corpus confirm
        prop_full = eval_pack(prop_pack, f"{tag}_full", all_cids, args.workers)
        accept, why, margin = gate_confirm(base_score, prop_full)
        rec["scores"]["confirm"] = {"detection": prop_full["detection"],
                                     "typing": prop_full["typing"],
                                     "judge_calls": prop_full["judge_calls"],
                                     "gate": why}
        print(f"confirm: {why} -> margin={margin}")
        if accept and margin <= 2:
            prop_full2 = eval_pack(prop_pack, f"{tag}_full_b",
                                   all_cids, args.workers)
            accept2, why2, _ = gate_confirm(base_score, prop_full2)
            rec["scores"]["confirm_rerun"] = {
                "detection": prop_full2["detection"],
                "typing": prop_full2["typing"],
                "judge_calls": prop_full2["judge_calls"], "gate": why2}
            print(f"confirm re-run (margin<=2): {why2}")
            accept = accept and accept2
            prop_full = prop_full2  # keep the stricter estimate

        rec["decision"] = "accepted" if accept else "rejected_confirm"
        (PROPOSALS / f"round{rnd}_v{ver}.json").write_text(
            json.dumps(rec, indent=1))
        if accept:
            out = ROOT / f"prompt_pack_v{ver}.json"
            out.write_text(json.dumps(prop_pack, indent=1))
            pack, base_score = prop_pack, prop_full
            print(f"ACCEPTED -> {out.name} ({time.time()-t0:.0f}s)")
        else:
            rejections.append({"diff_fields": rec["diff_fields"],
                               "rationale": rationale[:400],
                               "veto": why})
            print(f"REJECTED at confirm ({time.time()-t0:.0f}s)")

    print(f"\nfinal pack v{pack.get('version')} "
          f"({jev_loop.pack_hash(pack)})")


if __name__ == "__main__":
    main()
