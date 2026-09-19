"""Scorer for jev_loop result dirs — shared by evolve_loop and ad-hoc use.

score_dir(dir) -> metrics over all <cid>.json files found:
  detection    clips correctly flagged corrupt/clean (composite_type vs truth)
  typing       composite_type == manifest operator (normalized)
  judge_calls  total VLM pair-judge calls used
  jev_calls    total Jev decisions
  per_op       per-operator detection counts
  failures     [{clip_id, op, expected, got, kind}] — kind = detection|typing

Truth normalization: reverse_segment -> reverse. A clip with no composite
type (loop died before verdict) counts as predicted 'none'.

CLI:  python jev_score.py <results_dir> [--csv]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "corpus_v3" / "manifest.json").read_text())
CLIPS = {c["id"]: c for c in MANIFEST["clips"]}

NORM = {"reverse_segment": "reverse"}


def truth_op(cid: str) -> str:
    return NORM.get(CLIPS[cid]["operator"], CLIPS[cid]["operator"])


def arbitrate(composite, ndis: int):
    """Seam-count arbitration (idempotent): room_swap = 2+ confirmed
    discontinuities, splice = <2. Applied here so result files written
    before arbitration was baked into run_clip score consistently."""
    if composite == "splice" and ndis >= 2:
        return "room_swap"
    if composite == "room_swap" and ndis < 2:
        return "splice"
    return composite


def score_dir(results_dir) -> dict:
    d = Path(results_dir)
    rows = []
    for f in sorted(d.glob("*.json")):
        cid = f.stem
        if cid not in CLIPS:
            continue
        r = json.loads(f.read_text())
        exp = truth_op(cid)
        got = arbitrate(r.get("composite_type") or "none",
                        len(r.get("discontinuous_at") or []))
        det_ok = (exp != "none") == (got != "none")
        type_ok = got == exp
        rows.append({"clip_id": cid, "op": exp, "got": got,
                     "det_ok": det_ok, "type_ok": type_ok,
                     "judge_calls": r.get("judge_calls", 0),
                     "jev_calls": r.get("jev_calls", 0),
                     "jev_errors": r.get("jev_errors", 0),
                     "n_candidates": r.get("n_candidates", 0)})

    per_op = {}
    for row in rows:
        st = per_op.setdefault(row["op"], [0, 0, 0])  # [det_ok, type_ok, n]
        st[2] += 1
        st[0] += row["det_ok"]
        st[1] += row["type_ok"]

    failures = [{"clip_id": r["clip_id"], "op": r["op"],
                 "expected": r["op"], "got": r["got"],
                 "kind": "detection" if not r["det_ok"] else "typing"}
                for r in rows if not r["type_ok"]]

    return {
        "n": len(rows),
        "detection": sum(r["det_ok"] for r in rows),
        "typing": sum(r["type_ok"] for r in rows),
        "judge_calls": sum(r["judge_calls"] for r in rows),
        "jev_calls": sum(r["jev_calls"] for r in rows),
        "jev_errors": sum(r["jev_errors"] for r in rows),
        "candidates": sum(r["n_candidates"] for r in rows),
        "per_op": {k: {"det": v[0], "type": v[1], "n": v[2]}
                   for k, v in per_op.items()},
        "failures": failures,
        "rows": rows,
    }


def report(sc: dict) -> str:
    lines = [f"n={sc['n']}  detection={sc['detection']}/{sc['n']}  "
             f"typing={sc['typing']}/{sc['n']}  "
             f"judge_calls={sc['judge_calls']} "
             f"({sc['judge_calls']/max(sc['candidates'],1):.0%} of candidates)  "
             f"jev_calls={sc['jev_calls']}  jev_errors={sc['jev_errors']}"]
    for op, v in sorted(sc["per_op"].items()):
        lines.append(f"  {op:12s} det={v['det']}/{v['n']}  type={v['type']}/{v['n']}")
    if sc["failures"]:
        lines.append("failures:")
        for f in sc["failures"]:
            lines.append(f"  [{f['kind']:9s}] {f['clip_id']:42s} "
                         f"{f['expected']} -> {f['got']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: jev_score.py <results_dir>")
    print(report(score_dir(sys.argv[1])))
