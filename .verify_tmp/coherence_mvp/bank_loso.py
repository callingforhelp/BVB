"""LOSO evaluation of the strip-episode reasoning bank.

Per held-out source S (6 folds):
  1. build_bank(all sources - S)  -> results/banks/loso_<S>/
  2. jev_loop --bank on S's clips -> results/banks/eval_<S>/
  3. score and compare vs the no-bank baseline on the same clips

Answers: does retrieval-grounded precedent injection improve typing
(the room_swap/splice/static-swap residual class) on unseen sources?

Run: ~/s1-spike/.venv/bin/python bank_loso.py --pack prompt_pack_v2.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import jev_loop  # noqa: E402
import jev_score  # noqa: E402
import strip_bank as sb  # noqa: E402

BANKS = ROOT / "results" / "banks"
BANKS.mkdir(parents=True, exist_ok=True)


def sources() -> list[str]:
    return sorted({c["source"] for c in jev_loop.CLIPS.values()})


def clips_of(source: str) -> list[str]:
    return sorted(c for c, v in jev_loop.CLIPS.items()
                  if v["source"] == source)


def run_eval(pack, bank_dir, cids, out_dir, workers=8):
    bank = sb.StripBank.load(bank_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cids if not (out_dir / f"{c}.json").exists()]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(jev_loop.run_clip, c, pack, out_dir, bank): c
                    for c in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    print(f"    eval FAIL {futs[f]}: {e}")
    return jev_score.score_dir(out_dir)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / "prompt_pack_v2.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--baseline", default=None,
                    help="no-bank results dir for per-source comparison")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--only-source", default=None)
    args = ap.parse_args()

    pack = jev_loop.load_pack(args.pack)
    print(f"pack v{pack.get('version')} ({jev_loop.pack_hash(pack)})")

    all_src = sources()
    fold_src = [args.only_source] if args.only_source else all_src
    agg = {"detection": 0, "typing": 0, "n": 0, "judge_calls": 0}
    base_agg = {"detection": 0, "typing": 0, "n": 0, "judge_calls": 0}

    for held in fold_src:
        bank_dir = BANKS / f"loso_{held}"
        if not (bank_dir / "index.json").exists():
            print(f"[{held}] building bank from {len(all_src)-1} sources...")
            bank, skipped = sb.build_bank(set(all_src) - {held}, bank_dir)
            print(f"[{held}] bank: {len(bank._ids)} episodes "
                  f"({skipped} strips skipped)")
        else:
            bank = sb.StripBank.load(bank_dir)
            print(f"[{held}] bank cached: {len(bank._ids)} episodes")
        if args.build_only:
            continue

        eval_dir = BANKS / f"eval_{held}_{jev_loop.pack_hash(pack)}"
        sc = run_eval(pack, bank_dir, clips_of(held), eval_dir, args.workers)
        print(f"[{held}] banked: det={sc['detection']}/{sc['n']} "
              f"type={sc['typing']}/{sc['n']} judges={sc['judge_calls']}")
        for k in ("detection", "typing", "n", "judge_calls"):
            agg[k] += sc[k]

        if args.baseline:
            bsc_all = jev_score.score_dir(args.baseline)
            keep = set(clips_of(held))
            bsc = {"detection": 0, "typing": 0, "n": 0, "judge_calls": 0}
            for r in bsc_all["rows"]:
                if r["clip_id"] in keep:
                    bsc["n"] += 1
                    bsc["detection"] += r["det_ok"]
                    bsc["typing"] += r["type_ok"]
                    bsc["judge_calls"] += r["judge_calls"]
            print(f"[{held}]  base : det={bsc['detection']}/{bsc['n']} "
                  f"type={bsc['typing']}/{bsc['n']} "
                  f"judges={bsc['judge_calls']}")
            for k in base_agg:
                base_agg[k] += bsc[k]

    if not args.build_only:
        print(f"\n=== LOSO aggregate ===")
        print(f"banked: det={agg['detection']}/{agg['n']} "
              f"type={agg['typing']}/{agg['n']} judges={agg['judge_calls']}")
        if args.baseline:
            print(f"base  : det={base_agg['detection']}/{base_agg['n']} "
                  f"type={base_agg['typing']}/{base_agg['n']} "
                  f"judges={base_agg['judge_calls']}")


if __name__ == "__main__":
    main()
