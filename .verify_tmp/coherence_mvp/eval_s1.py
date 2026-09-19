"""Phase 3b eval: replay corpus clips through a Tinker-served model and score.

Patches jev_loop.jev with the S1Jev adapter (s1_serve.py) so run_clip drives
the Tinker sampler through the exact OpenJev serving format. Verdicts still
come from the cached pairjudge stores; only the Jev decision calls change.

    .venv/bin/python eval_s1.py --model-path tinker://.../final \
        --out results/jevloop_s1_tuned [--only <cids>] [--bank]

With no --model-path it evaluates the base model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import jev_loop as J          # noqa: E402
import jev_score              # noqa: E402
import s1_serve               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None,
                    help="tinker:// sampler checkpoint; omit for base model")
    ap.add_argument("--pack", default=str(ROOT / "prompt_pack_v2.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--source", default=None,
                    help="restrict to clips from one manifest source")
    ap.add_argument("--bank", default=None,
                    help="strip-bank dir for precedent injection")
    ap.add_argument("--workers", type=int, default=4,
                    help="low default: sampler calls are serialized upstream")
    args = ap.parse_args()

    adapter = s1_serve.S1Jev(model_path=args.model_path)
    J.jev = adapter.answers

    pack = J.load_pack(args.pack)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = None
    if args.bank:
        import strip_bank as sb
        bank = sb.StripBank.load(args.bank)

    cids = args.only or sorted(J.CLIPS)
    if args.source:
        cids = [c for c in cids if J.CLIPS[c]["source"] == args.source]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    todo = [c for c in cids if not (out_dir / f"{c}.json").exists()]
    print(f"{len(todo)} clips to run -> {out_dir}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(J.run_clip, c, pack, out_dir, bank): c for c in todo}
        done = 0
        for f in as_completed(futs):
            cid = futs[f]
            try:
                f.result()
            except Exception as e:
                print(f"  FAIL {cid}: {e}")
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(todo)}")

    sc = jev_score.score_dir(out_dir)
    print(json.dumps({k: v for k, v in sc.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
