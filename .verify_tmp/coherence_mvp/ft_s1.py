"""Phase 3b dataset emitter — SystemOne-native format.

The messages-JSONL from ft_dataset.py targets chat-SFT ("assistant emits the
answers JSON"), but OpenJev's real serving contract (github.com/ekzhang/
openjev-sglang, src/openjev/prompts.py) is different:

  * the serialized `state` becomes ONE user message
  * each question is an INDEPENDENT branch off the shared prefix, answered
    by a single label token (A, B, ..., Z, AA, ...)

So the faithful fine-tune datum is (prefix + question-suffix) -> one label
token, with loss only on that position. This file replays transcripts via
ft_dataset.process_file and emits per-question rows:

  {"s": <state json str>, "q": <question key>, "question": {...},
   "gold": <gold option key>, "clip_id", "source", "iter", "oracle"}

ft_tinker.py compiles them to (input_ids, label_id) datums with the exact
PromptCompiler port in s1_compile.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import ft_dataset as F  # noqa: E402  (process_file, CLIPS, transcript_files)
from s1_compile import option_pairs, gold_key  # noqa: E402


def s1_rows(row: dict) -> list[dict]:
    """One S1 row per (question, gold answer) pair in a replayed state."""
    state_str = json.dumps(row["state"])
    out = []
    for qkey, question in row["questions"].items():
        answer = row["answers"].get(qkey)
        if answer is None:
            continue
        opts = option_pairs(question)
        gold = gold_key(answer)
        if gold not in {k for k, _ in opts}:
            continue
        out.append({
            "s": state_str, "q": qkey, "question": question,
            "gold": gold, "clip_id": row["clip_id"], "source": row["source"],
            "iter": row["iter"], "oracle": row["oracle"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ft_dataset"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["plain", "rag"])
    args = ap.parse_args()
    rng = random.Random(20260918)

    banks = {}
    if "rag" in args.variants:
        import strip_bank as sb
        for src in {c["source"] for c in F.CLIPS.values()}:
            banks[src] = sb.StripBank.load(
                ROOT / "results" / "banks" / f"loso_{src}")

    files = F.transcript_files()
    files += sorted((ROOT / "results" / "jevloop_gatev2_nobank").glob("*.json"))
    print(f"{len(files)} transcript files")

    for variant in args.variants:
        rows_by_clip: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(F.process_file, f, variant, banks,
                              not args.no_aug,
                              random.Random(rng.randint(0, 1 << 30))): f
                    for f in files}
            for fut in as_completed(futs):
                try:
                    for r in fut.result():
                        rows_by_clip.setdefault(r["clip_id"], []).append(r)
                except Exception as e:
                    print(f"  !! {futs[fut].name}: {e}")
        seen, rows = set(), []
        for cid, rs in rows_by_clip.items():
            for r in rs:
                for s1 in s1_rows(r):
                    h = hashlib.sha1(json.dumps(
                        [s1["s"], s1["q"], s1["question"], s1["gold"]],
                        sort_keys=True).encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    rows.append(s1)
        vdir = Path(args.out) / f"{variant}_s1"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "all.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        sources = sorted({c["source"] for c in F.CLIPS.values()})
        for held in sources:
            fdir = vdir / f"loso_{held}"
            fdir.mkdir(exist_ok=True)
            tr = [r for r in rows if r["source"] != held]
            va = [r for r in rows if r["source"] == held]
            (fdir / "train.jsonl").write_text(
                "\n".join(json.dumps(r) for r in tr) + "\n")
            (fdir / "val.jsonl").write_text(
                "\n".join(json.dumps(r) for r in va) + "\n")
        qs = Counter(r["q"] for r in rows)
        print(f"[{variant}] {len(rows)} s1 rows | per-question: {dict(qs)}")


if __name__ == "__main__":
    main()
