#!/usr/bin/env python3
"""Build OOD test clips: 30s windows at offsets the corpus never used,
on the 3 sources whose duration leaves a fresh gap.

Clips (window id w3, appended to corpus_v3/manifest.json):
  09c1414f1b__w3__{clean,reverse_segment,loop}
  13c3e046d7__w3__{clean,reverse_segment,room_swap}
  21d970d8de__w3__{clean,splice}
"""
import json
from pathlib import Path

import build_corpus_v3 as B

ROOT = B.ROOT
SRC = B.SRC_DIR
OUT = B.OUT_DIR

# fresh windows: sources are 181.7/153.6/152.4s; corpus used
# [0,30],[mid,mid+30],[end-30,end] — these sit in the untouched gaps.
# Donor sources (for room_swap/splice inserts) use start 0 — the donor
# frames need only be a different room, not unseen footage.
WIN = {
    "09c1414f1b": 110.0,
    "13c3e046d7": 95.0,
    "21d970d8de": 95.0,
    "0d2ee665be": 0.0,
    "1ada7a0617": 0.0,
}


def window(src: str) -> Path:
    dst = B.SCRATCH / f"{src}__w3__window.mp4"
    if not dst.exists():
        B.cut(SRC / f"{src}.mp4", dst, WIN[src], B.WINDOW_S)
    return dst


def main():
    B.SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = []
    spec = [
        ("09c1414f1b", ["clean", "reverse_segment", "loop"]),
        ("13c3e046d7", ["clean", "reverse_segment", "room_swap"]),
        ("21d970d8de", ["clean", "splice"]),
    ]
    for src, ops in spec:
        w = window(src)
        for op in ops:
            cid = f"{src}__w3__{op}"
            dst = OUT / f"{cid}.mp4"
            if op == "clean":
                if not dst.exists():
                    B.run(["cp", str(w), str(dst)])
                breaks = []
            elif op == "reverse_segment":
                breaks = B.reverse_segment(w, dst, cid)
            elif op == "loop":
                breaks = B.loop_segment(w, dst, cid)
            elif op == "room_swap":
                other = window("1ada7a0617" if src != "1ada7a0617"
                               else "09c1414f1b")
                breaks = B.room_swap(w, other, dst, cid)
            elif op == "splice":
                other = window("09c1414f1b" if src != "09c1414f1b"
                               else "13c3e046d7")
                breaks = B.splice(w, other, dst, cid)
            rows.append({"id": cid, "source": src, "window": 3,
                         "operator": "none" if op == "clean" else op,
                         "arm": "ood", "breaks_s": breaks, "path": f"{cid}.mp4"})
            print(f"built {cid} breaks={breaks}")

    mpath = OUT / "manifest.json"
    m = json.loads(mpath.read_text())
    have = {c["id"] for c in m["clips"]}
    m["clips"] += [r for r in rows if r["id"] not in have]
    mpath.write_text(json.dumps(m, indent=1))
    print(f"manifest: {len(m['clips'])} clips")


if __name__ == "__main__":
    main()
