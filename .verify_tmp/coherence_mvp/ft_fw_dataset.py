"""Emit Fireworks chat-format SFT datasets from s1 rows.

Fireworks SFT takes OpenAI-style {"messages": [...]} JSONL. We map the
OpenJev contract faithfully:

    user      <- state JSON                      (weight 0: no loss)
    user      <- EVAL_INSTR + question suffix    (weight 0: no loss)
                  "Question: ...\n\nOptions:\nA: ...\n...\n\nAnswer:\n"
    assistant <- gold label letter (A, B, ...)

weight 0 on user messages replicates the Tinker datum (loss only on the
label token). At inference the same two user messages are sent; the model
emits one label token — identical train/serve shape.

Stdlib only.
"""
from __future__ import annotations

import itertools
import json
import string
import sys
from pathlib import Path

from s1_compile import EVAL_INSTR, option_pairs

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "ft_dataset" / "plain_s1"
OUT = HERE / "results" / "ft_fw"


def label_letters(n: int) -> list[str]:
    """A..Z, AA, AB, ... — matches s1_compile.build_labels ordering."""
    letters = list(string.ascii_uppercase)
    for p in itertools.product(string.ascii_uppercase, repeat=2):
        letters.append("".join(p))
        if len(letters) >= n:
            break
    return letters[:n]


def question_suffix(question: dict) -> str:
    """EVAL_INSTR + 'Question: ... Options: ... Answer:' — mirrors the
    OpenJev suffix layout (question text inside the second user turn,
    'Answer:' immediately before the assistant turn)."""
    opts = option_pairs(question)
    labels = label_letters(len(opts))
    lines = [f"Question: {question['instructions']}", "", "Options:"]
    for (key, desc), label in zip(opts, labels):
        text = (key if desc is None else desc).replace("\n", "\n   ")
        lines.append(f"{label}: {text}")
    return EVAL_INSTR + "\n".join(lines) + "\n\nAnswer:\n"


def row_to_fw(row: dict) -> dict | None:
    opts = option_pairs(row["question"])
    keys = [k for k, _ in opts]
    if row["gold"] not in keys:
        return None
    label = label_letters(len(keys))[keys.index(row["gold"])]
    return {"messages": [
        {"role": "user", "content": row["s"], "weight": 0},
        {"role": "user", "content": question_suffix(row["question"]),
         "weight": 0},
        {"role": "assistant", "content": label},
    ]}


def convert(src: Path, dst: Path) -> None:
    n_in = n_out = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            n_in += 1
            ex = row_to_fw(json.loads(line))
            if ex is None:
                continue
            g.write(json.dumps(ex) + "\n")
            n_out += 1
    print(f"{src.name} -> {dst.name}: {n_out}/{n_in} rows")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    convert(SRC / "all.jsonl", OUT / "all.jsonl")
    for fold in sorted(p.name for p in SRC.iterdir() if p.is_dir()):
        for part in ("train", "val"):
            src = SRC / fold / f"{part}.jsonl"
            if src.exists():
                convert(src, OUT / f"{fold}_{part}.jsonl")


if __name__ == "__main__":
    sys.exit(main())
