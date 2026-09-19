"""Quick progress probe for a Tinker sampler checkpoint via the
OpenAI-compatible endpoint — answers "is the checkpoint learning the task"
without standing up the full S1Jev adapter.

Sends an s1 JSONL row as a chat request (state user message + eval/question
user message, reasoning_effort=false) and checks whether the generated label
letter maps to the oracle gold option. NOTE: the chat endpoint cannot inject
the literal "Answer:\\n" suffix that serving uses, so this is a smoke signal
— for faithful numbers use eval_s1.py / S1Jev instead.

    python3 s1_probe.py --model-path tinker://.../sampler_weights/final \
        --file results/ft_dataset/plain_s1/loso_09c1414f1b/val.jsonl --lines 0-49
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s1_compile import EVAL_INSTR, option_pairs  # noqa: E402

BASE_URL = ("https://tinker.thinkingmachines.dev/services/tinker-prod"
            "/oai/api/v1/chat/completions")


def api_key() -> str:
    if os.environ.get("TINKER_API_KEY"):
        return os.environ["TINKER_API_KEY"]
    import yaml
    return yaml.safe_load(open(Path.home() / ".dsh" / ".credentials.yaml"))[
        "refs"]["TINKER_API_KEY"]


def question_block(question: dict) -> str:
    lines = [f"Question: {question['instructions']}", "", "Options:"]
    for i, (key, desc) in enumerate(option_pairs(question)):
        text = (key if desc is None else desc).replace("\n", "\n   ")
        lines.append(f"{string.ascii_uppercase[i]}: {text}")
    return "\n".join(lines)


def probe(model_path: str, row: dict) -> dict:
    body = json.dumps({
        "model": model_path,
        "messages": [
            {"role": "user", "content": row["s"]},
            {"role": "user",
             "content": EVAL_INSTR + question_block(row["question"])},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "logprobs": True,
        "reasoning_effort": False,
    }).encode()
    req = urllib.request.Request(
        BASE_URL, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key()}",
                 "User-Agent": "s1-probe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}


def predict(resp: dict, question: dict) -> tuple[str | None, str]:
    """-> (predicted option key, raw generated text)."""
    if "error" in resp:
        return None, resp["error"]
    text = resp["choices"][0]["message"].get("content") or ""
    letters = re.findall(r"[A-Z]{1,2}", text)
    opts = option_pairs(question)
    if letters:
        idx = 0
        for ch in letters[-1]:           # A=0 .. Z=25, AA=26 ..
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        idx -= 1
        if 0 <= idx < len(opts):
            return opts[idx][0], text
    return None, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="tinker:// sampler checkpoint path")
    ap.add_argument("--file", required=True, help="s1 jsonl file")
    ap.add_argument("--line", type=int, default=None)
    ap.add_argument("--lines", default=None, help="range like 0-49")
    ap.add_argument("--verbose", action="store_true",
                    help="print every row instead of only misses")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file)]
    if args.lines:
        lo, hi = (int(x) for x in args.lines.split("-"))
        idxs = range(lo, min(hi + 1, len(rows)))
    elif args.line is not None:
        idxs = [args.line]
    else:
        idxs = range(min(20, len(rows)))

    n = hit = 0
    per_q: dict[str, list[int]] = {}
    for i in idxs:
        row = rows[i]
        resp = probe(args.model_path, row)
        pred, raw = predict(resp, row["question"])
        ok = pred == row["gold"]
        n += 1
        hit += ok
        stat = per_q.setdefault(row["q"], [0, 0])
        stat[0] += ok
        stat[1] += 1
        if args.verbose or not ok:
            shown = raw if "error" in resp else repr(raw[:60])
            print(f"[{i}] {row['clip_id']} iter{row['iter']} q={row['q']} "
                  f"gold={row['gold']} pred={pred} "
                  f"{'OK' if ok else 'MISS'} -> {shown}")
    print(f"\n{hit}/{n} correct")
    for q, (h, c) in sorted(per_q.items()):
        print(f"  {q}: {h}/{c}")


if __name__ == "__main__":
    main()
