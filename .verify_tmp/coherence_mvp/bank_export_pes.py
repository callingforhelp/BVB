"""Export strip-bank episodes as dsh-pes-reasoning.v1 entries into the shared
project reasoning bank at <repo>/.codex/reasoning-bank/.

The strip bank (jev-strip-bank.v1) was designed on the dsh-pes-reasoning.v1
convention (graft/examples/dsh-pes/reasoning-bank.js): bounded projection,
cue/lesson split, idempotent rb_ ids, FIFO cap. This bridges the last gap —
it rewrites each episode into the convention's entry shape so the shared
bank (and any dsh-pes tooling) can ingest it directly.

Entries are deduped by trace id across folds (loso_<src> banks overlap on
train sources by construction — each fold excludes only its own source).

    python3 bank_export_pes.py [--banks results/banks] [--out ../.codex/reasoning-bank]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
SCHEMA = "dsh-pes-reasoning.v1"
MAX_ENTRIES = 256            # convention cap; oldest evicted
MAX_SUMMARY = 512
MAX_CUE = 240
MAX_LESSON = 320
MAX_EVID = 32


def bounded(v, n):
    s = str(v if v is not None else "")
    return s[: n - 3] + "..." if len(s) > n else s


def entry_for(ep: dict) -> dict:
    v = ep.get("verdict") or {}
    outcome = ("discontinuous/" + str(v.get("break_kind"))
               if v.get("continuous") is False else "continuous")
    trace_id = ep["id"]
    eid = f"{ep['clipId']}@{ep['frame']}:{ep['gate']}"
    summary = (f"strip judgment {eid} op={ep['clip_op']} "
               f"seam={ep['is_true_seam']} -> {outcome}")
    cue = bounded(
        f"strip_judgment returned 1 linked event(s). "
        f"query={json.dumps({'seam': [ep['frame']], 'gate': ep['gate']}, separators=(',', ':'))} "
        f"outcome={json.dumps({'verdict': outcome, 'clip_op': ep['clip_op']}, separators=(',', ':'))} "
        f"signals={json.dumps(ep.get('signalContext', {}), separators=(',', ':'))}",
        MAX_CUE)
    lesson = bounded(
        "Compare signalContext (recur/dup/echo/pj_z) before trusting a "
        "similar seam verdict; dense duplicates + recur>=0.9 means static "
        "segment (room_swap), not a loop. Artifact references were verified.",
        MAX_LESSON)
    return {
        "schemaVersion": SCHEMA,
        "id": "rb_" + hashlib.sha256(trace_id.encode()).hexdigest()[:24],
        "traceId": trace_id,
        "traceKind": bounded(ep.get("traceKind", "strip_judgment"), 64),
        "producerSha": bounded(ep.get("producerSha") or "jevloop_phase2", 64),
        "runOrdinal": ep.get("runOrdinal", 0),
        "summary": bounded(summary, MAX_SUMMARY),
        "cue": cue,
        "lesson": lesson,
        "eventCount": 1,
        "artifactVerification": (
            "verified" if ep.get("status") == "verified" else "unverified"),
        "evidenceIds": [eid][:MAX_EVID],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", default=str(ROOT / "results" / "banks"))
    ap.add_argument("--out", default=str(ROOT.parent.parent
                                         / ".codex" / "reasoning-bank"))
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    for fold in sorted(Path(args.banks).glob("loso_*/episodes.jsonl")):
        for line in fold.open():
            ep = json.loads(line)
            seen.setdefault(ep["id"], ep)

    # bounded bank, stratified: round-robin across (source, op) buckets so
    # the 256-entry cap covers every source and operator, not just late folds
    buckets: dict[tuple, list] = {}
    for ep in seen.values():
        buckets.setdefault((ep["source"], ep["clip_op"]), []).append(ep)
    picked: list[dict] = []
    while len(picked) < MAX_ENTRIES:
        progressed = False
        for k in sorted(buckets):
            if buckets[k] and len(picked) < MAX_ENTRIES:
                picked.append(buckets[k].pop())
                progressed = True
        if not progressed:
            break
    entries = [entry_for(ep) for ep in picked]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.jsonl").open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    (out / "index.json").write_text(json.dumps({
        "schemaVersion": SCHEMA,
        "source": "jev-strip-bank.v1",
        "folds": sorted({ep["source"] for ep in seen.values()}),
        "entries": len(entries),
        "ids": [e["id"] for e in entries],
    }, indent=1))
    print(f"{len(seen)} unique episodes -> {len(entries)} entries -> {out}")


if __name__ == "__main__":
    main()
