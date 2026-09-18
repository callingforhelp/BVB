"""Semantic rate-strip probe for timewarp (flow-interp / speedup coverage).

Same 4-frame strip format as pairjudge, different question: two pairs at
IDENTICAL in-pair temporal stride straddling a candidate boundary.
  A pair = frames inside the BEFORE window, B pair = inside AFTER.
The VLM compares action/scene progress per unit wall-clock — a finite-
difference semantic rate meter, self-anchored to the clip's own rate.

Arms (per the two-sided verifier rule):
  boundary : pairs straddle the true boundary (timewarp@600) -> expect
             "more" (0.5x region -> 2x region)
  control  : same probe at t=300, both pairs inside the uniform-rate
             slowed region -> expect "same"
  clean    : same probe at t=600 on clean clips -> expect "same"

Run: ~/s1-spike/.venv/bin/python rate_probe.py --workers 8
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
SPIKE = Path.home() / "s1-spike"
sys.path.insert(0, str(SPIKE))
CORPUS = ROOT / "corpus_v3"
OUT = ROOT / "results" / "rateprobe"
OUT.mkdir(parents=True, exist_ok=True)

STRIDE = 60          # in-pair gap: 2s at 30fps
MARGIN = 30          # keep pairs this far from the boundary
BOUNDARY = 600       # timewarp seam frame (20s)
CONTROL = 300        # inside uniform slowed region

PROMPT = """You are shown 4 frames from one video, in order A1, A2, B1, B2.
A1 and A2 were taken {dt}s apart at time ~{ta}s (before a point T).
B1 and B2 were taken {dt}s apart at time ~{tb}s (after T).
The in-pair time gap is IDENTICAL for both pairs.

Compare how much the scene, the actor's hands, and objects PROGRESS or
move between A1->A2 versus B1->B2. Judge action/object progress, not
overall camera shake. If the video plays at a different speed after T,
pair B will show more (or less) progress for the same wall-clock gap.

Answer ONLY with JSON:
{{"progress_B_vs_A": "much_less"|"less"|"same"|"more"|"much_more",
 "what_changed": "one line",
 "confidence": 0.0-1.0}}"""


def extract_frames(mp4: Path, idxs: list[int]) -> list[tuple[str, str]]:
    idxs = [max(0, min(899, t)) for t in idxs]
    sel = "+".join(f"eq(n,{t})" for t in idxs)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-vf", f"select='{sel}',scale=480:-1",
         "-f", "image2pipe", "-vcodec", "mjpeg",
         "-q:v", "5", "-"],
        capture_output=True, check=True)
    blobs, start = [], None
    buf = p.stdout
    i = 0
    while i < len(buf) - 1:
        if buf[i] == 0xFF and buf[i + 1] == 0xD8:
            start = i
        elif buf[i] == 0xFF and buf[i + 1] == 0xD9 and start is not None:
            blobs.append(buf[start:i + 2])
            start = None
        i += 1
    labels = ["A1", "A2", "B1", "B2"]
    return [(l, base64.b64encode(b).decode())
            for l, b in zip(labels, blobs)][:4]


def probe_one(clip_id: str, center: int, arm: str, mp4: Path) -> dict:
    from routes import ROUTES
    from seg_spike import LabeledZenmuxProvider, FailoverVisionProvider
    from system_one_adapter.providers.base import Message

    idxs = [center - STRIDE - MARGIN, center - MARGIN,
            center + MARGIN, center + MARGIN + STRIDE]
    strip = extract_frames(mp4, idxs)
    if len(strip) < 4:
        return {"frame": center, "arm": arm, "error": f"strip={len(strip)}"}
    providers = []
    for name in ("zenmux", "oc-vision-exp", "ark-plan"):
        cfg = ROUTES[name]
        providers.append(LabeledZenmuxProvider(
            api_key=cfg["key"](), labeled_frames=strip,
            model_name=cfg["model"], base_url=cfg["url"],
            send_thinking=cfg["thinking"],
            extra_headers=cfg["headers"]() if cfg.get("headers") else None))
    provider = FailoverVisionProvider(providers)
    prompt = PROMPT.format(dt=STRIDE / 30, ta=(idxs[0] + idxs[1]) / 60,
                           tb=(idxs[2] + idxs[3]) / 60)
    t0 = time.time()
    try:
        res = provider.request([Message(role="user", content=prompt)],
                               schema={}, structured=False)
        m = re.search(r"\{[^{}]*\}", res.text, re.S)
        verdict = json.loads(m.group(0)) if m else {"parse_error": res.text[:200]}
        return {"frame": center, "arm": arm, "verdict": verdict,
                "route": provider.model_name, "secs": round(time.time() - t0, 1)}
    except Exception as e:
        return {"frame": center, "arm": arm,
                "error": f"{type(e).__name__}: {e}",
                "secs": round(time.time() - t0, 1)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real", action="store_true",
                    help="probe the realplanted corpus (450f@25fps, break@350)")
    args = ap.parse_args()

    jobs = []
    if args.real:
        real_dir = ROOT / "results" / "realplanted"
        for mp4 in sorted(real_dir.glob("*__timewarp.mp4")):
            cid = mp4.stem
            jobs.append((cid, 350, "boundary", mp4))
            jobs.append((cid, 175, "control", mp4))
        for mp4 in sorted(real_dir.glob("*__clean.mp4")):
            jobs.append((mp4.stem, 350, "clean", mp4))
    else:
        manifest = json.loads((CORPUS / "manifest.json").read_text())
        for c in manifest["clips"]:
            if c["operator"] == "timewarp":
                jobs.append((c["id"], BOUNDARY, "boundary",
                             CORPUS / f"{c['id']}.mp4"))
                jobs.append((c["id"], CONTROL, "control",
                             CORPUS / f"{c['id']}.mp4"))
            elif c["operator"] == "none":
                jobs.append((c["id"], BOUNDARY, "clean",
                             CORPUS / f"{c['id']}.mp4"))
    todo = []
    for cid, center, arm, mp4 in jobs:
        f = OUT / f"{cid}__{arm}.json"
        if not f.exists():
            todo.append((cid, center, arm, mp4))
    print(f"{len(todo)} probe calls ({len(jobs)-len(todo)} cached)")

    def run(job):
        cid, center, arm, mp4 = job
        r = probe_one(cid, center, arm, mp4)
        r["clip_id"] = cid
        (OUT / f"{cid}__{arm}.json").write_text(json.dumps(r, indent=1))
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            v = r.get("verdict", {})
            print(f"[{i+1}/{len(todo)}] {r['clip_id']:40s} {r['arm']:9s} "
                  f"-> {v.get('progress_B_vs_A', r.get('error','?'))} "
                  f"({v.get('confidence','-')}, {r.get('secs','?')}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
