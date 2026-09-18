#!/usr/bin/env python3
"""Gate 0 verification for corpus_v2: media parity + metadata leakage."""
from __future__ import annotations

import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

FFPROBE = "/opt/homebrew/bin/ffprobe"
ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_v2"
REPORT = ROOT / "results" / "gate0_corpus_v2.json"


def probe(path: Path) -> dict:
    raw = subprocess.run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration:format=duration,format_name",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    data = json.loads(raw.stdout)
    stream = data["streams"][0]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "stream_duration": float(stream.get("duration", data["format"]["duration"])),
        "format_duration": float(data["format"]["duration"]),
        "format": data["format"]["format_name"],
    }


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    rows = []
    failures = []
    for clip in manifest["clips"]:
        path = CORPUS / clip["path"]
        info = probe(path)
        row = {**clip, "probe": info}
        rows.append(row)
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"{clip['id']}: missing/empty")
        if info["codec"] != "h264" or (info["width"], info["height"]) != (640, 480):
            failures.append(f"{clip['id']}: media mismatch {info}")
        if info["fps"] != "30/1":
            failures.append(f"{clip['id']}: fps={info['fps']}")
        if info["frames"] != 900:
            failures.append(f"{clip['id']}: frames={info['frames']}")
        if abs(info["format_duration"] - 30.0) > 0.05:
            failures.append(f"{clip['id']}: duration={info['format_duration']}")

    by_operator = defaultdict(list)
    for row in rows:
        by_operator[row["operator"]].append(row["probe"]["frames"])
    frame_sets = {operator: Counter(values) for operator, values in by_operator.items()}
    leakage = {}
    for field in ("frames", "format_duration", "codec", "fps"):
        values = {row["id"]: row["probe"][field] for row in rows}
        if len(set(values.values())) > 1:
            leakage[field] = sorted(set(map(str, values.values())))

    stats = {
        "clips": len(rows),
        "failures": failures,
        "frames_by_operator": {k: dict(v) for k, v in frame_sets.items()},
        "metadata_leakage": leakage,
        "duration_range_s": [
            min(r["probe"]["format_duration"] for r in rows),
            max(r["probe"]["format_duration"] for r in rows),
        ],
        "mean_duration_s": statistics.mean(r["probe"]["format_duration"] for r in rows),
        "boundary_counts": {
            op: sorted({len(r["breaks_s"]) for r in rows if r["operator"] == op})
            for op in sorted(by_operator)
        },
        "rows": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in
                      ("clips", "failures", "metadata_leakage", "duration_range_s", "boundary_counts")},
                     indent=2))
    return 1 if failures or leakage else 0


if __name__ == "__main__":
    raise SystemExit(main())
