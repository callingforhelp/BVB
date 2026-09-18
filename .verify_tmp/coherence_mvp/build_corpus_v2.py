#!/usr/bin/env python3
"""Build the coherence corpus v2 with matched media and complete boundaries."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT.parent.parent / "VSI-Bench" / "scannetpp"
OUT_DIR = ROOT / "corpus_v2"
SCRATCH = OUT_DIR / "_scratch"
WINDOW_S = 30.0
FPS = 30
FRAMES = int(WINDOW_S * FPS)
ENC = [
    "-vf", (f"scale=640:480:flags=lanczos,fps={FPS},"
            f"tpad=stop=-1:stop_mode=clone,fps={FPS},setpts=N/({FPS}*TB)"),
    "-frames:v", str(FRAMES),
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
]


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed:\n{' '.join(command)}\n{result.stderr[-1600:]}")


def duration(path: Path) -> float:
    result = subprocess.run([
        FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def cut(src: Path, dst: Path, start: float = 0.0, seconds: float | None = None,
        setpts: str | None = None, frames: int | None = None) -> None:
    filters = ["scale=640:480:flags=lanczos", f"fps={FPS}"]
    if setpts:
        filters.append(setpts)
    if frames is not None:
        filters.append("tpad=stop=-1:stop_mode=clone")
    command = [FFMPEG, "-y", "-v", "error", "-ss", f"{start:.6f}"]
    if seconds is not None:
        command += ["-t", f"{seconds:.6f}"]
    command += ["-i", str(src), "-vf", ",".join(filters)]
    if frames is not None:
        command += ["-frames:v", str(frames)]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-an", str(dst)]
    run(command)


def concat(parts: list[Path], dst: Path, tag: str) -> None:
    listing = SCRATCH / f"_list_{tag}.txt"
    listing.write_text("".join(f"file '{part}'\n" for part in parts))
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), *ENC, str(dst)])


def clean(src: Path, start: float, dst: Path) -> list[float]:
    cut(src, dst, start, WINDOW_S, frames=FRAMES)
    return []


def splice(base: Path, src: Path, src_start: float, dst: Path, tag: str) -> list[float]:
    first = SCRATCH / f"{tag}_first.mp4"
    second = SCRATCH / f"{tag}_second.mp4"
    cut(base, first, 0, 15)
    cut(src, second, src_start + WINDOW_S, 15)
    concat([first, second], dst, tag)
    return [15.0]


def reverse_segment(base: Path, dst: Path, tag: str) -> list[float]:
    pre = SCRATCH / f"{tag}_pre.mp4"
    middle = SCRATCH / f"{tag}_middle.mp4"
    reversed_middle = SCRATCH / f"{tag}_reversed.mp4"
    post = SCRATCH / f"{tag}_post.mp4"
    cut(base, pre, 0, 10)
    cut(base, middle, 10, 6)
    run([FFMPEG, "-y", "-v", "error", "-i", str(middle),
         "-vf", f"reverse,fps={FPS}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(reversed_middle)])
    cut(base, post, 16, 14)
    concat([pre, reversed_middle, post], dst, tag)
    return [10.0, 16.0]


def loop_segment(base: Path, dst: Path, tag: str) -> list[float]:
    pre = SCRATCH / f"{tag}_pre.mp4"
    segment = SCRATCH / f"{tag}_segment.mp4"
    post = SCRATCH / f"{tag}_post.mp4"
    cut(base, pre, 0, 10)
    cut(base, segment, 10, 5)
    cut(base, post, 15, 10)
    concat([pre, segment, segment, post], dst, tag)
    return [15.0]


def room_swap(base: Path, other: Path, dst: Path, tag: str) -> list[float]:
    pre = SCRATCH / f"{tag}_pre.mp4"
    inserted = SCRATCH / f"{tag}_inserted.mp4"
    post = SCRATCH / f"{tag}_post.mp4"
    cut(base, pre, 0, 10)
    cut(other, inserted, 10, 6)
    cut(base, post, 16, 14)
    concat([pre, inserted, post], dst, tag)
    return [10.0, 16.0]


def timewarp(base: Path, dst: Path, tag: str) -> list[float]:
    slow = SCRATCH / f"{tag}_slow.mp4"
    fast = SCRATCH / f"{tag}_fast.mp4"
    cut(base, slow, 0, 10, "setpts=2.0*PTS")
    cut(base, fast, 10, 20, "setpts=0.5*PTS")
    concat([slow, fast], dst, tag)
    return [20.0]


def main() -> int:
    sources = sorted(SRC_DIR.glob("*.mp4"))
    if len(sources) != 5:
        raise RuntimeError(f"expected 5 sources, found {len(sources)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    bases: dict[str, Path] = {}
    starts: dict[str, float] = {}
    rows = []
    for src in sources:
        src_duration = duration(src)
        start = max(0.0, (src_duration - WINDOW_S) / 2.0)
        dst = OUT_DIR / f"{src.stem}__clean.mp4"
        boundaries = clean(src, start, dst)
        bases[src.stem] = dst
        starts[src.stem] = start
        rows.append({
            "id": f"{src.stem}__clean", "source": src.stem,
            "operator": "none", "arm": "baseline", "breaks_s": boundaries,
            "path": dst.name, "src_start_s": round(start, 6),
        })

    for src in sources:
        base = bases[src.stem]
        other = bases[next(name for name in sorted(bases) if name != src.stem)]
        for operator in ("splice", "reverse_segment", "loop", "room_swap", "timewarp"):
            tag = f"{src.stem}__{operator}"
            dst = OUT_DIR / f"{tag}.mp4"
            if operator == "splice":
                boundaries = splice(base, src, starts[src.stem], dst, tag)
            elif operator == "reverse_segment":
                boundaries = reverse_segment(base, dst, tag)
            elif operator == "loop":
                boundaries = loop_segment(base, dst, tag)
            elif operator == "room_swap":
                boundaries = room_swap(base, other, dst, tag)
            else:
                boundaries = timewarp(base, dst, tag)
            rows.append({
                "id": tag, "source": src.stem, "operator": operator,
                "arm": "test", "breaks_s": boundaries, "path": dst.name,
            })

    manifest = {
        "version": 2,
        "window_s": WINDOW_S,
        "fps": FPS,
        "resolution": [640, 480],
        "codec": "h264/yuv420p/no-audio",
        "clips": rows,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} clips -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
