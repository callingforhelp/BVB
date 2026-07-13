"""Minimal multi-turn bash agent (Stage-1 harness), mini-swe-agent style.

The harness is a thin substrate and lets the MODEL decide everything about
effort: how many turns to take, how many frames to look at, when it is done. The
only hard budget is cost (USD), computed by litellm -- exactly mini-swe-agent's
philosophy (cost_limit is the stop; step_limit defaults to off).

Two primitives, both driven by the model via fenced blocks:
  - ```bash : run a command in the sandbox; stdout/stderr is returned.
  - ```frames : request video frames to look at; the harness extracts them and
    injects them as images on the next turn. (A chat model cannot see a jpg on
    disk; injecting pixels is the one thing the harness must mediate -- but WHICH
    frames is the model's call.)

The model finishes by replying DONE, but only after it has actually seen >=1
frame (a scene built without looking at the video is invalid).
"""

from __future__ import annotations

import base64
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ..models import Model, image_part, text_part
from ..sandbox import Sandbox
from ..tasks import BVBTask


SYSTEM_PROMPT = """You are a video-understanding agent evaluated on reconstructing
an indoor scene from a video as Blender Python (bpy). You drive a Linux sandbox
with Blender installed.

Interface:
- ```bash block: the harness runs it in the sandbox and returns stdout/stderr.
  Run Blender headless as: blender_run --python /workspace/scratch/build.py
  (wraps `xvfb-run -a blender --background`).
- ```frames block: request video frames to look at. The harness extracts them
  and shows them to you as images on the next turn. Content is either explicit
  timestamps in seconds (e.g. `0, 4.5, 12, 30`) or `count=N` for N evenly-spaced
  frames (optionally with `start=` `end=` seconds). Look at as many frames as you
  need; request more whenever you want.
- When the scene is finished and saved, reply with the single word DONE.

Work in this order — do NOT skip looking at the video:
1. FIRST request frames with a ```frames block, on their own, and WAIT. Do not
   write any bpy or say DONE in the same message as a ```frames request — the
   frames you request are only shown to you on the next turn.
2. Look at the returned frames, then build the scene with ```bash.
3. Request more frames to check details whenever useful, then revise.
4. Only after you have actually seen frames and saved result.blend, say DONE.
A scene built without looking at any frame is a failure.

You decide how much effort to spend: how many frames, how many turns. The raw
video is also at /workspace/video/video.mp4 inside the sandbox.

Scene/output rules (these affect scoring, follow them exactly):
- The final scene MUST be saved to /workspace/output/result.blend
  (e.g. bpy.ops.wm.save_as_mainfile(filepath="/workspace/output/result.blend")).
- Build ALL geometry from basic primitives only (cube, plane, cylinder, cone,
  uv_sphere, torus) and assemblies of them. Multi-part objects (a chair = seat +
  back + legs) are encouraged; group each object's parts in a Collection named
  after the object.
- No imported models, no sculpting/arbitrary meshes, no geometry nodes,
  particles, physics, or image textures. Materials = a single Principled BSDF
  with numeric values only.
- 1 Blender unit = 1 meter. Keep Unit Scale = 1.0. Use radians for rotations.
- Include at least one camera and one light.
- Reproduce object counts, sizes, positions, and spatial relationships as
  faithfully as you can from the video.

Camera trajectory / temporal reconstruction (do NOT skip this):
- The video is a moving camera walking through the scene. Reconstruct that
  camera motion as an ANIMATION, not a single static viewpoint.
- Insert keyframes for the camera's location and rotation_euler over time, e.g.:
    cam.keyframe_insert(data_path="location", frame=f)
    cam.keyframe_insert(data_path="rotation_euler", frame=f)
  and set bpy.context.scene.frame_start / frame_end to span the motion.
"""

BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
FRAMES_BLOCK = re.compile(r"```frames\s*\n(.*?)```", re.DOTALL)
KV = re.compile(r"(count|start|end)\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
NUM = re.compile(r"-?[0-9]*\.?[0-9]+")
MAX_FEEDBACK_CHARS = 4000
# Safety guard against a single pathological request extracting thousands of
# frames (a DoS guard, not an effort budget -- total frames are unbounded).
MAX_FRAMES_PER_REQUEST = 60


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


def _clamp(ts: float, duration: float) -> float:
    ts = max(0.0, ts)
    if duration > 0:
        ts = min(ts, max(0.0, duration - 0.05))
    return round(ts, 3)


def resolve_frame_request(content: str, duration: float) -> list[float]:
    """Parse a ```frames block into a list of timestamps (seconds)."""
    kv = {key.lower(): float(value) for key, value in KV.findall(content)}
    if "count" in kv:
        count = int(kv["count"])
        start = kv.get("start", 0.0)
        end = kv.get("end", duration if duration > 0 else float(count))
        if count <= 0:
            return []
        if count == 1:
            return [_clamp((start + end) / 2, duration)]
        step = (end - start) / (count - 1)
        return [_clamp(start + step * i, duration) for i in range(count)]
    return [_clamp(float(match), duration) for match in NUM.findall(content)]


def extract_frames_at(
    video_path: Path, out_dir: Path, timestamps: list[float], *, start_index: int = 0, scale: int = 768
) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[float, Path]] = []
    for offset, ts in enumerate(timestamps):
        ts = max(0.0, ts)
        dest = out_dir / f"frame_{start_index + offset:03d}_{ts:.3f}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "3", "-vf", f"scale={scale}:-1", str(dest)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and dest.exists():
            frames.append((ts, dest))
    return frames


def _truncate(text: str, limit: int = MAX_FEEDBACK_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _frame_parts(frames: list[tuple[float, Path]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for ts, path in frames:
        parts.append(text_part(f"[frame t={ts:.2f}s]"))
        parts.append(image_part(base64.b64encode(path.read_bytes()).decode("ascii")))
    return parts


def run_bash_agent(
    task: BVBTask,
    sandbox: Sandbox,
    client: Model,
    *,
    frames_dir: Path,
    cost_limit: float = 3.0,
    max_turns: int = 0,          # 0 = unlimited; cost_limit is the real stop
    exec_timeout: float = 300.0,
    wall_clock_s: float | None = None,
) -> dict[str, Any]:
    start_time = time.time()
    duration = _probe_duration(task.video_path) if task.video_path else 0.0
    frames_extracted = 0   # pulled from the video
    frames_seen = 0        # actually present in a message the model then answered
    pending_frames = 0     # injected, not yet consumed by a model turn
    cost = 0.0
    in_tokens = 0
    out_tokens = 0

    def budget_note() -> str:
        remaining = cost_limit - cost
        warn = " You are LOW on budget: save result.blend now and say DONE." if remaining <= 0.25 * cost_limit else ""
        return f"[budget] spent ${cost:.3f} of ${cost_limit:.2f} (${remaining:.3f} left).{warn}"

    parts: list[dict[str, Any]] = [text_part(
        f"Scene: {task.scene_name} ({task.dataset}). Video duration {duration:.1f}s. "
        f"No frames shown yet — request frames with a ```frames block (e.g. "
        f"`count=12`, or timestamps `0 5 12 30`). You have a cost budget of "
        f"${cost_limit:.2f} for this scene; spend it as you see fit but save "
        f"result.blend before it runs out. Then reconstruct the scene and say DONE."
    )]
    messages: list[dict[str, Any]] = [{"role": "user", "parts": parts}]

    turns = 0
    commands_run = 0
    error: str | None = None
    done = False
    stop_reason = "max_turns" if max_turns else "cost"
    while True:
        turns += 1
        if max_turns and turns > max_turns:
            turns -= 1
            stop_reason = "max_turns"
            break
        if wall_clock_s is not None and time.time() - start_time > wall_clock_s:
            stop_reason = "wall_clock"
            break
        if cost >= cost_limit:
            stop_reason = "cost"
            break
        # Frames injected last turn are now in the context the model will read.
        frames_seen += pending_frames
        pending_frames = 0
        try:
            reply, info = client.complete(SYSTEM_PROMPT, messages)
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
            stop_reason = "error"
            break
        cost += float(info.get("cost", 0.0) or 0.0)
        in_tokens += info.get("input_tokens", 0)
        out_tokens += info.get("output_tokens", 0)
        messages.append({"role": "assistant", "parts": [text_part(reply)]})

        bash_blocks = [b.strip() for b in BASH_BLOCK.findall(reply) if b.strip()]
        frame_blocks = [b.strip() for b in FRAMES_BLOCK.findall(reply) if b.strip()]

        if not bash_blocks and not frame_blocks:
            if "DONE" in reply and frames_seen > 0 and sandbox.output_exists():
                done = True
                stop_reason = "done"
                break
            nudge = (
                "You have not viewed any frames yet. Request frames with a "
                "```frames block before building or finishing."
                if frames_seen == 0 else
                "No ```bash or ```frames block found, or you said DONE before "
                "/workspace/output/result.blend existed. Build/save the scene "
                "with bash, then say DONE."
            )
            messages.append({"role": "user", "parts": [text_part(nudge)]})
            continue

        next_parts: list[dict[str, Any]] = []

        # Perception: extract requested frames and inject them as images.
        for content in frame_blocks:
            requested = resolve_frame_request(content, duration)[:MAX_FRAMES_PER_REQUEST]
            if not requested:
                continue
            got = extract_frames_at(
                task.video_path, frames_dir, requested, start_index=frames_extracted
            ) if task.video_path else []
            frames_extracted += len(got)
            pending_frames += len(got)
            next_parts.extend(_frame_parts(got))

        # Action: run bash blocks in the sandbox.
        feedback_chunks = []
        for block in bash_blocks:
            result = sandbox.exec(block, timeout=exec_timeout)
            commands_run += 1
            feedback_chunks.append(
                f"$ {block}\n[exit {result.returncode}]\n"
                f"{_truncate(result.stdout)}\n{_truncate(result.stderr)}"
            )
        if feedback_chunks:
            next_parts.append(text_part(_truncate("\n\n".join(feedback_chunks), 8000)))

        next_parts.append(text_part(budget_note()))
        if not next_parts:
            next_parts.append(text_part("(no frames extracted; check your request)"))
        messages.append({"role": "user", "parts": next_parts})

        # Finish only once the model has actually seen >=1 frame.
        if "DONE" in reply and sandbox.output_exists() and frames_seen > 0:
            done = True
            stop_reason = "done"
            break

    produced = sandbox.output_exists()
    return {
        "scene_name": task.scene_name,
        "turns": turns,
        "commands_run": commands_run,
        "frames_seen": frames_seen,
        "frames_extracted": frames_extracted,
        "done": done,
        "stop_reason": stop_reason,
        "produced_blend": produced,
        "error": error,
        "cost_usd": round(cost, 4),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "duration_s": round(time.time() - start_time, 1),
    }
