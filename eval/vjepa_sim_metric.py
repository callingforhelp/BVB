#!/usr/bin/env python3
"""Vision-level BVB metric: V-JEPA similarity between original and rendered video.

For each Stage-1 ``.blend`` submission:
  1. Sparse-render the scene camera (cached as ``camera_renders/<id>.mp4`` by default).
  2. Encode original + rendered clips with a frozen V-JEPA 2.1 encoder.
  3. Compute layout / motion / combined similarity scores.
  4. Never write features to disk; use ``--no-keep-renders`` to skip caching.

Writes ``vision_sim.jsonl`` + ``vision_sim_summary.json`` into the run directory
(per-scene rows include ``layout_sim``, ``motion_sim``, and ``vision_sim``).

Intended to run on a GPU Linux box::

  pip install -r eval/requirements-vjepa.txt
  python eval/vjepa_sim_metric.py \\
    --run sandbox/results/mini-harness-...-run01 \\
    --vsi-bench VSI-Bench \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

from render_blend_video import encode_png_sequence_to_mp4, render_blend_to_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA_METADATA = Path(__file__).resolve().parent / "test.jsonl"
DEFAULT_VSI_BENCH = REPO_ROOT / "VSI-Bench"
DEFAULT_MODEL = "apiantonio/vjepa2.1-vit-gigantic-384"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def scene_datasets(qa_metadata_path: Path) -> dict[str, str]:
    datasets: dict[str, str] = {}
    for record in read_jsonl(qa_metadata_path):
        scene = str(record.get("scene_name") or "")
        dataset = record.get("dataset")
        if scene and dataset and scene not in datasets:
            datasets[scene] = str(dataset)
    return datasets


def locate_video(scene_name: str, dataset: str | None, vsi_bench: Path) -> Path | None:
    candidates: list[Path] = []
    if dataset:
        candidates.append(vsi_bench / dataset / f"{scene_name}.mp4")
    for ds in ("arkitscenes", "scannet", "scannetpp"):
        candidates.append(vsi_bench / ds / f"{scene_name}.mp4")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def records_from_run(run_dir: Path) -> list[dict[str, str]]:
    blends_dir = run_dir / "blends"
    if not blends_dir.is_dir():
        raise SystemExit(f"No blends directory: {blends_dir}")
    records = [
        {"id": blend.stem, "submission_path": str(blend.resolve())}
        for blend in sorted(blends_dir.glob("*.blend"))
    ]
    if not records:
        raise SystemExit(f"No .blend submissions found in {blends_dir}")
    return records


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        index_str, count_str = value.split("/", 1)
        index = int(index_str)
        count = int(count_str)
    except ValueError as exc:
        raise SystemExit("--shard must look like i/N, e.g. 1/8") from exc
    if index < 1 or count < 1 or index > count:
        raise SystemExit("--shard must satisfy 1 <= i <= N")
    return index, count


def select_records(
    records: list[dict[str, str]],
    *,
    limit: int | None,
    sample: int | None,
    scene_ids: list[str] | None,
    seed: int,
    shard: tuple[int, int] | None,
) -> list[dict[str, str]]:
    if scene_ids is not None:
        by_id = {row["id"]: row for row in records}
        missing = [scene_id for scene_id in scene_ids if scene_id not in by_id]
        if missing:
            raise SystemExit(f"Missing requested submission ids: {', '.join(missing)}")
        selected = [by_id[scene_id] for scene_id in scene_ids]
    else:
        selected = list(records)
        if sample is not None:
            rng = random.Random(seed)
            if sample > len(selected):
                raise SystemExit(f"--sample {sample} exceeds {len(selected)} submissions")
            selected = rng.sample(selected, sample)
            selected = sorted(selected, key=lambda row: row["id"])
        elif limit is not None:
            selected = selected[:limit]
    if shard is not None:
        index, count = shard
        selected = [row for i, row in enumerate(selected) if i % count == (index - 1)]
    return selected


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    done: set[str] = set()
    for row in read_jsonl(output_path):
        scene_id = row.get("id") or row.get("scene_id")
        if scene_id:
            done.add(str(scene_id))
    return done


def sample_frame_indices(num_frames_total: int, num_frames: int) -> list[int]:
    if num_frames_total <= 0:
        raise ValueError("Video has no frames")
    if num_frames_total == 1:
        return [0] * num_frames
    if num_frames_total <= num_frames:
        indices = list(range(num_frames_total))
        while len(indices) < num_frames:
            indices.append(num_frames_total - 1)
        return indices
    if num_frames == 1:
        return [0]
    return [
        int(round(i * (num_frames_total - 1) / (num_frames - 1)))
        for i in range(num_frames)
    ]


def decode_png_sequence(frames_dir: Path, num_frames: int) -> Any:
    """Return ``(T, H, W, 3)`` uint8 RGB frames from a Blender PNG sequence."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "opencv-python-headless and numpy are required for frame decoding. "
            "Install eval/requirements-vjepa.txt"
        ) from exc

    paths = sorted(frames_dir.glob("frame*.png"))
    if not paths:
        raise RuntimeError(f"No PNG frames in {frames_dir}")
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    indices = sample_frame_indices(len(frames), num_frames)
    return np.stack([frames[int(i)] for i in indices], axis=0)


def decode_video_frames(path: Path, num_frames: int) -> Any:
    """Return ``(T, H, W, 3)`` uint8 RGB frames sampled uniformly.

    Decodes sequentially (no ``CAP_PROP_POS_FRAMES`` seeks) because keyframe
    snapping is unreliable across codecs and quietly returns the wrong frame.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "opencv-python-headless and numpy are required for frame decoding. "
            "Install eval/requirements-vjepa.txt"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        frames: list[Any] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            raise RuntimeError(f"No frames decoded: {path}")
        indices = sample_frame_indices(len(frames), num_frames)
        return np.stack([frames[int(i)] for i in indices], axis=0)
    finally:
        capture.release()


def cosine_similarity(a: Any, b: Any) -> float:
    import torch.nn.functional as F

    a = F.normalize(a.flatten().float(), dim=0)
    b = F.normalize(b.flatten().float(), dim=0)
    return float((a * b).sum().item())


def infer_token_grid(num_tokens: int, num_frames: int, tubelet_size: int = 2) -> tuple[int, int, int]:
    temporal = max(1, num_frames // tubelet_size)
    spatial = num_tokens // temporal
    side = int(round(spatial**0.5))
    if side * side * temporal != num_tokens:
        # Fallback: treat as flat tokens with no spatial grid.
        return temporal, num_tokens, 1
    return temporal, side, side


def compute_vjepa_similarity(
    original_frames: Any,
    rendered_frames: Any,
    *,
    model: Any,
    processor: Any,
    device: str,
    dtype: Any,
) -> dict[str, float]:
    """Return layout / motion / combined vision similarities.

    ``layout_sim``: temporally pooled spatial patch-map cosine.
    ``motion_sim``: global token-mean cosine.
    ``vision_sim``: ``0.5 * (layout_sim + motion_sim)``.
    Scores are typically in ``[0, 1]`` (cosine range ``[-1, 1]``).
    """
    import torch

    inputs = processor(
        [list(original_frames), list(rendered_frames)],
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values_videos"].to(device=device, dtype=dtype)

    with torch.inference_mode():
        outputs = model(pixel_values_videos=pixel_values, skip_predictor=True)
        tokens = outputs.last_hidden_state  # (2, N, D)

    original_tokens = tokens[0]
    rendered_tokens = tokens[1]
    num_frames = int(original_frames.shape[0])
    temporal, height, width = infer_token_grid(original_tokens.shape[0], num_frames)

    motion_sim = cosine_similarity(original_tokens.mean(dim=0), rendered_tokens.mean(dim=0))

    if height > 1 and width > 1 and temporal * height * width == original_tokens.shape[0]:
        original_map = original_tokens.reshape(temporal, height, width, -1).mean(dim=0)
        rendered_map = rendered_tokens.reshape(temporal, height, width, -1).mean(dim=0)
        flat_o = original_map.reshape(-1, original_map.shape[-1])
        flat_r = rendered_map.reshape(-1, rendered_map.shape[-1])
        import torch.nn.functional as F

        layout_sim = float(
            F.cosine_similarity(
                F.normalize(flat_o.float(), dim=1),
                F.normalize(flat_r.float(), dim=1),
                dim=1,
            ).mean().item()
        )
    else:
        layout_sim = motion_sim

    vision_sim = float(0.5 * (layout_sim + motion_sim))
    return {
        "layout_sim": float(layout_sim),
        "motion_sim": float(motion_sim),
        "vision_sim": vision_sim,
    }


def load_encoder(model_id: str, device: str, dtype_name: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoVideoProcessor

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_name]
    processor = AutoVideoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device).eval()
    return model, processor, dtype


def _resolve_cached_video(keep_renders: Path | None, scene_id: str) -> Path | None:
    """Prefer flat ``<id>.mp4``; fall back to legacy ``<id>/video.mp4``."""
    if keep_renders is None:
        return None
    flat = keep_renders / f"{scene_id}.mp4"
    if flat.is_file() and flat.stat().st_size > 0:
        return flat
    legacy = keep_renders / scene_id / "video.mp4"
    if legacy.is_file() and legacy.stat().st_size > 0:
        return legacy
    return None


def evaluate_scene(
    *,
    scene_id: str,
    blend_path: Path,
    original_video: Path,
    model: Any,
    processor: Any,
    device: str,
    dtype: Any,
    num_frames: int,
    blender: str | None,
    render_resolution: int,
    render_timeout: float,
    keep_renders: Path | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": scene_id,
        "submission_path": str(blend_path),
        "original_video": str(original_video),
        "status": "error",
        "vision_sim": None,
        "layout_sim": None,
        "motion_sim": None,
    }

    try:
        cached_video = _resolve_cached_video(keep_renders, scene_id)
        legacy_frames = (
            keep_renders / scene_id / "frames" if keep_renders is not None else None
        )
        cached_pngs = (
            sorted(legacy_frames.glob("frame*.png"))
            if legacy_frames is not None and legacy_frames.is_dir()
            else []
        )

        if cached_video is not None:
            meta = {"ok": True, "cached": True, "source": "video.mp4"}
            rendered_frames = decode_video_frames(cached_video, num_frames)
        elif len(cached_pngs) >= num_frames:
            meta = {"ok": True, "cached": True, "source": "png"}
            rendered_frames = decode_png_sequence(legacy_frames, num_frames)
        else:
            with tempfile.TemporaryDirectory(prefix=f"bvb_vjepa_{scene_id}_") as temp_dir:
                frames_dir = Path(temp_dir) / "frames"
                meta = render_blend_to_frames(
                    blend_path,
                    frames_dir,
                    blender=blender,
                    resolution=render_resolution,
                    num_samples=num_frames,
                    timeout=render_timeout,
                )
                meta["source"] = "png"
                if not meta.get("ok"):
                    row["status"] = "render_error"
                    row["error"] = (
                        meta.get("stderr_tail") or meta.get("stdout_tail") or "render failed"
                    )[-500:]
                    return row
                rendered_frames = decode_png_sequence(frames_dir, num_frames)
                if keep_renders is not None:
                    keep_renders.mkdir(parents=True, exist_ok=True)
                    fps = int((meta.get("render") or {}).get("fps") or 24)
                    out_mp4 = keep_renders / f"{scene_id}.mp4"
                    tmp_mp4 = Path(temp_dir) / "video.mp4"
                    encode_png_sequence_to_mp4(frames_dir, tmp_mp4, fps=fps)
                    shutil.move(str(tmp_mp4), str(out_mp4))

        if not meta.get("ok"):
            row["status"] = "render_error"
            row["error"] = (meta.get("stderr_tail") or meta.get("stdout_tail") or "render failed")[
                -500:
            ]
            return row

        original_frames = decode_video_frames(original_video, num_frames)
        sims = compute_vjepa_similarity(
            original_frames,
            rendered_frames,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
        )
        row["layout_sim"] = sims["layout_sim"]
        row["motion_sim"] = sims["motion_sim"]
        row["vision_sim"] = sims["vision_sim"]
        row["status"] = "ok"
        row["render_cached"] = bool(meta.get("cached"))
        row["render_source"] = meta.get("source")
        return row
    except SystemExit as exc:  # noqa: BLE001 — e.g. missing Blender must not kill the shard
        # SystemExit is BaseException, not Exception; convert to a scored row.
        row["status"] = "missing_render"
        row["error"] = str(exc) or "SystemExit (often: no camera_renders cache and no Blender)"
        return row
    except Exception as exc:  # noqa: BLE001 — per-scene isolation for cluster runs
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("vision_sim") is not None]

    def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        mean = float(statistics.fmean(values))
        std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        return mean, std

    def _filled(key: str) -> list[float]:
        # A scene that never scored counts as 0 rather than dropping out of the
        # mean: otherwise a submission that fails to render is rewarded for it.
        return [
            float(row[key])
            if row.get("status") == "ok" and row.get(key) is not None
            else 0.0
            for row in rows
        ]

    def _scored(key: str) -> list[float]:
        return [float(row[key]) for row in ok_rows if row.get(key) is not None]

    vision_mean, vision_std = _mean_std(_filled("vision_sim"))
    layout_mean, layout_std = _mean_std(_filled("layout_sim"))
    motion_mean, motion_std = _mean_std(_filled("motion_sim"))
    scored_vision, _ = _mean_std(_scored("vision_sim"))
    scored_layout, _ = _mean_std(_scored("layout_sim"))
    scored_motion, _ = _mean_std(_scored("motion_sim"))
    return {
        "num_scenes": len(rows),
        "num_ok": len(ok_rows),
        "num_error": sum(row.get("status") != "ok" for row in rows),
        "vision_sim": vision_mean,
        "vision_sim_std": vision_std,
        "layout_sim": layout_mean,
        "layout_sim_std": layout_std,
        "motion_sim": motion_mean,
        "motion_sim_std": motion_std,
        "scored_only": {
            "vision_sim": scored_vision,
            "layout_sim": scored_layout,
            "motion_sim": scored_motion,
        },
        "by_status": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({str(row.get("status")) for row in rows})
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute V-JEPA vision similarity for a Stage-1 run."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run",
        type=Path,
        help="Stage-1 run directory containing blends/. Writes vision_sim*.json here.",
    )
    source.add_argument(
        "--submissions",
        type=Path,
        help="JSONL manifest with id and submission_path.",
    )
    parser.add_argument("--vsi-bench", type=Path, default=DEFAULT_VSI_BENCH)
    parser.add_argument("--qa-metadata", type=Path, default=DEFAULT_QA_METADATA)
    parser.add_argument("--output", type=Path, help="Per-scene JSONL (default: <run>/vision_sim.jsonl).")
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Summary JSON (default: <run>/vision_sim_summary.json).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BVB_VJEPA_MODEL", DEFAULT_MODEL),
        help=f"HF model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: auto | cuda | cuda:0 | cpu. auto picks cuda when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Encoder compute dtype (bfloat16 recommended on Ampere+ GPUs).",
    )
    parser.add_argument("--num-frames", type=int, default=64, help="Frames sampled per video clip (V-JEPA 2.1 default clip length).")
    parser.add_argument("--blender", default=os.getenv("BLENDER_BIN"))
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--render-timeout", type=float, default=600.0)
    parser.add_argument(
        "--keep-renders",
        type=Path,
        nargs="?",
        const=Path("__AUTO__"),
        default=Path("__AUTO__"),
        help=(
            "Directory for cached camera PNG frames. Default with --run: "
            "<run>/camera_renders/. Pass an explicit path to override. "
            "Use --no-keep-renders to delete after scoring."
        ),
    )
    parser.add_argument(
        "--no-keep-renders",
        action="store_true",
        help="Delete rendered frames after each scene (disables cache).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--sample", type=int)
    selection.add_argument("--scene-ids", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", help="Evaluate shard i/N into independent output files.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print the plan without loading the encoder or rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    datasets = scene_datasets(args.qa_metadata)
    shard = parse_shard(args.shard)

    if args.run is not None:
        records = records_from_run(args.run)
        if shard and args.output is None:
            output_path = args.run / f"vision_sim.shard-{shard[0]}-of-{shard[1]}.jsonl"
        else:
            output_path = args.output or (args.run / "vision_sim.jsonl")
        if shard and args.summary_output is None:
            summary_output = args.run / f"vision_sim_summary.shard-{shard[0]}-of-{shard[1]}.json"
        else:
            summary_output = args.summary_output or (args.run / "vision_sim_summary.json")
        if args.no_keep_renders:
            keep_renders: Path | None = None
        elif args.keep_renders == Path("__AUTO__"):
            keep_renders = args.run / "camera_renders"
        else:
            keep_renders = args.keep_renders
    else:
        assert args.submissions is not None
        records = [
            {"id": str(row["id"]), "submission_path": str(row["submission_path"])}
            for row in read_jsonl(args.submissions)
        ]
        output_path = args.output or Path("vision_sim.jsonl")
        summary_output = args.summary_output or Path("vision_sim_summary.json")
        if args.no_keep_renders or args.keep_renders == Path("__AUTO__"):
            keep_renders = None if args.no_keep_renders else Path("camera_renders")
        else:
            keep_renders = args.keep_renders
    selected = select_records(
        records,
        limit=args.limit,
        sample=args.sample,
        scene_ids=args.scene_ids,
        seed=args.seed,
        shard=shard,
    )

    completed = load_completed_ids(output_path) if args.resume else set()
    pending = [row for row in selected if row["id"] not in completed]

    plan = {
        "num_submissions": len(records),
        "num_selected": len(selected),
        "num_pending": len(pending),
        "num_skipped_resume": len(selected) - len(pending),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "num_frames": args.num_frames,
        "output": str(output_path),
        "summary_output": str(summary_output),
        "keep_renders": str(keep_renders) if keep_renders else None,
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        for row in pending[:20]:
            video = locate_video(row["id"], datasets.get(row["id"]), args.vsi_bench)
            print(f"{row['id']}: blend={row['submission_path']} video={video}")
        if len(pending) > 20:
            print(f"... {len(pending) - 20} more")
        return

    model, processor, dtype = load_encoder(args.model, args.device, args.dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output_path.exists() else "w"
    rows: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        rows.extend(read_jsonl(output_path))

    with output_path.open(mode, encoding="utf-8") as out_f:
        for index, record in enumerate(pending, start=1):
            scene_id = record["id"]
            blend_path = Path(record["submission_path"])
            original_video = locate_video(scene_id, datasets.get(scene_id), args.vsi_bench)
            print(f"[{index}/{len(pending)}] {scene_id}")
            if original_video is None:
                row = {
                    "id": scene_id,
                    "submission_path": str(blend_path),
                    "status": "missing_original_video",
                    "vision_sim": None,
                    "layout_sim": None,
                    "motion_sim": None,
                }
            else:
                row = evaluate_scene(
                    scene_id=scene_id,
                    blend_path=blend_path,
                    original_video=original_video,
                    model=model,
                    processor=processor,
                    device=args.device,
                    dtype=dtype,
                    num_frames=args.num_frames,
                    blender=args.blender,
                    render_resolution=args.render_resolution,
                    render_timeout=args.render_timeout,
                    keep_renders=keep_renders,
                )
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            rows.append(row)
            if args.stop_on_error and row.get("status") != "ok":
                break

    summary = summarize(rows)
    summary["model"] = args.model
    summary["device"] = args.device
    summary["num_frames"] = args.num_frames
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
