#!/usr/bin/env python3
"""Cache exact float16 V-JEPA layout and motion descriptors per model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from pairwise_vjepa_similarity import (
    DEFAULT_META,
    DEFAULT_RESULTS,
    canonical_scenes,
    encode_video,
    load_systems,
    parse_shard,
)
from vjepa_sim_metric import DEFAULT_MODEL, load_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--qa-metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--shard", help="Model shard i/N.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = load_systems(args.systems)
    shard = parse_shard(args.shard)
    selected = list(enumerate(systems))
    if shard is not None:
        index, count = shard
        selected = selected[index - 1 :: count]
    scenes = canonical_scenes(args.qa_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if shard is None or shard[0] == 1:
        manifest = {
            "systems": systems,
            "scenes": scenes,
            "model": args.model,
            "num_frames": args.num_frames,
            "feature_dtype": "float16",
            "layout_definition": "temporally pooled normalized spatial patch map",
            "motion_definition": "normalized global token mean",
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    model, processor, dtype = load_encoder(args.model, args.device, args.dtype)
    for system_number, (system_idx, system) in enumerate(selected, start=1):
        run = system["run"]
        output = args.output_dir / f"{run}.safetensors"
        error_output = args.output_dir / f"{run}.errors.json"
        if args.resume and output.is_file():
            print(f"[skip] {run}: {output.name} exists", flush=True)
            continue

        layout: np.ndarray | None = None
        motion: np.ndarray | None = None
        valid = np.zeros(len(scenes), dtype=np.uint8)
        errors: list[dict[str, str]] = []

        for scene_idx, scene in enumerate(scenes):
            video = (
                args.results_dir
                / run
                / "camera_renders"
                / f"{scene}.mp4"
            )
            if not video.is_file() or video.stat().st_size == 0:
                continue
            try:
                scene_layout, scene_motion = encode_video(
                    video,
                    model=model,
                    processor=processor,
                    device=args.device,
                    dtype=dtype,
                    num_frames=args.num_frames,
                )
                if layout is None or motion is None:
                    layout = np.zeros(
                        (len(scenes), *scene_layout.shape),
                        dtype=np.float16,
                    )
                    motion = np.zeros(
                        (len(scenes), *scene_motion.shape),
                        dtype=np.float16,
                    )
                if scene_layout.shape != layout.shape[1:]:
                    raise RuntimeError(
                        f"layout shape changed: {scene_layout.shape} "
                        f"vs {layout.shape[1:]}"
                    )
                layout[scene_idx] = scene_layout.astype(np.float16)
                motion[scene_idx] = scene_motion.astype(np.float16)
                valid[scene_idx] = 1
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"scene": scene, "error": f"{type(exc).__name__}: {exc}"}
                )

            done = scene_idx + 1
            if done % args.progress_every == 0 or done == len(scenes):
                print(
                    f"[{system_number}/{len(selected)}] {run}: "
                    f"{done}/{len(scenes)} valid={int(valid.sum())}",
                    flush=True,
                )

        if layout is None or motion is None:
            raise SystemExit(f"{run}: no camera render could be encoded")
        save_file(
            {"layout": layout, "motion": motion, "valid": valid},
            str(output),
            metadata={
                "run": run,
                "model": args.model,
                "num_frames": str(args.num_frames),
                "system_index": str(system_idx),
            },
        )
        error_output.write_text(
            json.dumps(errors, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[done] {run}: valid={int(valid.sum())} "
            f"bytes={output.stat().st_size}",
            flush=True,
        )


if __name__ == "__main__":
    main()
