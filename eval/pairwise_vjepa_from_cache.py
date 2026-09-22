#!/usr/bin/env python3
"""Compute cross-model V-JEPA matrices from cached exact descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    systems = manifest["systems"]
    scenes = manifest["scenes"]
    features = {}
    for system in systems:
        run = system["run"]
        path = args.cache_dir / f"{run}.safetensors"
        if not path.is_file():
            raise SystemExit(f"missing feature cache: {path}")
        features[run] = load_file(str(path))

    n = len(systems)
    vision_sum = np.zeros((n, n), dtype=np.float64)
    layout_sum = np.zeros((n, n), dtype=np.float64)
    motion_sum = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)

    for scene_idx, _scene in enumerate(scenes):
        valid_indices = [
            idx
            for idx, system in enumerate(systems)
            if bool(features[system["run"]]["valid"][scene_idx])
        ]
        if valid_indices:
            layout_np = np.stack(
                [
                    features[systems[idx]["run"]]["layout"][scene_idx]
                    for idx in valid_indices
                ]
            )
            motion_np = np.stack(
                [
                    features[systems[idx]["run"]]["motion"][scene_idx]
                    for idx in valid_indices
                ]
            )
            layout = torch.from_numpy(layout_np).to(
                device=args.device,
                dtype=torch.float32,
            )
            motion = torch.from_numpy(motion_np).to(
                device=args.device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                layout_matrix = (
                    torch.einsum("mpd,npd->mn", layout, layout)
                    / layout.shape[1]
                )
                motion_matrix = motion @ motion.T
                vision_matrix = 0.5 * (layout_matrix + motion_matrix)
            layout_cpu = layout_matrix.cpu().numpy()
            motion_cpu = motion_matrix.cpu().numpy()
            vision_cpu = vision_matrix.cpu().numpy()
            index_array = np.asarray(valid_indices)
            selection = np.ix_(index_array, index_array)
            layout_sum[selection] += layout_cpu
            motion_sum[selection] += motion_cpu
            vision_sum[selection] += vision_cpu
            counts[selection] += 1

        done = scene_idx + 1
        if done % args.progress_every == 0 or done == len(scenes):
            print(f"[{done}/{len(scenes)}]", flush=True)

    means = {}
    for values, name in (
        (vision_sum, "vision_mean"),
        (layout_sum, "layout_mean"),
        (motion_sum, "motion_mean"),
    ):
        mean = np.full((n, n), np.nan, dtype=np.float64)
        np.divide(values, counts, out=mean, where=counts > 0)
        means[name] = mean.tolist()

    payload = {
        **manifest,
        "counts": counts.tolist(),
        "vision_sum": vision_sum.tolist(),
        "layout_sum": layout_sum.tolist(),
        "motion_sum": motion_sum.tolist(),
        **means,
        "encoded_scenes": [
            int(features[system["run"]]["valid"].sum())
            for system in systems
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
