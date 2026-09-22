#!/usr/bin/env python3
"""Compute cross-model V-JEPA similarity between reconstruction videos.

Each scene is encoded once per available model. Pairwise layout, motion, and
combined similarities are then accumulated over scenes where both models have
a successful camera render. The script supports one process per GPU via
``--shard i/N`` and can merge shard JSON files without loading the encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vjepa_sim_metric import (
    DEFAULT_MODEL,
    decode_video_frames,
    infer_token_grid,
    load_encoder,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_RESULTS = REPO_ROOT / "sandbox" / "results"
DEFAULT_META = HERE / "test.jsonl"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_scenes(path: Path) -> list[str]:
    return sorted({str(row["scene_name"]) for row in read_jsonl(path)})


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        index, count = (int(part) for part in value.split("/", 1))
    except ValueError as exc:
        raise SystemExit("--shard must look like i/N") from exc
    if not (1 <= index <= count):
        raise SystemExit("--shard must satisfy 1 <= i <= N")
    return index, count


def load_systems(path: Path) -> list[dict[str, str]]:
    systems = json.loads(path.read_text(encoding="utf-8"))
    required = {"run", "label", "family"}
    if not isinstance(systems, list) or not systems:
        raise SystemExit("systems file must contain a non-empty JSON list")
    for row in systems:
        if not isinstance(row, dict) or not required <= set(row):
            raise SystemExit(f"each system needs {sorted(required)}")
    runs = [str(row["run"]) for row in systems]
    if len(runs) != len(set(runs)):
        raise SystemExit("duplicate run in systems file")
    return [
        {key: str(row[key]) for key in ("run", "label", "family")}
        for row in systems
    ]


def encode_video(
    path: Path,
    *,
    model: Any,
    processor: Any,
    device: str,
    dtype: Any,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    frames = decode_video_frames(path, num_frames)
    inputs = processor([list(frames)], return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(device=device, dtype=dtype)
    with torch.inference_mode():
        tokens = model(
            pixel_values_videos=pixel_values,
            skip_predictor=True,
        ).last_hidden_state[0]

    motion = F.normalize(tokens.mean(dim=0).float(), dim=0)
    temporal, height, width = infer_token_grid(tokens.shape[0], num_frames)
    if height > 1 and width > 1 and temporal * height * width == tokens.shape[0]:
        layout = tokens.reshape(temporal, height, width, -1).mean(dim=0)
        layout = F.normalize(layout.reshape(-1, layout.shape[-1]).float(), dim=1)
    else:
        layout = motion[None, :]
    return (
        layout.cpu().numpy().astype(np.float32, copy=False),
        motion.cpu().numpy().astype(np.float32, copy=False),
    )


def empty_accumulators(n: int) -> dict[str, np.ndarray]:
    return {
        "vision_sum": np.zeros((n, n), dtype=np.float64),
        "layout_sum": np.zeros((n, n), dtype=np.float64),
        "motion_sum": np.zeros((n, n), dtype=np.float64),
        "counts": np.zeros((n, n), dtype=np.int64),
    }


def score_scenes(args: argparse.Namespace) -> None:
    systems = load_systems(args.systems)
    scenes = canonical_scenes(args.qa_metadata)
    shard = parse_shard(args.shard)
    if shard is not None:
        index, count = shard
        scenes = scenes[index - 1 :: count]
    if args.limit is not None:
        scenes = scenes[: args.limit]

    model, processor, dtype = load_encoder(args.model, args.device, args.dtype)
    accumulators = empty_accumulators(len(systems))
    encode_ok = np.zeros(len(systems), dtype=np.int64)
    errors: dict[str, list[dict[str, str]]] = {
        row["run"]: [] for row in systems
    }

    for scene_idx, scene in enumerate(scenes, start=1):
        features: list[tuple[np.ndarray, np.ndarray] | None] = []
        for system_idx, row in enumerate(systems):
            video = (
                args.results_dir
                / row["run"]
                / "camera_renders"
                / f"{scene}.mp4"
            )
            if not video.is_file() or video.stat().st_size == 0:
                features.append(None)
                continue
            try:
                features.append(
                    encode_video(
                        video,
                        model=model,
                        processor=processor,
                        device=args.device,
                        dtype=dtype,
                        num_frames=args.num_frames,
                    )
                )
                encode_ok[system_idx] += 1
            except Exception as exc:  # noqa: BLE001
                features.append(None)
                errors[row["run"]].append(
                    {"scene": scene, "error": f"{type(exc).__name__}: {exc}"}
                )

        for row_idx, row_features in enumerate(features):
            if row_features is None:
                continue
            layout_a, motion_a = row_features
            for col_idx in range(row_idx + 1):
                col_features = features[col_idx]
                if col_features is None:
                    continue
                layout_b, motion_b = col_features
                if layout_a.shape != layout_b.shape:
                    errors[systems[row_idx]["run"]].append(
                        {
                            "scene": scene,
                            "error": (
                                "layout shape mismatch with "
                                f"{systems[col_idx]['run']}: "
                                f"{layout_a.shape} vs {layout_b.shape}"
                            ),
                        }
                    )
                    continue
                layout_sim = float(np.sum(layout_a * layout_b, axis=1).mean())
                motion_sim = float(np.dot(motion_a, motion_b))
                vision_sim = 0.5 * (layout_sim + motion_sim)
                for key, value in (
                    ("layout_sum", layout_sim),
                    ("motion_sum", motion_sim),
                    ("vision_sum", vision_sim),
                ):
                    accumulators[key][row_idx, col_idx] += value
                    if row_idx != col_idx:
                        accumulators[key][col_idx, row_idx] += value
                accumulators["counts"][row_idx, col_idx] += 1
                if row_idx != col_idx:
                    accumulators["counts"][col_idx, row_idx] += 1

        if scene_idx % args.progress_every == 0 or scene_idx == len(scenes):
            print(
                f"[{scene_idx}/{len(scenes)}] "
                f"encoded={int(encode_ok.sum())}",
                flush=True,
            )

    payload = {
        "systems": systems,
        "model": args.model,
        "num_frames": args.num_frames,
        "shard": args.shard,
        "num_scenes": len(scenes),
        "encode_ok": encode_ok.tolist(),
        "errors": errors,
        **{key: value.tolist() for key, value in accumulators.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


def merge_shards(inputs: list[Path], output: Path) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not payloads:
        raise SystemExit("no shard files to merge")
    systems = payloads[0]["systems"]
    model = payloads[0]["model"]
    num_frames = payloads[0]["num_frames"]
    n = len(systems)
    accumulators = empty_accumulators(n)
    encode_ok = np.zeros(n, dtype=np.int64)
    errors: dict[str, list[dict[str, str]]] = {
        row["run"]: [] for row in systems
    }
    num_scenes = 0
    for payload in payloads:
        if payload["systems"] != systems:
            raise SystemExit("shard systems do not match")
        if payload["model"] != model or payload["num_frames"] != num_frames:
            raise SystemExit("shard encoder settings do not match")
        num_scenes += int(payload["num_scenes"])
        encode_ok += np.asarray(payload["encode_ok"], dtype=np.int64)
        for key in accumulators:
            accumulators[key] += np.asarray(payload[key], dtype=accumulators[key].dtype)
        for run, rows in payload["errors"].items():
            errors[run].extend(rows)

    counts = accumulators["counts"]
    means = {}
    for source, target in (
        ("vision_sum", "vision_mean"),
        ("layout_sum", "layout_mean"),
        ("motion_sum", "motion_mean"),
    ):
        values = np.full((n, n), np.nan, dtype=np.float64)
        np.divide(
            accumulators[source],
            counts,
            out=values,
            where=counts > 0,
        )
        means[target] = values.tolist()

    merged = {
        "systems": systems,
        "model": model,
        "num_frames": num_frames,
        "num_scenes": num_scenes,
        "encode_ok": encode_ok.tolist(),
        "errors": errors,
        **{key: value.tolist() for key, value in accumulators.items()},
        **means,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"merged {len(inputs)} shards -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--qa-metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--shard")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", nargs="+", type=Path)
    args = parser.parse_args()
    if args.merge is None and args.systems is None:
        parser.error("--systems is required unless --merge is used")
    return args


def main() -> None:
    args = parse_args()
    if args.merge is not None:
        merge_shards(args.merge, args.output)
    else:
        score_scenes(args)


if __name__ == "__main__":
    main()
