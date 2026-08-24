#!/usr/bin/env python3
"""Build self-contained ranking HTML packs.

Modes:
  Single form (default): one survey.html, fixed 3x3 scenes.
  Multi form (--forms N): pool of scenes (--pool-per-source 8 → 24),
  each form samples 3 per source → survey-r01.html … survey-rNN.html.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import string
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HUMAN = Path(__file__).resolve().parent
FORM = HUMAN / "form" / "rank.html"
RESULTS = REPO / "sandbox" / "results"
TEST_JSONL = REPO / "eval" / "test.jsonl"
VSI_BENCH = REPO / "VSI-Bench"
INSTRUMENT_ID = "bvb-human-rank-v1"
INSTRUMENT_VERSION = "1.0"
SOURCE_ORDER = ["arkitscenes", "scannet", "scannetpp"]

DEFAULT_MODELS = [
    ("mini-harness-grok-4.5-reasoning-high-run01", "Grok high"),
    ("mini-harness-gpt-5.6-sol-reasoning-xhigh-run01", "Sol xhigh"),
    ("mini-harness-gemini-3.1-pro-preview-reasoning-high-run01", "Gemini-3.1"),
    ("mini-harness-claude-opus-4-6-run01", "Opus-4.6"),
    ("mini-harness-qwen3.5-397b-a17b-run01", "Qwen3.5-397B"),
]


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_ffmpeg() -> str | None:
    env = os.environ.get("FFMPEG_BIN")
    if env and Path(env).is_file():
        return env
    which = shutil.which("ffmpeg")
    if which:
        return which
    local = REPO / ".conda" / "envs" / "bvb-sandbox" / "bin" / "ffmpeg"
    if local.is_file():
        return str(local)
    return None


def find_ffprobe(ffmpeg: str | None) -> str | None:
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)
    env = os.environ.get("FFPROBE_BIN")
    if env and Path(env).is_file():
        return env
    return shutil.which("ffprobe")


def load_scene_sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in _iter_jsonl(TEST_JSONL):
        scene = str(rec.get("scene_name", ""))
        dataset = rec.get("dataset")
        if scene and dataset and scene not in out:
            out[scene] = str(dataset)
    return out


def locate_reference(scene: str, source: str | None) -> Path | None:
    candidates = []
    if source:
        candidates.append(VSI_BENCH / source / f"{scene}.mp4")
    for ds in SOURCE_ORDER:
        candidates.append(VSI_BENCH / ds / f"{scene}.mp4")
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def render_path(run_name: str, scene: str) -> Path:
    return RESULTS / run_name / "camera_renders" / f"{scene}.mp4"


def video_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def scene_basic_bundle(tests: list[dict]) -> dict | None:
    basic = [t for t in tests if t.get("test_type") == "basic_validity"]
    if not basic:
        return None
    evaluated = [t for t in basic if t.get("status") in {"pass", "fail"}]
    if not evaluated:
        return {"evaluated": False, "passed": False}
    return {
        "evaluated": True,
        "passed": all(t.get("status") == "pass" for t in evaluated),
    }


def scene_test_rate(tests: list[dict]) -> float | None:
    non_basic = [t for t in tests if t.get("test_type") != "basic_validity"]
    ev = [t for t in non_basic if t.get("status") in {"pass", "fail"}]
    passed = sum(1 for t in ev if t["status"] == "pass")
    n = len(ev)
    bundle = scene_basic_bundle(tests)
    if bundle and bundle["evaluated"]:
        n += 1
        if bundle["passed"]:
            passed += 1
    if n == 0:
        return None
    return passed / n


def mean_scene_test(scene: str, run_names: list[str]) -> float | None:
    rates = []
    for run in run_names:
        path = RESULTS / run / "unit_tests.jsonl"
        tests = [t for t in _iter_jsonl(path) if str(t.get("id") or t.get("scene_name")) == scene]
        if not tests:
            continue
        rate = scene_test_rate(tests)
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None
    return sum(rates) / len(rates)


def tertile(values: list[float], value: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 1
    lo = ordered[max(0, len(ordered) // 3 - 1)]
    hi = ordered[max(0, (2 * len(ordered)) // 3 - 1)]
    if value <= lo:
        return 0
    if value <= hi:
        return 1
    return 2


def parse_models(raw: list[str] | None) -> list[tuple[str, str]]:
    if not raw:
        return list(DEFAULT_MODELS)
    alias = {run: label for run, label in DEFAULT_MODELS}
    out = []
    for item in raw:
        if "=" in item:
            run, label = item.split("=", 1)
            out.append((run.strip(), label.strip()))
        else:
            out.append((item.strip(), alias.get(item.strip(), item.strip())))
    return out


def probe_field(ffprobe: str | None, src: Path, entry: str, section: str) -> str | None:
    if not ffprobe:
        return None
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            f"{section}={entry}",
            "-of",
            "csv=p=0",
            str(src),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip().split(",")[0]
    return value or None


def probe_duration(ffprobe: str | None, src: Path) -> float | None:
    raw = probe_field(ffprobe, src, "duration", "format")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def probe_nb_frames(ffprobe: str | None, src: Path) -> int | None:
    raw = probe_field(ffprobe, src, "nb_frames", "stream")
    if raw and raw.isdigit():
        return int(raw)
    duration = probe_duration(ffprobe, src)
    rate = probe_field(ffprobe, src, "avg_frame_rate", "stream") or ""
    if duration and "/" in rate:
        num, den = rate.split("/", 1)
        try:
            fps = float(num) / float(den)
        except ValueError:
            return None
        if fps > 0:
            return max(1, int(duration * fps))
    return None


def compress_clip(ffmpeg: str, ffprobe: str | None, src: Path, dst: Path, *, sparse: bool) -> None:
    extra: list[str] = []
    if sparse:
        n_frames = probe_nb_frames(ffprobe, src)
        duration = probe_duration(ffprobe, src)
        if n_frames and n_frames > 80:
            step = max(1, round(n_frames / 64))
            vf = f"select='not(mod(n\\,{step}))',setpts=N/12/TB,scale=480:-2"
        elif duration and duration > 12:
            extra = ["-t", "8"]
            vf = "scale=480:-2,fps=12"
        else:
            vf = "scale=480:-2"
    else:
        vf = "scale=512:-2"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-vf",
        vf,
        *extra,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "24",
        str(dst),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not dst.is_file() or dst.stat().st_size < 1000:
        raise RuntimeError(f"ffmpeg failed on {src.name}: {completed.stderr[-400:]}")


def data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{payload}"


def sample_scenes(
    rng: random.Random,
    sources: dict[str, str],
    models: list[tuple[str, str]],
    per_source: int,
    scene_ids: list[str] | None,
    pool_by_source: dict[str, list[str]] | None = None,
) -> list[str]:
    if scene_ids:
        return list(scene_ids)
    if pool_by_source:
        picked: list[str] = []
        for source in SOURCE_ORDER:
            pool = list(pool_by_source.get(source, []))
            if len(pool) < per_source:
                raise SystemExit(f"Pool too small for {source}: need {per_source}, have {len(pool)}")
            rng.shuffle(pool)
            picked.extend(pool[:per_source])
        return picked

    run_names = [m[0] for m in models]
    need = len(models)
    by_source: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    for scene, source in sources.items():
        if locate_reference(scene, source) is None:
            continue
        if sum(1 for run, _ in models if video_ok(render_path(run, scene))) < need:
            continue
        by_source[source].append((scene, mean_scene_test(scene, run_names)))
    picked = []
    for source in SOURCE_ORDER:
        pool = list(by_source.get(source, []))
        if len(pool) < per_source:
            print(f"[warn] only {len(pool)} eligible scenes for {source}")
        scores = [row[1] for row in pool if row[1] is not None]
        buckets: dict[int, list[tuple[str, float | None]]] = defaultdict(list)
        for row in pool:
            bucket = tertile(scores, row[1]) if row[1] is not None and scores else 1
            buckets[bucket].append(row)
        take: list[str] = []
        for bucket in range(3):
            candidates = buckets.get(bucket, [])
            rng.shuffle(candidates)
            if candidates and len(take) < per_source:
                take.append(candidates[0][0])
        if len(take) < per_source:
            leftover = [row[0] for row in pool if row[0] not in take]
            rng.shuffle(leftover)
            take.extend(leftover[: per_source - len(take)])
        picked.extend(take[:per_source])
    return picked


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scene_blind_codes(
    rng: random.Random,
    models: list[tuple[str, str]],
    letters: list[str],
) -> dict[str, dict]:
    available = list(models)
    codes = letters[:]
    rng.shuffle(codes)
    rng.shuffle(available)
    return {
        code: {"label": label, "run": run}
        for code, (run, label) in zip(codes, available)
    }


def build_scene_cache(
    scene: str,
    source: str,
    models: list[tuple[str, str]],
    rng: random.Random,
    letters: list[str],
    cache_dir: Path,
    ffmpeg: str,
    ffprobe: str | None,
    sources: dict[str, str],
) -> tuple[dict, dict] | None:
    ref = locate_reference(scene, source or sources.get(scene, ""))
    if ref is None:
        print(f"[skip] {scene}: no reference")
        return None
    if sum(1 for run, _ in models if video_ok(render_path(run, scene))) < len(models):
        print(f"[skip] {scene}: missing renders")
        return None

    codes_map = scene_blind_codes(rng, models, letters)
    scene_cache = cache_dir / scene
    scene_cache.mkdir(parents=True, exist_ok=True)
    ref_path = scene_cache / "ref.mp4"
    if not ref_path.is_file():
        compress_clip(ffmpeg, ffprobe, ref, ref_path, sparse=True)
    for code, info in codes_map.items():
        run = info["run"]
        clip = scene_cache / f"{run}.mp4"
        if not clip.is_file():
            compress_clip(ffmpeg, ffprobe, render_path(run, scene), clip, sparse=False)

    blind_entry = {
        "source": source or sources.get(scene, ""),
        "codes": codes_map,
    }
    return blind_entry, codes_map


def items_from_cache(
    scene_ids: list[str],
    blind: dict[str, dict],
    cache_dir: Path,
) -> list[dict]:
    items = []
    for scene in scene_ids:
        entry = blind[scene]
        scene_cache = cache_dir / scene
        ref_uri = data_uri(scene_cache / "ref.mp4")
        candidates = []
        for code, info in entry["codes"].items():
            candidates.append(
                {
                    "code": code,
                    "video": data_uri(scene_cache / f"{info['run']}.mp4"),
                }
            )
        items.append(
            {
                "scene_id": scene,
                "source": entry["source"],
                "reference": ref_uri,
                "candidates": candidates,
            }
        )
    return items


def write_survey_html(template: str, pack: dict, path: Path) -> float:
    html = template.replace("__PACK_JSON__", json.dumps(pack, ensure_ascii=False), 1)
    path.write_text(html, encoding="utf-8")
    return path.stat().st_size / (1024 * 1024)


def build(args: argparse.Namespace) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("Need ffmpeg. Set FFMPEG_BIN or install ffmpeg.")
    ffprobe = find_ffprobe(ffmpeg)
    rng = random.Random(args.seed)
    models = parse_models(args.models)
    sources = load_scene_sources()
    if not sources:
        raise SystemExit(f"No scenes in {TEST_JSONL}")

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty. Pass --force to overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    if args.force and args.forms > 1:
        for stale in (out / "survey.html", out / "scene_ids.txt"):
            if stale.is_file():
                stale.unlink()

    wave_id = args.wave_id or out.name
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    letters = list(string.ascii_uppercase[: len(models)])
    template = FORM.read_text(encoding="utf-8")
    if "__PACK_JSON__" not in template:
        raise SystemExit(f"{FORM} missing __PACK_JSON__ placeholder")

    multi = args.forms > 1
    pool_per = args.pool_per_source if multi else args.per_source

    scene_ids = None
    if args.scene_ids:
        scene_ids = [s.strip() for s in Path(args.scene_ids).read_text().split() if s.strip()]
    elif args.scenes:
        scene_ids = args.scenes

    pool_scenes = sample_scenes(rng, sources, models, pool_per, scene_ids)
    if len(pool_scenes) < 3:
        raise SystemExit("Not enough scenes with original + all model renders.")

    cache_dir = out / "_clips_cache"
    blind: dict[str, dict] = {}
    pool_by_source: dict[str, list[str]] = defaultdict(list)

    print(f"compressing pool ({len(pool_scenes)} scenes)...")
    for scene in pool_scenes:
        source = sources.get(scene, "")
        built = build_scene_cache(
            scene, source, models, rng, letters, cache_dir, ffmpeg, ffprobe, sources
        )
        if not built:
            continue
        blind_entry, _ = built
        blind[scene] = blind_entry
        pool_by_source[blind_entry["source"]].append(scene)

    pool_scenes = sorted(blind.keys())
    if multi:
        for source in SOURCE_ORDER:
            if len(pool_by_source.get(source, [])) < args.per_source:
                raise SystemExit(
                    f"Pool needs {args.per_source} scenes per source for {source}, "
                    f"have {len(pool_by_source.get(source, []))}"
                )

    need = args.per_source * len(SOURCE_ORDER)
    assignments: list[dict] = []
    coverage: dict[str, int] = {s: 0 for s in pool_scenes}
    html_paths: list[Path] = []

    print(f"building {args.forms} forms...")
    for i in range(1, args.forms + 1):
        slot = f"r{i:02d}"
        pack_id = f"{wave_id}-{slot}" if multi else wave_id
        form_rng = random.Random(args.seed + i * 9973)
        if multi:
            picked = sample_scenes(
                form_rng,
                sources,
                models,
                args.per_source,
                None,
                pool_by_source,
            )
        else:
            picked = pool_scenes

        items = items_from_cache(picked, blind, cache_dir)
        if len(items) != need:
            raise SystemExit(f"{pack_id}: built {len(items)} items, expected {need}")

        for s in picked:
            coverage[s] += 1

        pack = {
            "pack_id": pack_id,
            "wave_id": wave_id,
            "rater_slot": slot if multi else None,
            "instrument_id": INSTRUMENT_ID,
            "instrument_version": INSTRUMENT_VERSION,
            "language": args.language,
            "created_utc": created,
            "n_models": len(models),
            "pool_size": len(pool_scenes),
            "items": items,
        }
        survey_name = f"survey-{slot}.html" if multi else "survey.html"
        survey_path = out / survey_name
        mb = write_survey_html(template, pack, survey_path)
        html_paths.append(survey_path)
        assignments.append(
            {
                "pack_id": pack_id,
                "rater_slot": slot if multi else None,
                "survey_file": survey_name,
                "scene_ids": picked,
            }
        )
        print(f"  {survey_name} ({mb:.2f} MB)")

    write_json(
        out / "BLIND_MAP.json",
        {
            "wave_id": wave_id,
            "instrument_id": INSTRUMENT_ID,
            "instrument_version": INSTRUMENT_VERSION,
            "created_utc": created,
            "models": [{"run": run, "label": label} for run, label in models],
            "pool_size": len(pool_scenes),
            "scenes": blind,
            "warning": "Experimenter only. Do not send this file or _clips_cache.",
        },
    )
    write_json(out / "assignments.json", {"wave_id": wave_id, "assignments": assignments})
    (out / "pool_scene_ids.txt").write_text("\n".join(pool_scenes) + "\n", encoding="utf-8")

    if multi:
        send = (
            "Send ONE file per person:\n"
            "  survey-r01.html … survey-r16.html\n"
            "Each has a fixed ID (r01–r16). Do not swap files.\n"
            "Do not send BLIND_MAP.json or _clips_cache/.\n"
        )
    else:
        send = "Send survey.html only. Do not send BLIND_MAP.json or _clips_cache/.\n"
    (out / "SEND.txt").write_text(send, encoding="utf-8")

    print(f"wave:   {wave_id}")
    print(f"pool:   {len(pool_scenes)} scenes")
    print(f"forms:  {args.forms} × {need} questions")
    if multi:
        print("coverage (scene × forms):")
        for scene, n in sorted(coverage.items(), key=lambda x: -x[1]):
            print(f"  {scene:16s} {n}")
    print("Keep BLIND_MAP.json private.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=3, help="Scenes per source in each form.")
    parser.add_argument("--pool-per-source", type=int, default=8, help="Pool per source when --forms > 1.")
    parser.add_argument("--forms", type=int, default=1, help="Number of questionnaires.")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--scene-ids", type=Path)
    parser.add_argument("--wave-id", default="")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.per_source < 1 or args.pool_per_source < args.per_source:
        raise SystemExit("--pool-per-source must be >= --per-source")
    if args.forms < 1:
        raise SystemExit("--forms must be >= 1")
    build(args)


if __name__ == "__main__":
    main()
