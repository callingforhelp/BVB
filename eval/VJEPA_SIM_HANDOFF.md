# V-JEPA Vision-Sim Handoff (temporary)

> Temporary notes for a cluster agent continuing this work.
> Official docs live in [`README.md`](README.md); this file is the short context dump.

## What changed / intent

BVB eval is now **dual-level**:

| Level | Metric | Artifacts |
|------|--------|-----------|
| **Code** | unit-test pass rate (unchanged) | `<run>/unit_tests.jsonl`, `<run>/summary.json` |
| **Vision** | V-JEPA paired cosine sim (new) | `<run>/vision_sim.jsonl`, `<run>/vision_sim_summary.json` |

**Do not break** Stage-1 agent runs under `sandbox/results/` or the code-level evaluator (`unit_test_metric.py`). Vision sim writes **separate** files only.

Old Video-QA retention (`videoqa_metric.py`) is **deprecated** — keep the file, do not treat it as the official vision metric.

## Pipeline (vision)

```
.blend (scene.camera)
  -> sparse EEVEE PNG frames across full timeline   [CPU / Blender]
  -> sample same #frames from original VSI-Bench mp4
  -> frozen V-JEPA 2.1 ViT-G encode both clips      [GPU]
  -> vision_sim = 0.5 * (layout_cosine + motion_cosine)
  -> write one score per scene; features never on disk
```

Default encoder: `apiantonio/vjepa2.1-vit-gigantic-384`  
Layout = temporally pooled spatial patch-map cosine; motion = global token-mean cosine.

## Files

| Path | Role |
|------|------|
| [`batch_render_camera.py`](batch_render_camera.py) | Mac-side batch Blender render → `camera_renders/` (no GPU) |
| [`vjepa_sim_metric.py`](vjepa_sim_metric.py) | Orchestrator (render → encode → score) |
| [`render_blend_video.py`](render_blend_video.py) | Blender camera render → PNG sequence (optional MP4 wrapper) |
| [`requirements-vjepa.txt`](requirements-vjepa.txt) | Cluster deps (torch / transformers / opencv) — **not** the sandbox Stage-1 venv |
| [`unit_test_metric.py`](unit_test_metric.py) | Code-level eval — leave alone unless asked |
| [`test.jsonl`](test.jsonl) | Scene → dataset mapping for locating `VSI-Bench/<dataset>/<scene>.mp4` |

## How to run on the cluster

Preferred workflow (Mac renders once → HF → cluster scores):

```bash
# --- on Mac (CPU / Blender only) ---
# Overnight helper (parallel workers + HF sync):
#   caffeinate -i bash eval/overnight_render_and_sync.sh
python eval/batch_render_camera.py --all-runs --resume --workers 6 --num-frames 64
# ~15s/scene @512px, ~160KB mp4/scene → camera_renders/<id>.mp4
python scripts/sync_eval_results.py --camera-renders-only

# --- on GPU cluster ---
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/download_results.py --run <run>
pip install -r eval/requirements-vjepa.txt
```

### GPU instance choice (Lambda-style)

| Goal | Rent | Why |
|------|------|-----|
| **Fastest wall-clock** | **8x A100 40GB** | Same $/GPU-hr as 1x; ~8× throughput via scene shards |
| Cheaper / smoke | 1x A100 40GB | Enough VRAM for ViT-G @ 64 frames |
| Avoid | 1x A10 24GB (OOM risk), V100 16GB (too small) |

Do **not** use 8 GPUs inside one process — launch **8 processes**, one per GPU:

```bash
RUN=sandbox/results/<run>
for i in $(seq 1 8); do
  CUDA_VISIBLE_DEVICES=$((i-1)) python eval/vjepa_sim_metric.py \
    --run "$RUN" \
    --vsi-bench VSI-Bench \
    --device cuda --dtype bfloat16 --num-frames 64 \
    --shard "$i/8" --resume &
done
wait
# Each shard writes vision_sim.shard-i-of-8.jsonl + vision_sim_summary.shard-i-of-8.json
# TODO: merge shards into vision_sim.jsonl / vision_sim_summary.json (or score one GPU per run)
```

Warm `camera_renders/<id>.mp4` ⇒ encoder skips Blender.

One-shot single-GPU (render+score on the same machine) still works:

```bash
pip install -r eval/requirements-vjepa.txt
cd eval
python vjepa_sim_metric.py --run ../sandbox/results/<run> --vsi-bench ../VSI-Bench --dry-run --limit 2
python vjepa_sim_metric.py --run ../sandbox/results/<run> --vsi-bench ../VSI-Bench --device cuda --dtype bfloat16
```

Useful flags: `--resume`, `--shard i/N`, `--limit`, `--sample`, `--scene-ids`.

Defaults that matter:

- `--num-frames 64` — uniform sparse samples across the **full** camera timeline
- Renders cached as `<run>/camera_renders/<scene_id>.mp4` and **reused**
- Features never written; scores in `vision_sim*.json`
- `--no-keep-renders` deletes local render cache after scoring (not recommended)

**VRAM note:** 64-frame ViT-G yields ~18k tokens/clip. Prefer **A100 40GB**. If OOM, try `--dtype float16`, `apiantonio/vjepa2.1-vit-giant-384`, or `--num-frames 32`.
## Hard constraints for the next agent

1. **Never overwrite** `unit_tests.jsonl` / `summary.json`.
2. **Never modify** existing `blends/*.blend` from agent runs.
3. Prefer fixing/extending `vjepa_sim_metric.py` + `render_blend_video.py` over inventing a second vision pipeline.
4. If changing the score definition, bump a note in `vision_sim_summary.json` (e.g. `metric_version`) so old numbers are not mixed silently.
5. ViT-G (~2B) needs a real GPU; CPU is only for `--dry-run` / path checks.

## Render cost (measured on Mac, Blender 5.1 EEVEE @ 512px)

Sampled 19 blends from a 288-scene run (`mini-harness-claude-opus-4-6-run01`):

- Timeline length in `.blend` files is often **huge**: median ~**1644** frames, max ~**4360** (do **not** render the full animation).
- Sparse render ≈ **0.2 s/frame** after startup.
- **64 frames** (current default): ~**12–15 s/scene**, ~**12 MB** PNG/scene → ~**3.5 GB**/288-scene run.
- 16 frames: ~3–4 s/scene, ~3 MB/scene (legacy smoke setting).

### Recommendation

**Pre-render on Mac → sync `camera_renders/` to `yunlong10/BVB-results` → cluster only runs the encoder.** Disk is cheap; Blender time is the bottleneck if you re-render on every metric tweak.

```bash
python eval/batch_render_camera.py --run sandbox/results/<run> --resume
python scripts/sync_eval_results.py --run <run> --camera-renders-only
```

## Open follow-ups (nice, not blocking)

- Merge helper for `vision_sim.shard-*-of-N.jsonl` → `vision_sim.jsonl` + summary (8-GPU path)
- Batch shell for multi-run × 8-GPU encode (like `sandbox/run_eval_batch.sh`)
- Plot `vision_sim` into `scripts/plot_eval_results.py`
- Paper section `misc/sections/2_benchmark.tex` → Vision-Level Metrics still a stub
- If ViT-G @ 64 frames OOMs, document the fallback model / frame count in the summary

## Quick sanity checks

```bash
# Paths only
python eval/vjepa_sim_metric.py --run sandbox/results/<run> --vsi-bench VSI-Bench --dry-run --limit 3

# Blender render only
python eval/render_blend_video.py \
  --blend sandbox/results/<run>/blends/<scene>.blend \
  --output /tmp/<scene>.mp4 \
  --num-samples 16
```
