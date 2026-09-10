# BVB Evaluation

The current BVB protocol evaluates reconstructions along **two axes**:

| Axis | Measurement | Output summary |
| --- | --- | --- |
| **Dual VQA (DV)** | Retention of answers the judge gets right on the source | `dual_vqa_summary.json` |
| **Latent Similarity (LS)** | Mean layout and motion similarity from frozen V-JEPA 2.1 features | `vision_sim_summary.json` |

**Overall** is `((sqrt(DV) + sqrt(LS)) / 2) ** 2`, with both scores on a
0–100 scale. Both axes evaluate the rendered reconstruction against the source
video, independently of the reconstruction agent.

All commands below start from the **repository root**. The paper evaluates
288 scenes and 5,130 questions from [`test.jsonl`](test.jsonl), using the same
reconstruction sandbox and a $3 Stage-1 spend ceiling per scene. Evaluation
does not send scores back to the reconstruction agent.

## 1. Render the animated scene camera

Requires host Blender and FFmpeg. Use a Python 3.10+ environment; the
[Stage-1 environment](../sandbox/README.md) can also run this rendering step.

```bash
python eval/batch_render_camera.py \
  --run sandbox/results/run_001 \
  --num-frames 64 \
  --resume
```

Set `BLENDER_BIN` or pass `--blender /path/to/blender` if necessary. The renderer
samples the full camera timeline with EEVEE and caches
`<run>/camera_renders/<scene>.mp4`. `--keep-pngs` retains intermediate frames
for debugging; the batch renderer otherwise removes them after making the MP4.
No judge API or GPU encoder is called in this step.

## 2. Score Dual VQA

Install the judge dependencies in an evaluation environment:

```bash
python -m pip install openai opencv-python-headless
export OPENAI_API_KEY="..."
```

The manuscript uses `gpt-5.4-mini` and 16 uniformly sampled frames per video.
Source answers are shared across runs under
`sandbox/results/_dual_vqa_shared/original_answers.jsonl`.

```bash
python eval/dual_vqa_metric.py \
  --ensure-originals-only \
  --model gpt-5.4-mini \
  --n-frames 16

python eval/dual_vqa_metric.py \
  --run sandbox/results/run_001 \
  --model gpt-5.4-mini \
  --n-frames 16
```

These commands call the judge API for uncached answers. Resume is enabled by
default. Keep the judge and sampling settings fixed when reusing shared banks;
use a separate results directory for a different judging protocol.

Outputs inside each run:

- `dual_vqa.jsonl`: answers, source-correct flags, and retention for each question.
- `dual_vqa_summary.json`: retention, accuracy, per-task values, and coverage.

DV is `|C_source ∩ C_render| / |C_source|`, reported as a percentage. Its
denominator is the set of **source-correct questions**, not all questions and
not the number of scenes. A scene with no source-correct questions has
undefined per-scene DV; it should not be labeled as a zero-retention example.
The reported overall DV is pooled over questions rather than an unweighted
mean of scene percentages.

`--text-only` optionally builds a chance-floor bank; the current leaderboard
uses retention directly. The deprecated offline prediction aggregator has
been retired to Git history and local archives.
`import_dual_vqa_from_pilot.py` is only for migrating existing pilot logs and
is not a required setup step for a fresh checkout.

## 3. Score Latent Similarity

Use a dedicated GPU environment, separate from the Stage-1 agent environment:

```bash
python -m pip install -r eval/requirements-vjepa.txt
python eval/vjepa_sim_metric.py \
  --run sandbox/results/run_001 \
  --vsi-bench VSI-Bench \
  --model apiantonio/vjepa2.1-vit-gigantic-384 \
  --num-frames 64 \
  --device cuda \
  --dtype bfloat16 \
  --resume
```

The frozen V-JEPA 2.1 ViT-G encoder compares source and rendered clips:

- **Layout:** cosine similarity of temporally pooled spatial patch maps.
- **Motion:** cosine similarity of global token-mean features.
- **LS:** the mean of Layout and Motion.

Outputs inside each run:

- `vision_sim.jsonl`: one score or failure record per selected scene.
- `vision_sim_summary.json`: `vision_sim`, `layout_sim`, and `motion_sim` over
  all selected scenes, filling failed rows with zero. `scored_only` is a
  separate diagnostic over successful rows and is not the benchmark mean.

Raw summary scores use a 0–1 scale; multiply by 100 for tables. Cached MP4s
avoid another Blender render. Render caching is enabled by default;
`--no-keep-renders` disables cache reuse and saving, while existing MP4s stay
on disk. Newly rendered PNGs are temporary. Features are never written to
disk. Use `--limit`, `--sample`, or `--scene-ids` only for explicitly labeled
subsets.

See [the render/cluster guide](VJEPA_SIM_HANDOFF.md) for moving camera renders,
running independent GPU shards, and merging their results.

## 4. Check coverage and compute Overall

Use the same full test pool for every configuration. The paper assigns zero
to failed reconstructions on both axes; dropping failures changes the task.
Record coverage alongside scores, and resolve evaluation infrastructure errors
before reporting benchmark results.

The current DV runner discovers questions from existing camera renders, and
its raw summary aggregates successfully scored question records. A partial
or failed run can therefore have a smaller denominator. **Do not treat that
partial-run summary as a complete 288-scene result.** Check it against
`test.jsonl` and the complete source-answer bank. Likewise, an LS summary's
`num_scenes` must cover the intended pool, including failure records.

Once both summaries cover the intended evaluation pool, compute the aggregate
without further model calls:

```python
import json
import math
from pathlib import Path

run = Path("sandbox/results/run_001")
dv = 100 * json.loads((run / "dual_vqa_summary.json").read_text())["retention_rate"]
ls = 100 * json.loads((run / "vision_sim_summary.json").read_text())["vision_sim"]
overall = ((math.sqrt(dv) + math.sqrt(max(0.0, ls))) / 2) ** 2
print({"Dual VQA": dv, "Latent Similarity": ls, "Overall": overall})
```

The square-root mean is applied to the two configuration-level scores, not
averaged over per-scene Overall scores. It is neither the geometric mean nor
the arithmetic mean. Negative cosine similarity is clipped to zero before
taking its square root, as defined in the paper.

The [results CSV](../assets/bvb-results.csv) preserves all 46 manuscript
configurations and their full-precision values. The
[project page](https://yoloytang.me/BVB/#leaderboard) adds interactive ranking
and cost comparisons.

## Blind human ranking

The paper reports 15 raters, each ranking five anonymized reconstructions on
nine scenes. LS correlates with preference at the scene-model level
(Spearman ρ = 0.83). Overall matches the ordering of the five studied
configurations (ρ = 1.00); Astra was not included in that study.
See [the human-study guide](human/README.md) and [protocol](human/PROTOCOL.md).
