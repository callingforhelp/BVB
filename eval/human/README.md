# BVB blind ranking

The paper analyzes responses from **15 raters**. Each rater ranks **5 anonymized
reconstructions** on **9 scenes** drawn from a **24-scene pool** (8 each from
ARKitScenes, ScanNet, and ScanNet++), with 3 scenes per source. Sixteen survey
forms were generated; the number of forms is not the number of returned raters.

The five configurations in the paper are GPT-5.6 Sol xhigh, Grok-4.5 high,
Gemini 3.1 Pro high, Claude Opus 4.6 high, and Qwen3.5-397B-A17B high. Astra
and Opus 5 were not in this study. See [PROTOCOL.md](PROTOCOL.md) for the
design. Commands below run from the repository root.

New surveys sample scenes at random within each source using a fixed seed and
do not read model scores. The sampling policy and seed are recorded in
`BLIND_MAP.json` and `assignments.json`. Existing study assignments follow the
saved packs; rerunning the original seed does not recreate them.

## Generate surveys

Requires FFmpeg plus complete camera renders and source videos for the selected
models. The generator uses the Python standard library. Write a new output
directory for each wave; add `--force` only when rebuilding an existing pack.

```bash
python eval/human/make_pack.py \
  --out eval/human/packs/wave2 \
  --forms 16 \
  --pool-per-source 8 \
  --per-source 3 \
  --seed 20260822
```

This writes `survey-r01.html` through `survey-r16.html`. Each file is a
self-contained HTML survey with embedded clips. Send one form per rater.
`assignments.json` records the draw. Keep `BLIND_MAP.json`,
`assignments.json`, and `_clips_cache/` with the experimenter.

Raters save and return JSON after ranking; `pack_id` identifies the form.
Store raw responses with the `BLIND_MAP.json` that generated them. Adding a
model requires a new survey.

## Score returned responses

Put received JSON files in that wave's `responses/` directory, then run:

```bash
python eval/human/score_human.py \
  --pack-dir eval/human/packs/wave2 \
  --responses eval/human/packs/wave2/responses \
  --out eval/human/packs/wave2/scored-dv-ls
```

Point at that wave's own directories for a completed study. The script does
not modify raw responses. Metric correlations also need `dual_vqa.jsonl` and
`vision_sim.jsonl` for the studied models under `sandbox/results/<run>/`.

| Output | Contents |
| --- | --- |
| `human_mean_rank.csv` | Mean rank (lower is better) and first-place rate |
| `human_pairwise.csv` | Pairwise win rates implied by the rankings |
| `human_by_scene_model.csv` | Per-scene, per-model mean human rank and automatic metrics |
| `human_metric_correlation.csv` | Scene–model Spearman correlation of automatic metrics with negated mean rank |
| `summary.json` | Rater count, scene coverage, means, correlations, and Overall definition |

**Overall uses only DV and LS:** `((sqrt(DV) + sqrt(LS)) / 2) ** 2`.
The script stores 0–1 scores and clips negative LS cosine values to zero
before the square root. Automatic-metric outputs contain only DV, LS, and
Overall. Overall is left blank when DV or LS is missing. DV is also blank for
a scene with no source-correct questions.

The script's correlations use **scene–model** pairs. The paper separately
compares the five configurations' global orderings; that Overall ρ = 1.00 is
not the same number as the scene–model correlations here. The paper's
scene–model LS correlation is ρ = 0.83. Write updated analyses to a new
output directory and keep the raw responses and historical results.
