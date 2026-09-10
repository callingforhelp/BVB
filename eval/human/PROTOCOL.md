# BVB blind ranking protocol

Instrument: `bvb-human-rank-v1`. The reported study contains 15 raters. Each
rater ranks five reconstructions for nine scenes: three each from ARKitScenes,
ScanNet, and ScanNet++. The shared pool contains 24 scenes, eight per source.
Sixteen survey forms were prepared; the paper analyzes 15 returned responses.

## Task and models

The rater sees a source clip and five anonymized camera renders, then orders
the reconstructions from most to least similar to the source. The study
compares GPT-5.6 Sol xhigh, Grok-4.5 high, Gemini 3.1 Pro high, Claude Opus 4.6
high, and Qwen3.5-397B-A17B high. Astra and Opus 5 were not part of this study.

Rankings assess visual fidelity to the source. Human judgments validate the
automatic metrics; Scene Test is not a ground-truth human preference or an
axis in the current Overall score. Pairwise wins are derived from each total
order; the instrument does not collect ties or absolute 1–5 ratings.

## Blinding and delivery

The generator creates self-contained HTML files with compressed clips
embedded. Send only the assigned survey HTML to each rater. Keep the
`BLIND_MAP.json`, `assignments.json`, and clip cache with the experimenter.
Raters open the HTML locally and return the downloaded JSON.

Model codes and scene assignments are recorded in the pack. Adding a model
requires a new survey; preserve every response with its original blind map.
See [README.md](README.md) for generation and analysis commands.

## Sampling

Multi-form mode uses `--forms 16 --pool-per-source 8 --per-source 3` for the
24-scene pool and nine scenes per rater. Every sampled scene must have a camera
render from all five models. `assignments.json` records the actual allocation.

The existing generator uses mean legacy Scene Test scores to stratify scenes
within a source. This is a historical sampling choice, not the current metric
definition. Reproducing the collected study requires its saved assignments
and blind map. Single-form mode without `--forms` instead builds one nine-scene
survey and does not reproduce the multi-form assignment design.

## Analysis

`score_human.py` writes mean rank (1 is best), pairwise wins, and per-scene,
per-model mean ranks. It computes Spearman correlation between negated human
mean rank and DV, LS, and their two-axis Overall when those scores exist.
Scene Test may appear as a separate historical diagnostic.

Automatic values in these files are on a 0–1 scale. Overall is
`((sqrt(DV) + sqrt(max(0, LS))) / 2) ** 2`; it does not require Scene Test.
Missing automatic metrics remain undefined, including DV for a scene with no
source-correct questions. Correlations use only pairs with an available score.

Distinguish two analysis units when quoting the paper:

- **Scene–model pairs:** LS correlates with negative mean human rank at
  Spearman ρ = 0.83.
- **Five configurations:** the ordering by benchmark Overall agrees with the
  ordering by human mean rank at ρ = 1.00.

The scoring script reports the former unit. It does not derive the latter
by averaging its per-scene Overall scores: benchmark Overall combines pooled
DV and aggregate LS. Report the actual sample sizes and preserve raw responses.
