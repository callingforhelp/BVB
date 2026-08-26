# BVB blind ranking protocol

Instrument `bvb-human-rank-v1`. Nine scenes, three from each source
(ARKitScenes, ScanNet, ScanNet++). One ranking question per scene.

The rater sees the original VSI-Bench clip and n anonymized camera renders
(default: one representative per model family). They produce a total order
from most to least similar to the original. Scene Test remains the mechanical
ground truth. This instrument only asks which reconstruction looks closer.

## Why ranking, and why nine

A closed set of family representatives can be ranked in one sitting.
Nine items keep the session under half an hour at n=5 (54 short clips).
Pairwise human wins are read off the ranks (ties do not occur inside one
rater's permutation). Absolute 1–5 sheets are not used.

Adding a model later requires a new `survey.html`. Old ranking JSONs stay
with their own `BLIND_MAP.json`.

## Blinding and delivery

Build one self-contained `survey.html` with compressed videos embedded.
Send that file only. `BLIND_MAP.json` stays with the experimenter.
Raters open the HTML locally and return the downloaded ranking JSON.

## Sampling

Default `--per-source 3`. Within each source, prefer one easy / mid / hard
scene by mean Scene Test on the chosen runs, and require every model to have
a camera render.

## Analysis

`score_human.py` computes mean rank (1 is best), a pairwise win matrix, and
Spearman correlation of negated mean rank vs Dual VQA / Scene Test / latent
similarity when those jsonl files exist. Do not invent human numbers.
