# Rung-1 Deterministic Coherence Protocol — FROZEN before corrupt evaluation

Status: FINAL regardless of outcome. Frozen before any corrupted clip is scored.
Any later change is logged as a protocol amendment and results computed before
the change are labelled with the protocol version that produced them.

## Corpus

`corpus_v2/` — 5 sources × 6 arms = 30 clips. All: 900 frames, 30.0 s, 30 fps,
640×480, h264/yuv420p, no audio. Verified in `results/gate0_corpus_v2.json`.
Planted boundaries per clip live in `corpus_v2/manifest.json` field `breaks_s`.

## Signals (all deterministic, OpenCV 5.0.0, grayscale 160×120)

| signal | definition | raw direction |
|---|---|---|
| `photo_jump` | mean abs pixel diff of consecutive frames | high = anomalous |
| `orb_inliers` | ORB(500) + BF Hamming knn ratio 0.75 + findHomography RANSAC inliers, consecutive frames | low = anomalous |
| `flow_reversal` | Farneback (pyr .5, lvl 3, win 15, it 3, poly_n 5, poly_sigma 1.2); median flow vector m[t]; 1 − cos(m[t], median(m[t−30..t−1])) | high = anomalous |
| `recurrence` | for lags L ∈ {60..300 step 30} frames: r_L[t] = −MSE(f[t], f[t−L]); z per lag, signal = max over lags | high = anomalous |
| `motion_scale` | median ‖flow‖ per transition; |m[t] − median(m[t−30..t−1])| / (1.4826·MAD(m[t−30..t−1]) + 1e−9) | high = anomalous |

## Normalization

Per clip, per signal: z[t] = ±(x[t] − median(x)) / max(1.4826·MAD(x), 1e−9).
Sign chosen so larger z = more anomalous.

## Thresholds — leave-one-source-out, no in-sample fitting

For each held-out source s: calibration = the 4 OTHER sources' clean clips.
τ = max over calibration clips of max_t z[t]. Threshold is frozen per fold
before the held-out clean or any corrupt clip is scored.

## Detection rule

Fired boundaries = contiguous runs of z[t] > τ; each run contributes one
detection at its argmax. A detection hits a planted break iff
|t_det − t_break| ≤ 0.5 s. Greedy one-to-one matching.

## Chance baseline (mandatory before reporting recall)

For a clip with K detections and B planted breaks over T=30 s with w=0.5 s
tolerance: P_chance = 1 − (1 − 2w/T)^(K·B) approximated per break and combined;
reported alongside precision so dense-fire recall is not mistaken for skill.

## Preassigned operator→check pairs (only these count for pass/fail)

- splice → orb_inliers (photo_jump support only)
- room_swap → orb_inliers (photo_jump support only)
- reverse_segment → flow_reversal
- loop → recurrence
- timewarp → motion_scale

All five signals are still computed on all clips, but non-preassigned
combinations are exploratory and CANNOT promote a result.

## Pass criterion (per pair, over 5 folds)

1. Held-out clean specificity 5/5 (clean clip never crosses τ).
2. Sensitivity ≥ 4/5 (corrupt clip fires a hit within tolerance).
3. Detection precision ≥ 0.5 AND hit rate > analytic chance baseline.
4. Margin: median z_max/τ ≥ 1.5 on detected corruptions.

## Inverse-operation controls (only for pairs that pass)

Repair: reverse→re-reverse interval; loop→drop duplicated span; timewarp→
invert speed map; splice/room_swap→restore clean base span (kept at 30 s via
cloned-tail pad, same encoder). Criterion: max z within ±2 s of each former
break falls below τ AND drops ≥ 50 % vs corrupt value; median |Δz| outside
±2 s reported as a sanity bound, not a gate.

## Reporting

`results/rung1_v1.json` — per-fold thresholds, raw z curves in
`results/signals/*.npz`, detections, margins, precision/recall, chance
baseline, inverse-control deltas. Labels: CALIBRATION for threshold fits,
VALIDATION-HELDOUT for the fold evals.
