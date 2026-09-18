# Rung-1 Protocol v2 — FROZEN amendments before re-scoring

Status: FINAL regardless of outcome. Frozen after v1 evaluation
(`results/rung1_v1.json`, all pairs failed) and BEFORE any v2 signal is
scored. v1 results remain labelled `rung1_v1`. Corpus, thresholds,
tolerance, detection rule, chance baseline, pass criteria, and
inverse-control criteria are UNCHANGED from RUNG1_PROTOCOL.md.

## Why amend (v1 diagnosis, measured)

1. `recurrence`: duplicated span is pixel-identical (median MSE 1.1 vs
   6643 control) but −MSE robust-z compresses near-zero MSE to z≈2.5.
2. `motion_scale`: MAD-normalised window quotient is degenerate —
   near-zero MAD on smooth clean segments produces τ=226.
3. `flow_reversal`: trailing-median cosine responds asymmetrically
   (exit seam z=9.16, entry z=−0.43) and τ inflated by one clean outlier.

## Amended signal definitions (v2)

| signal | v2 definition | direction |
|---|---|---|
| `recur_frac` | for lags L ∈ {60..300 step 30}: frac_L[t] = fraction of pixels with |f[t]−f[t−L]| < 8; signal = max over lags | high |
| `motion_step` | symmetric windowed median magnitude step: |median(mags[t..t+30]) − median(mags[t−30..t])| in px/frame | high |
| `flow_flip` | symmetric windowed direction flip: 1 − cos(median(m[t..t+30]), median(m[t−30..t])) with per-window median flow vector | high |

`orb_inliers` and `photo_jump` UNCHANGED. Robust-z normalisation,
τ = calibration-clean max (LOSO), 0.5 s tolerance, run-based detection,
chance baseline, pass criteria, and inverse-control criteria unchanged.

## Amended preassigned pairs

- splice → orb_inliers (unchanged)
- room_swap → orb_inliers (unchanged)
- reverse_segment → **flow_flip** (replaces flow_reversal)
- loop → **recur_frac** (replaces recurrence)
- timewarp → **motion_step** (replaces motion_scale)

## Notes on expected-but-unfixed limits

v1 showed splice seams (same-source jump cut) are the argmax of
orb_inliers on all 5 clips but sit below the clean maximum. That is a
measured property of this corruption class; v2 keeps the signal and
threshold rule so the result stays honest — a same-room jump cut that
cannot be thresholded above natural discontinuities is reported as a
detection limit, not repaired post hoc.
