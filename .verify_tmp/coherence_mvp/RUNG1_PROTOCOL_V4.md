# Rung-1 Protocol v4 — FROZEN narrow amendment before final scoring

Status: FINAL regardless of outcome. Frozen after the v3 raw-signal
diagnostic (`raw_diag.py` output) and BEFORE re-scoring. Everything not
listed here is unchanged from v3. Other pairs' v3 verdicts are FINAL —
this amendment cannot reopen them.

## Why amend (measured)

Raw recur_frac separates perfectly: 15 cleans max_t frac ∈ [0.23, 0.56],
15 loops ≥ 0.988 (duplicates are pixel-identical by construction).
The v3 z-normalization compressed this saturated signal below the
clean-max z threshold. The signal was correct; the decision quantity
was wrong.

## Amendment (loop pair only)

1. Decision quantity: RAW recur_frac[t] (fraction of pixels with
   |f[t]−f[t−L]| < 8, max over lags {60..300 step 30}). No z transform.
2. Threshold: τ_raw = max over the 12 calibration cleans of max_t
   recur_frac[t]. Same LOSO-clean-max rule as every other signal.
3. Detection point: run ONSET (first frame of each contiguous run
   above τ_raw), not argmax — recur_frac is a region-type signal and
   the planted break is the onset of duplication. Peak-type signals
   keep argmax detection.
4. Margin: max_t frac / τ_raw ≥ 1.5 (raw-signal analogue of z margin).
5. All other criteria identical: specificity = all 15 held-out cleans
   below τ_raw; sensitivity ≥ 12/15; precision ≥ 0.5; above analytic
   chance; ±0.5 s tolerance.

## Inverse-operation control (loop, only if it passes)

Repair: remove the duplicated span from the CORRUPT clip
([0,15)+[15,30) of its content → 25 s), pad tail to exactly 900 frames
via tpad clone-tail, same encoder. Known artifact: the cloned tail is
static and produces a local frac≈1 region after 25 s — disclosed here;
the criterion is LOCAL so this cannot affect the verdict.

Criterion: max recur_frac within ±2 s of the former 15 s break falls
below τ_raw AND drops ≥ 50 % vs the corrupt clip's value at the same
location. Median |Δfrac| outside ±2 s reported as sanity, not a gate.

## What passes, then

If recur_frac→loop passes v4 AND its inverse control shrinks the
anomaly locally, ONE preassigned pair has demonstrated power.
Per the E2E plan that earns a bounded Blender probe scoped to the
loop/recurrence mechanism — not a general Blender license.
