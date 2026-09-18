# Rung-1 Protocol v3 — FROZEN corpus expansion before scoring

Status: FINAL regardless of outcome. Frozen after v2 evaluation
(`results/rung1_v2.json`, all pairs failed) and BEFORE corpus_v3 is scored.
Signals, thresholds, tolerance, detection rule, chance baseline, and
pass proportions UNCHANGED from v2. Only the corpus and fold arithmetic
change. v1/v2 results remain labelled `rung1_v1` / `rung1_v2`.

## Why amend (v2 diagnosis, measured)

Held-out clean specificity was 4/5 on every signal in both v1 and v2:
with τ = calibration-clean max and only 4 calibration clips, one
naturally-discontinuous clean clip poisons every fold. Signals respond
at true breaks (margins 1.7–1.9 on hits; duplicate span pixel-identical
MSE 1.1) — the measured blocker is calibration sample size, not signal.

## Corpus v3

- Each of the 5 sources contributes 3 DISJOINT 30 s windows:
  starts {0, (D−30)/2, D−30} — all sources ≥ 90 s so windows never overlap.
- 15 clean clips; every window receives all 5 operators → 75 corrupts.
- 90 clips total, same codec/fps/resolution/frame-count guarantees as v2.
- Splice semantics preserved (same-source temporal jump): second half is
  the first 15 s of the NEXT window of the same source, keeping the jump
  within-source at ≥ 15 s temporal distance and inside source duration.
- Other operators unchanged (all within-window). Break labels unchanged:
  splice [15], reverse [10,16], loop [15], room_swap [10,16], timewarp [20].

## Fold arithmetic (LOSO by source, unchanged rule)

- Calibration per fold = 12 clean clips (4 other sources × 3 windows).
- Held-out per fold = 3 cleans + 15 corrupts.
- τ = max over the 12 calibration cleans of max_t z[t].
- Pass criteria per preassigned pair (same proportions as v2):
  specificity = all 15 held-out cleans unflagged; sensitivity ≥ 12/15;
  precision ≥ 0.5; above chance; median margin ≥ 1.5.

## Signals and pairs (identical to v2)

splice→orb_inliers, room_swap→orb_inliers, reverse→flow_flip,
loop→recur_frac, timewarp→motion_step. photo_jump support only.
Normalization, detection, matching, chance baseline: unchanged.
