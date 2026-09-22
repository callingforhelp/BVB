# Agentic S1 training addendum — 2026-09-22

## Active checkout

- Worktree: `/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_agentic_s1`
- Branch: `codex/fireworks-agentic-s1-v1`
- Stacked draft PR: `https://github.com/callingforhelp/BVB/pull/1`
- Base: `flinter-evidence-policy` at plan start
- Plan commit: `6dfb241f540ed3a38c2ab823fecce718dcdda0f8`
- Tested implementation commit: `0ebbc35c706e297f1b74e6dc6bf7a11ede03e7f6`

Do not edit or clean the dirty sibling reference worktree at
`../coherence_mvp_flinter`.

## Current objective

Train an owned Jev-like next-evidence policy as a
Qwen3.6-35B-A3B LoRA on Fireworks. The model selects the next valid physical
frame to inspect from observed evidence. Code continues to own candidate
construction, budget, safety floors, stopping, abstention, and final verdict
arbitration.

This is scorer/harness training, not raw-video model training and not direct
final-class supervision.

## Completed in this branch

- Strict model-visible state renderer with fixed-width physical frame IDs.
- Candidate-order and whole-timeline translation invariance.
- Hard-label exporter for unique-best counterfactual actions.
- Source-disjoint held-out split and explicit shortcut baselines.
- Read-only exact tokenizer/cost planner with GET-only Fireworks inventory.
- Duplicate-safe content-addressed dataset, job, and output-model IDs.
- 48 passing tests with third-party pytest plugin autoload disabled.
- Frozen artifact package in `results/agentic_s1_v1_fold_09c/`.

## Important data-quality correction

The originally proposed `1ada7a0617` validation fold was invalid: a global
training-label-frequency lookup selected 29/30 held-out answers because the
synthetic corpus reuses absolute seam coordinates. That would let a model
appear competent without reading the state.

The final package instead holds out `09c1414f1b` and augments both splits with
candidate-order permutations and semantics-preserving timeline translations.
The strongest measured shortcut on the final 72-row validation set is 17/72
(23.61%); 64/72 target IDs are unseen in training.

## Frozen package and exact plan

- Training: 460 rows, 326,729 tokens,
  SHA-256 `eeabaaea9004c50a8932a849230dc6e91fe0017ee054f5c2b70e228ad565f4d4`.
- Validation: 72 rows, 50,085 tokens,
  SHA-256 `e03658748b1f9aced2cb8d0d1bf604874ad03fc5782aa68d739cbc37810858e9`.
- Every answer and offered candidate is exactly eight Qwen tokens; zero rows
  mix candidate token lengths.
- Base: `accounts/fireworks/models/qwen3p6-35b-a3b`.
- Managed LoRA SFT: rank 8, batch 16, two epochs, approximately 58 updates.
- Billable training tokens: 653,458.
- Published-rate point estimate: $1.96 (exact: $1.960374).
- Planning range: $1.37–$2.55.
- Train dataset ID: `jev-act-v1-eeabaaea`.
- Validation dataset ID: `jev-act-v1-e0365874`.
- Job/output model ID: `jev-agentic-s1-bbe6341d88b3`.
- Active approval hash:
  `f66147ebf62fb7242f02461abd12c26a5773e13c4139a7e2ae1b90f032b98ead`.

The replacement-account Fireworks preflight found the base READY and
supervised-LoRA tunable, an empty account inventory, and no proposed
resource-ID collisions.

## Hard boundary and exact next action

The account-specific training-only plan for `dave-z-d5jskgf9ohx3` was approved
with hash
`f66147ebf62fb7242f02461abd12c26a5773e13c4139a7e2ae1b90f032b98ead`.
The two content-addressed datasets are READY:

- `jev-act-v1-eeabaaea`: 460 training rows;
- `jev-act-v1-e0365874`: 72 validation rows.

Exactly one managed SFT job, `jev-agentic-s1-bbe6341d88b3`, was created at
2026-09-22T16:28:52Z and completed at 2026-09-22T16:49:52Z. Its verified state
is `JOB_STATE_COMPLETED`, status `OK`, and 100% progress. Fireworks reports a
training cost of `$1.954854012`, below the approved `$2.548486` ceiling.

The resulting output model is
`accounts/dave-z-d5jskgf9ohx3/models/jev-agentic-s1-bbe6341d88b3`.
The read-only serving/evaluation preflight is complete. No addon-compatible
shape was returned, so the prepared plan uses sequential live-merge
preemptible deployments: base first, tuned second, never concurrently. The
validated recommended shape is one BF16 B200; the conservative 60 GPU-minute
evaluation envelope is `$13.02` at the published `$0.217/minute` rate. The
preemptible account treatment remains an explicit unknown to verify in usage.

The exact plan is `FIREWORKS_AGENTIC_S1_EVAL_PLAN.md`. Do not provision a
deployment or run paid inference until that plan is separately approved.
Hugging Face publication, GMI, and all-data training remain separate later
approvals.

---

# Flinter evidence-policy MVP — handoff

Date: 2026-09-19
Worktree: `/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter`
Branch: `flinter-evidence-policy`

## Objective

Implement the bounded Flinter evidence-gathering architecture without changing the existing coherence detector:

```text
cheap signals / existing gates
  -> candidate union
  -> bounded evidence policy chooses what to inspect
  -> evidence floor and abstention in code
  -> existing deterministic arbitration
  -> append-only reviewed trace
```

The model/Jev policy must not own candidate construction, safety floors, final arbitration, or timeline mechanics.

## Completed commits

```text
778911c Add bounded Flinter evidence-gathering MVP
5abd8db Evaluate Flinter proposals on coherence corpus
```

## Files in this worktree

### MVP implementation

- `flinter_mvp/core.py`
  - `Rubric`: required observation kinds and confidence floor.
  - `EvidenceItem`: timestamped observation with provenance and frame references.
  - `CandidateInterval`: proposed interval with validation.
  - `EvidenceState`: current evidence, candidates, judged IDs, and budget.
  - `generate_candidates`: deterministic proposal helper for generic interval tasks.
  - `choose_next_request`: bounded evidence request / select / review policy.
  - `record_trace`: append-only JSONL trace writer.
- `flinter_mvp/__init__.py`: exports the public MVP API.
- `tests/test_flinter_mvp.py`: four unit tests.
- `FLINTER_MVP.md`: design notes and current corpus result.

### Existing-corpus evaluation adapters

- `evaluate_existing_proposals.py`
  - Loads the existing `coherence_mvp/jev_loop.py`.
  - Reuses the existing gated candidate union exactly.
  - Uses the manifest only after proposal generation to measure coverage.
- `evaluate_candidate_coverage.py`
  - Naive local-peak baseline using cached signal arrays.
  - Included as a negative control; it is not the recommended proposal stage.

### Saved result

- `results/existing_proposal_coverage.json`
- `results/flinter_candidate_coverage.json`

## Existing testing corpus and code locations

The source coherence MVP is here:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp
```

Important files/directories:

```text
coherence_mvp/jev_loop.py                 existing Jev loop
coherence_mvp/jev_score.py                scorer/evaluation helpers
coherence_mvp/jevlike_loop.py             local/policy variant, if used
coherence_mvp/corpus_v3/manifest.json     98-clip manifest and labels
coherence_mvp/corpus_v3/                  source corpus metadata/media paths
coherence_mvp/results/signals_v3/         cached cheap-signal .npz files
coherence_mvp/results/pairjudge/          cached pair-judge results
coherence_mvp/results/pairjudge_scenecut/ cached scene-cut judgments
coherence_mvp/results/pairjudge_wide/     cached wide-window judgments
coherence_mvp/results/pairjudge_bracket/  cached bracket judgments
coherence_mvp/results/jevloop_v9/         existing loop transcripts/results
coherence_mvp/HANDOFF.md                  previous coherence-MVP handoff
coherence_mvp/RUNBOOK.md                  existing runbook
```

The current corpus has 98 clips:

- 18 clean clips.
- 80 changed clips.
- Changed classes include `splice`, `reverse_segment`, `loop`, `room_swap`, and `timewarp`.
- The corpus is a temporal-edit detection corpus, not an action-boundary annotation corpus.

The existing `jev_loop.py` builds candidates from the union of cached gate outputs. It already contains important safety behavior such as seam review, evidence floors, and deterministic arbitration. Read it before changing integration behavior.

## Commands already run

Run the four MVP unit tests:

```bash
cd "/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter"
python3 -m pytest -q tests/test_flinter_mvp.py
```

Expected result:

```text
4 passed
```

Run the source-faithful candidate-coverage check:

```bash
cd "/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter"
/Users/oldap/s1-spike/.venv/bin/python evaluate_existing_proposals.py \
  --root "/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp" \
  --output results/existing_proposal_coverage.json
```

Observed result:

```text
98 clips evaluated
80 changed clips
78/80 changed clips covered within ±10 frames
coverage = 97.5%
```

Run the deliberately naive local-peak negative control:

```bash
cd "/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter"
python3 evaluate_candidate_coverage.py \
  --root "/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp" \
  --output results/flinter_candidate_coverage.json
```

Observed result:

```text
80 changed clips
6/80 fully covered
coverage = 7.5%
```

This negative control demonstrates that the mature existing candidate gates must be retained.

## Current implementation limits

1. The generic `flinter_mvp` policy is deterministic and not yet connected to the production `jev_loop.py`.
2. The existing-corpus evaluation measures proposal coverage only; it does not measure a new scorer or evidence policy’s detection accuracy.
3. No action-boundary labels exist in this corpus, so “complete action interval” has not yet been evaluated.
4. No new VLM/API calls were made by this worktree. Existing cached judgments were used.
5. The generic `generate_candidates` helper is intentionally not a replacement for the existing candidate union.
6. The current policy’s stopping semantics are a starting point and should be reconciled with the existing `jev_loop.py` evidence-floor and seam-review logic before production use.

## Recommended next steps

### 1. Build an adapter, not a replacement

Create an adapter in this worktree that converts the existing `jev_loop.py` state into the MVP state format:

```text
existing free_signals
existing candidate records
existing revealed verdicts
existing budget
existing rubric
    -> EvidenceState
```

Map existing cached candidate verdicts into `EvidenceItem` records with explicit provenance. Do not expose hidden manifest labels or ground truth to the policy.

### 2. Replay the same cached judgments

Implement a replay mode that compares:

- existing `jev_loop.py` policy;
- bounded evidence-policy adapter;
- deterministic baseline;
- optionally the existing S1/local policy.

Use the exact same candidates and cached pair-judge outcomes. Report separately:

- final corrupted/not-corrupted accuracy;
- break-type accuracy;
- clean false positives;
- abstention rate;
- judge calls used;
- candidate coverage;
- errors due to missing proposals versus bad evidence policy.

This must be a fair same-evidence comparison.

### 3. Preserve existing arbitration

Initially, let the new policy choose which cached candidate judgment to reveal, but keep the existing final arbitration and evidence floors unchanged. This isolates the value of information acquisition.

Only change arbitration in a separate experiment.

### 4. Add an action-boundary corpus

Create a small reviewed dataset for one action family, for example “place object on surface.” Each record should contain:

- source video ID and source group;
- rough action window;
- timestamped evidence and frame references;
- candidate intervals;
- rubric version;
- reviewer-selected interval or abstention;
- correction and reason;
- exact VLM/scorer outputs.

Keep proposal coverage separate from candidate ranking. A scorer cannot select a missing interval.

### 5. Add interchangeable scorer backends

Define a common interface:

```python
score(state, question, candidates) -> distribution + status
```

Backends should include:

- deterministic heuristic;
- existing Jev API;
- local S1/NanoJev-style policy later.

Compare them on identical evidence and candidates. Do not assume Jev adds value until it beats the strongest simple baseline or reduces reviewer effort/cost.

### 6. Evaluate OOD and cost

Use source-grouped splits. Report:

- held-out source/video performance;
- candidate recall;
- ranking quality;
- abstention precision;
- reviewer correction time;
- VLM calls;
- Jev calls;
- total cost;
- latency.

Do not treat agreement with Jev as correctness. Keep independent reviewed outcomes for the final test set.

## Design rules to preserve

- Evidence is timestamped and retains frame/window provenance.
- Do not place the answer label in the evidence state; that causes leakage.
- Candidate proposal coverage and candidate ranking are separate metrics.
- A probability over candidates is a relative-choice distribution, not automatically independent acceptability probabilities.
- Use CE and direct Brier before RLCD-inspired sampled objectives.
- Keep event prediction, action policy, and final edit success separate.
- Deterministic code owns constraints, safety, state transitions, and final arbitration.
- Abstain when the evidence floor is not met or multiple candidates remain plausible.
- Do not replace the existing candidate gates with naive signal peaks.
- Do not add Blender, a video world model, or full multi-clip planning until the bounded interval/evidence loop demonstrates value.

## Definition of done for the next integration milestone

The next milestone is complete when a replay command can run on the existing 98 clips and produce a side-by-side JSON report containing:

```text
same candidates
same cached judgments
existing policy result
new bounded-policy result
final arbitration result
judge-call count
abstention status
coverage/error decomposition
```

The milestone should include tests for:

- no candidates -> review;
- budget exhausted -> review;
- missing required observation -> gather;
- evidence floor met -> select;
- multiple valid candidates -> review;
- already-judged candidates are not requested again;
- hidden ground-truth fields are not present in policy state;
- identical cached evidence gives deterministic replay.

## Git state

Current branch:

```text
flinter-evidence-policy
```

Latest commit:

```text
5abd8db Evaluate Flinter proposals on coherence corpus
```

Before continuing, run:

```bash
git status --short --branch
git log --oneline -3
```

Do not modify the original `coherence_mvp` worktree unless explicitly requested. This worktree is the experimental implementation branch.


## Companion experiment document

The probability-learning background that informs the training/evaluation plan is documented separately here:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter/TLCD_EXPERIMENT.md
```

The user referred to this as the “TLCD experiment”; the supplied research note calls the method **RLCD** (“Reinforcement Learning for Calibrated Decisions”). No original file with that exact name was found in the source worktree, so `TLCD_EXPERIMENT.md` is a handoff reconstruction from the supplied note and should be replaced or amended if the canonical source is located.

### TLCD/RLCD context that the next agent must preserve

- It is an independent research candidate, not a recovered official TypeSafe/Jev implementation.
- The sampled proper-reward estimator is mathematically checked, but the experiments do not show it beats direct CE or direct Brier.
- Direct CE and exact direct Brier are the first controls for small finite candidate sets.
- The target distribution `q(.|x)` is the underlying conditional event distribution; an observed label `Y` is one sample from it.
- A teacher/Jev distribution is an imitation target, not automatically ground truth.
- Event prediction, preferred-action distribution, action policy, and final workflow success are different contracts.
- The sampled estimator assumes a complete candidate set, independent predictive draws, an outcome independent of the draws conditional on `x`, and a correctly detached baseline.
- In the Qwen event run, all arms started from the same existing NanoJev checkpoint; it was one seed and a narrow event family. The OOD slice included deterministic cases, so it does not prove broad OOD generalization.
- For Flinter, begin with CE + direct Brier on reviewed/known outcomes. Keep RLCD-inspired sampling as a later comparison, not a first implementation dependency.
- For multi-step editing, use deterministic timeline simulation and separate policy/planning evaluation before attempting policy-gradient training.

### Full handoff context for another agent

Read these documents in this order:

1. This file:
   `/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter/HANDOFF.md`
2. Probability-learning experiment:
   `/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter/TLCD_EXPERIMENT.md`
3. Existing detector handoff:
   `/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp/HANDOFF.md`
4. Existing runbook:
   `/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp/RUNBOOK.md`
5. Existing loop implementation:
   `/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp/jev_loop.py`

The requested next implementation is the replay adapter: reuse the existing candidate union and cached judgments, put the new bounded evidence policy above it, preserve current arbitration, and produce a side-by-side report of accuracy, abstention, candidate coverage, judge calls, and error decomposition.


## Final verification on Mac worktree

After disabling unrelated pytest plugin autoload, the focused suite passed:

```text
24 passed in 1.07s
```

The replay loader was made independent of the unavailable live `system_one_adapter` by stubbing `routes.cred`; cached replay does not call the live API. Final artifact:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter/results/replay_side_by_side_final.json
```

Final replay remains:

```text
existing: n=98 det=95/98 type=95/98 clean_fp=0 abstained=0 judge_calls=474
bounded:  n=98 det=95/98 type=94/98 clean_fp=0 abstained=21 judge_calls=437
coverage: 78/80 changed fully covered
fidelity mismatches: []
bounded deterministic: True
```

## 2026-09-19 late session: typing-parity diagnosis + fix

### Diagnosis of the 81/98 typing gap (pre-fix artifact)

room_swap was 3/16. All 16 room_swap clips carry their two seams (~f300
enter / ~f480 exit) in the candidate union with cached `scene_change`
verdicts; the failures were acquisition/stopping, not arbitration:

- 9x `second_seam_not_judged`: `choose_next_request` selects as soon as ONE
  candidate meets the floor (`len(complete)==1 and judged>=min(2,n)`). With
  `ndis==1` the deterministic verdict is `splice`; arbitration cannot remap
  to room_swap without a second revealed seam.
- 3x `multiple candidates meet the rubric` abstains: both seams WERE
  revealed (`discont=[299,479]`, arbitrated type room_swap) and the generic
  core treats two complete candidates as ambiguity. For this task >=2
  confirmed seams IS the room_swap signature, so the review discarded a
  correct answer.
- 1x cap abstain (`0d2ee665be__w2`, 12 candidates): six false predicted
  seams consumed the whole reveal budget before the first true seam.

The 25 abstentions were 18 clean clips (all cached verdicts continuous ->
floor never met -> review -> composite `none`; correct by design, zero FP)
+ 4 room_swap above + 3 splice cap-reviews that never found a seam (the
same information-limit class as the existing arm's misses).

Safety surface for gathering more reveals: only the 13 two-seam room_swap
clips have >=2 discontinuous verdicts in cache (splices <=1, loops <=1,
timewarps/cleans 0; reverses <=2 but all have echo>10 which locks the
verdict regardless of ndis). `dup_trailing_run>=50` exists only on the 3
static `09c1414f1b` swaps. So extra reveals can only change composites on
room_swap clips.

### Fix (flinter_mvp/jev_replay.py only; core.py generic semantics unchanged)

- `conclusion_determined(st)`: the composite is locked when
  `dup_swap_signature` holds, `ndis>=2`, or every candidate is judged.
- `run_bounded` gathers (seam-target order, unchanged) while the composite
  is undetermined; it consults `choose_next_request` only at terminal
  states, where floor-met (`floor_candidate_id`) maps any outcome to
  `select` -- this resolves the multi-complete review -- and
  floor-unmet stays `review`. Iteration cap with floor met now emits the
  best-effort type instead of abstaining.
- `MAX_ITERS` 8 -> 10 for the bounded arm: the deterministic acquisition is
  weaker than Jev's; the two cap-boundary room_swaps needed reveals 9-10.
  Aggregate spend stays below the recorded existing total (437 < 474).

Measured effect (same candidates, same cached verdicts, same arbitration):
type 81->94/98, room_swap 3->16/16, abstained 25->21, judge_calls 323->437,
det unchanged 95/98, clean_fp still 0, replay still deterministic, existing
arm fidelity mismatches still none.

### Residuals (bounded arm)

- `0d2ee665be__w2__reverse_segment` typed loop: recur_frac_max=0.97 wins the
  typing-guide precedence over echo=1065. echo-first would mistype the 9
  loop clips whose echo scores are astronomical, so this is an information-
  level signal conflict, not an acquisition gap.
- `13c3e046d7__w0__splice` seam_not_judged: 14 candidates, the lone
  discontinuous verdict sits at index 10 far from predicted seams; the
  seam-first walk does not reach it within 10 reveals (existing Jev found
  it with its learned ordering).
- `13c3e046d7__w2__splice`, `1ada7a0617__w0__splice` evidence_invisible:
  identical misses on the existing arm -- information-limit.

Alternating seam-prior/frame-order acquisition was simulated and rejected:
it fixed the two cap-boundary room_swaps inside budget 8 but pushed the
single discontinuous verdict past the cap on two splice clips (detection
regression to 93) and dropped a found seam on `1ada7a0617__w1__room_swap`.

Still true: do not claim the bounded arm as a replacement policy or begin
RLCD/TLCD reward experiments from this result alone.

### Validation

```text
26 passed (PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q)
```

Full 98-clip replay:

```text
existing: n=98 det=95/98 type=95/98 clean_fp=0 abstained=0 judge_calls=474
bounded:  n=98 det=95/98 type=94/98 clean_fp=0 abstained=21 judge_calls=437
coverage: 78/80 changed fully covered
fidelity mismatches: []
bounded deterministic: True
```

Per-class bounded result:

```text
loop:      det 16/16, type 16/16
none:      det 18/18, type 18/18
reverse:   det 17/17, type 16/17
room_swap: det 16/16, type 16/16
splice:    det 13/16, type 13/16
timewarp:  det 15/15, type 15/15
```

Artifacts: `results/replay_side_by_side_final.json` (canonical),
`results/replay_max10.json` (identical cap-10 probe).

## Canonical final handoff summary (2026-09-19 22:32 EDT)

Canonical worktree:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter
```

Branch: `flinter-evidence-policy`

Latest commit: `1accfba Improve bounded replay parity`

Parent worktree was not modified:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp
```

### Purpose

This is a bounded evidence-acquisition policy, not a complete video editor or a raw-video Jev model. Existing cheap candidate gates produce a fixed candidate union; the bounded policy chooses which candidate evidence to reveal; existing deterministic arbitration produces the final result. The experiment tests whether expensive VLM judge calls can be reduced without losing detection quality.

### Main files

- `flinter_mvp/jev_adapter.py` — parent-state to evidence-state adapter.
- `flinter_mvp/jev_replay.py` — bounded replay, seam-first selection, signal-aware stopping, arbitration, abstention.
- `replay_side_by_side.py` — 98-clip side-by-side replay.
- `tests/test_jev_replay.py` and `tests/test_flinter_mvp.py` — focused tests.
- `results/replay_max10.json` — latest full replay.
- `TLCD_EXPERIMENT.md` — RLCD/TLCD probability-learning context.

### Final measured results

Existing policy:

```text
n=98, detection=95/98, typing=95/98, clean FPs=0,
abstentions=0, judge calls=474
```

Bounded policy with a ten-reveal cap:

```text
n=98, detection=95/98, typing=94/98, clean FPs=0,
abstentions=21,
judge calls=437
```

Per-class bounded typing:

```text
loop 16/16; none 18/18; reverse 16/17;
room_swap 16/16; splice 13/16; timewarp 15/15
```

Additional checks:

```text
candidate coverage: 78/80 changed clips
replay fidelity mismatches: 0
bounded replay deterministic: true
focused tests: 26 passed
```

The ten-reveal change recovered room-swap typing from 3/16 to 16/16 while retaining detection parity and zero clean false positives. It reduces cached judge reveals by 37 (7.8%) relative to the existing policy. It is an improvement over the prior eight-reveal bounded result (81/98 typing, 323 calls), but not full policy parity because of one reverse miss and 21 abstentions.

### Next work

1. Freeze this deterministic baseline: 95/98 detection, 94/98 typing, 437 calls, zero clean FPs.
2. Diagnose the 21 abstentions and the single reverse typing miss; classify them as insufficient evidence, reveal-order, budget, or arbitration errors.
3. Compare against a simple fixed-order deterministic policy before adding a learned scorer.
4. If the contract is stable, train a local next-evidence/stop policy with CE and direct Brier first.
5. Keep TLCD/RLCD as a later controlled comparison; do not change reward functions yet.
6. Use source-grouped/OOD evaluation and independent reviewed outcomes. Do not measure success only by agreement with Jev.

### Interpretation boundary

The result supports cost-efficient evidence gathering on this structured corpus. It does not establish universal video understanding, broad OOD generalization, Jev superiority, or a need for RLCD/TLCD. Candidate coverage and evidence extraction remain separate failure modes from policy ranking.
