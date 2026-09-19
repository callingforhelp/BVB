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
