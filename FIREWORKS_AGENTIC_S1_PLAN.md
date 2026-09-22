# Fireworks Agentic S1 v1 plan

## Objective

Build and validate an owned Jev-like evidence-selection model: an open-weight
base model plus a portable LoRA adapter, trained remotely on Fireworks and
servable outside Fireworks. The learned model selects the next valid physical
frame to inspect. Deterministic code continues to enforce action validity,
budget, stopping safety, and final verdict arbitration.

This is a scorer/harness fine-tune, not a new raw-video encoder and not direct
training of the final temporal-edit class.

## Frozen evidence

- Committed bounded-policy baseline: `0758ecad762c4a6e7ef1bc184df399be05e7ec40`.
- Upstream baseline review: `yunlong10/BVB#8` (draft).
- Current fixed replay: 95/98 detection, 94/98 typing, zero clean false
  positives, 437 cached judge calls, 21 abstentions.
- Local linear CE reference: 96/98 detection, 95/98 typing, zero clean false
  positives, 421 calls, 20 abstentions.
- Local linear Brier reference: same accuracy, 422 calls.
- Counterfactual benchmark: 437 states, 98 clips, 5 source videos, 3,481
  forced first-action interventions, zero replay/resume mismatches.
- Hard-label MVP eligibility: 133 source-observed states have one unique best
  action. Exclude 283 non-decision/uniform states and 21 partial-tie states
  from v1 hard-label training.

The four-clip reasoning-bank pilot and old `jev-s1-*` datasets are evaluation
history, not v1 training supervision.

## Chosen training path

- Platform: Fireworks Managed LoRA SFT.
- Candidate base: `accounts/fireworks/models/qwen3p6-35b-a3b`.
- Local machine role: export, validate, tokenize, evaluate, and orchestrate;
  no local 35B training or model-sized weight storage.
- Output: downloadable LoRA adapter with exact base-model and tokenizer pins.
- Loss: one answer-token cross-entropy for the first MVP.

Custom soft-target CE/Brier through the Training API is deferred until this
hard-label proof either succeeds or identifies a concrete tie-calibration gap.

## Delivery sequence and gates

1. **Preserve the boundary**
   - Work only in this isolated branch/worktree.
   - Keep unrelated dirty research files in the source worktree untouched.
   - Land the plan in a stacked draft PR before implementation commits.

2. **Import the minimum reviewed implementation surface**
   - Counterfactual benchmark schema/loader.
   - Model-visible state renderer and stable physical frame IDs.
   - Focused tests needed to prove target and split fidelity.
   - Do not copy cached videos or broad experiment-result directories.

3. **Export Agentic S1 v1**
   - Select only unique-best next-inspection states.
   - Split by source video before augmentation.
   - Emit deterministic candidate-order permutations only in the train split.
   - Emit Fireworks CHAT rows: observed state, valid options, one label token.
   - Never put expected operation or final ground truth in model input.
   - Write source hashes, prompt version, tie policy, tokenizer, and exporter
     revision into a manifest.

4. **Run the data-quality gate**
   - Every output label resolves to one offered physical frame.
   - Train/validation sources are disjoint.
   - No exact prompt has conflicting hard labels.
   - Candidate reordering preserves the target frame.
   - Labels are single tokens for the selected tokenizer.
   - Exact row and token counts are recorded.

5. **Produce a read-only Fireworks plan**
   - Inventory existing datasets, SFT jobs, and output models.
   - Use content-addressed IDs: `jev-act-v1-<sha8>`.
   - Refuse an existing mismatched dataset, in-flight equivalent job, or
     colliding output model.
   - Calculate price from exact token count and epochs.
   - Emit an immutable plan hash. This step must not upload or create a job.

6. **Paid approval gate**
   - Present exact train/validation row counts, token count, hyperparameters,
     official rate, estimated maximum spend, resource IDs, and rollback plan.
   - Require explicit user approval before dataset upload or job creation.

7. **One validation-fold training run**
   - One predeclared held-out source; no sweep and no parallel duplicate run.
   - Start with a small batch (8-16) and enough optimizer updates rather than
     reusing the old 128-sample batch uncritically.
   - Preserve Fireworks job, dataset, and output-model IDs in the run manifest.

8. **Evaluate before producing a deployment artifact**
   - Hard gate: zero clean false positives.
   - Preserve at least 95/98 detection and 94/98 typing.
   - Use fewer than 437 judge calls; stretch target is at most 421.
   - Reject invalid/repeated actions and candidate-order dependence.
   - Run the available fresh-video safety check separately from corpus replay.

9. **Only after a passing fold**
   - Train one all-data production adapter under a second explicit cost plan.
   - Download and verify the adapter.
   - Publish a private, revision-pinned Hugging Face repository.
   - Determine whether GMI can load the LoRA against its Qwen3.6 base; merge
     weights only if the platform cannot load the adapter directly.
   - Obtain a separate GMI deployment estimate and approval before creation.

10. **Reasoning-bank experiment**
    - Compare identical model weights with and without approved bank retrieval.
    - Do not train on unreviewed pilot transcripts.
    - Promote a bank lesson only when held-out behavior improves without a
      clean-video or cost regression.

## Duplicate-trigger protections

- Dataset ID includes the byte hash of the immutable exported JSONL.
- Job ID includes dataset hash, base model, LoRA rank, and epoch count.
- `plan` is read-only and returns a plan hash.
- `train` requires that exact plan hash and rechecks remote state immediately
  before mutation.
- A matching pending/running/completed job is reused or reported, never
  recreated automatically.
- Missing validation files are fatal; there is no silent auto-carveout.
- Only one process may launch a run for a given plan.

## Cost envelope before exact tokenization

The current Fireworks managed LoRA SFT rate for a 16.1B-80B model is
$3 per million training tokens. The old 40.8M-token static dataset would cost
about $122.40 for one epoch and trains the wrong contract, so it will not be
reused. The expected v1 export is roughly 0.4-0.7M tokens per epoch; four
epochs would therefore be roughly $4.80-$8.40. This is only a planning range.
No paid action is allowed until the exact tokenizer-derived estimate is shown.

## Current status

- Baseline PR open: yes.
- Isolated training branch: yes.
- Plan committed/pushed: pending this first commit.
- Dataset exporter: not implemented.
- Exact token/cost plan: not yet available.
- Fireworks upload/job: not started.
- Hugging Face publication: not started.
- GMI model/deployment: not started.

## Exact next action

Commit and push this plan, open the stacked draft PR, then implement and test
the exporter plus read-only duplicate-safe Fireworks planner. Stop again at
the exact paid-cost approval gate.
