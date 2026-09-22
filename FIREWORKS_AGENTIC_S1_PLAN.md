# Fireworks Agentic S1 v1 — final pre-training plan

Date: 2026-09-22

## Decision

Build an owned Jev-like next-evidence scorer as a Qwen3.6-35B-A3B LoRA,
fine-tuned with Fireworks Managed SFT. The model receives only the observed
evidence state and currently valid physical frame choices. It scores which
frame should be inspected next. Deterministic code continues to own candidate
construction, action validity, reveal limits, safety floors, stopping,
abstention, and final temporal-edit arbitration.

This is the minimum viable proof for the learned harness. It is not a raw-video
encoder, not a direct final-class predictor, and not a claim that the model is
better until held-out replay passes.

## Existing control

Freeze the committed bounded-policy baseline at
`0758ecad762c4a6e7ef1bc184df399be05e7ec40` while evaluating this scorer:

- 95/98 detection;
- 94/98 typing;
- zero clean false positives;
- 437 cached judge calls;
- 21 abstentions.

The local linear controls remain:

- CE: 96/98 detection, 95/98 typing, zero clean false positives, 421 calls;
- Brier: the same accuracy, 422 calls.

The learned model must beat the relevant cost-adjusted control, not merely
produce plausible explanations.

## What was implemented

- `flinter_mvp/frame_prompt.py`
  - Strict allowlisted model-visible state.
  - Stable fixed-width physical IDs such as `frame_000474`.
  - Deterministic candidate-order permutations.
  - Semantics-preserving timeline-coordinate translations.
  - Recursive rejection of hidden labels or final-class leakage.
- `agentic_s1_export.py`
  - Selects only states with one unique lexicographic best action.
  - Splits by source video before augmentation.
  - Emits deterministic Fireworks CHAT JSONL.
  - Measures train-label-frequency and position shortcuts.
  - Rejects empty validation, conflicting labels, or excessive shortcut
    accuracy.
- `agentic_s1_plan.py`
  - Read-only exact Qwen tokenizer audit and cost calculation.
  - GET-only Fireworks inventory and collision checks.
  - Content-addressed resource IDs and immutable approval hash.
  - No upload or resource-creation capability.
- `tests/test_agentic_s1_export.py` and `tests/test_agentic_s1_plan.py`
  - Cover leakage, coordinate shifts, permutation invariance, split fidelity,
    shortcut rejection, hashes, exact token accounting, and GET-only planning.

## Data-quality finding and correction

The first proposed holdout, source `1ada7a0617`, was invalid for this answer
contract: a training-label-frequency lookup selected 29 of 30 held-out targets
without understanding the evidence. Several synthetic source videos reuse the
same absolute seam coordinates, so unshifted physical IDs create a shortcut.

The correction is two-part:

1. use source `09c1414f1b` as the held-out fold; and
2. apply both candidate-order permutations and deterministic whole-timeline
   translations to training and validation variants.

The translated rows preserve every relative temporal relationship and the
correct physical choice, but prevent memorizing globally frequent frame IDs.
The held-out source is also deliberately difficult: it contains the
corpus-exclusive static room-swap examples.

Final shortcut measurements on 72 validation rows:

| Shortcut | Accuracy |
|---|---:|
| Random expected choice | 10.84% |
| Train-label-frequency prior | 9.72% (7/72) |
| Best fixed candidate position | 15.28% (11/72) |
| Frequency then first position | 23.61% (17/72) |

Sixty-four of 72 validation targets are unseen shifted labels. The strongest
measured shortcut is 23.61%, below the predeclared 50% rejection threshold.

## Frozen dataset package

Artifact directory: `results/agentic_s1_v1_fold_09c/`

| Item | Value |
|---|---|
| Source benchmark | 437 states; SHA-256 `0490ce7f24cb2911c1a84b67ed1365b0f04eee7c9f5c45a9af42768a5660d05c` |
| Eligible unique-best states | 133 |
| Excluded | 270 uniform, 13 non-decision, 21 partial-tie |
| Training split | 115 states, 4 sources, 460 augmented rows |
| Validation split | 18 states, held source `09c1414f1b`, 72 augmented rows |
| Training dataset | `jev-act-v1-eeabaaea` |
| Training SHA-256 | `eeabaaea9004c50a8932a849230dc6e91fe0017ee054f5c2b70e228ad565f4d4` |
| Validation dataset | `jev-act-v1-e0365874` |
| Validation SHA-256 | `e03658748b1f9aced2cb8d0d1bf604874ad03fc5782aa68d739cbc37810858e9` |
| Prompt contract | `agentic-s1-frame-v3` |
| Answer contract | stable physical frame ID |
| Serving score contract | sequence log-probability over complete candidate strings |

The source benchmark itself is not copied into this branch. Its hash anchors
the export to the reviewed counterfactual benchmark in the sibling reference
worktree.

## Exact token audit

Tokenizer: `Qwen/Qwen3.6-35B-A3B`, pinned revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`.

| Split | Rows | Rendered tokens | Offered candidates | Answer/candidate length |
|---|---:|---:|---:|---:|
| Train | 460 | 326,729 | 4,276 | exactly 8 tokens each |
| Validation | 72 | 50,085 | 720 | exactly 8 tokens each |

No row mixes candidate sequence lengths. Therefore, raw complete-sequence
log-probability comparison does not contain a candidate-length preference.
The longest rendered row is 1,037 tokens, far below the base model's reported
262,144-token context.

## Exact approved Fireworks plan

| Field | Value |
|---|---|
| Account | `zhangye1987-q56dy8y3` |
| Workflow | Fireworks Managed LoRA SFT through the existing REST driver |
| Base model | `accounts/fireworks/models/qwen3p6-35b-a3b` |
| Live model state | READY; supervised LoRA tunable; LoRA supported |
| Purpose field | omitted; platform default |
| Epochs | 2 |
| LoRA rank | 8 |
| Batch size | 16 samples |
| Estimated optimizer steps | 58 |
| Learning rate | model-selected platform default |
| Scheduler | constant after warmup, API default |
| Warmup / optimizer / weight decay | platform-resolved, unknown before create |
| Maximum context | platform default; observed maximum is 1,037 tokens |
| Evaluation | explicit validation dataset; auto-carveout disabled |
| W&B | disabled |
| Job ID | `jev-agentic-s1-bbe6341d88b3` |
| Output model ID | `jev-agentic-s1-bbe6341d88b3` |
| Approval hash | `2ce848f686534b8140b64ddee39419f731d97ed43aedcca8cee493c83052cc98` |

The read-only inventory found 11 existing datasets, zero SFT jobs, and zero
account output models. None of the proposed dataset, job, or output-model IDs
collide with an existing resource.

Rank 8 follows the current managed-SFT default. Two epochs are a deliberate
minimal departure from the one-epoch default because 460 rows at batch 16
would otherwise provide only about 29 optimizer steps. This plan provides
about 58 updates without the overfit and spend risk of the earlier rank-32,
four-epoch draft.

## Cost

Fireworks currently lists Managed LoRA SFT for 16.1B–80B models at $3.00 per
million training tokens. This plan bills 653,458 training tokens:

- point estimate: **$1.96** (exact computed value: $1.960374);
- planning range: **$1.37–$2.55**;
- rate certainty: published;
- usage certainty: exact dataset-derived rendered tokens;
- pricing checked: `2026-09-22T14:43:56Z`;
- excluded: paid inference evaluation, deployment uptime, Hugging Face storage,
  and GMI serving.

The range is planning protection, not a quote. The immutable JSON plan is
`results/agentic_s1_v1_fold_09c/fireworks_plan.json`.

## Training and evaluation gates

After explicit approval of the exact approval hash, the training-only stage is:

1. re-run GET-only collision checks;
2. upload the two content-addressed datasets once;
3. create exactly one managed SFT job with the recorded ID;
4. monitor numeric progress and reconcile any `AlreadyExists` response rather
   than launching a duplicate.

After successful training, prepare a separate serving/evaluation cost plan and
obtain separate approval before provisioning a dedicated deployment or running
paid inference. The behavioral evaluation stage will then:

5. score the tuned model on the unchanged held-out replay;
6. require zero clean false positives, at least 95/98 detection, at least
   94/98 typing, fewer than 437 judge calls, valid actions, and candidate-order
   invariance;
7. run the fresh-video safety check separately;
8. reject the adapter if it fails those gates.

The 421-call CE result is the stretch target. Validation NLL alone is not the
acceptance criterion.

## What follows only after a passing fold

- Prepare a separately costed all-data adapter run.
- Export and verify the LoRA against the pinned Qwen base.
- Publish to a private revision-pinned Hugging Face repository.
- Verify GMI/SGLang can score complete candidate strings from the adapter.
- Obtain separate serving cost and deployment approval before creating GMI
  capacity.
- Compare identical weights with and without reviewed reasoning-bank retrieval.
- Consider soft-target CE/Brier or RLCD/RFT only if the hard-label MVP exposes
  a specific calibration or tie-handling failure.

## Current boundary

Completed locally:

- plan-first branch and draft PR;
- exporter, strict renderer, read-only planner, and tests;
- immutable train/validation package;
- exact token audit, live GET-only inventory, and exact cost plan;
- approval-locked remote executor and duplicate-safe reconciliation;
- two content-addressed Fireworks datasets uploaded and READY;
- explicit recovery for a confirmed empty `UPLOADING` dataset shell;
- one approval-locked managed SFT job created and completed on account
  `dave-z-d5jskgf9ohx3`;
- 51 passing tests.

Not performed:

- no paid inference or serving deployment;
- no Hugging Face publication;
- no GMI model or deployment;
- no claim of tuned-model accuracy.

The active account-specific plan was approved with hash
`f66147ebf62fb7242f02461abd12c26a5773e13c4139a7e2ae1b90f032b98ead`.
Its immutable plan is
`results/agentic_s1_v1_fold_09c/fireworks_plan_dave_account.json`. The existing
job `jev-agentic-s1-bbe6341d88b3` completed at 100% with status `OK`; Fireworks
reported `$1.954854012` training cost against the approved `$2.548486` ceiling.
The resulting output model is
`accounts/dave-z-d5jskgf9ohx3/models/jev-agentic-s1-bbe6341d88b3`.
The next action is a separate serving and held-out behavioral-evaluation cost
plan. Do not create a deployment or run paid inference before approval.
