# RLCD/TLCD-inspired probability experiment — context for handoff

> The user referred to this as the “TLCD experiment” in the handoff request. The supplied research note expands **RLCD** as “Reinforcement Learning for Calibrated Decisions.” No pre-existing `TLCD_EXPERIMENT.md` was present in the source worktree, so this file records the experiment details supplied in the conversation and the implications for Flinter.

## Scope and status

This is an independent RLCD-inspired probability-learning experiment. It is **not** a recovered or official TypeSafe/Jev training recipe.

The experiment compares:

1. observed-label cross-entropy;
2. direct vector Brier loss;
3. a sampled proper-reward / policy-gradient estimator for a Brier-like objective;
4. correctness-only REINFORCE as a deliberately weaker control.

The experiment establishes implementation correctness and useful probability learning. It does **not** establish that sampled training is better than direct CE or direct Brier.

## Contract and notation

For visible input `x`, the desired conditional distribution is:

```text
q(. | x)
```

The model predicts:

```text
p_theta(. | x) = softmax(logits)
```

The candidate set is complete for each question and can vary by question. Padding candidates are excluded before normalization.

An observed outcome is sampled as:

```text
Y ~ q(. | x)
```

The sampled prediction candidates are independent draws with replacement:

```text
A_i ~ p_theta(. | x), i = 1..M, M >= 2
```

The estimator requires that `Y` be independent of the predictive draws conditional on `x`. The draws represent predictions for the same event; they are not physical actions that change the event outcome.

## Necessary conditions for the sampled estimator

The conditional expectation argument depends on all of the following:

- the candidate set is complete;
- predictive draws are independent, including no correlated self-pairs in the agreement penalty;
- the observed outcome is independent of the model draws conditional on input;
- all sample-dependent reward terms are retained;
- the detached baseline does not depend on the particular `A_i` being differentiated;
- zero baseline is valid, though higher-variance;
- the baseline is not differentiated through.

Rewarding only `1[A == Y]` is not enough: it is linear in `p` and encourages a winning class rather than recovering the full target distribution `q`.

The scalar sampled surrogate is not itself the reported Brier score, and its mean need not equal the direct Brier loss numerically.

## Direct controls

Observed-label cross-entropy:

```text
L_CE = -log p[Y]
```

Vector Brier loss:

```text
L_Brier = sum_k (p[k] - 1[Y=k])^2
```

Their population optima recover `q` under the standard proper-scoring argument. For a binary observed outcome:

```text
E[(p - Y)^2 | x] = (p - q)^2 + q(1-q)
```

The second term does not depend on `p`.

For the finite candidate sets used here, exact Brier is preferable to unnecessary sampling variance as a first control.

## Mathematical and CPU checks

Recorded in the supplied research note as `calibrated_objectives_check.json`:

- float64 enumeration;
- `M=2`, `K=2/3/5`;
- `M=3`, `K=2`;
- with and without conditional baseline;
- maximum exact gradient error: `1.39e-16`;
- fixed-seed Monte Carlo checks;
- Boolean scaling;
- candidate relabeling;
- invalid-input checks;
- padding isolation;
- per-question mean checks;
- largest Monte Carlo gradient discrepancy: `2.003` standard errors.

## CPU learning benchmark

Recorded in the supplied research note as `calibrated_learning_benchmark.json`.

Setup:

```text
shared candidate MLP
known non-degenerate q
K = 2/3/5
train/dev/test = 1536/768/2048
seeds = 17/18/19
same data per seed
same initialization and batch schedule within seed
600 Adam updates
batch size = 32
learning rate = 0.003
sampled arms: M = 32
```

Test known-q squared L2, mean ± sample SD:

```text
Initial random scorer       0.038371 ± 0.005178
Observed CE                 0.002581 ± 0.000516
Direct Brier                0.003279 ± 0.000530
Paired proper-reward PG    0.003377 ± 0.000908
Correctness-only REINFORCE 0.017131 ± 0.013953
```

All CPU arms selected the minimum observed dev NLL, including step zero, before one test evaluation. Correctness-only selected steps `50/0/50`; its final-step dev NLL rose to `3.146/2.814/3.162`. Early selection limits the damage visible in the test row.

Interpretation: the paired estimator can learn probability information and is mathematically implemented correctly, but it is not superior to direct controls and has more variance in this experiment.

## Qwen event experiment

All three trained arms start from the same existing NanoJev checkpoint:

```text
v3_teacher_coords_multi_seed17
Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca
```

This is not an untouched Qwen baseline. It is the existing NanoJev decision checkpoint before event training.

Training:

```text
seed = 17
100 full-model steps
effective batch = 16
BF16 forward
sequence limit = 8192 tokens
AdamW backbone/head rates = 2e-5 / 2e-4
weight decay = 0.01
gradient norm clipping = 1.0
```

The unbiased-estimator proof concerns the raw gradient, not the nonlinear optimizer, clipping, or finite-step result.

Environment and targets:

- frozen simulator inputs specify maze or Snake geometry;
- random actuator executes named move with reliability `rho`, otherwise chooses another offered move uniformly;
- target event is one-step collision-free movement;
- exact `q` follows from simulator transitions;
- independently seeded actuator draw supplies observed `Y`;
- only observed outcomes enter training losses;
- exact probabilities remain available for evaluation outside the model request.

Data split:

```text
train = 1124 questions
dev = 372
calibration = 368
test = 364
OOD = 128
regular sizes = 8/16/32
OOD size = 50
```

The manifest preserves source-group isolation. Questions from the same map or episode are not independent maps. The OOD slice is not a clean test of rich stochastic reasoning or long-horizon play: all 64 OOD Snake questions have `q(true)=1`, so deterministic cases are included.

Results, cells are `NLL / vector Brier / known-q squared L2`:

```text
Arm                    Test                         OOD
Initial NanoJev        0.639286 / 0.446492 / 0.277078  0.679861 / 0.486668 / 0.319949
Observed CE            0.460450 / 0.299032 / 0.124226  0.332381 / 0.229906 / 0.072497
Direct Brier           0.450525 / 0.303270 / 0.138445  0.316682 / 0.208351 / 0.067186
Paired proper-reward PG0.427911 / 0.278123 / 0.118444  0.329677 / 0.201326 / 0.062022
```

All trained checkpoints were selected at step 100 by minimum observed dev NLL among steps 50/100. One seed and a narrow event family cannot establish a general ranking. Paired PG has lower known-q L2 in this run; direct Brier has lower OOD NLL.

The summary preserves:

- `gold_probs_kind=programmatic_conditional_distribution`;
- `gold_label_kind=observed_outcome`;
- family-specific metrics;
- action targets outside event calibration;
- vector Brier, which is twice scalar Bernoulli Brier;
- Boolean ECE using ten fixed `p(true)` bins against finite-sample observed frequencies.

## Implications for Flinter

### Use direct controls first

For Flinter boundary/event judgments, begin with:

```text
CE + direct Brier
```

Use sampled proper-reward training only as a research comparison after the direct baseline works.

### Keep event prediction separate from action policy

Examples of event predictions:

```text
Was the object released?
Is the action complete?
Is there a boundary in this interval?
Is the proposed interval acceptable?
```

Examples of action policies:

```text
Which candidate should be inspected next?
Which interval should be selected?
Should the loop gather more evidence?
```

A distribution over preferred actions is not automatically a distribution of event-success probabilities.

### Preserve observed evidence and independent outcomes

A VLM/Jev teacher distribution is a teacher target, not automatically the true `q`. Keep teacher outputs separate from human/programmatic outcomes. Evaluate against held-out independent labels or known simulator truth where possible.

### Use known-q diagnostics where available

For synthetic audiovisual data, known offsets/boundaries can provide exact target distributions. For ambiguous real videos, reviewed outcomes provide observations of an underlying acceptance distribution, but not necessarily exact `q` for an individual item.

### Do not introduce RL prematurely

For a small finite candidate set and direct outcomes, sampling adds variance. In multi-step editing, actions change future states; that is a policy/planning problem, not automatically the same event-calibration problem. Use deterministic timeline simulation and local supervised/ranking losses first.

## Relationship to the current Flinter MVP

The current Flinter MVP is still a deterministic evidence-policy core. It does not yet train a model. The intended later experiment is:

```text
same candidate set + same evidence
  -> heuristic baseline
  -> Jev policy
  -> local NanoJev-style policy
  -> reviewed outcome
```

The RLCD/TLCD experiment informs the probability metrics and training comparison, but it should not become a dependency for the first replay integration.

## Provenance warning

No original `TLCD_EXPERIMENT.md` was found in the searched source directories. This document is a handoff reconstruction from the supplied research note. If the original file exists elsewhere, replace or append this document rather than treating this reconstruction as the canonical source.

## Full paths

Flinter handoff:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter/HANDOFF.md
```

This experiment note:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp_flinter/TLCD_EXPERIMENT.md
```

Original coherence worktree:

```text
/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp
```

Do not modify the original coherence worktree unless explicitly requested.
以上

Need update HANDOFF to reference TLCD and full context maybe current handoff already. Add section perhaps
