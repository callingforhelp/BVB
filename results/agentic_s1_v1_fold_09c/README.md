# Agentic S1 v1 held-out package

This directory is the immutable pre-training package for the first owned
Jev-like next-evidence scorer experiment.

## Provenance

- Source: `action_utility_benchmark.json`, 437 counterfactual states.
- Source SHA-256:
  `0490ce7f24cb2911c1a84b67ed1365b0f04eee7c9f5c45a9af42768a5660d05c`.
- Exporter commit:
  `0ebbc35c706e297f1b74e6dc6bf7a11ede03e7f6`.
- Prompt contract: `agentic-s1-frame-v3`.
- Tie policy: unique lexicographic best action only.
- Held-out source: `09c1414f1b`.

## Files

- `train.jsonl`: 460 Fireworks CHAT rows, 326,729 rendered Qwen tokens,
  SHA-256 `eeabaaea9004c50a8932a849230dc6e91fe0017ee054f5c2b70e228ad565f4d4`.
- `val.jsonl`: 72 rows, 50,085 rendered Qwen tokens,
  SHA-256 `e03658748b1f9aced2cb8d0d1bf604874ad03fc5782aa68d739cbc37810858e9`.
- `manifest.json`: split, source, augmentation, contract, and content hashes.
- `quality_report.json`: row-level audit and shortcut measurements.
- `fireworks_plan.json`: exact GET-only remote preflight, tokenizer audit,
  resource IDs, cost, and approval hash.

## Why the timeline is translated

Candidate ordering alone was insufficient. The first proposed holdout could be
solved 29/30 times by memorizing globally frequent absolute seam frames. Each
non-canonical variant therefore shifts every timeline coordinate by the same
deterministic offset while preserving relative relationships and the correct
physical choice.

On this package, the strongest measured train-label/position shortcut is
17/72 (23.61%); 64/72 validation targets are unseen shifted IDs.

## Mutation boundary

This directory was produced locally. Its Fireworks inventory fields came only
from authenticated GET requests. No dataset was uploaded, no training or
inference job was created, and no deployment was provisioned.

Any remote mutation must match approval hash
`0f2382d47de58b5b845eac6d6690195679311f5637f96b18a9ccc2eba8a4d688`
and requires explicit user approval.
