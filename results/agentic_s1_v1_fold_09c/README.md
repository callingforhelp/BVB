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

The active account-specific training plan was approved with hash
`f66147ebf62fb7242f02461abd12c26a5773e13c4139a7e2ae1b90f032b98ead`.
Both content-addressed datasets are READY on account `dave-z-d5jskgf9ohx3`.
Exactly one managed SFT job, `jev-agentic-s1-bbe6341d88b3`, was created and is
running. Its approved training ceiling is $2.548486 for 653,458 tokens.

Monitor the existing job; do not create a second one. Paid inference, serving,
Hugging Face publication, GMI deployment, and all-data training require
separate later approval.
