# Temporal-edit coherence detection — agent notes

Mission: detect temporal edits (loop/reverse/room_swap/splice/timewarp) in egocentric
video with near-zero false positives on clean footage. Phase 3 = fine-tune the Jev
policy (Qwen3.6-35B-A3B LoRA on Tinker) to replace the OpenJev endpoint.

Read `.verify_tmp/coherence_mvp/HANDOFF.md` first — it is the running state doc.

## Environments (two interpreters, pick per task)

- `~/s1-spike/.venv/bin/python` — tinker 0.30.0 + `system_one_adapter`/`routes.cred`.
  Required for: `jev_loop.py`, `eval_s1.py`, `ft_rl.py` (they import jev_loop).
- system `python3` — tinker only. Use for `ft_tinker.py`, `s1_probe.py`.
- Credentials: `~/.dsh/.credentials.yaml` → `refs.TINKER_API_KEY`, `refs.TYPESAFE_API_KEY`,
  `refs.HF_TOKEN` (or env). NEVER print key values. HF_TOKEN is needed for the
  tokenizer download in `S1Compiler`.

## Tinker discipline (learned the expensive way)

- **Estimate tokens before launching** a training job and state the estimate to the
  user. One epoch over the s1 dataset ≈ 30–45M tokens (each row is a ~1.9k-token
  state prompt; loss on 1 label token, compute on the whole context).
- **Single flight per fold**: `pgrep -f ft_tinker.py` before launching. Two
  processes on one fold silently interleave in the same log and waste tokens.
- **Always pass `--save-every N --sample-every N`**. `save_weights_for_sampler`
  weights cannot resume training — `create_training_client_from_state` only
  accepts `save_state` (`weights/`) checkpoints. A run without state saves is a
  dead end for RL warm-start.
- `tinker auth status` distinguishes "key valid, account billing-blocked" (402,
  empty account name in the error) from bad credentials. Billing portal:
  https://tinker.thinkingmachines.ai/billing/balance — the key MUST belong to the
  billed account (user top-up on a different account won't unblock it).
- Weights live on Tinker as `tinker://` URIs. Locally keep only `sampler_path.txt`,
  `train.log`, eval transcripts — never model-sized files.

## Fireworks (alternative training backend — prepped, not yet launched)

- **Skills exist**: `~/.codex/skills/configure` (train/monitor workflow +
  mandatory plan-approval gate before job creation), `debug`, and
  `fireworks-training` (references: models/cost, managed RFT, deploy/teardown).
  Follow configure's final-plan gate: show the full resolved plan + cost and
  get explicit approval before `train`.
- Preflight verified 2026-09-19: account `zhangye1987-q56dy8y3`, datasets READY,
  managed-SFT quota ok (concurrent-runs=8, submissions=8), deploy quota b200=16.
  `training-*-count: 0` = dedicated Training API path only (not managed SFT).
  Payment-method-on-file unknown — job create rejects up front if absent.
- Managed RFT exists (GRPO + `--warm-start-from` a previous LoRA) — our ft_rl
  algorithm is portable IF the reward evaluator is wrapped per eval-protocol
  contract. `rftLoraManaged:false` on this model in the catalog; recheck
  live matrix before relying on it.
- Pure-REST driver `ft_fireworks.py` (no firectl signin needed):
  `upload` / `train --set <all|loso_src> --name <id>` / `status <job> [--watch]` / `jobs`.
  Account auto-discovered from the API key (`zhangye1987-q56dy8y3`).
- Datasets already uploaded: `jev-s1-all` (29,844 ex, ~40.8M tok) + 5 fold
  `*_train` / `*_val` pairs, format CHAT, avgTurns 3.
- Format: `[user:state(w0), user:EVAL+Question/Options/Answer:(w0), assistant:label]`
  — `weight:0` replicates single-token loss. Built by `ft_fw_dataset.py`.
- Job knobs mirror Tinker: loraRank 32, batchSizeSamples 128, 1 epoch,
  `evaluationDataset` per fold, `warmStartFrom` available. The job-create
  response carries `estimatedCost` — read it before confirming.
- Serving: `s1_fw_serve.FWJev(model_id)` — same answers() contract via
  chat completions + UNIFORM logit_bias on all label token-ids (preserves
  relative probs → exact normalized dist even at top_logprobs cap 5).
  `--jev fw:<model-id>` in jev_loop; `--fw-model <id>` in eval_s1.py.
  `results/ft_fw/label_ids.json` caches the 64 label token-ids.
- Inference needs a dedicated deployment (LoRA shape qwen3p6-35b-a3b-256k-lora,
  4×B200 — per-hour; undeploy after eval). `prompt_token_ids` exists if exact
  compiled-token fidelity is ever needed. `reasoning_effort:"none"` disables thinking.
- RL: REST has create-reinforcement-fine-tuning-job, but this model shows
  `rftLoraManaged:false` — keep RL on Tinker unless that changes.

## Evaluation fidelity (do not shortcut)

- **Faithful eval = `eval_s1.py` + `S1Jev`** — uses SamplingClient with the exact
  compiled OpenJev contract: state user-message + eval instruction +
  `enable_thinking=False` + per-question `Question/Options/Answer:` suffix →
  single label token, logprob-normalized.
- **`s1_probe.py` / OpenAI-compat chat endpoint is qualitative only.** The chat
  route cannot inject the literal `Answer:\n` suffix — smoke and fully-trained
  checkpoints produce identical miss patterns through it. Identical misses ≠
  model failure. Endpoint quirks: `/completions` rejects `logprobs`,
  `top_logprobs` unsupported, `reasoning_effort:false` disables thinking.
- **Val NLL misleads on single-token labels** (observed 4.8 val vs 0.03 train yet
  exact argmax parity). Judge by held-out detection/typing/judge-call counts.
- Score any run: `jev_score.py <transcripts_dir>` (auto-finds `manifest.json`).

## Known numbers & residuals

- OpenJev baselines: det 88/90, type 83/90 (v2), 375 judges; banked 319 judges.
- First tuned ckpt (loso `09c1414f1b` fold): det 88/90, type 83/90, **69 judges**,
  0 clean FPs. Hard gate: the 15 clean clips must stay FP-free.
- Residual classes: static room_swaps (mis-typed loop), one-seam swaps → splice,
  sub-perceptual splices → none, one reverse → loop. All information-limit.
- **Static-swap discriminator** (verified corpus-exclusive via `load_clip_state`):
  `recur_frac_max >= 0.9 AND dup_dense_last_frame is not None` → room_swap.
  Only 3 corpus examples, all in source `09c1414f1b` — LOSO cannot learn it;
  the production/alldata model can (188 signature rows in `all.jsonl`).

## Conventions

- Action vocab: `judge_0..7`, `conclude`, `abstain`. `abstain` is data-only —
  prod `questions_for` never offers it (stagnation gate v2 was verified a
  structural no-op; a reachable gate is a future decision).
- Bank: `.codex/reasoning-bank/` is `dsh-pes-reasoning.v1` — 256 stratified
  entries, exported by `bank_export_pes.py` (source×op round-robin, not FIFO).
- Production path: `jev_loop.py --jev s1:tinker://<ckpt>` drops the tuned
  sampler in as the policy — same state/question logic, no OpenJev calls.

## Hygiene

- DO NOT `git clean`. `.verify_tmp/` holds the corpus, signal caches, banks —
  expensive to rebuild.
- Never commit unrelated WIP: `eval/dual_vqa_*.py`,
  `sandbox/bvb_sandbox/agents/bash_agent.py` are user-owned dirty files.
- Update `HANDOFF.md` before ending a session — next agent reads it first.
