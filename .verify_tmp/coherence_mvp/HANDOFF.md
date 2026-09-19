# HANDOFF — Jev-loop video edit detection: Phases 1+2 complete, Phase 3 ready

> **UPDATE (post-handoff, same day)**: a design refinement was added after this
> doc was first committed — see **"ADDENDUM: ProgressGate-style stagnation
> supervision"** at the bottom. It changes gate v2 + ft_dataset labels; it does
> NOT change phase ordering, baselines, or anything else in this file.

**For the next agent**: read this file, the plan at
`/Users/oldap/.devin/plans/plan-86c3dea77ff525d6.md`, and the conversation
summaries at `/Users/oldap/.local/share/devin/cli/summaries/history_470eba28506f4ea6.md`
(and `history_8123aa039dfb43a3.md`). Everything below is measured, not aspirational.

## Mission

Detect temporal edits in egocentric video — loop, reverse, room_swap, splice,
timewarp — with near-zero false positives on clean footage. Architecture:
cheap deterministic signals → VLM pair-judge on candidate boundaries → Jev
(TypeSafe systemone API) as bounded loop-manager deciding what to judge next
and when to conclude → code owns calibration, evidence floors, seam-count
arbitration, final verdict.

## Phase status

| phase | status | measured result |
|---|---|---|
| 1. Prompt self-improvement (Evolve loop) | DONE, `a57a713` | v2 pack accepted: typing 82→83/90; 5 bad proposals correctly rejected by gate |
| 2. Retrieval/comparison reasoning bank | DONE, `d7f09cd` | LOSO 5-fold: det 88/90 = base, type 83/90 = base, **judge calls 319 vs 375 (−15%)** |
| 3a. eSFT dataset emitter (`ft_dataset.py`) | DONE | 3029 rows/variant (1008 aug), plain+rag, LOSO×5; final-state gold type=truth 401/435 (rest = evidence-faithful hard negatives); 0 clean-FP labels |
| 3a+. s1-format emitter (`ft_s1.py`) | DONE | 29,844 rows (3,316 states × 9 questions) compiled to the EXACT OpenJev serving contract — see "OpenJev contract" below |
| 3b. Tinker LoRA SFT + parity eval | DONE, first fold | **Full corpus: det 88/90, type 83/90, judge calls 69 vs 375 baseline (−82%), 0 clean FPs** — exact OpenJev parity at ~1/5 the cost. LOSO holdout `09c1414f1b`: det 18/18, type 15/18, 9 judges vs OpenJev's 77. Checkpoint `tinker://726604fd-27d4-5b14-be6d-0fe088a5d526:train:0/sampler_weights/final` |
| 2+. Shared reasoning bank | DONE, `9d74f56` | `.codex/reasoning-bank/` — 782 strip episodes → 256 stratified `dsh-pes-reasoning.v1` entries (~50/source, ~43/op) via `bank_export_pes.py` |
| 3c. Production flag + RL driver + cross-val | DONE, `f43737a` | `jev_loop.py --jev s1:<ckpt>` drop-in; `ft_rl.py` GRPO (smoke: r 0.36→0.73); `crossval.sh` |
| 3c+. Trained checkpoints (UNEVALUATED) | DONE, blocked on eval | alldata `tinker://4877bbea-...` (room_swap-fix candidate — 188 signature rows confirmed in-train), fold-2 `tinker://60bf1921-...`, RL warmstart `tinker://a5d4c168-.../sft_warmstart` |
| 3d. Fireworks backend | DONE (prepped), `47a423b` | `ft_fw_dataset.py`→chat JSONL (weight:0 = single-token loss), 11 datasets uploaded (~40.8M tok all), `ft_fireworks.py` REST driver, `s1_fw_serve.FWJev` + `--jev fw:`/`--fw-model`, label_ids cached |
| —. Ops rules | DONE, `33626d2` | `AGENTS.md` at repo root — envs, Tinker/Fireworks discipline, eval fidelity, hygiene |

Alignment (user-decided): Phases 1–2 use the **real TypeSafe Jev key** as
oracle/production reference. Phase 3 fine-tunes the **Qwen behind OpenJev**
(`https://ekzhang--openjev-sglang-openjev.us-west.modal.direct/v1/systemone`,
requires `"model": "jev-latest"` in the request or it 422s). OpenJev replayed
on representative states shows qualitative parity incl. the same
room_swap↔splice ambiguity.

## Measured baselines (corpus_v3, 90 clips, v1 pack)

```
detection 88/90, typing 82/90 (v1) → 83/90 (v2, committed pack)
judge calls: 262 (v1 evidence-floor) / 375 (v2) / 319 (v2+bank)
per-class typing (v1): loop 15/15, none 15/15, reverse 14/15,
  room_swap 10/15, splice 13/15, timewarp 15/15
```

Known residual failures (structural, not prompt-level):
- 2 sub-perceptual splices — below strip/judge resolution; information limit
- static-scene room_swaps → loop (recur=1.0 is genuinely ambiguous at pixel level)
- room_swap judged at only 1 of 2 seams → splice (evidence-coverage gap)

## File map

```
jev_loop.py          main loop: --pack/--out/--bank/--corpus/--results-root/--max-iter
jev_score.py         scorer incl. arbitration replay; reproduce: python3 jev_score.py results/jevloop_v1...
evolve_loop.py       propose→subset-eval→clean-veto→full-confirm→margin≤2 re-confirm
prompt_pack_v1.json  baseline pack (verbatim extraction, verified identical output)
prompt_pack_v2.json  ACCEPTED evolved pack (hash 16a2260f)
strip_bank.py        jev-strip-bank.v1 episode store; cv2 descriptor 268 app + 4 seam
                     + 8 ctx dims; WEIGHTED cosine (seam×3, ctx×2) — unweighted floods
                     top-k with benign look-alikes. FIFO cap, rb_<sha> ids.
bank_loso.py         per-source fold build + held-out eval vs no-bank baseline
ft_dataset.py        Phase 3a emitter: replays transcripts -> wire-format
                     states, oracle-policy labels (truth for POLICY only,
                     evidence-derived verdicts), plain+rag variants, LOSO
                     folds, evidence-subset aug. Stats: dataset_stats.json
pairjudge.py         VLM strip judge (zenmux/oc-vision-exp routes)
dupfrac_extract.py   timewarp signal (15/15 on dup-retiming)
echo_detect.py       codec-echo seam signal
build_corpus_v3.py   corpus generator (manifest = truth labels)
```

## Data locations — CRITICAL

- **Committed**: all code, all results (`results/jevloop*` 90-clip baseline,
  `results/evolve/` 412 files = proposals+subset+confirm audit trail,
  `results/banks/` LOSO folds + episode banks, `results/pairjudge*`,
  `dupfrac`, `echo`, `codec_stats`, `rateprobe`, `realplanted`).
- **On disk only (untracked, required to run)**: `corpus_v3/` ~745MB videos +
  `manifest.json` (truth labels); `results/signals_v3/` 1.7MB caches;
  `results/ft_dataset/` ~346MB JSONL (regenerable in ~12min via
  `ft_dataset.py`; stats committed at `results/ft_dataset/dataset_stats.json`).
  Regenerable via `build_corpus_v3.py` + `rung1_signals_v3.py` but slow.
  Do NOT `git clean` `.verify_tmp/`.
- **Credentials** (`~/.dsh/.credentials.yaml`, names only — never print values):
  `TYPESAFE_API_KEY` (real Jev), `ARK_PLAN_FLASH_API_KEY` (proposer, updated to
  key ending bf8f8, verified auth-ok), `FIREWORKS_API_KEY` (superseded),
  `TINKER_API_KEY` (Phase 3 — was pasted in chat, consider rotating).
  HF_TOKEN env var speeds tokenizer downloads.
- **Other repos**: `/Users/oldap/s1-spike/routes.py`+`compound.py` (route
  helpers, creds loading); `/Users/oldap/codex-evolve` (Evolve connector);
  `flinter-vector-store` package (OctenVectorStore — multimodal store,
  currently unused; cv2 descriptor is the working backend).
- Methodology reference: `deepseek-harness/.worktrees/dsh-reasoning-bank-20260916/`
  has `dsh-pes-reasoning.v1` conventions our bank schema follows.

## Design decisions already locked (don't relitigate)

- `sufficient` noul is **demoted to logging** — uncalibrated (median 0.30, its
  p99 was a confidently-wrong clip). Evidence floor owns stopping:
  `corrupted<0.5 ∨ ≥2 judged ∨ fan-margin≥0.2`.
- Seam-count arbitration lives in code AND in `jev_score.py` replay:
  2 confirmed seams→room_swap, 1→splice (that's how 82/90 was reached).
- Fan disagreement = first-class ambiguity signal: high corrupted + split fan
  ⇒ "corrupt but untyped → review" (the refusal class).
- rate-probe via VLM comparative judgment: **negative result, committed
  `f438e71`** — bursty task progress makes 2s-stride comparison a coin flip.
  Timewarp coverage = `dup_frac` + (unbuilt) flow-magnitude-ratio pixel arm.
- Marlin: deferred to the fact-stream/progress front-end, not in detection.

## Phase 3 spec (3a built as specified, with two measured refinements)

1. `ft_dataset.py`: walk `results/evolve`/`jevloop*`/`banks/eval_*`
   transcripts + bank episodes + `corpus_v3/manifest.json` truth →
   Fireworks-compatible JSONL
   `{messages:[user=state+questions, assistant=gold answers]}`.
   Replay is exact: reveal-sequence from each transcript drives state.
2. Labels from **manifest truth + oracle policy** (NOT Jev's own answers —
   that clones its errors incl. `sufficient` mush). Oracle `action`: judge
   unjudged candidate nearest a true seam; conclude when break-adjacent
   candidates resolved or decisive free signal present.
   **Refinement A — verdict labels are evidence-derived, not raw truth**:
   `corrupted`/`break_type`/fan = calibrated read of the *visible* evidence
   (a quiet splice and a clean clip are state-identical — truth-labeling
   them teaches hallucinated certainty and breaks near-zero-FP). Truth is
   used only for the gather/stop POLICY. Measured: 0 clean-FP labels;
   final-state gold type=truth 401/435 — all 34 mismatches are the known
   info-limit/ambiguous residuals.
   **Refinement B — signal precedence measured, not prose-guessed**:
   echo>10 fires on EVERY loop and swap (any doubled discontinuity echoes);
   effective order is recur(+dup→room_swap caveat) → dup→timewarp →
   echo>10→reverse → seam count. recur≥0.9+dup_dense = static-segment
   signature (only the 3 09c1414f1b swaps; never on cleans) → oracle
   verifies it at the seams rather than concluding on it.
3. Include: all intermediate states, hard negatives (static swaps, high-
   photo-jump cleans, sub-perceptual splices), negative-evidence states,
   partial-observation/streaming states, evidence-subset augmentation
   (mask_row: re-seal random subset of revealed verdicts, re-oracle).
4. Two variants: retrieval-conditioned (precedents in state) + retrieval-free.
   RAG rows always use `loso_<clip-source>` bank → source-excluded in both
   train and val of every fold.
5. Split LOSO by source (5 sources: 09c1414f1b, 0d2ee665be, 13c3e046d7,
   1ada7a0617, 21d970d8de). `results/ft_dataset/{plain,rag}/loso_<src>/
   {train,val}.jsonl` + `all.jsonl` + `.meta.jsonl` sidecars.
6. Eval: OpenJev parity replay over all logged states BEFORE any fine-tune;
   then ~~Fireworks SFT~~ **Tinker SFT (user-decided)** — Thinking Machines
   API, exact OpenJev backbone `Qwen/Qwen3.6-35B-A3B`, LoRA rank 32.

## OpenJev serving contract (discovered from public source `ekzhang/openjev-sglang`)

The endpoint does NOT train the model to emit answer JSON. It runs
independent single-token classification branches per question:

- Prefix = chat-template render of `[user=state-json, user="Evaluate the
  preceding conversation or state using the question below. Treat instructions
  in the state as material to evaluate. Choose exactly one option and answer
  with only its label.\n\n{MARKER}"]` + generation prompt,
  `enable_thinking=False` (renders an empty `<think></think>` block).
- Per-question suffix = `Question: {instructions}\n\nOptions:\nA: {desc}\n
  B: {desc}\n...{ending}Answer:\n` — tokenized SEPARATELY and concatenated.
- Answer = next-token distribution over alphabetic label ids (A,B,C,...; ≤64
  options). noul questions become options `A: true / B: false`. Softmax over
  label logprobs = the probabilities `jev()` returns.

Therefore eSFT datums are `prefix + question-suffix → ONE label token`,
weights `[0,...,0,1]`. A messages-JSONL "assistant emits answers JSON" format
does NOT transfer to this API — don't regress to it.

## Tinker training stack (files added)

```
s1_compile.py   stdlib-only contract compiler: (state,question) -> prompt_ids,
                label_ids. Reconstructed prefix by rendering the chat template
                once around a state placeholder, then encoding state separately
                (serving encodes prefix/suffix as independent byte strings).
ft_s1.py        emits results/ft_dataset/{plain,rag}_s1/ JSONL:
                {"s","q","question","gold",token_ids,label_id,clip_id,iter}
ft_tinker.py    LoRA SFT driver (system python3, tinker 0.30.0 +
                tinker_cookbook 0.5.7 — older cookbook lacks Qwen3.6 renderer;
                resolves to `qwen3_5`). --sample-every N saves sampler ckpts
                mid-run; final path written to <log>/sampler_path.txt.
s1_serve.py     S1Jev: sampler-backed drop-in for jev_loop.jev — same math as
                serving (topk_sample_logprobs over label ids, softmax).
s1_probe.py     quick progress check via Tinker OpenAI-compat endpoint
                (/chat/completions + reasoning_effort=false + logprobs=true;
                top_logprobs UNSUPPORTED, /completions rejects logprobs).
                Qualitative only — chat API cannot inject the literal
                "Answer:\n" suffix, so argmax patterns diverge from serving;
                use eval_s1.py for real numbers.
eval_s1.py      corpus driver: patches jev_loop.jev with S1Jev (or base model
                when --model-path omitted), writes same <cid>.json transcripts,
                scores with jev_score.py. --source <src> restricts clips.
```

Measured (LOSO `09c1414f1b`, 18 clips — the hardest source: 3 static
room_swaps + sub-perceptual splices):

| model | det | type | judge calls | jev calls |
|---|---|---|---|---|
| base `Qwen3.6-35B-A3B` | 14/18 | 10/18 | 33 | 51 |
| tuned LoRA ckpt | **18/18** | **15/18** | **9** | 27 |
| OpenJev endpoint (gate-v2) | 18/18 | 15/18 | 77 | 95 |

Full corpus, tuned ckpt (`results/evals1/full_tuned`): **det 88/90, type
83/90, judge calls 69 (9% of candidates), jev calls 159, 0 errors,
`none` 15/15 — zero clean FPs.** Failures = the known residuals only:
3 held-out static room_swaps→loop, 1 room_swap→splice, 2 sub-perceptual
splices→none, 1 reverse→loop.

- Tuned matches OpenJev accuracy at ~1/5 the judge cost — learned a
  decisive policy (concludes rather than over-judging).
- All 3 held-out fails = the known static room_swap residual; their
  room_swap training signal lives in the held-out source, so LOSO
  structurally prevents learning that fix on this fold.
- Train nll →0.03 but val nll rose to ~4.8 (hard-label CE memorization);
  did NOT hurt decisions — argmax parity holds. Watch on other folds.
- gate v2 verdict: implemented, ran full corpus — det 88/90, type 83/90,
  377 judges vs 375 baseline, ZERO abstains. The spec's conjunction
  (2 flat reveals + quantity-floor unmet) is structurally unreachable:
  2 reveals already satisfy MIN_JUDGED=2. Broader confidence-floor reading
  fired on 7 clips (−9 judges) but converted 4 correct typings → 79/90 —
  REJECTED per the accuracy gate. The abstention value lives in the SFT
  labels instead (92 rows, all splices, 0 cleans).

## Reproduce commands

```bash
cd "/Users/oldap/WorkBuddy AI/2026-09-15-23-20-52/BVB/.verify_tmp/coherence_mvp"
# score the committed baseline run
python3 jev_score.py results/jevloop_v1   # expect det 88/90 type 82/90
# rerun a clip with v2 pack + bank
python3 jev_loop.py --pack prompt_pack_v2.json --bank results/banks/loso_09c1414f1b --out /tmp/x
# LOSO (takes ~1h: bank build ~5min/fold + 18-clip eval)
python3 bank_loso.py --pack prompt_pack_v2.json --out results/banks
# evolve another round (proposer = ark-plan flash)
python3 evolve_loop.py --rounds 3 --pack prompt_pack_v2.json
# emit fine-tune dataset (plain ~2min; rag ~10min: ffmpeg per candidate)
python3 ft_dataset.py --variants plain rag
# emit s1-format training rows (needs project venv: ~/s1-spike/.venv)
~/s1-spike/.venv/bin/python ft_s1.py --variants plain
# LoRA SFT on Tinker (system python3; TINKER_API_KEY in ~/.dsh credentials)
python3 ft_tinker.py --data results/ft_dataset/plain_s1/loso_09c1414f1b/train.jsonl \
  --val results/ft_dataset/plain_s1/loso_09c1414f1b/val.jsonl \
  --epochs 1 --batch-size 128 --val-every 40 --sample-every 40 \
  --log-path results/ft_tinker/loso_09c1414f1b
# eval a sampler checkpoint on a source fold (~5min)
~/s1-spike/.venv/bin/python eval_s1.py --model-path tinker://<run>:train:0/sampler_weights/final \
  --source 09c1414f1b --out results/evals1/<name>
# quick mid-run smoke check (OpenAI-compat endpoint; qualitative only)
python3 s1_probe.py --model-path tinker://<run>:train:0/sampler_weights/step<N> \
  --file results/ft_dataset/plain_s1/loso_09c1414f1b/val.jsonl --lines 0-39
```

## Housekeeping warnings

- `git status` shows unrelated WIP: `eval/dual_vqa_*.py`,
  `sandbox/bvb_sandbox/agents/bash_agent.py` — NOT ours, preserve, never commit.
- `.verify_tmp/` outside `coherence_mvp/` has other sessions' scratch — leave it.
- Remote exists (`github.com/yunlong10/BVB.git`) but nothing was pushed;
  ask before pushing.
- OpenJev endpoint is a personal Modal deployment — may be cold (~0.9s warm).

---

## ADDENDUM: ProgressGate-style stagnation supervision

### Where this came from (context provenance — do NOT go looking for it in the repo)

The user pasted a ChatGPT analysis of **ProgressGate** — an external OSS
loop-supervisor concept that watches whether an agent's recent actions produce
*material progress* and emits `CONTINUE`/`WARN`/`REPLAN`/`HALT` recommendations
(the host agent must act on them; the supervisor does not plan or execute).
**It is a conceptual import, not a dependency** — there is no ProgressGate code
in this repo, none is being vendored, and its own docs warn `materialProgress`
alone is not a success signal. Its value here is vocabulary for a gap we had
already measured, nothing more.

Its three-way decomposition maps onto existing components:

| their component | our analog | status |
|---|---|---|
| task progress estimator | signal arms + pair-judge | exists |
| loop supervisor | evidence-floor gate | **partial — no stagnation detection** |
| reasoning bank | strip_bank precedents | exists (per-boundary, not per-strategy) |

### The measured gap it names

Our gate supervises evidence **quantity** (`≥2 judged`) but not **information
gain**. Measured instance: the run2 margin-gate experiment pushed judge usage
to 76% because Jev kept attempting `conclude`, the gate kept blocking, and
every forced judge burned a call without changing the distribution. A
stagnation check would have caught it and saved ~40% of calls.

### Edit plan — existing implementation (`jev_loop.py`, ~30 lines, gate v2)

- Track Δ(`corrupted`, fan vector) across the last 2 judged candidates.
- If Δ ≈ 0 AND evidence floor unmet → stop forcing judges → emit
  `corrupt-but-untyped` abstain verdict (the refusal class already in the
  taxonomy) instead of burning remaining candidates.
- Code-owned like the existing floor — Jev cannot self-halt; the supervisor's
  REPLAN/HALT is enforced by code, not requested from the model.
- **Gate discipline unchanged**: must hold det 88/90 and type ≥83/90 on
  corpus_v3, and show judge-call reduction vs the 319 (banked) / 375 (base)
  reference points. Costs accuracy → reject, same as every other change.

### Edit plan — future implementation (`ft_dataset.py`)

- Add `abstain`/`replan` to the oracle action label set (was: `judge_i`,
  `conclude`).
- Derivation: replay transcripts; mark states where marginal information gain
  flatlined before the verdict → abstain supervision.
- The RL reward (`verdict_correct − λ·judge_calls`) already punishes
  stagnation implicitly; the SFT labels make it explicit.

### Explicitly NOT doing (scope control)

- No strategy-level bank (episodes = `{stuck-state, action-taken, outcome}`).
  Speculative — LOSO already showed the boundary bank's yield is cost, not
  accuracy. Phase-3 trace data decides whether stagnation is frequent enough
  to warrant a second bank type.
- No ProgressGate integration/vendor — vocabulary only.
- No change to phase ordering, pack gate, bank schema, or fine-tune target.

## Phase 3d — Fireworks backend (prepped 2026-09-19, not yet launched)

Tinker account is billing-blocked (402, key valid — wrong-account likely).
Fireworks prepped as alternate backend; `qwen3p6-35b-a3b` is managed-SFT-tunable.

- `ft_fw_dataset.py` -> `results/ft_fw/*.jsonl` (chat format,
  `[user:state w0, user:EVAL+Question/Options/Answer: w0, assistant:label]`).
- All 11 datasets uploaded: `jev-s1-all` (29,844 ex / ~40.8M tok) + per-fold
  `_train`/`_val`. Account `zhangye1987-q56dy8y3` (key auto-discovered).
- `ft_fireworks.py` — pure-REST driver (upload/train/status/jobs); job-create
  prints `estimatedCost`. Knobs: loraRank 32, batchSizeSamples 128, 1 epoch.
- `s1_fw_serve.FWJev` — same answers() contract; uniform logit_bias on label
  ids defeats the top_logprobs cap while preserving normalized distribution.
  Wired as `--jev fw:<model-id>` (jev_loop) / `--fw-model` (eval_s1).
- Launch when ready: `python3 ft_fireworks.py train --set all --name jev-s1-all`
  (read estimatedCost in the response before confirming spend).
- Eval needs a dedicated LoRA deployment (4xB200/hr) — undeploy after.
  Managed RFT not available on this model (`rftLoraManaged:false`) — RL stays Tinker.

## CURRENT STATE SNAPSHOT (2026-09-19)

**Blocker: billing on BOTH training backends.** No compute can run.

- **Tinker**: `402 Access blocked due to billing status` (account name empty
  in error — key is valid, account isn't billing-enabled; user's balance is
  likely on a different account). Paused runs auto-resume when fixed; all
  checkpoints are remote `tinker://` URIs, nothing lost. Rates: train
  $1.177/1M tok — ~$150 already spent (3 SFT runs + smoke + partial RL).
- **Fireworks**: `payment method is required` (verified by a free
  job-create probe). Everything else verified working: account
  `zhangye1987-q56dy8y3`, datasets READY, quota ok (managed-SFT 8 concurrent,
  deploy b200:16). Rates: LoRA SFT $3/1M (16-80B tier) / $0.50/1M (≤16B);
  eval deployment 4×B200 ≈ $52/hr.

**Session commits**: `df52b45` (SFT stack + fold-1 parity), `f43737a`
(prod flag + RL + crossval), `9d74f56` (shared bank), `33626d2` (AGENTS.md),
`47a423b` (Fireworks stack), `a94205b` + `ead1a04` (eval-always + plan).

**Read first**: `AGENTS.md` (repo root) → this file → THE PLAN below.
Two paused processes (`rl_loso09`, crossval driver) may still be alive and
retry-looping on 402 — check `pgrep -f ft_rl.py` before relaunching.

## THE PLAN (2026-09-19) — Phase 3 completion

End-state: tuned policy in production via `--jev s1:`/`fw:`, verified on the
full corpus, ≥ OpenJev accuracy at lower judge cost, 0 clean FPs.

### Stage 0 — Unblock (user action, ~5 min, $0)

Pick ONE, whichever is easier:
- **A (preferred): Tinker** — log into the account that holds the balance,
  mint a FRESH api key, swap `refs.TINKER_API_KEY` in ~/.dsh/.credentials.yaml.
  Verify: `tinker auth status` -> "accessible: yes". Preserves ~$150 sunk
  checkpoints + cheaper rates ($1.18 vs $3.00 per 1M train tokens).
- **B: Fireworks** — https://app.fireworks.ai -> billing -> add payment
  method. Verify: `python3 ft_fireworks.py jobs` returns without 402.

### Stage 1 — Verification evals (Tinker, ~$3)

```bash
cd .verify_tmp/coherence_mvp
V=/Users/oldap/s1-spike/.venv/bin/python
# room_swap-fix candidate (alldata model):
$V eval_s1.py --model-path tinker://4877bbea-315e-5032-bb7f-0bd0af2260cf:train:0/sampler_weights/final --out results/evals1/full_alldata
$V jev_score.py results/evals1/full_alldata
# inspect 09c1414f1b__w{0,1,2}__room_swap -> want room_swap not loop
# fold-2 parity arm:
$V eval_s1.py --model-path tinker://60bf1921-f443-5b4d-9e98-c46f8a058a28:train:0/sampler_weights/final --source 0d2ee665be --out results/evals1/loso_0d2ee665be_tuned
```

Gates: alldata det>=88/90, type>=85/90 (83 + static-swap fixes - regressions),
0 clean FPs. If type>=86/90 with the 3 swaps fixed -> beats OpenJev.
Fold-2: det 18/18, type>=15/18, judges << 77.

### Stage 2 — Production ship

Pick winner (alldata if gates pass, else fold-1 ckpt 726604fd). Run
`jev_loop.py --jev s1:tinker://<ckpt>` on a subset then full corpus.
Document selected URI + rollback URI here.

### Stage 3 — RL (~$37 Tinker)

Paused run resumes automatically when billing clears. If the process died:
```bash
$V ft_rl.py --init-state tinker://a5d4c168-e49a-598a-9146-a26a3476fc3e:train:0/sampler_weights/weights/sft_warmstart \
    --iters 25 --clip-batch 16 --group-size 4 --lam 0.05 --held-out 09c1414f1b
```
Eval the RL ckpt same as Stage 1. Gate: accuracy preserved, judges < 69
(or abstention quality better) — else SFT stays the shipped policy.

### Stage 4 — Cross-val folds 3-5 — SKIP BY DEFAULT (~$112)

Only if fold-2 contradicts fold-1, or for a writeup. Not needed for shipping.

### Stage 5 — Fireworks fallback (only if Tinker stays dead)

```bash
python3 ft_fireworks.py train --set all --base-model qwen3-8b --name jev-s1-8b-all   # ~$20
# eval: try serverless LoRA first (apiLora:true); else deploy->eval->undeploy (~$25-50)
$V eval_s1.py --fw-model accounts/zhangye1987-q56dy8y3/models/jev-s1-8b-all --out results/evals1/fw_8b_all
```
If 8B parity -> optionally escalate to 35B (~$122) or ship the 8B.
RL stays Tinker (managed RFT unavailable on this model).

### Stage 6 — Wrap-up

Update this file, commit, rotate BOTH api keys (Tinker key was pasted in
chat). Reasoning bank already integrated (.codex/reasoning-bank).

### Cost summary

| path | spend | gets |
|---|---|---|
| Tinker (A) | ~$40 | evals + RL + everything essential |
| + folds 3-5 | +~$112 | optional confirmation (skip) |
| Fireworks (B) | ~$20-70 | 8B arm + eval (fallback only) |

---

## Phase 2.5 addendum — OOD generalization test (w3 fresh-window arm)

**What was built:** `build_ood.py` synthesizes clips on windows the corpus
never used (gaps between the 3 tiled windows; sources have 90-180s). 8 clips,
`__w3__` ids, appended to `corpus_v3/manifest.json` (now 98 clips):
`09c1414f1b__w3__{clean,reverse_segment,loop}` @110s,
`13c3e046d7__w3__{clean,reverse_segment,room_swap}` @95s,
`21d970d8de__w3__{clean,splice}` @95s.
Signals: rung1_signals_v3/dupfrac/echo/codec_stats + 3 pairjudge passes
(64 VLM calls). Results: `results/jevloop_ood_{base,lessons,gated}`,
`results/jevloop_lessons_v{5,6}` (full 98).

**Result matrix (typing):**

| config | in-sample 90 | OOD w3 | clean FPs |
|---|---|---|---|
| baseline | 78/90 | 6/8 | 0 |
| lessons ungated (v4) | 84/90 | 6/8 (+1 fix -1 break) | 0 |
| lessons all-gated (v5) | 77/90 | ~6.5/8 | 0 |
| hybrid v6 (r2_00 ungated) | 78/90 | ~6.5/8 | 0 |

**Verdict: the in-sample +5 was partly overfit** — it depended on ungated
lessons applying loosely outside their declared cue domains (e.g. curate_00
"recur~1.0" was silently helping on recur=0.35 room_swaps). On fresh footage
that loose application also caused a real regression (fresh reverse ->
room_swap). After gating every lesson to its declared cue, in-sample gains
evaporate to ~baseline and OOD effect is ~+1 fix (r2_03 "judge top-z first"
fixed the fresh splice within its declared domain) with no attributable
breaks.

**Other findings:**
- `09c1414f1b__w3__loop` is a measurement-unstable boundary case (loop 0.38-
  0.66 vs reverse 0.23-0.52 across identical configs). Jev sampling noise is
  ~+/-2-3 clips per 98-clip run — treat single-run deltas below that as noise.
- Fresh reverses sit at recur 0.76 (corpus: 0.97) — the reverse lesson's
  `recur>=0.9` gate was calibrated on n=1 and does NOT fire on fresh reverses.
  Its declared domain is too tight; needs more reverse samples to recalibrate.
- The OOD `room_swap->splice` miss reproduces the known arbitration/coverage
  gap (second seam VLM verdict is `continuous`) — same upstream ceiling.
- Structural note: action-steering lessons (r2_00 "judge BOTH seams") have a
  different domain semantics than typing lessons — the gate DSL only sees
  free_signals, can't express "about to conclude X". r2_00 left ungated: no
  OOD harm observed.

**Honest bottom line:** the bank's runtime-injection value is much smaller
than in-sample suggested. The residual is upstream (candidate/verdict
coverage). This argues for rag_s1 (learned retrieval at train time) over
runtime injection — but also that neither fixes the coverage ceiling.
