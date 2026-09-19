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
| 3. eSFT/RL on Qwen3.6-35B-A3B (OpenJev) | NOT STARTED | dataset emitter + provider choice are the next deliverables |

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
  `manifest.json` (truth labels); `results/signals_v3/` 1.7MB caches.
  Regenerable via `build_corpus_v3.py` + `rung1_signals_v3.py` but slow.
  Do NOT `git clean` `.verify_tmp/`.
- **Credentials** (`~/.dsh/.credentials.yaml`, names only — never print values):
  `TYPESAFE_API_KEY` (real Jev), `ARK_PLAN_FLASH_API_KEY` (proposer, updated to
  key ending bf8f8, verified auth-ok), `FIREWORKS_API_KEY` (Phase 3).
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

## Phase 3 spec (agreed, not built)

1. `ft_dataset.py`: walk `results/evolve`/`jevloop*` transcripts + bank
   episodes + `corpus_v3/manifest.json` truth → Fireworks-compatible JSONL
   `{messages:[user=state+questions, assistant=gold answers]}`.
2. Labels from **manifest truth + oracle policy** (NOT Jev's own answers —
   that clones its errors incl. `sufficient` mush). Oracle `action`: judge
   unjudged candidate nearest a true seam; conclude when break-adjacent
   candidates resolved or decisive free signal present.
3. Include: all intermediate states, hard negatives (static swaps, high-
   photo-jump cleans, sub-perceptual splices), negative-evidence states,
   partial-observation/streaming states, evidence-subset augmentation.
4. Two variants: retrieval-conditioned (precedents in state) + retrieval-free.
5. Split LOSO by source (5 sources: 09c1414f1b, 0d2ee665be, 13c3e046d7,
   1ada7a0617, 21d970d8de).
6. Eval: OpenJev parity replay over all logged states BEFORE any fine-tune;
   then Fireworks SFT (simplest) or Tinker (if RL reward
   `verdict_correct − λ·judge_calls` wanted).

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
