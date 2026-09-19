# Temporal-Edit Detector — Runbook

Status: v7 validated (98-clip corpus). Detached worktree, no git clean ever.
Venv for everything: `~/s1-spike/.venv/bin/python` (shell var `$V`).

## The pipeline (5 layers, run in order)

```
clips mp4 ──> free signals (4 extractors, local, free)
          ──> VLM pair-judge passes (4 gates, ~2s/call, zenmux->oc-vision->ark-plan)
          ──> jev_loop (agentic evidence-gathering inside code boundaries)
          ──> composite_type verdict
```

### 0. Corpus

```bash
$V build_corpus_v3.py     # 90 clips: 5 src x 3 win x 6 ops
$V build_ood.py           # 8 HELD-OUT w3 clips (fresh windows) -> manifest now 98
```

**w3 clips are the permanent held-out arm. Never train, tune, distill, or
calibrate on them.** Manifest: `corpus_v3/manifest.json` (`window==3`).

### 1. Free signals (local, $0)

```bash
$V rung1_signals_v3.py [filter]   # recur_frac, photo_jump, ORB, flow  -> results/signals_v3/
$V dupfrac_extract.py             # near-dup density                   -> results/dupfrac/
$V echo_detect.py                 # echo best_pair + score             -> results/echo/
$V rung1_codec_stats.py           # I-frame/scenecut packet stats      -> results/codec_stats/
```

All skip already-computed clips (safe to re-run after adding clips).

### 2. VLM pair-judge passes (4-frame strips, ~2s/call)

```bash
$V pairjudge.py [--only X]        # pixel_gate: top photo-jump/ORB peaks (cap ~5)
$V e2e_scenecut.py                # scenecut: codec I-frame candidates
$V splice_widen.py                # wide_gate: extra peaks on splice clips
$V pairjudge_bracket.py           # signal_seam: scenecut/dup_last/echo-pair
                                  #   positions no other pass covered
```

Verdicts cache to `results/pairjudge{,_scenecut,_wide,_bracket}/<cid>.json`.
Standard 4-frame strip ONLY — a 12-frame dense strip was tested and
REJECTED (same model: scene_change@1.0 on 4 frames, `continuous` 113/117
on 12; attention dilutes over the span).

### 3. Jev loop — the policy

```bash
# reference policy (TypeSafe SystemOne, free, network)
$V jev_loop.py --out results/RUN [--lessons results/bank_curate/lessons_v4.jsonl] \
               [--only cid1 cid2 ...] [--pack prompt_pack_v2.json] [--workers 10]

# tuned policy (SFT Qwen — the efficiency path, see below)
$V jev_loop.py --jev s1:tinker://<ckpt>/sampler_weights/final --out results/RUN2 ...
$V jev_loop.py --jev fw:accounts/<acct>/models/<id>            --out results/RUN3 ...
```

What lives in code (deterministic boundaries — do NOT move into the model):
- candidate union over the 4 gates; reveals are cached-verdict lookups
- seam-count arbitration: room_swap needs >=2 revealed discontinuous, else splice
- **seam second-look**: on a {splice, room_swap} conclude with <2 discont
  revealed, code reveals unjudged candidates (signal-seam positions first,
  MAX_SEAM_LOOKS=3) — "splice asserts a second seam is absent; prove it"
- info-gain stagnation -> abstain corrupt_untyped; evidence floor on conclude
- LESSON_GATES: a lesson injects only when its declared signal conditions
  hold; a lesson with no gate fires everywhere (r2_00 intentionally).

### 4. Scoring (IMPORTANT)

Score `composite_type` — the post-arbitration verdict — never
`final.break_type.choice` (raw model answer, pre-arbitration).
Detection = (gold!="none") == (pred!="none"); typing = exact match;
normalize `reverse_segment` -> `reverse`.

**Noise floor: +/-2-3 clips per 98-clip run** (Jev sampling; e.g.
`09c1414f1b__w3__loop` coin-flips loop/reverse across identical configs).
Any single-run delta <3 clips is noise. For eval claims: run both arms
(90 in-sample + 8 w3), report clean FPs separately, prefer 2 runs.

### 5. Reasoning bank (curator loop — training material, NOT runtime)

```bash
$V bank_curate.py propose  --misses <dirs> --active <lessons.jsonl> --out <cand.jsonl>
$V bank_curate.py validate --candidates <cand.jsonl>      # BOTH arms required
$V bank_curate.py deposit  --validated <cand.jsonl>       # PASS only
```

Runtime injection of lesson prose is RETIRED as a gains path (in-sample
+5 did not survive gating/OOD; loose application caused a real regression).
Bank value now = rag_s1 training rows + a failure memory the curator reads.

## Efficiency — SFT Qwen > TypeSafe Jev

The expensive axis is judge calls (VLM calls, ~2s each), not jev calls.
The tuned policy's proven value is cost, not accuracy:

| | TypeSafe Jev (v7) | SFT Qwen fold-1 (measured pre-v7) |
|---|---|---|
| detection | 88/90 | 88/90 (parity) |
| typing | 84/90 | 83/90 (parity) |
| judge calls | ~5-8/clip | **69 total vs 375 baseline (-82%)** |
| held-out source | 7/8 w3 | det 18/18, type 15/18, 9 judges vs 77 |

Why it works: the policy learned to conclude on fewer reveals — it judges
only what discriminates, then stops. The surrounding code (bracket
candidates, seam-look, arbitration, lesson gates) is identical for both
backends, so SFT inherits the coverage fixes AND keeps the -82% call
reduction. Expected upside over v7 is modest (+2-4 typing if the
static-swap discrimination learns); the cost win is the headline.

## SFT path (BLOCKED on billing — Stage 0)

```bash
$V ft_dataset.py                          # transcripts+truth -> JSONL variants
python3 ft_tinker.py train --fold N|all   # Tinker LoRA (billing-blocked)
python3 ft_fireworks.py train --set all --base-model qwen3-8b   # FW (402-blocked)
$V eval_s1.py --ckpt tinker://<uri> --out results/evals1/<name>
```

Stage 0 (user action, ~5 min): fix Tinker key (`refs.TINKER_API_KEY` in
~/.dsh/.credentials.yaml — balance lives on the other account) OR add a
Fireworks payment method. 3 checkpoints already trained, remote-safe:
alldata `tinker://4877bbea-...` (room_swap-fix candidate), fold-2
`60bf1921-...`, RL warmstart `a5d4c168-.../sft_warmstart`. ~$150 spent.

Ship gate: det >=88/90, type >=84/90 on BOTH arms (90 + w3), 0 clean FPs,
judges well under v7's ~5-8/clip, confirmed on >=2 runs (noise band).

## Rejected approaches (do not retry without new evidence)

- 12-frame dense judge strips (113/117 continuous at real seams)
- ungated lesson injection (loose application -> fresh reverse->room_swap)
- natural-language cues as preconditions (code gates or nothing)
- thresholds fit to n=1 clips (reverse gate recur>=0.9 never fires fresh)
- claiming gains from single runs (noise +/-2-3/98)

## Current residual (7 of 98)

- 3x static room_swap->loop (recur=1.0 + dup_dense set = loop signature;
  needs the trained discrimination — prime SFT target)
- 2x splice->none (invisible to VLM at every strip width; likely a hard
  floor unless a new signal or stronger judge arrives)
- 2x loop->reverse boundary coin-flips (dup-null loops; Jev noise range)
