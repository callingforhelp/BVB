"""GRPO-style RL on the s1 action policy, continuing from SFT weights in the
same training client (Tinker `weights/` state can't be rebuilt from sampler
checkpoints, so --sft-steps runs inline first when no --init-state is given).

Per iteration:
  1. save_weights_for_sampler -> sampling client for the current policy
  2. B clips x G rollouts: replay the jev_loop episode where the ACTION
     branch label is sampled (temp T) from the label-token distribution;
     verdict questions (corrupted/break_type/is_*) are evaluated greedily
     on the same policy — matching deployment, where one model does all.
     Gate mechanics replicate run_clip: conclude honored iff the evidence
     floor is met, else forced judge / flatline abstain / forced end.
  3. reward = [composite_type == truth] - lambda * n_judged
  4. advantage = r - group mean (per clip); datum = prompt + action token,
     weights [0,...,0,adv] -> exact on-policy REINFORCE on single-token
     actions (label distribution IS the policy distribution).

    python3 ft_rl.py --sft-file results/ft_dataset/plain_s1/loso_09c1414f1b/train.jsonl \
        --sft-steps 160 --iters 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
import torch
from tinker.types import SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).parent
SPIKE = Path.home() / "s1-spike"
sys.path.insert(0, str(SPIKE))

import jev_loop as J                       # noqa: E402
from s1_compile import S1Compiler          # noqa: E402
from ft_dataset import ABSTAIN_CRIT        # noqa: E402
from ft_tinker import api_key, row_to_datum, load_rows  # noqa: E402
from tinker_cookbook.supervised.common import datum_from_model_input_weights  # noqa: E402

MODEL = "Qwen/Qwen3.6-35B-A3B"
PACK = J.load_pack(str(ROOT / "prompt_pack_v2.json"))
TRUTH = {cid: ("reverse" if c["operator"] == "reverse_segment" else c["operator"])
         for cid, c in J.CLIPS.items()}
MIN_JUDGED, FAN_MARGIN, FLAT_EPS = J.MIN_JUDGED, J.FAN_MARGIN, J.FLAT_EPS


def action_question(st: dict) -> dict:
    q = J.questions_for(st, PACK)["action"]
    q["criteria"]["abstain"] = ABSTAIN_CRIT     # match SFT action vocabulary
    return q


def label_dist(sampler, compiler, state_str: str, q: dict):
    ids, label_ids, keys = compiler.compile(state_str, q)
    resp = sampler.sample(
        prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        topk_sample_logprobs=min(len(label_ids), 64)).result()
    topk = resp.sequences[0].topk_logprobs
    pairs = topk[0] if topk and topk[0] else []
    lp = {t: v for t, v in pairs}
    return ids, label_ids, keys, [lp.get(t, -30.0) for t in label_ids]


def softmax(xs: list[float], temp: float = 1.0) -> list[float]:
    peak = max(xs)
    ws = [math.exp((v - peak) / temp) for v in xs]
    tot = math.fsum(ws)
    return [w / tot for w in ws]


def verdict_answers(sampler, compiler, state_str: str, st: dict) -> dict:
    """Greedy evaluation of the non-action questions (deployment parity)."""
    out = {}
    for qkey, q in J.questions_for(st, PACK).items():
        if qkey == "action":
            continue
        _, label_ids, keys, lps = label_dist(sampler, compiler, state_str, q)
        probs = softmax(lps)
        dist = dict(zip(keys, probs))
        if q["type"] == "noul":
            out[qkey] = {"type": "noul", "noul": dist["true"]}
        else:
            ent = -math.fsum(p * math.log(p) for p in probs if p > 0)
            out[qkey] = {"type": "choice",
                         "choice": max(dist, key=dist.__getitem__),
                         "probabilities": dist,
                         "confidence": min(1.0, max(0.0, 1 - ent / math.log(len(probs))))}
    return out


def composite_of(ans: dict, st: dict, abstained: bool) -> str:
    """Mirror run_clip's verdict arbitration."""
    discont = [c["frame"] for c in st["candidates"]
               if c["revealed"] and c["revealed"].get("continuous") is False]
    fan = {t: ans.get(f"is_{t}", {}).get("noul") for t in PACK["type_desc"]}
    fan_type = max(fan, key=lambda k: fan.get(k) or 0) if any(
        v is not None for v in fan.values()) else None
    ch = ans.get("break_type", {}).get("choice")
    cor = ans.get("corrupted", {}).get("noul", 0)
    comp = fan_type if (ch == "none" and cor > 0.5 and fan_type) else ch
    if comp == "splice" and len(discont) >= 2:
        comp = "room_swap"
    elif comp == "room_swap" and len(discont) < 2:
        comp = "splice"
    return "corrupt_untyped" if abstained else comp


def rollout(cid: str, sampler, compiler, temp: float, lam: float,
            max_iters: int, rng: random.Random) -> dict:
    """One episode: returns per-step (ids, action_label_id) + reward."""
    st = J.load_clip_state(cid)
    steps, deltas, prev = [], [], None
    ans, abstained, done = {}, False, False
    judged = 0

    for it in range(max_iters):
        unjudged = [i for i, c in enumerate(st["candidates"])
                    if not c["revealed"]]
        if not unjudged and it == 0:
            break                                  # nothing to judge
        state_str = json.dumps(J.state_view(st, PACK))
        ids, label_ids, keys, lps = label_dist(
            sampler, compiler, state_str, action_question(st))
        probs = softmax(lps, temp)
        i = rng.choices(range(len(keys)), weights=probs)[0]
        steps.append((ids, label_ids[i]))
        act = keys[i]

        # belief tracking for the flatline gate needs verdict answers —
        # evaluate them only when the loop might stop (conclude) or for
        # the flatline check after reveals.
        need_ans = act in ("conclude", "abstain") or it > 0
        if need_ans:
            ans = verdict_answers(sampler, compiler, state_str, st)
            fan = {t: ans.get(f"is_{t}", {}).get("noul")
                   for t in PACK["type_desc"]}
            belief = (ans.get("corrupted", {}).get("noul", 0), fan)
            if prev is not None:
                deltas.append(abs(belief[0] - prev[0]) + sum(
                    abs((belief[1].get(t) or 0) - (prev[1].get(t) or 0))
                    for t in PACK["type_desc"]))
            prev = belief

        if act == "conclude":
            vals = sorted((v for v in fan.values() if v is not None),
                          reverse=True)
            margin = (vals[0] - vals[1]) if len(vals) > 1 else 1.0
            cor_p = ans.get("corrupted", {}).get("noul", 0)
            if cor_p < 0.5 or judged >= MIN_JUDGED or margin >= FAN_MARGIN:
                done = True
                break
            if not unjudged:
                done = True
                break                            # forced end
            if len(deltas) >= 2 and all(d < FLAT_EPS for d in deltas[-2:]):
                abstained = done = True
                break                            # gate v2 abstain
            st["candidates"][unjudged[0]]["revealed"] = \
                st["candidates"][unjudged[0]]["verdict"]
            judged += 1
            continue                             # forced judge
        if act == "abstain":
            abstained = done = True
            break
        if not unjudged:
            break
        if act.startswith("judge_"):
            j = int(act.split("_")[1])
            if 0 <= j < len(st["candidates"]) and not st["candidates"][j]["revealed"]:
                st["candidates"][j]["revealed"] = \
                    st["candidates"][j]["verdict"]
                judged += 1
                continue
        # invalid/duplicate -> reveal next unjudged (production parity)
        st["candidates"][unjudged[0]]["revealed"] = \
            st["candidates"][unjudged[0]]["verdict"]
        judged += 1

    if not ans:
        state_str = json.dumps(J.state_view(st, PACK))
        ans = verdict_answers(sampler, compiler, state_str, st)
    comp = composite_of(ans, st, abstained)
    r = (1.0 if comp == TRUTH[cid] else 0.0) - lam * judged
    return {"steps": steps, "reward": r, "composite": comp,
            "judged": judged, "truth": TRUTH[cid], "clip_id": cid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--init-state", default=None,
                    help="tinker:// weights/ checkpoint to resume; skips SFT")
    ap.add_argument("--sft-file", default=None,
                    help="s1 jsonl for inline SFT warm-start")
    ap.add_argument("--sft-steps", type=int, default=0)
    ap.add_argument("--sft-batch", type=int, default=128)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--clip-batch", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.05,
                    help="judge-call cost in the reward")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="episode clips (default: all non-heldout corpus)")
    ap.add_argument("--held-out", default="09c1414f1b")
    ap.add_argument("--log-path",
                    default=str(ROOT / "results" / "ft_tinker" / "rl"))
    args = ap.parse_args()

    os.environ.setdefault("TINKER_API_KEY", api_key())
    log_path = Path(args.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / "train.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    rng = random.Random(args.seed)
    service = tinker.ServiceClient()
    if args.init_state:
        tc = service.create_training_client_from_state(args.init_state)
        log(f"resumed weights: {args.init_state}")
    else:
        tc = service.create_lora_training_client(
            base_model=args.model, rank=args.lora_rank)
        log(f"new LoRA client: {args.model} rank={args.lora_rank}")
    tokenizer = tc.get_tokenizer()
    compiler = S1Compiler(tokenizer)

    # -------- optional inline SFT warm-start --------
    if args.sft_steps and args.sft_file:
        rows = load_rows(args.sft_file, args.held_out, "train")
        random.Random(args.seed).shuffle(rows)
        log(f"SFT warm-start: {len(rows)} rows, {args.sft_steps} steps")
        for step in range(args.sft_steps):
            batch_rows = rows[step * args.sft_batch:
                              (step + 1) * args.sft_batch]
            if not batch_rows:
                break
            batch = [d for d in (row_to_datum(compiler, r, args.max_length)
                                 for r in batch_rows) if d is not None]
            lr_mult = max(0.0, 1.0 - step / max(args.sft_steps, 1))
            adam = tinker.AdamParams(learning_rate=args.sft_lr * lr_mult,
                                     beta1=0.9, beta2=0.95, eps=1e-8)
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            tc.optim_step(adam).result()
            fb.result()
            if step % 20 == 0:
                log(f"  sft {step}/{args.sft_steps}")
        st = tc.save_state("sft_warmstart").result()
        log(f"SFT state saved: {st.path}  (reusable via --init-state)")

    clips = args.clips or sorted(
        c for c in J.CLIPS if J.CLIPS[c]["source"] != args.held_out)
    log(f"RL: {len(clips)} episode clips, G={args.group_size}, "
        f"B={args.clip_batch}, temp={args.temp}, lam={args.lam}, lr={args.lr}")

    t0 = time.time()
    for it in range(args.iters):
        sp = tc.save_weights_for_sampler(f"rl{it:04d}").result()
        sampler = service.create_sampling_client(model_path=sp.path)
        batch_clips = rng.sample(clips, min(args.clip_batch, len(clips)))

        jobs = [(cid, g) for cid in batch_clips for g in range(args.group_size)]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            eps = list(ex.map(
                lambda j: rollout(j[0], sampler, compiler, args.temp,
                                  args.lam, args.max_iters,
                                  random.Random(rng.random())), jobs))

        datums, rewards, judged_n, correct = [], [], [], 0
        by_clip: dict[str, list[dict]] = {}
        for e in eps:
            by_clip.setdefault(e["clip_id"], []).append(e)
        for cid, group in by_clip.items():
            mean_r = sum(e["reward"] for e in group) / len(group)
            for e in group:
                adv = e["reward"] - mean_r
                rewards.append(e["reward"])
                judged_n.append(e["judged"])
                correct += e["composite"] == e["truth"]
                if adv == 0.0:
                    continue
                for ids, tok in e["steps"]:
                    full = ids + [tok]
                    if len(full) > args.max_length:
                        continue
                    w = torch.zeros(len(full))
                    w[-1] = adv
                    datums.append(datum_from_model_input_weights(
                        tinker.ModelInput.from_ints(full), w, None))

        if datums:
            adam = tinker.AdamParams(learning_rate=args.lr,
                                     beta1=0.9, beta2=0.95, eps=1e-8)
            fb = tc.forward_backward(datums, loss_fn="cross_entropy")
            tc.optim_step(adam).result()
            fb.result()
        log(f"iter {it}/{args.iters} r={sum(rewards)/len(rewards):.3f} "
            f"acc={correct}/{len(eps)} judges={sum(judged_n)/len(judged_n):.1f} "
            f"datums={len(datums)} ({time.time()-t0:.0f}s)")


    out = tc.save_weights_for_sampler("rl_final").result()
    log(f"SAMPLER CHECKPOINT: {out.path}")
    (log_path / "sampler_path.txt").write_text(out.path + "\n")
    log("done")


if __name__ == "__main__":
    main()
