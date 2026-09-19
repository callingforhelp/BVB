"""Phase 3: Tinker SFT on the OpenJev backbone (Qwen/Qwen3.6-35B-A3B).

Reads per-question s1 JSONL rows emitted by ft_s1.py, compiles each to the
exact OpenJev serving format (s1_compile.S1Compiler — a port of OpenJev's
PromptCompiler), and trains a LoRA adapter with cross-entropy on the single
answer-label token. Produces a tinker:// sampler checkpoint that can serve
behind a /v1/systemone-shaped adapter (s1_serve.py) or be downloaded for
sglang.

Run under SYSTEM python3 (tinker + tinker_cookbook installed there), not
the project venv. TINKER_API_KEY env var or ~/.dsh/.credentials.yaml ref.

    python3 ft_tinker.py --data results/ft_dataset/plain_s1/all.jsonl \
        --val results/ft_dataset/plain_s1/loso_09c1414f1b/val.jsonl \
        --held-out 09c1414f1b --epochs 1 --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import tinker
from tinker_cookbook.supervised.common import (
    compute_mean_nll,
    datum_from_model_input_weights,
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from s1_compile import S1Compiler, row_to_example  # noqa: E402

MODEL = "Qwen/Qwen3.6-35B-A3B"


def api_key() -> str:
    if os.environ.get("TINKER_API_KEY"):
        return os.environ["TINKER_API_KEY"]
    import yaml
    creds = yaml.safe_load(
        open(Path.home() / ".dsh" / ".credentials.yaml"))
    return creds["refs"]["TINKER_API_KEY"]


def load_rows(path: str, held_out: str | None, split: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    if held_out:
        if split == "train":
            rows = [r for r in rows if r["source"] != held_out]
        elif split == "val":
            rows = [r for r in rows if r["source"] == held_out]
    return rows


def row_to_datum(compiler: S1Compiler, row: dict, max_length: int):
    ex = row_to_example(compiler, row)
    if ex is None:
        return None
    ids, label_id = ex
    full = ids + [label_id]
    if len(full) > max_length:
        return None                      # don't train truncated labels
    mi = tinker.ModelInput.from_ints(full)
    w = torch.zeros(len(full))
    w[-1] = 1.0                          # loss on the label token only
    return datum_from_model_input_weights(mi, w, None)


def label_token_metrics(fwd_result, batch, compiler, rows_in_batch):
    """argmax-over-labels accuracy at the answer position."""
    # forward_backward returns per-position logprobs of the TARGET tokens,
    # not full vocab — accuracy needs a sampler; report mean NLL instead.
    logprobs = [x["logprobs"] for x in fwd_result.loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in batch]
    return compute_mean_nll(logprobs, weights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None,
                    help="held-out val jsonl (or reuse --data with --held-out)")
    ap.add_argument("--held-out", default=None,
                    help="source id to exclude from train / keep for val")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train rows (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-path", default=str(ROOT / "results" / "ft_tinker"))
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=0,
                    help="also save a sampler checkpoint every N steps so "
                         "progress can be probed mid-run (OpenAI-compat API "
                         "or SamplingClient)")
    ap.add_argument("--val-every", type=int, default=0,
                    help="run val-NLL every N steps")
    ap.add_argument("--resume", default=None,
                    help="tinker:// state checkpoint to resume")
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

    service = tinker.ServiceClient()
    if args.resume:
        tc = service.create_training_client_from_state_with_optimizer(
            args.resume)
        log(f"resumed training client from {args.resume}")
    else:
        tc = service.create_lora_training_client(
            base_model=args.model, rank=args.lora_rank)
        log(f"new LoRA client: {args.model} rank={args.lora_rank}")

    tokenizer = tc.get_tokenizer()
    compiler = S1Compiler(tokenizer)
    log(f"compiler ready: {len(compiler.labels)} labels, "
        f"ending={compiler.ending!r}")

    train_rows = load_rows(args.data, args.held_out, "train")
    random.Random(args.seed).shuffle(train_rows)
    if args.limit:
        train_rows = train_rows[: args.limit]
    val_rows = []
    if args.val:
        val_rows = load_rows(args.val, args.held_out, "val")
    elif args.held_out:
        val_rows = load_rows(args.data, args.held_out, "val")
    log(f"train rows: {len(train_rows)} | val rows: {len(val_rows)}")

    n_batches = len(train_rows) // args.batch_size
    total_steps = n_batches * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    log(f"{n_batches} batches/epoch x {args.epochs} epochs "
        f"= {total_steps} steps")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for b in range(n_batches):
            if step >= total_steps:
                break
            rows = train_rows[b * args.batch_size:
                              (b + 1) * args.batch_size]
            batch = [d for d in (row_to_datum(compiler, r, args.max_length)
                                 for r in rows) if d is not None]
            if not batch:
                continue
            lr_mult = max(0.0, 1.0 - step / max(total_steps, 1))
            adam = tinker.AdamParams(learning_rate=args.lr * lr_mult,
                                     beta1=0.9, beta2=0.95, eps=1e-8)
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            op = tc.optim_step(adam)
            fb_res = fb.result()
            op.result()
            nll = label_token_metrics(fb_res, batch, compiler, rows)
            if step % 10 == 0 or step == total_steps - 1:
                log(f"epoch {epoch} step {step}/{total_steps} "
                    f"nll={nll:.4f} lr={args.lr * lr_mult:.2e} "
                    f"tok={sum(d.model_input.length for d in batch)} "
                    f"({time.time() - t0:.0f}s)")
            if args.val_every and val_rows and step % args.val_every == 0 \
                    and step > 0:
                vd = [d for d in (row_to_datum(compiler, r, args.max_length)
                                  for r in val_rows[:512]) if d is not None]
                vf = tc.forward(vd, loss_fn="cross_entropy").result()
                vnll = label_token_metrics(vf, vd, compiler, None)
                log(f"  val nll={vnll:.4f} on {len(vd)} rows")
            if args.save_every and step % args.save_every == 0 and step > 0:
                p = tc.save_state(f"step{step:06d}").result()
                log(f"  saved state: {p.path}")
            if args.sample_every and step % args.sample_every == 0 \
                    and step > 0:
                p = tc.save_weights_for_sampler(
                    f"step{step:06d}").result()
                log(f"  saved sampler: {p.path}")
            step += 1

    out = tc.save_weights_for_sampler("final").result()
    log(f"SAMPLER CHECKPOINT: {out.path}")
    (log_path / "sampler_path.txt").write_text(out.path + "\n")
    log("done")


if __name__ == "__main__":
    main()
