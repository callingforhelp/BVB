#!/bin/bash
# Serial LOSO cross-validation over the remaining 4 folds.
# For each held-out source: train LoRA ckpt, then eval_s1 on that source,
# score with jev_score. Logs per fold under results/ft_tinker/loso_<src>/
# and results/evals1/loso<src>_tuned/.
set -u
cd "$(dirname "$0")"
PY_VENV=/Users/oldap/s1-spike/.venv/bin/python

for src in 0d2ee665be 13c3e046d7 1ada7a0617 21d970d8de; do
  echo "=== fold $src: train ==="
  python3 ft_tinker.py \
    --data "results/ft_dataset/plain_s1/loso_${src}/train.jsonl" \
    --val  "results/ft_dataset/plain_s1/loso_${src}/val.jsonl" \
    --epochs 1 --batch-size 128 --val-every 40 --sample-every 80 \
    --log-path "results/ft_tinker/loso_${src}" \
    > "results/ft_tinker/loso_${src}.driver.log" 2>&1
  ckpt=$(cat "results/ft_tinker/loso_${src}/sampler_path.txt" 2>/dev/null)
  if [ -z "$ckpt" ]; then echo "fold $src: NO CHECKPOINT, skipping eval"; continue; fi
  echo "=== fold $src: eval ($ckpt) ==="
  HF_TOKEN=$HF_TOKEN "$PY_VENV" eval_s1.py --model-path "$ckpt" \
    --source "$src" --out "results/evals1/loso_${src}_tuned" --workers 4 \
    >> "results/ft_tinker/loso_${src}.driver.log" 2>&1
  echo "=== fold $src: score ==="
  "$PY_VENV" jev_score.py "results/evals1/loso_${src}_tuned" \
    | tee "results/evals1/loso_${src}_tuned.score.txt"
done
echo "ALL FOLDS DONE"
