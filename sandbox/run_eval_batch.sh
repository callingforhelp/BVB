#!/usr/bin/env bash
# Batch Stage-2 unit-test evaluation over completed Stage-1 runs.
#
# Usage (from sandbox/):
#   caffeinate -i ./run_eval_batch.sh
#   ./run_eval_batch.sh --dry-run
#   ./run_eval_batch.sh mini-harness-gpt-5.6-terra-reasoning-high-run01
#   ./run_eval_batch.sh --all-complete   # every local run with >=278 blends
#
# Skips runs that already have summary.json. Interrupted shards resume
# automatically on the next pass.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd .. && pwd)"
EVAL_PY="${ROOT}/eval/unit_test_metric.py"
MERGE_PY="${ROOT}/eval/merge_eval_shards.py"
PYTHON="${ROOT}/sandbox/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

JUDGE_MODEL="gpt-5.4-mini"
REASONING_EFFORT="none"
SHARDS=8
MIN_BLENDS=288
MIN_BLENDS_SET=0
STOP_ON_ERROR=1
DRY_RUN=0
FORCE=0
ALL_COMPLETE=0
RESUME=1
LOG_DIR="log"
declare -a RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./run_eval_batch.sh [options] [run-dir ...]

Options:
  --all-complete     Queue every results/mini-harness-* with enough blends
                     (default threshold 278 = full 288 minus at most 10;
                      override with --min-blends)
  --model NAME       Judge model (default: gpt-5.4-mini)
  --reasoning VALUE  Judge reasoning effort (default: none)
  --shards N         Parallel scene shards (default: 8)
  --min-blends N     Skip runs with fewer blends
                     (default: 288, or 278 with --all-complete)
  --no-stop-on-error Continue a run after a scene judge/API error
  --no-resume        Never resume; always start shard outputs fresh
  --force            Re-evaluate even if summary.json already exists
  --dry-run          Print the planned queue and exit
  -h, --help         Show this help

If no run dirs are given and --all-complete is not set, uses the default
priority queue (terra curve, then luna, then cross-model baselines).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-complete) ALL_COMPLETE=1; shift ;;
    --model) JUDGE_MODEL="${2:?}"; shift 2 ;;
    --reasoning) REASONING_EFFORT="${2:?}"; shift 2 ;;
    --shards) SHARDS="${2:?}"; shift 2 ;;
    --min-blends) MIN_BLENDS="${2:?}"; MIN_BLENDS_SET=1; shift 2 ;;
    --no-stop-on-error) STOP_ON_ERROR=0; shift ;;
    --no-resume) RESUME=0; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; RUN_ARGS+=("$@"); break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *) RUN_ARGS+=("$1"); shift ;;
  esac
done

# --all-complete also accepts near-complete runs missing at most 10 blends.
if [[ "$ALL_COMPLETE" -eq 1 && "$MIN_BLENDS_SET" -eq 0 ]]; then
  MIN_BLENDS=278
fi

# Default priority after sol curve is finished.
DEFAULT_QUEUE=(
  mini-harness-gpt-5.6-terra-reasoning-high-run01
  mini-harness-gpt-5.6-terra-reasoning-xhigh-run01
  mini-harness-gpt-5.6-terra-reasoning-medium-run01
  mini-harness-gpt-5.6-terra-reasoning-low-run01
  mini-harness-gpt-5.6-luna-reasoning-high-run01
  mini-harness-gpt-5.6-luna-reasoning-medium-run01
  mini-harness-gpt-5.6-luna-reasoning-low-run01
  mini-harness-claude-opus-4-8-reasoning-high-run01
  mini-harness-claude-opus-4-8-run01
  mini-harness-claude-sonnet-5-reasoning-high-run01
  mini-harness-gemini-3.1-pro-preview-reasoning-high-run01
  mini-harness-gpt-5.5-reasoning-high-run01
  mini-harness-gpt-5.5-reasoning-medium-run01
  mini-harness-gpt-5.5-reasoning-low-run01
  mini-harness-gpt-5.5-run01
  mini-harness-qwen3.5-397b-a17b-run01
  mini-harness-qwen3-vl-235b-a22b-thinking-run01
  mini-harness-grok-4.5-reasoning-high-run01
  mini-harness-minimax-m3-run01
  mini-harness-seed-2.0-lite-run01
  mini-harness-seed-2.0-mini-run01
  mini-harness-kimi-k2.5-run01
  mini-harness-glm-5v-turbo-run01
  mini-harness-glm-4.6v-run01
)

blend_count() {
  local run_dir="$1"
  if [[ ! -d "$run_dir/blends" ]]; then
    echo 0
    return
  fi
  find "$run_dir/blends" -maxdepth 1 -type f -name '*.blend' | wc -l | tr -d ' '
}

short_log_name() {
  local name="$1"
  name="${name#mini-harness-}"
  name="${name%-run01}"
  printf '%s' "$name" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_'
}

build_queue() {
  declare -a raw=()
  if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
    raw=("${RUN_ARGS[@]}")
  elif [[ "$ALL_COMPLETE" -eq 1 ]]; then
    while IFS= read -r path; do
      raw+=("$(basename "$path")")
    done < <(find results -maxdepth 1 -type d -name 'mini-harness-*' | sort)
  else
    raw=("${DEFAULT_QUEUE[@]}")
  fi

  QUEUE=()
  for item in "${raw[@]}"; do
    local run_dir
    if [[ "$item" == results/* || "$item" == /* ]]; then
      run_dir="$item"
    else
      run_dir="results/$item"
    fi
    local name
    name="$(basename "$run_dir")"

    if [[ ! -d "$run_dir" ]]; then
      echo "[skip] missing: $name"
      continue
    fi
    local blends
    blends="$(blend_count "$run_dir")"
    if [[ "$blends" -lt "$MIN_BLENDS" ]]; then
      echo "[skip] blends=$blends < $MIN_BLENDS: $name"
      continue
    fi
    if [[ "$FORCE" -eq 0 && -f "$run_dir/summary.json" ]]; then
      echo "[skip] already evaluated: $name"
      continue
    fi
    QUEUE+=("$run_dir")
  done
}

evaluate_run() {
  local run_dir="$1"
  local name slug cache_dir log_prefix
  name="$(basename "$run_dir")"
  slug="$(short_log_name "$name")"
  cache_dir="$run_dir/introspection-cache"
  log_prefix="$LOG_DIR/eval-${slug}.s${SHARDS}"
  mkdir -p "$LOG_DIR" "$cache_dir"

  echo
  echo "======== evaluating $name ========"
  echo "shards=$SHARDS judge=$JUDGE_MODEL reasoning=$REASONING_EFFORT"

  local -a pids=()
  local i
  for i in $(seq 1 "$SHARDS"); do
    local shard_out="$run_dir/unit_tests.shard-$i-of-$SHARDS.jsonl"
    local shard_cfg="${shard_out}.config.json"
    local -a cmd=(
      "$PYTHON" "$EVAL_PY"
      --run "$run_dir"
      --mode execute
      --model "$JUDGE_MODEL"
      --reasoning-effort "$REASONING_EFFORT"
      --cache-dir "$cache_dir"
      --shard "$i/$SHARDS"
    )
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      cmd+=(--stop-on-error)
    fi
    # --resume requires an existing sidecar config; fresh shards must omit it.
    if [[ "$RESUME" -eq 1 && -f "$shard_cfg" ]]; then
      cmd+=(--resume)
    fi
    nohup "${cmd[@]}" >"${log_prefix}_$i.log" 2>&1 &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[error] one or more shards failed for $name; see ${log_prefix}_*.log" >&2
    return 1
  fi

  "$PYTHON" "$MERGE_PY" --run "$run_dir" --cleanup
  echo "[done] $name -> $run_dir/summary.json"
}

QUEUE=()
build_queue

echo "Queued ${#QUEUE[@]} run(s):"
if [[ ${#QUEUE[@]} -gt 0 ]]; then
  for run_dir in "${QUEUE[@]}"; do
    echo "  - $(basename "$run_dir")  (blends=$(blend_count "$run_dir"))"
  done
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ ${#QUEUE[@]} -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

failed=0
for run_dir in "${QUEUE[@]}"; do
  if ! evaluate_run "$run_dir"; then
    failed=1
    echo "[abort] stopping batch after failure in $(basename "$run_dir")" >&2
    break
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo
echo "Batch complete."
