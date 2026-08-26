#!/usr/bin/env bash
# Batch V-JEPA vision-sim over Stage-1 runs, one run at a time, 1 process per GPU.
#
# Usage (from BVB/):
#   ./eval/run_vjepa_batch.sh --all          # paper Table main_results only
#   ./eval/run_vjepa_batch.sh --dry-run --all
#   ./eval/run_vjepa_batch.sh mini-harness-grok-4.5-reasoning-high-run01
#   ./eval/run_vjepa_batch.sh --gpus 8 --num-frames 64 --all
#
# Skips runs that already have vision_sim_summary.json. Interrupted shards resume
# with --resume. Prefers runs that already have camera_renders/ (no Blender needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EVAL_PY="${ROOT}/eval/vjepa_sim_metric.py"
MERGE_PY="${ROOT}/eval/merge_vision_sim_shards.py"
PYTHON="$(command -v python3)"
VSI_BENCH="${ROOT}/VSI-Bench"
RESULTS="${ROOT}/sandbox/results"

GPUS=8
NUM_FRAMES=64
DTYPE=bfloat16
DEVICE=cuda
MIN_RENDERS=1
MIN_RENDERS_SET=0
FORCE=0
DRY_RUN=0
ALL=0
ALL_LOCAL=0
RESUME=1
STOP_ON_ERROR=0
REQUIRE_RENDERS=1
LOG_DIR="${ROOT}/sandbox/log"
declare -a RUN_ARGS=()

# Paper Table~\ref{tab:main_results} order (Vision-Level column). Not in table → skip.
TABLE_QUEUE=(
  mini-harness-grok-4.5-reasoning-high-run01
  mini-harness-gpt-5.2-reasoning-high-run01
  mini-harness-gpt-5.4-run01
  mini-harness-claude-opus-4-6-run01
  mini-harness-gpt-5.6-sol-reasoning-xhigh-run01
  mini-harness-qwen3.5-397b-a17b-run01
  mini-harness-gpt-5.5-reasoning-low-run01
  mini-harness-gemini-3.1-pro-preview-reasoning-high-run01
  mini-harness-claude-sonnet-4-6-run01
  mini-harness-gpt-5.6-sol-reasoning-high-run01
  mini-harness-gpt-5.5-reasoning-medium-run01
  mini-harness-claude-sonnet-5-reasoning-high-run01
  mini-harness-qwen3.6-plus-run01
  mini-harness-claude-sonnet-4-6-reasoning-high-run01
  mini-harness-glm-5v-turbo-run01
  mini-harness-claude-sonnet-5-run01
  mini-harness-minimax-m3-run01
  mini-harness-gpt-5.5-reasoning-high-run01
  mini-harness-claude-opus-4-7-run01
  mini-harness-gpt-5.6-terra-reasoning-xhigh-run01
  mini-harness-claude-opus-4-8-reasoning-high-run01
  mini-harness-gpt-5.6-sol-reasoning-medium-run01
  mini-harness-qwen3.5-27b-run01
  mini-harness-claude-opus-4-7-reasoning-high-run01
  mini-harness-claude-opus-4-8-run01
  mini-harness-qwen3.5-122b-a10b-run01
  mini-harness-gpt-5.6-sol-run01
  mini-harness-gpt-5.6-terra-reasoning-high-run01
  mini-harness-kimi-k2.5-run01
  mini-harness-qwen3.5-35b-a3b-run01
  mini-harness-gpt-5.6-sol-reasoning-low-run01
  mini-harness-gpt-5.2-run01
  mini-harness-gpt-5.6-luna-reasoning-high-run01
  mini-harness-glm-4.6v-run01
  mini-harness-gemini-3-flash-preview-reasoning-high-run01
  mini-harness-gpt-5.6-terra-reasoning-medium-run01
  mini-harness-grok-4.3-reasoning-high-run01
  mini-harness-gpt-5.6-terra-reasoning-low-run01
  mini-harness-gpt-5.5-run01
  mini-harness-gpt-5.6-luna-reasoning-medium-run01
  mini-harness-qwen3-vl-235b-a22b-thinking-run01
  mini-harness-gpt-5.6-luna-reasoning-low-run01
  mini-harness-seed-2.0-mini-run01
  mini-harness-gpt-5.4-mini-run01
  mini-harness-grok-4.3-run01
  mini-harness-seed-2.0-lite-run01
)

usage() {
  cat <<'EOF'
Usage: ./eval/run_vjepa_batch.sh [options] [run-dir ...]

Options:
  --all              Queue paper Table main_results runs only (46 models)
  --all-local        Queue every local mini-harness-* (not just the table)
  --gpus N           Parallel scene shards / GPUs (default: 8)
  --num-frames N     Frames per clip (default: 64)
  --dtype DTYPE      bfloat16|float16|float32 (default: bfloat16)
  --min-renders N    Skip runs with fewer camera_renders (default: 1)
  --allow-no-renders Also queue runs missing camera_renders (needs Blender)
  --resume           Resume incomplete shards (default; accepted for clarity)
  --no-resume        Never resume; overwrite shard outputs
  --force            Re-score even if vision_sim_summary.json exists
  --stop-on-error    Abort a shard on the first non-ok scene
  --continue-on-fail Do not abort the whole batch if one run's shards fail
  --dry-run          Print the planned queue and exit
  -h, --help         Show this help

If no run dirs are given and neither --all nor --all-local is set, prints help.
EOF
}

CONTINUE_ON_FAIL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL=1; shift ;;
    --all-local) ALL_LOCAL=1; shift ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --num-frames) NUM_FRAMES="${2:?}"; shift 2 ;;
    --dtype) DTYPE="${2:?}"; shift 2 ;;
    --min-renders) MIN_RENDERS="${2:?}"; MIN_RENDERS_SET=1; shift 2 ;;
    --allow-no-renders) REQUIRE_RENDERS=0; shift ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --force) FORCE=1; shift ;;
    --stop-on-error) STOP_ON_ERROR=1; shift ;;
    --continue-on-fail) CONTINUE_ON_FAIL=1; shift ;;
    --abort-on-fail) CONTINUE_ON_FAIL=0; shift ;;
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

if [[ "$ALL" -eq 0 && "$ALL_LOCAL" -eq 0 && ${#RUN_ARGS[@]} -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ "$MIN_RENDERS_SET" -eq 0 ]]; then
  MIN_RENDERS=1
fi

render_count() {
  local run_dir="$1"
  if [[ ! -d "$run_dir/camera_renders" ]]; then
    echo 0
    return
  fi
  find "$run_dir/camera_renders" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.png' \) | wc -l | tr -d ' '
}

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
  elif [[ "$ALL" -eq 1 ]]; then
    raw=("${TABLE_QUEUE[@]}")
  elif [[ "$ALL_LOCAL" -eq 1 ]]; then
    while IFS= read -r path; do
      raw+=("$(basename "$path")")
    done < <(find "$RESULTS" -maxdepth 1 -type d -name 'mini-harness-*' | sort)
  fi

  QUEUE=()
  for item in "${raw[@]}"; do
    local run_dir
    if [[ "$item" == /* ]]; then
      run_dir="$item"
    elif [[ "$item" == sandbox/results/* ]]; then
      run_dir="${ROOT}/$item"
    elif [[ "$item" == results/* ]]; then
      run_dir="${ROOT}/sandbox/$item"
    else
      run_dir="${RESULTS}/$item"
    fi

    local name
    name="$(basename "$run_dir")"

    if [[ ! -d "$run_dir" ]]; then
      echo "[skip] missing: $name"
      continue
    fi
    local blends renders
    blends="$(blend_count "$run_dir")"
    renders="$(render_count "$run_dir")"
    if [[ "$REQUIRE_RENDERS" -eq 1 && "$renders" -lt "$MIN_RENDERS" ]]; then
      echo "[skip] camera_renders=$renders < $MIN_RENDERS: $name"
      continue
    fi
    if [[ "$blends" -eq 0 && "$renders" -eq 0 ]]; then
      echo "[skip] empty run: $name"
      continue
    fi
    if [[ "$FORCE" -eq 0 && -f "$run_dir/vision_sim_summary.json" ]]; then
      echo "[skip] already scored: $name"
      continue
    fi
    QUEUE+=("$run_dir")
  done
}

warm_model_once() {
  # Avoid 8 processes racing to download ViT-G on first launch.
  local marker="${HOME}/.cache/huggingface/hub/.bvb_vjepa_warmed"
  if [[ -f "$marker" ]]; then
    return 0
  fi
  echo "[warm] downloading / loading V-JEPA once before sharding..."
  mkdir -p "$(dirname "$marker")"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" - <<'PY'
from vjepa_sim_metric import DEFAULT_MODEL, load_encoder
load_encoder(DEFAULT_MODEL, "cuda", "bfloat16")
print("warm ok:", DEFAULT_MODEL)
PY
  touch "$marker"
}

score_run() {
  local run_dir="$1"
  local name slug log_prefix
  name="$(basename "$run_dir")"
  slug="$(short_log_name "$name")"
  log_prefix="${LOG_DIR}/vjepa-${slug}.g${GPUS}"
  mkdir -p "$LOG_DIR"

  echo
  echo "======== vjepa $name ========"
  echo "gpus=$GPUS frames=$NUM_FRAMES dtype=$DTYPE renders=$(render_count "$run_dir")"

  local -a pids=()
  local i
  for i in $(seq 1 "$GPUS"); do
    local gpu=$((i - 1))
    local shard_out="$run_dir/vision_sim.shard-$i-of-$GPUS.jsonl"
    local -a cmd=(
      "$PYTHON" "$EVAL_PY"
      --run "$run_dir"
      --vsi-bench "$VSI_BENCH"
      --device "$DEVICE"
      --dtype "$DTYPE"
      --num-frames "$NUM_FRAMES"
      --shard "$i/$GPUS"
    )
    if [[ "$RESUME" -eq 1 ]]; then
      cmd+=(--resume)
    elif [[ -f "$shard_out" ]]; then
      rm -f "$shard_out" \
        "$run_dir/vision_sim_summary.shard-$i-of-$GPUS.json"
    fi
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      cmd+=(--stop-on-error)
    fi
    echo "[launch] GPU $gpu shard $i/$GPUS -> ${log_prefix}_$i.log"
    (
      cd "${ROOT}/eval"
      CUDA_VISIBLE_DEVICES="$gpu" nohup "${cmd[@]}" >"${log_prefix}_$i.log" 2>&1
    ) &
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
  echo "[done] $name -> $run_dir/vision_sim_summary.json"
}

QUEUE=()
build_queue

echo "Queued ${#QUEUE[@]} run(s)  (gpus=$GPUS python=$PYTHON):"
if [[ ${#QUEUE[@]} -gt 0 ]]; then
  for run_dir in "${QUEUE[@]}"; do
    echo "  - $(basename "$run_dir")  (blends=$(blend_count "$run_dir") renders=$(render_count "$run_dir"))"
  done
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ ${#QUEUE[@]} -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

# Import path for warm + merge helpers.
export PYTHONPATH="${ROOT}/eval${PYTHONPATH:+:$PYTHONPATH}"

warm_model_once

failed=0
for run_dir in "${QUEUE[@]}"; do
  if ! score_run "$run_dir"; then
    failed=1
    if [[ "$CONTINUE_ON_FAIL" -eq 1 ]]; then
      echo "[warn] continuing after failure in $(basename "$run_dir")" >&2
      continue
    fi
    echo "[abort] stopping batch after failure in $(basename "$run_dir")" >&2
    break
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo
echo "Batch complete."
