#!/usr/bin/env bash
# Batch Dual VQA over Stage-1 runs that already have camera_renders/.
#
# Usage (from BVB/):
#   ./eval/run_dual_vqa_batch.sh --all-local
#   ./eval/run_dual_vqa_batch.sh mini-harness-gpt-5.6-sol-reasoning-xhigh-run01
#   ./eval/run_dual_vqa_batch.sh --dry-run --all-local
#
# Skips runs that already have dual_vqa_summary.json unless --force.
# Shared originals are filled once via --ensure-originals-only before the queue.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EVAL_PY="${ROOT}/eval/dual_vqa_metric.py"
PYTHON="$(command -v python3)"
RESULTS="${ROOT}/sandbox/results"
LOG_DIR="${ROOT}/sandbox/log"

MODEL="${DUAL_VQA_MODEL:-gpt-5.4-mini}"
N_FRAMES="${DUAL_VQA_N_FRAMES:-16}"
WORKERS="${DUAL_VQA_WORKERS:-6}"
FORCE=0
DRY_RUN=0
ALL_LOCAL=0
RESUME=1
declare -a RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./eval/run_dual_vqa_batch.sh [options] [run-dir ...]

Options:
  --all-local     Queue every local mini-harness-* with camera_renders/
  --model NAME    Judge VLM (default: gpt-5.4-mini)
  --n-frames N    Sparse frames per video (default: 16)
  --workers N     Concurrent API calls (default: 6)
  --force         Re-score even if dual_vqa_summary.json exists
  --no-resume     Do not resume partial dual_vqa.jsonl
  --dry-run       Print the planned queue and exit
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-local) ALL_LOCAL=1; shift ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --n-frames) N_FRAMES="${2:?}"; shift 2 ;;
    --workers) WORKERS="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *) RUN_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$LOG_DIR"

queue=()
if (( ${#RUN_ARGS[@]} > 0 )); then
  for name in "${RUN_ARGS[@]}"; do
    if [[ -d "$name" ]]; then
      queue+=("$(basename "$name")")
    else
      queue+=("$name")
    fi
  done
elif (( ALL_LOCAL )); then
  for dir in "$RESULTS"/mini-harness-*; do
    [[ -d "$dir/camera_renders" ]] || continue
    queue+=("$(basename "$dir")")
  done
else
  usage
  exit 2
fi

filtered=()
for run in "${queue[@]}"; do
  summary="$RESULTS/$run/dual_vqa_summary.json"
  if (( FORCE == 0 )) && [[ -f "$summary" ]]; then
    echo "[skip] $run (dual_vqa_summary.json exists)"
    continue
  fi
  if [[ ! -d "$RESULTS/$run/camera_renders" ]]; then
    echo "[skip] $run (no camera_renders/)"
    continue
  fi
  filtered+=("$run")
done

echo "queue=${#filtered[@]} model=$MODEL frames=$N_FRAMES workers=$WORKERS"
if (( DRY_RUN )); then
  printf '  %s\n' "${filtered[@]}"
  exit 0
fi

if (( ${#filtered[@]} == 0 )); then
  echo "nothing to do"
  exit 0
fi

echo "[batch] ensuring shared originals…"
"$PYTHON" "$EVAL_PY" \
  --ensure-originals-only \
  --results-dir "$RESULTS" \
  --model "$MODEL" \
  --n-frames "$N_FRAMES" \
  --workers "$WORKERS"

for run in "${filtered[@]}"; do
  log="$LOG_DIR/dual_vqa_${run}.log"
  echo "[batch] scoring $run → $log"
  args=(
    --run "$run"
    --results-dir "$RESULTS"
    --model "$MODEL"
    --n-frames "$N_FRAMES"
    --workers "$WORKERS"
  )
  if (( FORCE )); then
    args+=(--force)
  fi
  if (( RESUME == 0 )); then
    args+=(--no-resume)
  fi
  "$PYTHON" "$EVAL_PY" "${args[@]}" 2>&1 | tee -a "$log"
done

echo "[batch] done"
