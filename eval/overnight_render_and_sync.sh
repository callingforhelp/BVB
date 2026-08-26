#!/usr/bin/env bash
# Overnight: render all Stage-1 camera videos, then sync to Hugging Face.
# Safe for existing results: only writes <run>/camera_renders/<id>.mp4.
#
# Usage (from repo root):
#   nohup caffeinate -i bash eval/overnight_render_and_sync.sh &
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/sandbox/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi
LOG_DIR="${ROOT}/sandbox/log"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/camera_render_overnight_${STAMP}.log"

WORKERS="${WORKERS:-6}"
NUM_FRAMES="${NUM_FRAMES:-64}"

exec >>"$LOG" 2>&1
echo "log=$LOG"
echo "start=$(date) workers=$WORKERS num_frames=$NUM_FRAMES pid=$$"

echo "[smoke] rendering 1 scene"
"$PY" "${ROOT}/eval/batch_render_camera.py" \
  --all-runs --resume --workers 1 --limit 1 --num-frames "$NUM_FRAMES"
echo "[smoke] done rc=$?"

echo "[render] all runs start=$(date)"
# IMPORTANT: do not pipe this through tee — multiprocessing + pipes can kill workers.
"$PY" -u "${ROOT}/eval/batch_render_camera.py" \
  --all-runs --resume --workers "$WORKERS" --num-frames "$NUM_FRAMES" \
  --progress-every 20
RENDER_RC=$?
echo "[render] exit_code=$RENDER_RC end=$(date)"

echo "[count]"
"$PY" - <<'PY'
from pathlib import Path
import json
root = Path("sandbox/results")
total = ok = err = 0
per = {}
for run in sorted(root.glob("mini-harness-*")):
    blends = list((run / "blends").glob("*.blend")) if (run / "blends").is_dir() else []
    vids = list((run / "camera_renders").glob("*.mp4")) if (run / "camera_renders").is_dir() else []
    errs = list((run / "camera_renders").glob("*.error.json")) if (run / "camera_renders").is_dir() else []
    total += len(blends)
    ok += len(vids)
    err += len(errs)
    per[run.name] = {"blends": len(blends), "videos": len(vids), "errors": len(errs)}
print(json.dumps({"total_blends": total, "videos": ok, "errors": err, "runs": per}, indent=2))
PY

echo "[sync] uploading camera_renders video.mp4 to HF start=$(date)"
"$PY" "${ROOT}/scripts/sync_eval_results.py" --camera-renders-only
SYNC_RC=$?
echo "[sync] exit_code=$SYNC_RC end=$(date)"

echo "[done] render_rc=$RENDER_RC sync_rc=$SYNC_RC $(date)"
exit 0
