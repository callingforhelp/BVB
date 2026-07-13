#!/usr/bin/env bash
# Generic BVB Stage-1 runner for any litellm model.
#
# Usage:
#   ./run_model.sh <litellm-model> [extra run_agent.py flags]
#
# Examples:
#   OPENAI_API_KEY=sk-...     ./run_model.sh gpt-5.5
#   ANTHROPIC_API_KEY=sk-...  ./run_model.sh anthropic/claude-sonnet-4.6
#   GEMINI_API_KEY=...        ./run_model.sh gemini/gemini-2.5-pro --limit 3
#
# Output goes to results/mini-harness-<model-slug>-run01/ by default. If
# --reasoning is passed, output goes to
# results/mini-harness-<model-slug>-reasoning-<effort>-run01/ so default and
# non-default reasoning runs never collide.
# The API key env var must match the provider (litellm reads it automatically).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?Usage: ./run_model.sh <litellm-model> [extra flags]}"
shift

# model -> filesystem-safe slug, dropping any provider/ prefix for brevity
BASE_MODEL="${MODEL##*/}"
SLUG=$(printf '%s' "$BASE_MODEL" | tr -c 'A-Za-z0-9.-' '-')
REASONING=""
PREV=""
for ARG in "$@"; do
  if [[ "$PREV" == "--reasoning" ]]; then
    REASONING="$ARG"
    break
  fi
  case "$ARG" in
    --reasoning=*)
      REASONING="${ARG#--reasoning=}"
      break
      ;;
  esac
  PREV="$ARG"
done
if [[ -z "$REASONING" ]]; then
  case "$BASE_MODEL" in
    gpt-5.5) REASONING="medium" ;;
    gemini-3.1-pro-preview|gemini-3.1-pro) REASONING="high" ;;
    *) REASONING="none" ;;
  esac
fi
if [[ "$REASONING" != "none" ]]; then
  REASONING_SLUG=$(printf '%s' "$REASONING" | tr -c 'A-Za-z0-9.-' '-')
  OUT="results/mini-harness-${SLUG}-reasoning-${REASONING_SLUG}-run01"
else
  OUT="results/mini-harness-${SLUG}-run01"
fi

.venv/bin/python run_agent.py \
  --model "$MODEL" \
  --output "$OUT" \
  --cost-limit 3.0 \
  --resume \
  "$@"
