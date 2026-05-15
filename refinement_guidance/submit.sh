#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./submit.sh <scene_id>

Pulls the latest changes, commits blend/<scene_id>.blend, and pushes it.
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

scene_id="$1"
scene_id="${scene_id%.blend}"
scene_id="${scene_id%.mp4}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

blend_file="blend/${scene_id}.blend"

if [[ ! -f "$blend_file" ]]; then
  echo "Error: Blender file not found: $blend_file" >&2
  exit 1
fi

echo "Pulling latest changes..."
git pull --rebase --autostash

if git diff --quiet -- "$blend_file" && git diff --cached --quiet -- "$blend_file"; then
  echo "No changes found for $blend_file; nothing to commit."
  exit 0
fi

echo "Committing $blend_file..."
git add -- "$blend_file"
git commit -m "$scene_id"

echo "Pushing to remote..."
git push

echo "Done. Remember to mark your name in the Notion Refiner column."
