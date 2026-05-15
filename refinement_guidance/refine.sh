#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./refine.sh <scene_id>

Opens blend/<scene_id>.blend in Blender and the matching source video from
VSI-Bench/{arkitscenes,scannet,scannetpp}/<scene_id>.mp4.
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

blend_file="$repo_root/blend/${scene_id}.blend"
dataset_root="$repo_root/VSI-Bench"

if [[ ! -f "$blend_file" ]]; then
  echo "Error: Blender file not found: $blend_file" >&2
  exit 1
fi

if [[ ! -d "$dataset_root" ]]; then
  echo "Error: VSI-Bench folder not found: $dataset_root" >&2
  echo "Clone VSI-Bench into the BVB repository root first:" >&2
  echo "  git clone https://huggingface.co/datasets/nyu-visionx/VSI-Bench" >&2
  exit 1
fi

video_file=""
for source in arkitscenes scannet scannetpp; do
  candidate="$dataset_root/$source/${scene_id}.mp4"
  if [[ -f "$candidate" ]]; then
    video_file="$candidate"
    break
  fi
done

if [[ -z "$video_file" ]]; then
  echo "Error: source video not found for ID: $scene_id" >&2
  echo "Expected one of:" >&2
  echo "  $dataset_root/arkitscenes/${scene_id}.mp4" >&2
  echo "  $dataset_root/scannet/${scene_id}.mp4" >&2
  echo "  $dataset_root/scannetpp/${scene_id}.mp4" >&2
  exit 1
fi

echo "Opening Blender file: $blend_file"
echo "Opening source video:  $video_file"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if open -Ra "Blender"; then
    open -a "Blender" "$blend_file"
  else
    echo "Warning: Blender app was not found by macOS. Opening the .blend file with the default app." >&2
    open "$blend_file"
  fi

  open "$video_file"
elif command -v blender >/dev/null 2>&1; then
  blender "$blend_file" &

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$video_file" >/dev/null 2>&1 &
  else
    echo "Video file: $video_file"
  fi
else
  echo "Error: this script could not find Blender." >&2
  echo "Install Blender or add the 'blender' command to PATH." >&2
  echo "Blender file: $blend_file" >&2
  echo "Video file:   $video_file" >&2
  exit 1
fi
