#!/usr/bin/env python3
"""Batch-export Blender .blend files to reproducible bpy Python scripts.

This wraps Blender in background mode and reuses the BVB export addon from
../addons/export_bpy_code.py. It is intended for evaluation result folders such
as:

  results/claude-sonnet-4.6/blend/*.blend -> results/claude-sonnet-4.6/bpy/*.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_RESULT_DIR = Path("results/claude-sonnet-4.6")


BLENDER_EXPORT_SCRIPT = r"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import bpy


def arg_after(name: str) -> str:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Missing script args after --")
    args = argv[argv.index("--") + 1 :]
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
    raise SystemExit(f"Missing required argument: {name}")


output_path = Path(arg_after("--output"))
addon_path = Path(arg_after("--addon"))

spec = importlib.util.spec_from_file_location("bvb_export_bpy_code", addon_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Unable to load addon: {addon_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.register()

output_path.parent.mkdir(parents=True, exist_ok=True)
result = bpy.ops.export_scene.bpy_code(
    filepath=str(output_path),
    include_clear_scene=True,
    include_camera=True,
    include_lights=True,
    include_render_settings=True,
    skip_hidden=False,
    decimal_places="6",
)

if "FINISHED" not in result:
    raise SystemExit(f"Export operator failed: {result}")

print(f"BVB_EXPORT_OK {output_path}")
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-export .blend files to bpy scripts.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR / "blend",
        help="Directory containing .blend files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR / "bpy",
        help="Directory where .py exports will be written.",
    )
    parser.add_argument(
        "--addon",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "addons" / "export_bpy_code.py",
        help="Path to export_bpy_code.py.",
    )
    parser.add_argument(
        "--blender",
        default=os.getenv("BLENDER_BIN"),
        help=(
            "Blender executable. Can also be set with BLENDER_BIN. "
            "If omitted, common PATH and macOS app locations are tried."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .py files.")
    parser.add_argument("--limit", type=int, help="Export at most N files, useful for smoke tests.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSONL manifest with one row per attempted export.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned exports without running Blender.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def resolve_blender_executable(value: str | None) -> str:
    candidates = []
    if value:
        candidates.append(value)
    candidates.extend(
        [
            "blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.3.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.2.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.1.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.0.app/Contents/MacOS/Blender",
            "/Applications/Blender 3.6.app/Contents/MacOS/Blender",
        ]
    )
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        path = Path(candidate)
        if path.is_file():
            return str(path)
    raise SystemExit(
        "Could not find Blender. Pass --blender /path/to/Blender or set BLENDER_BIN."
    )


def write_manifest_row(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    addon_path = resolve_path(args.addon)
    manifest_path = resolve_path(args.manifest) if args.manifest else None
    blender_executable = resolve_blender_executable(args.blender)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not addon_path.is_file():
        raise SystemExit(f"Addon file does not exist: {addon_path}")

    blend_files = sorted(input_dir.glob("*.blend"))
    if args.limit is not None:
        blend_files = blend_files[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path and manifest_path.exists():
        manifest_path.unlink()

    planned = []
    for blend_path in blend_files:
        output_path = output_dir / f"{blend_path.stem}.py"
        if output_path.exists() and not args.overwrite:
            continue
        planned.append((blend_path, output_path))

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "num_blend_files": len(blend_files),
                "num_exports_planned": len(planned),
                "blender": blender_executable,
                "overwrite": args.overwrite,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    if args.dry_run:
        for blend_path, output_path in planned[:20]:
            print(f"{blend_path} -> {output_path}")
        if len(planned) > 20:
            print(f"... {len(planned) - 20} more")
        return

    failures = []
    with tempfile.TemporaryDirectory(prefix="bvb_export_") as temp_dir:
        blender_script = Path(temp_dir) / "export_one_blend.py"
        blender_script.write_text(BLENDER_EXPORT_SCRIPT, encoding="utf-8")

        for index, (blend_path, output_path) in enumerate(planned, start=1):
            print(f"[{index}/{len(planned)}] Exporting {blend_path.name}")
            command = [
                blender_executable,
                "--background",
                str(blend_path),
                "--python",
                str(blender_script),
                "--",
                "--output",
                str(output_path),
                "--addon",
                str(addon_path),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            row = {
                "blend_path": str(blend_path),
                "bpy_path": str(output_path),
                "returncode": completed.returncode,
                "ok": completed.returncode == 0 and output_path.exists(),
            }
            write_manifest_row(manifest_path, row)
            if not row["ok"]:
                failures.append(row)
                print(completed.stdout[-2000:])
                print(completed.stderr[-2000:], file=sys.stderr)

    summary = {
        "attempted": len(planned),
        "succeeded": len(planned) - len(failures),
        "failed": len(failures),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path) if manifest_path else None,
    }
    print(json.dumps(summary, indent=2))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
