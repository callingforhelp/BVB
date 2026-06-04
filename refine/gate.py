"""`bvbrefine gate` — export the live .blend to bpy, then score unit tests.

Default: offline deterministic scoring of `function` tests (no LLM); `proposition`
tests are reported as self-judged. With --judge, defer to eval/unit_test_metric.py
(LLM grounding + judging) for fully benchmark-faithful results.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bvb_lib  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
UNIT_TESTS = REPO / "eval" / "unit_tests.jsonl"
ADDON = REPO / "addons" / "export_bpy_code.py"


def export_bpy(scene_id: str, out_dir: Path):
    """Export blend/<id>.blend → <out_dir>/<id>.py via background Blender."""
    blend = REPO / "blend" / f"{scene_id}.blend"
    if not blend.is_file():
        print(f"❌ {blend} not found")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        one = Path(td) / f"{scene_id}.blend"
        shutil.copy2(blend, one)
        subprocess.run(
            ["python", str(REPO / "eval" / "batch_export_blend_to_bpy.py"),
             "--input-dir", str(Path(td)), "--output-dir", str(out_dir), "--overwrite"],
            check=True,
        )
    bpy_path = out_dir / f"{scene_id}.py"
    return bpy_path if bpy_path.is_file() else None


def cmd_gate(scene_id: str, judge: bool = False, floor_group=None) -> int:
    refine = REPO / "blend" / f"refine_{scene_id}"
    bpy_dir = refine / "bpy"
    bpy_path = export_bpy(scene_id, bpy_dir)
    if bpy_path is None:
        print("❌ bpy export failed — is Blender + export addon available?")
        return 1
    tests = bvb_lib.load_unit_tests(UNIT_TESTS, scene_id)

    if judge:
        pairs = refine / "pairs.jsonl"
        pairs.write_text(
            json.dumps(bvb_lib.build_pairs_row(scene_id, f"bpy/{scene_id}.py")) + "\n",
            encoding="utf-8",
        )
        out = refine / "gate_results.jsonl"
        rc = subprocess.run(
            ["python", str(REPO / "eval" / "unit_test_metric.py"),
             "--pairs", str(pairs), "--unit-tests", str(UNIT_TESTS),
             "--output", str(out)],
            cwd=str(refine),
        ).returncode
        print(f"\n→ official (judged) results written to {out} (rc={rc})")
        return rc

    print(f"\nOFFLINE GATE for {scene_id} ({len(tests)} tests):")
    failed = 0
    for t in tests:
        if t.get("evaluator") == "function":
            res = bvb_lib.evaluate_function_offline(REPO, bpy_path, t, floor_group)
            status = res.get("status")
            print(f"  [{t['test_id']}] {t['test_type']}: {status.upper()} "
                  f"(actual={res.get('actual')}, expected={t.get('expected')})")
            failed += status != "pass"
        else:
            print(f"  [{t['test_id']}] {t['test_type']}: SELF-JUDGED "
                  f"(proposition; rerun with --judge for official)")
    print(f"\n{len(tests) - failed}/{len(tests)} function tests passing offline "
          f"(proposition tests need --judge).")
    return 1 if failed else 0
