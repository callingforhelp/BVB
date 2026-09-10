"""Integration checks for the current QA-driven, two-axis workflow."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "sandbox"))
from bvb_sandbox.tasks import load_tasks
from human import make_pack, score_human
import merge_vision_sim_shards


def load_script(name, hf):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"huggingface_hub": hf}):
        spec.loader.exec_module(module)
    return module


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CurrentWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_tasks_use_unique_metadata_scenes_and_video_filter(self):
        meta = self.root / "qa.jsonl"
        jsonl(meta, [
            {"scene_name": "z", "dataset": "scannet"},
            {"scene_name": "a", "dataset": "arkitscenes"},
            {"scene_name": "z", "dataset": "scannet"},
        ])
        videos = self.root / "videos"
        (videos / "scannet").mkdir(parents=True)
        (videos / "scannet/z.mp4").write_bytes(b"fixture")
        all_tasks = load_tasks(qa_metadata_path=meta, vsi_bench=videos, require_video=False)
        self.assertEqual([t.scene_name for t in all_tasks], ["a", "z"])
        self.assertEqual([t.dataset for t in all_tasks], ["arkitscenes", "scannet"])
        self.assertEqual([t.scene_name for t in load_tasks(qa_metadata_path=meta, vsi_bench=videos)], ["z"])
        selected = load_tasks(qa_metadata_path=meta, vsi_bench=videos, require_video=False, scenes=["z", "a"], limit=1)
        self.assertEqual([t.scene_name for t in selected], ["z"])

    def test_sampling_is_balanced_repeatable_and_score_independent(self):
        sources = {f"{source}-{i}": source for source in make_pack.SOURCE_ORDER for i in range(6)}
        models = [("one", "One"), ("two", "Two")]
        with patch.object(make_pack, "locate_reference", return_value=Path("source.mp4")), patch.object(make_pack, "video_ok", return_value=True), patch.object(make_pack, "_iter_jsonl", side_effect=AssertionError("Sampling must not load scores")):
            first = make_pack.sample_scenes(random.Random(7), sources, models, 3, None)
            second = make_pack.sample_scenes(random.Random(7), dict(reversed(list(sources.items()))), models, 3, None)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 9)
        self.assertEqual(Counter(sources[s] for s in first), {s: 3 for s in make_pack.SOURCE_ORDER})

    def test_sampling_requires_renders_from_every_model(self):
        sources = {f"{source}-{i}": source for source in make_pack.SOURCE_ORDER for i in range(4)}
        with patch.object(make_pack, "locate_reference", return_value=Path("source.mp4")), patch.object(make_pack, "video_ok", side_effect=lambda p: not (p.stem.endswith('-0') and p.parent.parent.name == 'two')):
            picked = make_pack.sample_scenes(random.Random(1), sources, [("one", "One"), ("two", "Two")], 3, None)
        self.assertEqual(len(picked), 9)
        self.assertFalse(any(s.endswith('-0') for s in picked))

    def test_existing_explicit_assignments_are_preserved(self):
        saved = ["scene-b", "scene-a"]
        self.assertEqual(make_pack.sample_scenes(random.Random(9), {}, [], 3, saved), saved)
        pools = {source: [f"{source}-{i}" for i in range(8)] for source in make_pack.SOURCE_ORDER}
        picked = make_pack.sample_scenes(random.Random(9), {}, [], 3, None, pools)
        self.assertEqual(len(picked), 9)
        self.assertTrue(all(s in sum(pools.values(), []) for s in picked))

    def test_human_report_contains_only_current_metrics(self):
        pack = self.root / "pack"; pack.mkdir()
        blind = {"models": [{"run": "one", "label": "One"}, {"run": "two", "label": "Two"}], "scenes": {}}
        rankings = []
        for scene in ["a", "b", "c"]:
            blind["scenes"][scene] = {"codes": {"A": {"label": "One"}, "B": {"label": "Two"}}}
            rankings.append({"scene_id": scene, "rank": ["A", "B"]})
        (pack / "BLIND_MAP.json").write_text(json.dumps(blind))
        response = self.root / "response.json"
        response.write_text(json.dumps({"instrument_id": "bvb-human-rank-v1", "rankings": rankings}))
        for run, retained, ls in [("one", True, .81), ("two", False, .49)]:
            jsonl(self.root / run / "dual_vqa.jsonl", [{"scene_name": s, "original_correct": True, "retained": retained} for s in ["a", "b", "c"]])
            jsonl(self.root / run / "vision_sim.jsonl", [{"id": s, "vision_sim": ls} for s in ["a", "b", "c"]])
        out = self.root / "scores"
        with patch.object(score_human, "RESULTS", self.root), patch.object(sys, "argv", ["score_human.py", "--pack-dir", str(pack), "--responses", str(response), "--out", str(out)]), contextlib.redirect_stdout(io.StringIO()):
            score_human.main()
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["n_scenes"], 3)
        self.assertEqual(summary["overall_axes"], ["dual_vqa", "vision_sim"])
        self.assertEqual(summary["spearman_neg_rank_vs_overall"], 1.0)
        self.assertNotIn("scene_test", "".join(p.read_text() for p in out.iterdir()))

    def test_sync_discovers_current_metrics_and_optional_renders(self):
        hf = types.SimpleNamespace(HfApi=Mock())
        sync = load_script("sync_eval_results", hf)
        for run, artifact in [("old", "summary.json"), ("ls", "vision_sim_summary.json"), ("dv", "dual_vqa_summary.json")]:
            folder = self.root / f"mini-harness-{run}"; folder.mkdir(); (folder / artifact).write_text('{}')
        (self.root / "mini-harness-render/camera_renders").mkdir(parents=True)
        self.assertEqual({p.name for p in sync.discover_runs(self.root, include_camera_renders=False)}, {"mini-harness-ls", "mini-harness-dv"})
        self.assertEqual(len(sync.discover_runs(self.root, include_camera_renders=True)), 3)
        self.assertEqual(set(sync.EVAL_PATTERNS), {"vision_sim.jsonl", "vision_sim_summary.json", "dual_vqa.jsonl", "dual_vqa_summary.json"})
        hf.HfApi.assert_not_called()

    def test_download_filters_retired_artifacts(self):
        from fnmatch import fnmatch
        hf = types.SimpleNamespace(snapshot_download=Mock())
        download = load_script("download_results", hf)
        with patch.object(sys, "argv", ["download_results.py", "--run", "mini-harness-one"]):
            download.main()
        args = hf.snapshot_download.call_args.kwargs
        self.assertEqual(args['allow_patterns'], ['mini-harness-one/**'])
        excluded = lambda filename: any(fnmatch(filename, pattern) for pattern in args['ignore_patterns'])
        for name in ['unit_tests.jsonl', 'summary.json', 'summary.shard-1-of-8.json', 'introspection-cache/a.json']:
            self.assertTrue(excluded('mini-harness-one/' + name), name)
        for name in ['config.json', 'blends/a.blend', 'dual_vqa.jsonl', 'vision_sim_summary.json', 'camera_renders/a.mp4']:
            self.assertFalse(excluded('mini-harness-one/' + name), name)
        self.assertFalse(excluded('_dual_vqa_shared/original_answers.jsonl'))

    def test_ls_shards_merge_without_retired_helpers(self):
        jsonl(self.root / "vision_sim.shard-1-of-2.jsonl", [{"id": "a", "status": "ok", "vision_sim": .8, "layout_sim": .7, "motion_sim": .9}])
        jsonl(self.root / "vision_sim.shard-2-of-2.jsonl", [{"id": "b", "status": "render_error", "vision_sim": None}])
        with patch.object(sys, "argv", ["merge_vision_sim_shards.py", "--run", str(self.root), "--shard-count", "2"]), contextlib.redirect_stdout(io.StringIO()):
            merge_vision_sim_shards.main()
        summary = json.loads((self.root / "vision_sim_summary.json").read_text())
        self.assertEqual(summary['num_scenes'], 2)
        self.assertEqual(summary['vision_sim'], .4)


if __name__ == "__main__":
    unittest.main()
