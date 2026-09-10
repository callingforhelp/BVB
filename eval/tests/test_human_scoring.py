"""Regression checks for human-study calibration against the current axes."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human import score_human


class HumanAutoScoresTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = Path(self.temp.name)
        self.run = self.results / "fixture"
        self.run.mkdir()
        self.blind = {"models": [{"run": "fixture", "label": "model"}]}
        self.write("dual_vqa.jsonl", [
            {"scene_name": "scene", "original_correct": True, "retained": i == 0}
            for i in range(4)
        ])
        self.write("vision_sim.jsonl", [{"scene_id": "scene", "vision_sim": 1.0}])

    def write(self, name, rows):
        (self.run / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def score(self):
        with patch.object(score_human, "RESULTS", self.results):
            return score_human.auto_scores(self.blind, ["scene"])[("scene", "model")]

    def test_overall_works_without_legacy_scene_tests(self):
        row = self.score()
        self.assertIsNone(row["scene_test"])
        self.assertEqual(row["dual_vqa"], 0.25)
        self.assertEqual(row["overall"], 0.5625)

    def test_legacy_scores_do_not_change_overall(self):
        for status in ("pass", "fail"):
            with self.subTest(status=status):
                self.write("unit_tests.jsonl", [{
                    "scene_id": "scene", "test_type": "object_count", "status": status,
                }])
                row = self.score()
                self.assertEqual(row["scene_test"], float(status == "pass"))
                self.assertEqual(row["overall"], 0.5625)

    def test_no_source_correct_questions_is_undefined(self):
        self.write("dual_vqa.jsonl", [{"scene_name": "scene", "original_correct": False}])
        row = self.score()
        self.assertIsNone(row["dual_vqa"])
        self.assertIsNone(row["overall"])

    def test_zero_retention_still_combines_with_ls(self):
        self.write("dual_vqa.jsonl", [{
            "scene_name": "scene", "original_correct": True, "retained": False,
        }])
        self.assertEqual(self.score()["overall"], 0.25)

    def test_negative_cosine_is_clipped_for_overall(self):
        self.write("vision_sim.jsonl", [{"scene_id": "scene", "vision_sim": -0.2}])
        row = self.score()
        self.assertEqual(row["vision_sim"], -0.2)
        self.assertEqual(row["overall"], 0.0625)

    def test_missing_ls_is_undefined(self):
        (self.run / "vision_sim.jsonl").unlink()
        self.assertIsNone(self.score()["overall"])


if __name__ == "__main__":
    unittest.main()
