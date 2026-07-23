from __future__ import annotations

import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from generate_unit_tests import (  # noqa: E402
    appearance_answer_order,
    appearance_categories,
    direction_refs,
    relative_distance_refs,
    route_spec,
    turn_sequence,
)


class GenerateUnitTestsTest(unittest.TestCase):
    def test_direction_refs(self) -> None:
        self.assertEqual(
            direction_refs(
                "If I am standing by the stove and facing the sofa, "
                "is the tv to my front-left, front-right, back-left, or back-right?"
            ),
            ("stove", "sofa", "television"),
        )

    def test_relative_distance_refs(self) -> None:
        self.assertEqual(
            relative_distance_refs(
                "Measuring from the closest point of each object, which of these objects "
                "(chair, stool, stove, sofa) is the closest to the tv?"
            ),
            ("television", ["chair", "stool", "stove", "sofa"]),
        )

    def test_appearance_parser(self) -> None:
        self.assertEqual(
            appearance_categories(
                "What will be the first-time appearance order of the following categories "
                "in the video: ceiling light, cup, heater, door?"
            ),
            ["ceiling_light", "cup", "heater", "door"],
        )
        self.assertEqual(
            appearance_answer_order("cup, door, heater, then ceiling light"),
            ["cup", "door", "heater", "ceiling_light"],
        )

    def test_route_parser_and_fuzzy_rejection(self) -> None:
        supported = route_spec(
            "You are a robot beginning at the bed facing the tv. "
            "1. Go forward until the TV 2. [please fill in] "
            "3. Go forward until the shower 4. [please fill in] "
            "5. Go forward until the toilet. You have reached the final destination."
        )
        self.assertEqual(
            supported["route_steps"],
            [
                {"ref": "tv", "relation": "near"},
                {"ref": "shower", "relation": "near"},
                {"ref": "toilet", "relation": "near"},
            ],
        )
        fuzzy = route_spec(
            "You are a robot beginning at the tv facing the bed. "
            "1. [please fill in] 2. Go forward until the trash bin is on your right. "
            "You have reached the final destination."
        )
        self.assertEqual(
            fuzzy["route_steps"],
            [{"ref": "trash bin is on your right", "relation": "right"}],
        )
        passing = route_spec(
            "You are a robot beginning at the chair facing the door. "
            "1. Go forward passing the bed 2. [please fill in] "
            "3. Go forward until the table. You have reached the final destination."
        )
        self.assertEqual(
            passing["route_steps"],
            [
                {"ref": "bed", "relation": "pass"},
                {"ref": "table", "relation": "near"},
            ],
        )
        self.assertEqual(turn_sequence("Turn Left, Turn Right"), ["turn_left", "turn_right"])


if __name__ == "__main__":
    unittest.main()
