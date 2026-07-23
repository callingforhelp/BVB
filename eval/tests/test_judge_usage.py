from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from eval_utils import call_chat_completion_with_usage  # noqa: E402
from unit_test_metric import (  # noqa: E402
    JudgeResponseError,
    call_grounding_requests,
    judge_cost,
    parse_shard,
    records_from_run,
    select_records,
    summarize,
    validate_grounding_results,
    validate_resume_config,
)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": '{"results":[]}'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
            }
        ).encode()


class MalformedResponse(FakeResponse):
    def read(self) -> bytes:
        return b'{"unexpected": true}'


class JudgeUsageTest(unittest.TestCase):
    @patch("eval_utils.urllib.request.urlopen", return_value=FakeResponse())
    def test_chat_completion_usage_and_output_limit(self, mocked_urlopen) -> None:
        result = call_chat_completion_with_usage(
            base_url="https://example.test/v1",
            api_key="test",
            model="gpt-5.4-mini",
            system_prompt="system",
            user_prompt="user",
            temperature=None,
            max_completion_tokens=8192,
            reasoning_effort="none",
        )
        self.assertEqual(result.content, '{"results":[]}')
        self.assertEqual(result.prompt_tokens, 1000)
        self.assertEqual(result.completion_tokens, 200)
        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["max_completion_tokens"], 8192)
        self.assertEqual(payload["reasoning_effort"], "none")

    @patch("eval_utils.time.sleep")
    @patch("eval_utils.urllib.request.urlopen")
    def test_transient_http_error_is_retried(self, mocked_urlopen, mocked_sleep) -> None:
        mocked_urlopen.side_effect = [
            urllib.error.HTTPError(
                "https://example.test",
                429,
                "rate limited",
                {"Retry-After": "0"},
                BytesIO(b"rate limited"),
            ),
            FakeResponse(),
        ]
        result = call_chat_completion_with_usage(
            base_url="https://example.test/v1",
            api_key="test",
            model="gpt-5.4-mini",
            system_prompt="system",
            user_prompt="user",
            temperature=None,
            max_retries=1,
        )
        self.assertEqual(result.prompt_tokens, 1000)
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called_once_with(0.0)

    @patch("eval_utils.time.sleep")
    @patch("eval_utils.urllib.request.urlopen")
    def test_malformed_success_response_is_retried(self, mocked_urlopen, mocked_sleep) -> None:
        mocked_urlopen.side_effect = [MalformedResponse(), FakeResponse()]
        result = call_chat_completion_with_usage(
            base_url="https://example.test/v1",
            api_key="test",
            model="gpt-5.4-mini",
            system_prompt="system",
            user_prompt="user",
            temperature=None,
            max_retries=1,
        )
        self.assertEqual(result.completion_tokens, 200)
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_cost_and_summary_aggregation(self) -> None:
        cost = judge_cost(
            "gpt-5.4-mini",
            1_000_000,
            1_000_000,
            input_price_per_million=None,
            output_price_per_million=None,
        )
        self.assertEqual(cost, 5.25)
        with self.assertRaises(ValueError):
            judge_cost(
                "unknown-model",
                100,
                100,
                input_price_per_million=None,
                output_price_per_million=None,
            )
        summary = summarize(
            [
                {
                    "unit_tests": [],
                    "judge_usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 200,
                        "cost_usd": 0.00165,
                    },
                },
                {
                    "unit_tests": [],
                    "judge_usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 100,
                        "cost_usd": 0.000825,
                    },
                },
            ]
        )
        self.assertEqual(summary["judge_usage"]["prompt_tokens"], 1500)
        self.assertEqual(summary["judge_usage"]["completion_tokens"], 300)
        self.assertEqual(summary["judge_usage"]["cost_usd"], 0.002475)

    def test_appearance_summary_separates_coverage_from_order_accuracy(self) -> None:
        summary = summarize(
            [
                {
                    "judge_model": "gpt-5.4-mini",
                    "judge_usage": {},
                    "unit_tests": [
                        {
                            "test_type": "obj_appearance_order",
                            "status": "pass",
                            "evidence": {"first_visible_frames": {"chair": 1, "table": 2}},
                        },
                        {
                            "test_type": "obj_appearance_order",
                            "status": "fail",
                            "evidence": {"first_visible_frames": {"chair": 1, "lamp": None}},
                        },
                    ],
                }
            ]
        )
        diagnostics = summary["appearance_diagnostics"]
        self.assertEqual(diagnostics["visible_category_slots"], 3)
        self.assertEqual(diagnostics["total_category_slots"], 4)
        self.assertEqual(diagnostics["num_complete_order_tests"], 1)
        self.assertEqual(diagnostics["pass_rate_given_complete_order"], 1.0)

    def test_run_directory_is_scanned_without_manifest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            blends = run / "blends"
            blends.mkdir()
            (blends / "scene_b.blend").write_bytes(b"b")
            (blends / "scene_a.blend").write_bytes(b"a")
            records = records_from_run(run)
        self.assertEqual([row["id"] for row in records], ["scene_a", "scene_b"])
        self.assertTrue(all("submission_path" in row for row in records))

    def test_random_sample_is_reproducible_across_runs(self) -> None:
        records = [{"id": f"scene_{index:02d}"} for index in range(20)]
        first = select_records(records, limit=None, sample=5, seed=42)
        second = select_records(list(reversed(records)), limit=None, sample=5, seed=42)
        self.assertEqual(
            [row["id"] for row in first],
            [row["id"] for row in second],
        )
        selected = select_records(
            records,
            limit=None,
            sample=None,
            seed=0,
            scene_ids=["scene_03", "scene_01"],
        )
        self.assertEqual([row["id"] for row in selected], ["scene_03", "scene_01"])
        self.assertEqual(parse_shard("3/8"), (3, 8))
        with self.assertRaises(SystemExit):
            parse_shard("9/8")

    @patch("unit_test_metric.call_json_judge")
    def test_invalid_json_is_retried_as_split_batches(self, mocked_judge) -> None:
        mocked_judge.side_effect = [
            JudgeResponseError(
                "truncated",
                {"prompt_tokens": 100, "completion_tokens": 50, "finish_reason": "length"},
            ),
            (
                {"results": [{"grounding_id": "g1", "grounded_params": {}}]},
                {"prompt_tokens": 60, "completion_tokens": 10, "finish_reason": "stop"},
            ),
            (
                {"results": [{"grounding_id": "g2", "grounded_params": {}}]},
                {"prompt_tokens": 70, "completion_tokens": 20, "finish_reason": "stop"},
            ),
        ]
        args = Namespace(
            base_url="https://example.test/v1",
            api_key="test",
            model="gpt-5.4-mini",
            temperature=None,
            max_completion_tokens=8192,
            reasoning_effort="none",
        )
        results, usage = call_grounding_requests(
            scene_id="scene",
            manifest={"groups": []},
            grounding_requests=[
                {"grounding_id": "g1", "function": "count_objects"},
                {"grounding_id": "g2", "function": "count_objects"},
            ],
            args=args,
        )
        self.assertEqual([item["grounding_id"] for item in results], ["g1", "g2"])
        self.assertEqual(usage["prompt_tokens"], 230)
        self.assertEqual(usage["completion_tokens"], 80)

    def test_missing_and_duplicate_grounding_ids_are_rejected(self) -> None:
        requests = [
            {"grounding_id": "g1"},
            {"grounding_id": "g2"},
        ]
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        with self.assertRaises(JudgeResponseError):
            validate_grounding_results(
                requests,
                [{"grounding_id": "g1", "grounded_params": {}}],
                usage,
            )
        with self.assertRaises(JudgeResponseError):
            validate_grounding_results(
                requests,
                [
                    {"grounding_id": "g1", "grounded_params": {}},
                    {"grounding_id": "g1", "grounded_params": {}},
                    {"grounding_id": "g2", "grounded_params": {}},
                ],
                usage,
            )

    def test_resume_rejects_changed_evaluation_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.jsonl.config.json"
            path.write_text(json.dumps({"model": "mini"}), encoding="utf-8")
            validate_resume_config(path, {"model": "mini"})
            with self.assertRaises(SystemExit):
                validate_resume_config(path, {"model": "large"})


if __name__ == "__main__":
    unittest.main()
