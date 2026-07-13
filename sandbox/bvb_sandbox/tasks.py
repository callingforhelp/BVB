"""Build the Stage-1 task list from the existing BVB benchmark assets.

A task is the agent's input for one scene:
- ``scene_name``: e.g. ``41069025`` / ``scene0011_00``
- ``dataset``:    ``arkitscenes`` / ``scannet`` / ``scannetpp`` (from test.jsonl)
- ``video_path``: ``VSI-Bench/<dataset>/<scene_name>.mp4``

The set of scenes is exactly those that have unit tests in ``unit_tests.jsonl``
(excluding the ``*`` global tests), so we never run an agent on a scene we
cannot score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# repo root = BVB/BVB ; this file is BVB/BVB/sandbox/bvb_sandbox/tasks.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIT_TESTS = REPO_ROOT / "eval" / "unit_tests.jsonl"
DEFAULT_QA_METADATA = REPO_ROOT / "eval" / "test.jsonl"
DEFAULT_VSI_BENCH = REPO_ROOT / "VSI-Bench"


@dataclass
class BVBTask:
    scene_name: str
    dataset: str | None
    video_path: Path | None
    num_tests: int

    @property
    def has_video(self) -> bool:
        return self.video_path is not None and self.video_path.exists()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _scene_test_counts(unit_tests_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for test in _iter_jsonl(unit_tests_path):
        scene = str(test.get("scene_name"))
        if scene and scene != "*":
            counts[scene] = counts.get(scene, 0) + 1
    return counts


def _scene_datasets(qa_metadata_path: Path) -> dict[str, str]:
    datasets: dict[str, str] = {}
    for record in _iter_jsonl(qa_metadata_path):
        scene = str(record.get("scene_name"))
        dataset = record.get("dataset")
        if scene and dataset and scene not in datasets:
            datasets[scene] = str(dataset)
    return datasets


def locate_video(scene_name: str, dataset: str | None, vsi_bench: Path) -> Path | None:
    candidates = []
    if dataset:
        candidates.append(vsi_bench / dataset / f"{scene_name}.mp4")
    for ds in ("arkitscenes", "scannet", "scannetpp"):
        candidates.append(vsi_bench / ds / f"{scene_name}.mp4")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def load_tasks(
    *,
    unit_tests_path: Path = DEFAULT_UNIT_TESTS,
    qa_metadata_path: Path = DEFAULT_QA_METADATA,
    vsi_bench: Path = DEFAULT_VSI_BENCH,
    require_video: bool = True,
    scenes: list[str] | None = None,
    limit: int | None = None,
) -> list[BVBTask]:
    counts = _scene_test_counts(unit_tests_path)
    datasets = _scene_datasets(qa_metadata_path)

    selected = scenes if scenes else sorted(counts)
    tasks: list[BVBTask] = []
    for scene in selected:
        dataset = datasets.get(scene)
        video = locate_video(scene, dataset, vsi_bench)
        task = BVBTask(
            scene_name=scene,
            dataset=dataset,
            video_path=video,
            num_tests=counts.get(scene, 0),
        )
        if require_video and not task.has_video:
            continue
        tasks.append(task)

    if limit is not None:
        tasks = tasks[:limit]
    return tasks
