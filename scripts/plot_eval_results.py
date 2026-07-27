#!/usr/bin/env python3
"""Plot Stage-2 summary statistics into one output folder.

Reads every ``sandbox/results/mini-harness-*/summary.json`` and writes charts +
a CSV leaderboard under ``sandbox/eval_plots/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = Path("sandbox/results")
DEFAULT_OUT_DIR = Path("sandbox/eval_plots")

TEST_TYPES = [
    "basic_validity",
    "object_counting",
    "object_size_estimation",
    "object_abs_distance",
    "object_rel_distance",
    "object_rel_direction",
    "room_size_estimation",
    "route_planning",
    "obj_appearance_order",
]


def pretty_run_name(run_name: str) -> str:
    name = run_name
    if name.startswith("mini-harness-"):
        name = name[len("mini-harness-") :]
    if name.endswith("-run01"):
        name = name[: -len("-run01")]
    return name


def load_rows(results_dir: Path, min_scenes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("mini-harness-*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scenes = int(summary.get("num_scenes") or 0)
        if scenes < min_scenes:
            continue
        usage = summary.get("judge_usage") or {}
        by_type = summary.get("by_test_type") or {}
        agent_cost = 0.0
        agent_meta_dir = summary_path.parent / "agent_meta"
        if agent_meta_dir.is_dir():
            for meta_path in agent_meta_dir.glob("*.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                agent_cost += float(meta.get("cost_usd") or 0.0)
        row: dict[str, Any] = {
            "run": summary_path.parent.name,
            "label": pretty_run_name(summary_path.parent.name),
            "num_scenes": scenes,
            "pass_rate": float(summary.get("unit_test_pass_rate") or 0.0),
            "pass_rate_wo_basic": float(
                summary.get("unit_test_pass_rate_without_basic_validity") or 0.0
            ),
            "judge_cost_usd": float(usage.get("cost_usd") or 0.0),
            "agent_cost_usd": agent_cost,
            "agent_cost_per_scene": agent_cost / scenes if scenes else 0.0,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "num_error": int(summary.get("num_error") or 0),
        }
        # Keep old key for compatibility with earlier plot helpers.
        row["cost_usd"] = row["judge_cost_usd"]
        for test_type in TEST_TYPES:
            stats = by_type.get(test_type) or {}
            row[f"pr_{test_type}"] = float(stats.get("pass_rate") or 0.0)
        rows.append(row)
    rows.sort(key=lambda item: item["pass_rate"], reverse=True)
    return rows


def save_leaderboard_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def plot_leaderboard(rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    rates = [row["pass_rate"] * 100 for row in rows]
    fig_h = max(6.0, 0.28 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(rows))
    ax.barh(y, rates, color="#3B6D9C")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Unit-test pass rate (%)")
    ax.set_title("BVB Stage-2 leaderboard")
    style_axes(ax)
    for yi, rate in zip(y, rates):
        ax.text(rate + 0.3, yi, f"{rate:.1f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "01_leaderboard_pass_rate.png", dpi=160)
    plt.close(fig)


def plot_pass_vs_cost(rows: list[dict[str, Any]], out_dir: Path) -> None:
    # Judge cost (Stage-2) — small, mostly similar across runs.
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [row["judge_cost_usd"] for row in rows]
    ys = [row["pass_rate"] * 100 for row in rows]
    ax.scatter(xs, ys, s=36, color="#C45C26", alpha=0.85)
    for row in rows[:12]:
        ax.annotate(
            row["label"],
            (row["judge_cost_usd"], row["pass_rate"] * 100),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )
    ax.set_xlabel("Stage-2 judge cost (USD)")
    ax.set_ylabel("Unit-test pass rate (%)")
    ax.set_title("Pass rate vs judge cost")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "02_pass_rate_vs_judge_cost.png", dpi=160)
    plt.close(fig)

    # Agent cost (Stage-1) — the real money.
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [row["agent_cost_usd"] for row in rows]
    ys = [row["pass_rate"] * 100 for row in rows]
    ax.scatter(xs, ys, s=36, color="#2F6F4E", alpha=0.85)
    for row in sorted(rows, key=lambda item: item["pass_rate"], reverse=True)[:15]:
        ax.annotate(
            row["label"],
            (row["agent_cost_usd"], row["pass_rate"] * 100),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )
    ax.set_xlabel("Stage-1 agent cost (USD, full run)")
    ax.set_ylabel("Unit-test pass rate (%)")
    ax.set_title("Pass rate vs agent cost")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "02b_pass_rate_vs_agent_cost.png", dpi=160)
    plt.close(fig)


def plot_agent_cost_charts(rows: list[dict[str, Any]], out_dir: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["agent_cost_usd"], reverse=True)
    labels = [row["label"] for row in ordered]
    costs = [row["agent_cost_usd"] for row in ordered]
    fig_h = max(6.0, 0.28 * len(ordered) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(ordered))
    ax.barh(y, costs, color="#2F6F4E")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Stage-1 agent cost (USD)")
    ax.set_title("Agent cost by run")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "11_agent_cost_leaderboard.png", dpi=160)
    plt.close(fig)

    # Efficiency: pass-rate points per $100 agent spend.
    eff_rows = [
        row
        for row in rows
        if row["agent_cost_usd"] > 1.0
    ]
    eff_rows.sort(
        key=lambda item: (item["pass_rate"] * 100) / item["agent_cost_usd"],
        reverse=True,
    )
    labels = [row["label"] for row in eff_rows]
    eff = [(row["pass_rate"] * 100) / row["agent_cost_usd"] for row in eff_rows]
    fig_h = max(6.0, 0.28 * len(eff_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(eff_rows))
    ax.barh(y, eff, color="#7A5C2E")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pass-rate points per agent USD  (pass% / $)")
    ax.set_title("Cost efficiency (higher = more score per dollar)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "12_cost_efficiency.png", dpi=160)
    plt.close(fig)

    # Per-scene agent cost vs pass rate.
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [row["agent_cost_per_scene"] for row in rows]
    ys = [row["pass_rate"] * 100 for row in rows]
    ax.scatter(xs, ys, s=36, color="#6B4C9A", alpha=0.85)
    for row in sorted(rows, key=lambda item: item["pass_rate"], reverse=True)[:12]:
        ax.annotate(
            row["label"],
            (row["agent_cost_per_scene"], row["pass_rate"] * 100),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )
    ax.set_xlabel("Agent cost per scene (USD)")
    ax.set_ylabel("Unit-test pass rate (%)")
    ax.set_title("Pass rate vs per-scene agent cost")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "13_pass_rate_vs_agent_cost_per_scene.png", dpi=160)
    plt.close(fig)


def plot_pass_without_basic(rows: list[dict[str, Any]], out_dir: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["pass_rate_wo_basic"], reverse=True)
    labels = [row["label"] for row in ordered]
    rates = [row["pass_rate_wo_basic"] * 100 for row in ordered]
    fig_h = max(6.0, 0.28 * len(ordered) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(ordered))
    ax.barh(y, rates, color="#5B8C5A")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pass rate without basic_validity (%)")
    ax.set_title("Hard metrics only")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "03_leaderboard_without_basic.png", dpi=160)
    plt.close(fig)


def plot_test_type_heatmap(rows: list[dict[str, Any]], out_dir: Path, top_n: int = 20) -> None:
    top = rows[:top_n]
    matrix = np.array([[row[f"pr_{name}"] * 100 for name in TEST_TYPES] for row in top])
    fig_w = max(10.0, 0.9 * len(TEST_TYPES) + 4)
    fig_h = max(5.0, 0.35 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(TEST_TYPES)))
    ax.set_xticklabels(TEST_TYPES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([row["label"] for row in top], fontsize=8)
    ax.set_title(f"Pass rate by test type (top {len(top)} overall)")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="%")
    fig.tight_layout()
    fig.savefig(out_dir / "04_test_type_heatmap_top20.png", dpi=160)
    plt.close(fig)


def parse_reasoning_family(label: str) -> tuple[str, str] | None:
    # gpt-5.6-sol-reasoning-high -> (gpt-5.6-sol, high)
    match = re.match(
        r"^(?P<family>.+?)-reasoning-(?P<effort>none|low|medium|high|xhigh)$",
        label,
    )
    if match:
        return match.group("family"), match.group("effort")
    # bare model with no reasoning suffix is treated as none for known families
    for family in (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.2",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "grok-4.3",
    ):
        if label == family:
            return family, "none"
    return None


def plot_reasoning_curves(rows: list[dict[str, Any]], out_dir: Path) -> None:
    order = ["none", "low", "medium", "high", "xhigh"]
    families: dict[str, dict[str, float]] = {}
    for row in rows:
        parsed = parse_reasoning_family(row["label"])
        if parsed is None:
            continue
        family, effort = parsed
        families.setdefault(family, {})[effort] = row["pass_rate"] * 100

    # Prefer families with at least 3 reasoning points.
    selected = {
        family: points
        for family, points in families.items()
        if sum(effort in points for effort in order) >= 3
    }
    if not selected:
        selected = {
            family: points
            for family, points in families.items()
            if sum(effort in points for effort in order) >= 2
        }
    if not selected:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for family, points in sorted(selected.items()):
        xs = [effort for effort in order if effort in points]
        ys = [points[effort] for effort in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=family)
    ax.set_xlabel("Reasoning effort")
    ax.set_ylabel("Unit-test pass rate (%)")
    ax.set_title("Reasoning scaling curves")
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "05_reasoning_curves.png", dpi=160)
    plt.close(fig)


def plot_error_counts(rows: list[dict[str, Any]], out_dir: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["num_error"], reverse=True)[:25]
    labels = [row["label"] for row in ordered]
    errs = [row["num_error"] for row in ordered]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(ordered))
    ax.barh(y, errs, color="#A33B3B")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("num_error tests")
    ax.set_title("Judge/runtime errors by run (top 25)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "06_error_counts.png", dpi=160)
    plt.close(fig)


def plot_type_difficulty(rows: list[dict[str, Any]], out_dir: Path) -> None:
    means = []
    for test_type in TEST_TYPES:
        values = [row[f"pr_{test_type}"] * 100 for row in rows]
        means.append((test_type, float(np.mean(values)), float(np.std(values))))
    means.sort(key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in means]
    avg = [item[1] for item in means]
    std = [item[2] for item in means]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, avg, yerr=std, color="#6B7C93", alpha=0.9, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean pass rate across runs (%)")
    ax.set_title("Which question types are hard? (mean ± std over models)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "07_type_difficulty.png", dpi=160)
    plt.close(fig)


def plot_per_type_leaderboards(rows: list[dict[str, Any]], out_dir: Path, top_n: int = 12) -> None:
    n_types = len(TEST_TYPES)
    ncols = 3
    nrows = int(np.ceil(n_types / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows))
    axes_list = list(np.ravel(axes))
    for ax, test_type in zip(axes_list, TEST_TYPES):
        ordered = sorted(rows, key=lambda item: item[f"pr_{test_type}"], reverse=True)[:top_n]
        labels = [row["label"] for row in ordered]
        rates = [row[f"pr_{test_type}"] * 100 for row in ordered]
        y = np.arange(len(ordered))
        ax.barh(y, rates, color="#3B6D9C")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(test_type, fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_list[n_types:]:
        ax.axis("off")
    fig.suptitle(f"Per-question-type leaderboards (top {top_n})", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "08_per_type_leaderboards.png", dpi=160)
    plt.close(fig)


def plot_top_models_by_type(rows: list[dict[str, Any]], out_dir: Path, top_models: int = 8) -> None:
    hard_types = [name for name in TEST_TYPES if name != "basic_validity"]
    top = rows[:top_models]
    x = np.arange(len(hard_types))
    width = 0.8 / max(len(top), 1)
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    for index, row in enumerate(top):
        rates = [row[f"pr_{name}"] * 100 for name in hard_types]
        ax.bar(
            x + index * width - 0.4 + width / 2,
            rates,
            width=width,
            label=row["label"],
            color=cmap(index % 10),
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(hard_types, rotation=25, ha="right")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title(f"Top {top_models} models by question type (excl. basic_validity)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "09_top_models_by_type.png", dpi=160)
    plt.close(fig)


def plot_type_rank_table(rows: list[dict[str, Any]], out_dir: Path, top_n: int = 10) -> None:
    """Compact rank chart: for each type, show #1/#2/#3 model names and scores."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    lines = ["Per-type top-3 models", ""]
    for test_type in TEST_TYPES:
        ordered = sorted(rows, key=lambda item: item[f"pr_{test_type}"], reverse=True)[:3]
        medal = ",  ".join(
            f"{row['label']} ({row[f'pr_{test_type}'] * 100:.1f}%)" for row in ordered
        )
        lines.append(f"{test_type}:  {medal}")
    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)
    ax.set_title("Question-type winners")
    fig.tight_layout()
    fig.savefig(out_dir / "10_type_winners.png", dpi=160)
    plt.close(fig)

    # Also dump a CSV focused on type ranks for the overall top_n.
    path = out_dir / "by_test_type_top_models.csv"
    top = rows[:top_n]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank_overall", "label", *TEST_TYPES])
        for index, row in enumerate(top, start=1):
            writer.writerow(
                [index, row["label"], *[f"{row[f'pr_{name}'] * 100:.2f}" for name in TEST_TYPES]]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--min-scenes",
        type=int,
        default=200,
        help="Ignore tiny incomplete runs below this scene count.",
    )
    args = parser.parse_args()

    rows = load_rows(args.results_dir, args.min_scenes)
    if not rows:
        raise SystemExit(f"No summaries found under {args.results_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the folder dedicated to this script's outputs.
    for stale in args.out_dir.glob("*"):
        if stale.is_file():
            stale.unlink()

    save_leaderboard_csv(rows, args.out_dir / "leaderboard.csv")
    plot_leaderboard(rows, args.out_dir)
    plot_pass_vs_cost(rows, args.out_dir)
    plot_pass_without_basic(rows, args.out_dir)
    plot_test_type_heatmap(rows, args.out_dir)
    plot_reasoning_curves(rows, args.out_dir)
    plot_error_counts(rows, args.out_dir)
    plot_type_difficulty(rows, args.out_dir)
    plot_per_type_leaderboards(rows, args.out_dir)
    plot_top_models_by_type(rows, args.out_dir)
    plot_type_rank_table(rows, args.out_dir)
    plot_agent_cost_charts(rows, args.out_dir)

    print(f"Wrote {len(list(args.out_dir.iterdir()))} files to {args.out_dir}")
    print(f"Top-5: {', '.join(row['label'] for row in rows[:5])}")


if __name__ == "__main__":
    main()
