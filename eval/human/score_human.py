#!/usr/bin/env python3
"""Score ranking JSONs from the self-contained survey.html.

Does not invent numbers. Unblinds with BLIND_MAP.json and writes mean rank,
pairwise wins, and optional Spearman vs Dual VQA, Latent Similarity, and
their two-axis Overall. Scene Test is retained as a legacy diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "sandbox" / "results"


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rankings(path: Path) -> list[dict]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    rows = []
    for file in files:
        if file.name == "BLIND_MAP.json":
            continue
        try:
            rec = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if rec.get("instrument_id") != "bvb-human-rank-v1":
            continue
        rec["_file"] = file.name
        rows.append(rec)
    return rows


def unblind(scene: str, code: str, blind: dict) -> str | None:
    entry = ((blind.get("scenes") or {}).get(scene) or {}).get("codes") or {}
    info = entry.get(code) or {}
    return info.get("label") or info.get("run")


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def scene_basic_bundle(tests: list[dict]) -> dict | None:
    basic = [t for t in tests if t.get("test_type") == "basic_validity"]
    if not basic:
        return None
    evaluated = [t for t in basic if t.get("status") in {"pass", "fail"}]
    if not evaluated:
        return {"evaluated": False, "passed": False}
    return {
        "evaluated": True,
        "passed": all(t.get("status") == "pass" for t in evaluated),
    }


def scene_test_rate(tests: list[dict]) -> float | None:
    non_basic = [t for t in tests if t.get("test_type") != "basic_validity"]
    ev = [t for t in non_basic if t.get("status") in {"pass", "fail"}]
    passed = sum(1 for t in ev if t["status"] == "pass")
    n = len(ev)
    bundle = scene_basic_bundle(tests)
    if bundle and bundle["evaluated"]:
        n += 1
        if bundle["passed"]:
            passed += 1
    if n == 0:
        return None
    return passed / n


def dual_vqa_retention(records: list[dict]) -> float | None:
    orig_ok = [r for r in records if r.get("original_correct") is True]
    if not orig_ok:
        return None
    retained = 0
    for rec in orig_ok:
        if rec.get("render_correct") is True or rec.get("retained") is True:
            retained += 1
    return retained / len(orig_ok)


def scene_key_from_record(rec: dict) -> str:
    if rec.get("original_video"):
        return Path(str(rec["original_video"])).stem
    for key in ("scene_name", "scene_id", "id"):
        if rec.get(key):
            return str(rec[key])
    return ""


def auto_scores(blind: dict, scenes: list[str]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for model in blind.get("models") or []:
        run, label = model.get("run"), model.get("label")
        if not run or not label:
            continue
        tests_by: dict[str, list] = defaultdict(list)
        for rec in _iter_jsonl(RESULTS / run / "unit_tests.jsonl"):
            scene = scene_key_from_record(rec)
            if not scene:
                continue
            nested = rec.get("unit_tests")
            if nested:
                tests_by[scene].extend(nested)
            elif rec.get("test_type"):
                tests_by[scene].append(rec)
        dual_by = defaultdict(list)
        for rec in _iter_jsonl(RESULTS / run / "dual_vqa.jsonl"):
            scene = str(rec.get("scene_name") or scene_key_from_record(rec) or "")
            if scene:
                dual_by[scene].append(rec)
        vis: dict[str, float | None] = {}
        for rec in _iter_jsonl(RESULTS / run / "vision_sim.jsonl"):
            scene = scene_key_from_record(rec)
            if scene:
                vis[scene] = rec.get("vision_sim")
        for scene in scenes:
            st = scene_test_rate(tests_by.get(scene, []))
            orig_ok = dual_by.get(scene, [])
            retention = dual_vqa_retention(orig_ok)
            ls = vis.get(scene)
            overall = None
            if retention is not None and ls is not None:
                # Current paper: DV + LS; clip negative cosine before sqrt.
                mean_root = (
                    math.sqrt(retention) + math.sqrt(max(0.0, ls))
                ) / 2.0
                overall = mean_root**2
            out[(scene, label)] = {
                "scene_test": st,
                "dual_vqa": retention,
                "vision_sim": ls,
                "overall": overall,
            }
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    blind = json.loads((args.pack_dir / "BLIND_MAP.json").read_text(encoding="utf-8"))
    recs = load_rankings(args.responses)
    if not recs:
        raise SystemExit(f"No bvb-human-rank-v1 JSON in {args.responses}")

    out_dir = args.out or (args.pack_dir / "scored")
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_points: dict[str, list[float]] = defaultdict(list)
    pair_wins: dict[tuple[str, str], list[float]] = defaultdict(list)
    scene_rank: dict[tuple[str, str], list[float]] = defaultdict(list)
    scenes = set()

    for rec in recs:
        for item in rec.get("rankings") or []:
            scene = item.get("scene_id")
            order = item.get("rank") or []
            labels = [unblind(scene, code, blind) for code in order]
            if None in labels or len(labels) != len(order):
                continue
            scenes.add(scene)
            for i, label in enumerate(labels):
                rank_points[label].append(i + 1)
                scene_rank[(scene, label)].append(i + 1)
            for i, a in enumerate(labels):
                for j, b in enumerate(labels):
                    if a == b:
                        continue
                    pair_wins[(a, b)].append(1.0 if i < j else 0.0)

    models = sorted(rank_points)
    by_model = []
    for label in models:
        vals = rank_points[label]
        by_model.append({
            "model": label,
            "n": len(vals),
            "mean_rank": None if not vals else round(mean(vals), 3),
            "best1_rate": None if not vals else round(sum(1 for v in vals if v == 1) / len(vals), 3),
        })
    by_model.sort(key=lambda r: (r["mean_rank"] is None, r["mean_rank"] or 99))
    write_csv(out_dir / "human_mean_rank.csv", by_model, ["model", "n", "mean_rank", "best1_rate"])

    win_rows = []
    for a in models:
        row = {"model": a}
        wins = []
        for b in models:
            if a == b:
                row[b] = 0.5
                continue
            xs = pair_wins.get((a, b), [])
            row[b] = None if not xs else round(mean(xs), 3)
            if xs:
                wins.append(mean(xs))
        row["mean_win"] = None if not wins else round(mean(wins), 3)
        win_rows.append(row)
    write_csv(out_dir / "human_pairwise.csv", win_rows, ["model", *models, "mean_win"])

    auto = auto_scores(blind, sorted(scenes))
    cal_rows = []
    h_over, st, dual, vis, ovr = [], [], [], [], []
    for (scene, label), ranks_ in sorted(scene_rank.items()):
        human = mean(ranks_)
        auto_row = auto.get((scene, label), {})
        cal_rows.append({
            "scene_id": scene,
            "model": label,
            "human_mean_rank": None if human is None else round(human, 3),
            **auto_row,
        })
        if human is None:
            continue
        # Lower rank is better; flip for correlation with accuracy-like scores.
        score = -human
        if auto_row.get("scene_test") is not None:
            h_over.append(score)
            st.append(float(auto_row["scene_test"]))
        if auto_row.get("dual_vqa") is not None:
            dual.append((score, float(auto_row["dual_vqa"])))
        if auto_row.get("vision_sim") is not None:
            vis.append((score, float(auto_row["vision_sim"])))
        if auto_row.get("overall") is not None:
            ovr.append((score, float(auto_row["overall"])))
    write_csv(
        out_dir / "human_by_scene_model.csv",
        cal_rows,
        ["scene_id", "model", "human_mean_rank", "scene_test", "dual_vqa", "vision_sim", "overall"],
    )

    corr_rows = []

    def corr_report(name: str, pairs: list[tuple[float, float]]) -> dict | None:
        if len(pairs) < 3:
            return None
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        rho = spearman(xs, ys)
        row = {
            "metric": name,
            "n": len(pairs),
            "spearman_rho": None if rho is None else round(rho, 3),
        }
        corr_rows.append(row)
        return row

    st_row = corr_report("scene_test", [(x, y) for x, y in zip(h_over, st)])
    dual_row = corr_report("dual_vqa", dual)
    vis_row = corr_report("latent_sim", vis)
    ovr_row = corr_report("overall", ovr)
    write_csv(out_dir / "human_metric_correlation.csv", corr_rows, ["metric", "n", "spearman_rho"])

    report = {
        "overall_axes": ["dual_vqa", "vision_sim"],
        "overall_aggregation": "square_root_mean",
        "correlation_unit": "scene_model",
        "n_raters": len(recs),
        "n_scenes": len(scenes),
        "n_models": len(models),
        "mean_rank": {r["model"]: r["mean_rank"] for r in by_model},
        "spearman_neg_rank_vs_scene_test": None if not st_row else st_row["spearman_rho"],
        "spearman_neg_rank_vs_dual_vqa": None if not dual_row else dual_row["spearman_rho"],
        "spearman_neg_rank_vs_vision_sim": None if not vis_row else vis_row["spearman_rho"],
        "spearman_neg_rank_vs_overall": None if not ovr_row else ovr_row["spearman_rho"],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
