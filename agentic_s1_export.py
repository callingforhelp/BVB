"""Deterministic Agentic S1 v1 hard-label CHAT dataset exporter.

Reads the counterfactual action-utility benchmark
(`results/action_utility_benchmark.json` from the flinter worktree) and
emits Fireworks CHAT-format JSONL:

    {"messages": [{"role": "user", "content": <frame-prompt>, "weight": 0},
                  {"role": "assistant", "content": "frame_<n>"}]}

Hard-label v1 policy: a state is eligible only when it has >=2 actions and
exactly ONE action attains the lexicographic utility minimum (port of
local_policy_ranker.utility_key).  Uniform (all-tied) and partial-tie
states are excluded, as are single-action non-decision states.

Split happens on SOURCE before augmentation. Training and validation receive
deterministic candidate-order and clip-coordinate invariance variants; variant
zero always preserves the original timeline. Labels are stable physical frame
IDs rather than arbitrary letters, so reordering never changes the target. A
serving engine with sequence scoring can normalize the scores of the complete
candidate strings into a Jev-like distribution.

Outputs: train.jsonl, val.jsonl, quality_report.json, manifest.json.
Stdlib only; no network.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from flinter_mvp.frame_prompt import (
    PROMPT_VERSION,
    hash_permutation,
    max_frame_reference,
    normalize_prompt_state,
    offered_frame_ids,
    permuted_prompt_state,
    render_prompt,
    shift_frame_id,
    shift_prompt_state,
    validate_prompt_state,
)

BASE_MODEL = "accounts/fireworks/models/qwen3p6-35b-a3b"
DEFAULT_AUGMENT = 4
DEFAULT_VAL_AUGMENT = 4
DEFAULT_MAX_SHORTCUT_ACCURACY = 0.50
MAX_PERM_NONCE = 1000

EXPORTER_FILE = Path(__file__).resolve()


# ------------------------------------------------------------------ labels

def _num(x: Any, default: float = 0.0) -> float:
    return (float(x) if isinstance(x, (int, float))
            and not isinstance(x, bool) and math.isfinite(float(x))
            else default)


def utility_key(action: Mapping[str, Any]) -> tuple:
    """Lexicographic utility ordering; smaller is better.

    Port of local_policy_ranker.utility_key (coherence_mvp_flinter): clean
    false-positives are worst, then detection, then typing, then judge
    spend, then how early fresh discontinuity evidence arrives.
    """
    fj = action.get("first_new_discontinuity_call")
    return (
        int(bool(action.get("clean_fp"))),
        -int(bool(action.get("det_correct"))),
        -int(bool(action.get("type_correct"))),
        _num(action.get("total_judges"), 999.0),
        999.0 if fj is None else _num(fj, 999.0),
    )


def unique_best(state: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    """Return (best_action, None) or (None, exclusion_reason)."""
    actions = state.get("actions")
    if not isinstance(actions, list) or len(actions) < 2:
        return None, "non_decision"
    keys = [utility_key(a) for a in actions]
    best = min(keys)
    nbest = keys.count(best)
    if nbest == len(keys):
        return None, "uniform"
    if nbest > 1:
        return None, "partial_tie"
    return actions[keys.index(best)], None


def label_for(state: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    """Resolve the best action to its offered physical frame ID."""
    obs = state["observable"]
    validate_prompt_state(obs)
    candidates = obs["candidates"]
    idx = action.get("index")
    if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(candidates):
        raise ValueError(f"bad action index {idx!r} in {state.get('clip')}")
    cand = candidates[idx]
    if cand["frame"] != action.get("frame"):
        raise ValueError(
            f"action index/frame mismatch in {state.get('clip')} step "
            f"{state.get('step')}: index {idx} -> frame {cand['frame']}, "
            f"action claims {action.get('frame')}")
    fid = cand["frame_id"]
    if fid not in offered_frame_ids(obs):
        raise ValueError(f"label {fid} not among offered choices")
    return fid


# -------------------------------------------------------------- augment

def variant_permutations(n: int, variants: int, seed_base: str) -> list[list[int]]:
    """Exactly `variants` permutations; element 0 is the identity.

    Unique orderings are factorial-capped, then cycled. Rows remain distinct
    because nonzero variants also receive deterministic timeline translations.
    """
    perms = [list(range(n))]
    cap = min(variants, math.factorial(n))
    nonce = 0
    while len(perms) < cap and nonce < MAX_PERM_NONCE:
        nonce += 1
        perm = hash_permutation(n, f"{seed_base}|{nonce}")
        if perm not in perms:
            perms.append(perm)
    return [perms[index % len(perms)] for index in range(variants)]


def timeline_offset(state: Mapping[str, Any], variant: int,
                    seed_base: str) -> int:
    """Deterministic semantics-preserving coordinate translation."""
    if variant == 0:
        return 0
    maximum = max_frame_reference(state)
    available = 10 ** 6 - 1 - maximum
    if available < 1:
        raise ValueError("prompt state has no room for timeline translation")
    minimum = 1000 if available >= 1000 else 1
    span = available - minimum + 1
    digest = hashlib.sha256(
        f"{seed_base}|timeline|{variant}".encode()).digest()
    return minimum + int.from_bytes(digest[:8], "big") % span


def state_rows(state: Mapping[str, Any], label: str, variants: int,
               *, augment: bool) -> list[dict[str, Any]]:
    """Emit CHAT rows for one state; variant 0 is the canonical order."""
    obs = state["observable"]
    n = len(obs["candidates"])
    seed = f"{PROMPT_VERSION}|{state['clip']}|{state['step']}"
    perms = (variant_permutations(n, variants, seed) if augment
             else [list(range(n))])
    rows = []
    for v, perm in enumerate(perms):
        pstate = permuted_prompt_state(obs, perm)
        offset = timeline_offset(obs, v, seed)
        pstate = shift_prompt_state(pstate, offset)
        variant_label = shift_frame_id(label, offset)
        prompt = render_prompt(pstate)
        offered = offered_frame_ids(pstate)
        if variant_label not in offered:
            raise ValueError(
                f"label {variant_label} lost under permutation {perm}")
        rows.append({
            "messages": [
                {"role": "user", "content": prompt, "weight": 0},
                {"role": "assistant", "content": variant_label},
            ],
            "_row": {"clip": state["clip"], "step": state["step"],
                     "variant": v, "label": variant_label,
                     "canonical_label": label,
                     "timeline_offset": offset,
                     "offered": offered,
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        })
    return rows


# ------------------------------------------------------------------ load

def load_benchmark(path: Path) -> dict[str, Any]:
    bench = json.loads(path.read_text())
    states = bench.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("benchmark has no states")
    for st in states:
        for key in ("clip", "source", "step", "observable", "actions"):
            if key not in st:
                raise ValueError(f"state missing {key!r}")
        if not isinstance(st["source"], str) or not st["source"]:
            raise ValueError(f"state {st.get('clip')} missing source")
        st["observable"] = normalize_prompt_state(st["observable"])
    return bench


def select_eligible(states: Sequence[Mapping[str, Any]]) -> tuple[list, dict]:
    eligible, excluded = [], {"non_decision": 0, "uniform": 0, "partial_tie": 0}
    for st in states:
        action, reason = unique_best(st)
        if action is None:
            excluded[reason] += 1
        else:
            eligible.append((st, action))
    return eligible, excluded


# ------------------------------------------------------------- quality

def check_conflicts(pairs: Sequence[tuple[str, str]]) -> list[dict]:
    """Identical prompt bytes must never carry two different labels."""
    seen: dict[str, str] = {}
    conflicts = []
    for prompt_sha, label in pairs:
        prev = seen.setdefault(prompt_sha, label)
        if prev != label:
            conflicts.append({"prompt_sha256": prompt_sha,
                              "labels": sorted({prev, label})})
    return conflicts


def shortcut_baseline(train_meta: list[dict], val_meta: list[dict],
                      max_accuracy: float) -> dict[str, Any]:
    """Measure whether a held-out label is guessable without reading state.

    The frequency prior uses only train-label counts and the held-out offered
    set. Ties preserve candidate order. A near-perfect score means the split is
    validating an absolute-frame shortcut rather than evidence selection.
    """
    if not val_meta:
        return {"status": "not_applicable", "passed": True}
    frequencies = Counter(row["label"] for row in train_meta)
    frequency_correct = 0
    hybrid_correct = 0
    position_correct = Counter()
    random_expected = 0.0
    unseen = 0
    for row in val_meta:
        offered = row["offered"]
        ranked = max(
            enumerate(offered), key=lambda item: (frequencies[item[1]], -item[0]))
        known_prediction = ranked[1] if frequencies[ranked[1]] > 0 else None
        frequency_correct += int(known_prediction == row["label"])
        hybrid_prediction = known_prediction or offered[0]
        hybrid_correct += int(hybrid_prediction == row["label"])
        for position, candidate in enumerate(offered):
            position_correct[position] += int(candidate == row["label"])
        random_expected += 1.0 / len(offered)
        unseen += int(row["label"] not in frequencies)
    total = len(val_meta)
    accuracy = frequency_correct / total
    hybrid_accuracy = hybrid_correct / total
    best_position, best_position_correct = max(
        position_correct.items(), key=lambda item: (item[1], -item[0]))
    strongest = max(accuracy, hybrid_accuracy, best_position_correct / total)
    return {
        "status": "measured",
        "frequency_prior_correct": frequency_correct,
        "frequency_prior_total": total,
        "frequency_prior_accuracy": accuracy,
        "frequency_then_first_correct": hybrid_correct,
        "frequency_then_first_accuracy": hybrid_accuracy,
        "max_allowed_frequency_prior_accuracy": max_accuracy,
        "best_fixed_position": best_position,
        "best_fixed_position_correct": best_position_correct,
        "best_fixed_position_accuracy": best_position_correct / total,
        "random_expected_accuracy": random_expected / total,
        "unseen_target_labels": unseen,
        "strongest_shortcut_accuracy": strongest,
        "passed": strongest <= max_accuracy,
    }


def quality_report(train_meta: list[dict], val_meta: list[dict],
                   excluded: dict, eligible: list, val_sources: set,
                   max_shortcut_accuracy: float) -> dict:
    labels_ok = all(
        m["label"] in m["offered"] for m in train_meta + val_meta)
    train_sources = {m["source"] for m in train_meta}
    val_src = {m["source"] for m in val_meta}
    disjoint = not (train_sources & val_src) and val_src <= val_sources
    conflicts = check_conflicts(
        [(m["prompt_sha256"], m["label"]) for m in train_meta + val_meta])
    shortcut = shortcut_baseline(
        train_meta, val_meta, max_shortcut_accuracy)
    return {
        "labels_resolve_to_offered_frame": labels_ok,
        "sources_disjoint": disjoint,
        "conflicting_label_prompts": conflicts,
        "shortcut_baseline": shortcut,
        "excluded": excluded,
        "eligible_states": len(eligible),
        "train_states": len({(m["clip"], m["step"]) for m in train_meta}),
        "val_states": len({(m["clip"], m["step"]) for m in val_meta}),
        "train_rows": len(train_meta),
        "val_rows": len(val_meta),
        "rows": {"train": train_meta, "val": val_meta},
        "passed": labels_ok and disjoint and not conflicts and shortcut["passed"],
    }


# ---------------------------------------------------------------- export

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=EXPORTER_FILE.parent,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def _write(path: Path, data: bytes, force: bool) -> None:
    if path.exists() and path.read_bytes() != data and not force:
        raise SystemExit(
            f"{path} exists with different content; pass --force to overwrite")
    path.write_bytes(data)


def export(benchmark_path: Path, out_dir: Path, val_sources: Sequence[str],
           *, augment: int = DEFAULT_AUGMENT, all_data: bool = False,
           force: bool = False,
           val_augment: int = DEFAULT_VAL_AUGMENT,
           max_shortcut_accuracy: float = DEFAULT_MAX_SHORTCUT_ACCURACY) -> dict:
    bench = load_benchmark(benchmark_path)
    bench_bytes = benchmark_path.read_bytes()
    eligible, excluded = select_eligible(bench["states"])

    held = set() if all_data else set(val_sources)
    source_names = {str(st["source"]) for st in bench["states"]}
    if all_data and val_sources:
        raise SystemExit("--all-data cannot be combined with --val-source")
    if not all_data and not held:
        raise SystemExit("pass --val-source <src> or --all-data")
    missing_sources = held - source_names
    if missing_sources:
        raise SystemExit(
            f"validation source(s) absent from benchmark: "
            f"{sorted(missing_sources)}")
    train_part = [(s, a) for s, a in eligible if s["source"] not in held]
    val_part = [(s, a) for s, a in eligible if s["source"] in held]

    train_rows, train_meta, val_rows, val_meta = [], [], [], []
    for part, variants, aug, rows, meta in (
            (train_part, augment, True, train_rows, train_meta),
            (val_part, val_augment, True, val_rows, val_meta)):
        for st, action in part:
            label = label_for(st, action)
            for row in state_rows(st, label, variants, augment=aug):
                m = dict(row.pop("_row"))
                m["source"] = st["source"]
                rows.append(row)
                meta.append(m)

    train_bytes = b"".join(
        (json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for r in train_rows)
    val_bytes = b"".join(
        (json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for r in val_rows)

    report = quality_report(
        train_meta, val_meta, excluded, eligible, held,
        max_shortcut_accuracy)
    if not all_data and not val_rows:
        report["passed"] = False
        report["validation_error"] = "held-out split has no eligible rows"
    report["source_sha256"] = _sha256(bench_bytes)
    report["benchmark_file"] = benchmark_path.name

    manifest = {
        "schema": "agentic-s1-export/v1",
        "prompt_version": PROMPT_VERSION,
        "tie_policy": "unique_lexicographic_best",
        "base_model": BASE_MODEL,
        "answer_contract": "stable_physical_frame_id",
        "serving_score_contract": "sequence_logprob_over_complete_candidate_strings",
        "exporter": {"file": EXPORTER_FILE.name,
                     "sha256": _sha256(EXPORTER_FILE.read_bytes()),
                     "git_revision": _git_revision()},
        "source": {"benchmark_file": benchmark_path.name,
                   "benchmark_sha256": _sha256(bench_bytes),
                   "states": len(bench["states"])},
        "split": {"mode": "all_data" if all_data else "holdout",
                  "train_sources": sorted({s["source"] for s, _ in train_part}),
                  "val_sources": sorted(held)},
        "augment": {"scheme": "sha256-hash-order", "seed_scope": "clip|step",
                    "train_variants_requested": augment,
                    "val_variants": val_augment,
                    "timeline_translation": "variant0_identity; other_variants_sha256_offset"},
        "shortcut_gate": {
            "max_frequency_prior_accuracy": max_shortcut_accuracy,
            "observed": report["shortcut_baseline"],
        },
        "outputs": {
            "train.jsonl": {"rows": len(train_rows), "sha256": _sha256(train_bytes)},
            "val.jsonl": {"rows": len(val_rows), "sha256": _sha256(val_bytes)},
        },
        "dataset_ids": {
            "train": f"jev-act-v1-{_sha256(train_bytes)[:8]}",
            "val": (f"jev-act-v1-{_sha256(val_bytes)[:8]}"
                    if val_rows else None),
        },
        "quality_passed": report["passed"],
    }

    if not report["passed"]:
        shortcut = report["shortcut_baseline"]
        validation_error = report.get("validation_error")
        raise SystemExit(
            "quality gate failed: "
            f"labels_ok={report['labels_resolve_to_offered_frame']} "
            f"sources_disjoint={report['sources_disjoint']} "
            f"conflicts={len(report['conflicting_label_prompts'])} "
            f"shortcut={shortcut} "
            f"validation_error={validation_error!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "train.jsonl", train_bytes, force)
    _write(out_dir / "val.jsonl", val_bytes, force)
    _write(out_dir / "quality_report.json",
           (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(), force)
    _write(out_dir / "manifest.json",
           (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), force)

    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", type=Path, required=True,
                    help="action_utility_benchmark.json path")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory for train/val/report/manifest")
    ap.add_argument("--val-source", action="append", default=[],
                    help="held-out source id; repeatable")
    ap.add_argument("--all-data", action="store_true",
                    help="put every eligible state in train (empty val)")
    ap.add_argument("--augment", type=int, default=DEFAULT_AUGMENT,
                    help="candidate-order variants per train state")
    ap.add_argument("--val-augment", type=int, default=DEFAULT_VAL_AUGMENT,
                    help="order/coordinate invariance variants per val state")
    ap.add_argument("--force", action="store_true",
                    help="overwrite differing existing outputs")
    ap.add_argument("--max-shortcut-accuracy", type=float,
                    default=DEFAULT_MAX_SHORTCUT_ACCURACY,
                    help="maximum held-out train-label-frequency accuracy")
    args = ap.parse_args(argv)
    if args.augment < 1:
        ap.error("--augment must be >= 1")
    if args.val_augment < 1:
        ap.error("--val-augment must be >= 1")
    if not 0 <= args.max_shortcut_accuracy <= 1:
        ap.error("--max-shortcut-accuracy must be in [0, 1]")
    manifest = export(args.benchmark, args.out, args.val_source,
                      augment=args.augment, all_data=args.all_data,
                      force=args.force,
                      val_augment=args.val_augment,
                      max_shortcut_accuracy=args.max_shortcut_accuracy)
    print(json.dumps({"dataset_ids": manifest["dataset_ids"],
                      "outputs": manifest["outputs"],
                      "split": manifest["split"]}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
