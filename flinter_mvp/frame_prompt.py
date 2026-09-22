"""Model-visible state renderer for the Agentic S1 frame-ID contract.

Imported minimal surface of `flinter_mvp.ranking_pilot.build_frame_prompt`
(from the coherence_mvp_flinter worktree): the exact instruction text, the
canonical JSON serialization, and the physical `frame_<n>` choice IDs.  The
networked scorer half of that module is deliberately not imported here.

Prompt states are pre-sanitized observable dicts of the shape:

    {"candidates": [{"frame": int, "frame_id": "frame_<zero-padded-n>",
                     "gate": <gate>, "revealed": false | {continuous,
                     break_kind}}, ...],
     "choices": [{"frame_id": "frame_<n>"}, ...],
     "free_signals": {...}}

`choices` lists exactly the unresolved candidates (`revealed` is False) in
candidate order.  Pure functions only -- no I/O, no network, no labels.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

PROMPT_VERSION = "agentic-s1-frame-v3"
FRAME_DIGITS = 6

# Exact instruction string from ranking_pilot.build_frame_prompt.  Changing
# it changes every exported prompt, so it is pinned by PROMPT_VERSION.
INSTRUCTION = (
    "Select the next inspection to resolve temporal-edit evidence. "
    "Consider second seams when relevant. Frame IDs are clip-local coordinates; "
    "their absolute values do not transfer between examples. Reply with exactly "
    "one frame ID from the choices; do not output a final type.\n\n"
)

# Fields that would leak the hidden outcome/label if they ever appeared in a
# model-visible prompt.  `revealed` verdicts keep only {continuous,
# break_kind} -- observed evidence, not ground truth.
FORBIDDEN_KEYS = (
    "abstain", "clean_fp", "composite", "det_correct", "expected",
    "first_new_discontinuity_call", "operator", "outcome", "total_judges",
    "truth", "type_correct",
)

TOP_LEVEL_KEYS = frozenset({"candidates", "choices", "free_signals"})
CANDIDATE_KEYS = frozenset({"frame", "frame_id", "gate", "revealed"})
CHOICE_KEYS = frozenset({"frame_id"})
REVEALED_KEYS = frozenset({"continuous", "break_kind"})
SIGNAL_KEYS = frozenset({
    "recur_frac_max", "recur_argmax_frame", "dup_dense_last_frame",
    "dup_trailing_run", "echo_best_score", "echo_best_pair",
    "scenecut_iframe_frames", "photo_jump_top3",
})


def canonical_json(state: Mapping[str, Any]) -> str:
    """Deterministic serialization: sorted keys, compact separators."""
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def render_prompt(prompt_state: Mapping[str, Any]) -> str:
    """INSTRUCTION + canonical JSON -- the exact serve-side prompt shape."""
    return INSTRUCTION + canonical_json(prompt_state)


def frame_id(frame: int) -> str:
    if (isinstance(frame, bool) or not isinstance(frame, int)
            or not 0 <= frame < 10 ** FRAME_DIGITS):
        raise ValueError(
            f"frame must be in [0, {10 ** FRAME_DIGITS}): {frame!r}")
    return f"frame_{frame:0{FRAME_DIGITS}d}"


def normalize_prompt_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade reviewed legacy ``frame_<n>`` IDs to canonical fixed width.

    The action-utility benchmark predates the fixed-width scoring contract.  We
    accept only its exact legacy physical ID or the canonical v2 ID, verify the
    original choices, then rebuild canonical choices.  Arbitrary identifiers
    still fail closed.
    """
    if not isinstance(state, Mapping):
        raise ValueError("prompt state must be an object")
    unknown_top = set(state) - TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"unknown top-level prompt keys: {sorted(unknown_top)}")
    candidates = state.get("candidates")
    choices = state.get("choices")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a nonempty list")

    canonical_candidates: list[dict[str, Any]] = []
    source_unresolved: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate must be an object")
        frame = candidate.get("frame")
        canonical = frame_id(frame)
        source_id = candidate.get("frame_id")
        if source_id not in {f"frame_{frame}", canonical}:
            raise ValueError(
                f"source frame_id mismatch: {source_id!r} vs frame {frame!r}")
        item = dict(candidate)
        item["frame_id"] = canonical
        if isinstance(item.get("revealed"), Mapping):
            item["revealed"] = dict(item["revealed"])
        else:
            source_unresolved.append(str(source_id))
        canonical_candidates.append(item)

    source_choices: list[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping) or set(choice) != CHOICE_KEYS:
            raise ValueError("each choice must contain only frame_id")
        source_choices.append(choice["frame_id"])
    if source_choices != source_unresolved:
        raise ValueError("source choices must equal unresolved candidates in order")

    out = {
        "candidates": canonical_candidates,
        "choices": [
            {"frame_id": candidate["frame_id"]}
            for candidate in canonical_candidates
            if candidate.get("revealed") is False
        ],
        "free_signals": dict(state.get("free_signals", {})),
    }
    validate_prompt_state(out)
    return out


def shift_frame_id(value: str, offset: int) -> str:
    prefix = "frame_"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"invalid frame ID: {value!r}")
    digits = value.removeprefix(prefix)
    if not digits.isdigit():
        raise ValueError(f"invalid frame ID: {value!r}")
    return frame_id(int(digits) + offset)


def _shift_number(value: Any, offset: int) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"frame reference must be numeric or null: {value!r}")
    return value + offset


def max_frame_reference(state: Mapping[str, Any]) -> int:
    """Largest integral frame coordinate used by a canonical prompt state."""
    values: list[int] = [candidate["frame"] for candidate in state["candidates"]]
    signals = state["free_signals"]
    for key in ("recur_argmax_frame", "dup_dense_last_frame"):
        value = signals.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    for key in ("echo_best_pair", "scenecut_iframe_frames"):
        for value in signals.get(key) or []:
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
    for item in signals.get("photo_jump_top3") or []:
        if isinstance(item, Mapping):
            value = item.get("frame")
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
    return max(values)


def shift_prompt_state(state: Mapping[str, Any], offset: int) -> dict[str, Any]:
    """Translate all frame-valued coordinates by one nonnegative offset."""
    validate_prompt_state(state)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("timeline offset must be a nonnegative integer")
    if offset == 0:
        return {
            "candidates": [
                {**candidate,
                 "revealed": (dict(candidate["revealed"])
                              if isinstance(candidate["revealed"], Mapping)
                              else False)}
                for candidate in state["candidates"]
            ],
            "choices": [dict(choice) for choice in state["choices"]],
            "free_signals": dict(state["free_signals"]),
        }

    candidates = []
    for candidate in state["candidates"]:
        shifted_frame = candidate["frame"] + offset
        item = dict(candidate)
        item["frame"] = shifted_frame
        item["frame_id"] = frame_id(shifted_frame)
        if isinstance(item["revealed"], Mapping):
            item["revealed"] = dict(item["revealed"])
        candidates.append(item)

    signals = dict(state["free_signals"])
    for key in ("recur_argmax_frame", "dup_dense_last_frame"):
        if key in signals:
            signals[key] = _shift_number(signals[key], offset)
    for key in ("echo_best_pair", "scenecut_iframe_frames"):
        if key in signals:
            signals[key] = [
                _shift_number(value, offset) for value in (signals[key] or [])
            ]
    if "photo_jump_top3" in signals:
        signals["photo_jump_top3"] = [
            {**item, "frame": _shift_number(item.get("frame"), offset)}
            for item in (signals["photo_jump_top3"] or [])
        ]

    shifted = {
        "candidates": candidates,
        "choices": [
            {"frame_id": candidate["frame_id"]}
            for candidate in candidates if candidate["revealed"] is False
        ],
        "free_signals": signals,
    }
    validate_prompt_state(shifted)
    return shifted


def validate_prompt_state(state: Mapping[str, Any]) -> None:
    """Fail-closed structural checks on one observable prompt state."""
    if not isinstance(state, Mapping):
        raise ValueError("prompt state must be an object")
    unknown_top = set(state) - TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"unknown top-level prompt keys: {sorted(unknown_top)}")
    signals = state.get("free_signals")
    if not isinstance(signals, Mapping):
        raise ValueError("free_signals must be an object")
    unknown_signals = set(signals) - SIGNAL_KEYS
    if unknown_signals:
        raise ValueError(f"unknown free-signal keys: {sorted(unknown_signals)}")
    candidates = state.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    seen: set[str] = set()
    unresolved: list[str] = []
    for cand in candidates:
        if not isinstance(cand, Mapping):
            raise ValueError("candidate must be an object")
        unknown_candidate = set(cand) - CANDIDATE_KEYS
        if unknown_candidate:
            raise ValueError(
                f"unknown candidate keys: {sorted(unknown_candidate)}")
        frame = cand.get("frame")
        fid = cand.get("frame_id")
        if (isinstance(frame, bool) or not isinstance(frame, int)
                or frame < 0):
            raise ValueError(f"bad physical frame id: {frame!r}")
        if fid != frame_id(frame):
            raise ValueError(f"frame_id mismatch: {fid!r} vs frame {frame!r}")
        if fid in seen:
            raise ValueError(f"duplicate frame_id {fid!r}")
        seen.add(fid)
        revealed = cand.get("revealed")
        if revealed is False:
            unresolved.append(fid)
        elif not isinstance(revealed, Mapping):
            raise ValueError(f"revealed must be false or an object in {fid!r}")
        else:
            unknown_revealed = set(revealed) - REVEALED_KEYS
            if unknown_revealed:
                raise ValueError(
                    f"unknown revealed keys in {fid!r}: "
                    f"{sorted(unknown_revealed)}")
    choices = state.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a nonempty list")
    for choice in choices:
        if not isinstance(choice, Mapping) or set(choice) != CHOICE_KEYS:
            raise ValueError("each choice must contain only frame_id")
    offered = [c["frame_id"] for c in choices]
    if offered != unresolved:
        raise ValueError("choices must equal unresolved candidates in order")
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in FORBIDDEN_KEYS:
                    raise ValueError(
                        f"hidden-label key {key!r} in prompt state")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(state)


def offered_frame_ids(state: Mapping[str, Any]) -> list[str]:
    """Frame IDs the model is allowed to answer, in offered order."""
    return [c["frame_id"] for c in state["choices"]]


def hash_permutation(n: int, seed: str) -> list[int]:
    """Deterministic permutation of range(n) seeded by a string.

    Hash-order shuffle: index i sorts by sha256(seed|i).  Same inputs always
    give the same permutation on any machine, with no RNG state.
    """
    order = list(range(n))
    order.sort(key=lambda i: hashlib.sha256(f"{seed}|{i}".encode()).digest())
    return order


def permuted_prompt_state(state: Mapping[str, Any],
                          perm: Sequence[int]) -> dict[str, Any]:
    """Return a new prompt state with candidates reordered by `perm`.

    Choices are rebuilt as the unresolved frame_ids in the new candidate
    order, so the offered set is identical -- only presentation order moves.
    """
    candidates = state["candidates"]
    n = len(candidates)
    if sorted(perm) != list(range(n)):
        raise ValueError("perm must be a permutation of candidate indices")
    new_candidates = [dict(candidates[i]) for i in perm]
    new_choices = [{"frame_id": c["frame_id"]} for c in new_candidates
                   if c.get("revealed") is False]
    out = dict(state)
    out["candidates"] = new_candidates
    out["choices"] = new_choices
    return out
