"""Offline replay of the jev_loop reveal cascade, two policy arms.

Both arms run on the SAME `load_clip_state` output -- the same gated
candidate union and the same cached pair-judge verdicts -- and share the
SAME deterministic arbitration (the seam-count / dup-signature remap that
`run_clip` applies after its loop).  The arms differ only in how they
choose the next candidate to reveal and when they stop:

- `existing`: replays a recorded jevloop run's reveal order and reuses its
  recorded Jev answers (`final`, `fan_probs`) as the arbitration inputs.
  This is a faithful re-simulation, not a new Jev call.
- `bounded`: the Flinter `choose_next_request` policy drives reveals over
  the adapter's `EvidenceState`; a deterministic implementation of the
  pack's typing guide supplies the arbitration inputs.

No network, no video access -- everything consumes the cached verdict
records.

Abstention semantics: for the existing arm, `abstained` replays the
recorded flag and maps to `corrupt_untyped` exactly as `run_clip` does.
For the bounded arm a `review` decision means "the policy could not
certify a conclusion"; detection is still arbitrated from the revealed
evidence, but the specific type is withheld -> `corrupt_untyped` when the
arbitrated type is an edit, `none` when it is not.  That mirrors the
existing refusal class without turning a clean-clip review into a false
positive.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .core import StopDecision, choose_next_request, record_trace
from .jev_adapter import COHERENCE_RUBRIC, to_evidence_state

# Reveal budget for the bounded arm.  jev_loop.MAX_ITERS is 8, but the
# deterministic seam-first acquisition is weaker than the learned Jev
# policy: on two-seam room_swaps it can spend most of the budget on false
# predicted seams before reaching either true seam.  10 keeps the loop
# bounded while covering the observed two-seam cases; total spend stays
# below the recorded existing-arm total (474).
MAX_ITERS = 10
DUP_TRAIL_MIN = 50     # mirrors jev_loop.DUP_TRAIL_MIN
SEAM_TOL = 10          # frames; mirrors the parent seam-target gate
ECHO_MIN = 10.0        # typing guide: echo_best_score > 10 => reverse
RECUR_LOOP_MIN = 0.9   # typing guide: recur_frac >= 0.9 => loop

TYPES = ("loop", "reverse", "room_swap", "splice", "timewarp")


# ---------------------------------------------------------------- helpers

def reveal(st: dict, index: int) -> dict:
    """Reveal one candidate's cached verdict (the replay's only 'judge')."""
    cand = st["candidates"][index]
    cand["revealed"] = cand["verdict"]
    return cand


def judged_candidates(st: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [c for c in st["candidates"] if c.get("revealed")]


def discontinuous_frames(st: Mapping[str, Any]) -> list[int]:
    return [
        int(c["frame"])
        for c in judged_candidates(st)
        if c["revealed"].get("continuous") is False
    ]


def scene_change_count(st: Mapping[str, Any]) -> int:
    return len(
        [
            c
            for c in judged_candidates(st)
            if c["revealed"].get("break_kind") == "scene_change"
        ]
    )


def dup_swap_signature(st: Mapping[str, Any]) -> bool:
    """v9 signature: >=1 revealed scene_change + long pixel-continuous tail."""
    return (
        scene_change_count(st) >= 1
        and (st["free_signals"].get("dup_trailing_run") or 0) >= DUP_TRAIL_MIN
    )


# ---------------------------------------------------------------- arbitration
# Code-exact copy of the deterministic arbitration at the tail of
# jev_loop.run_clip, parameterized so both replay arms share it.  Any
# change to that logic must be mirrored here (and flagged by the replay's
# fidelity check against recorded composite types).

def arbitrate(
    break_type_choice: "str | None",
    corrupted_noul: float,
    fan: Mapping[str, "float | None"],
    st: Mapping[str, Any],
    *,
    abstained: bool,
) -> str:
    """Deterministic final-type arbitration over the revealed evidence."""
    fan_type = (
        max(fan, key=lambda k: fan.get(k) or 0) if fan else None
    )
    composite = (
        fan_type
        if (break_type_choice == "none" and corrupted_noul > 0.5 and fan_type)
        else break_type_choice
    )
    ndis = len(discontinuous_frames(st))
    dup_swap = dup_swap_signature(st)
    if composite == "splice" and (ndis >= 2 or dup_swap):
        composite = "room_swap"
    elif composite == "room_swap" and ndis < 2 and not dup_swap:
        composite = "splice"
    elif composite in ("loop", "reverse", "timewarp") and dup_swap:
        composite = "room_swap"
    if abstained:
        composite = "corrupt_untyped"
    return composite or "none"


def seam_positions(st: Mapping[str, Any]) -> set[int]:
    """Signal-predicted seam frames copied from the parent gate contract."""
    fs = st["free_signals"]
    positions = {int(f) for f in fs.get("scenecut_iframe_frames") or []}
    dense_last = fs.get("dup_dense_last_frame")
    if dense_last is not None:
        positions.add(int(dense_last))
    for value in fs.get("echo_best_pair") or []:
        if isinstance(value, (int, float)):
            positions.add(int(value))
    return positions


def seam_target_index(st: Mapping[str, Any]) -> int | None:
    """Choose the unjudged candidate nearest a predicted seam."""
    positions = seam_positions(st)
    unjudged = [
        i for i, candidate in enumerate(st["candidates"])
        if not candidate.get("revealed")
    ]
    if not unjudged:
        return None
    if not positions:
        return unjudged[0]
    best = min(
        unjudged,
        key=lambda i: min(
            abs(int(st["candidates"][i]["frame"]) - position)
            for position in positions
        ),
    )
    distance = min(
        abs(int(st["candidates"][best]["frame"]) - position)
        for position in positions
    )
    return best if distance <= SEAM_TOL else unjudged[0]


def conclusion_determined(st: Mapping[str, Any]) -> bool:
    """True when no further reveal can change the shared arbitration output.

    A single confirmed discontinuity is NOT a determined state: a second
    revealed seam still flips splice <-> room_swap.  The composite is locked
    once a second discontinuity is confirmed, the dup-swap signature is
    complete, or every candidate has been judged.  Signal-typed conclusions
    are handled by the fast path in run_bounded; they are reveal-invariant
    because the dup-swap signature needs dup_trailing_run >= DUP_TRAIL_MIN,
    which is also what vetoes signal_type.
    """
    if dup_swap_signature(st):
        return True
    if len(discontinuous_frames(st)) >= 2:
        return True
    return all(c.get("revealed") for c in st["candidates"])


def floor_candidate_id(st: Mapping[str, Any], min_confidence: float) -> "str | None":
    """First candidate whose revealed verdict meets the evidence floor."""
    for i, c in enumerate(st["candidates"]):
        revealed = c.get("revealed") or {}
        if revealed.get("continuous") is False and float(
            revealed.get("confidence") or 0.0
        ) >= min_confidence:
            return str(i)
    return None


def signal_type(st: Mapping[str, Any]) -> str | None:
    """Return a high-precision type suggested by free deterministic signals.

    These are policy-visible signals from the parent loop, not manifest labels.
    They are used only as an early-stop prior after at least one VLM reveal;
    final arbitration still runs over the revealed state.
    """
    fs = st["free_signals"]
    # A long duplicate tail is a room-swap contradiction signal. Do not take
    # the loop/reverse/timewarp fast path before inspecting enough seams to
    # distinguish a room swap from a single-seam edit.
    if (fs.get("dup_trailing_run") or 0) >= DUP_TRAIL_MIN:
        return None
    if (fs.get("recur_frac_max") or 0) >= RECUR_LOOP_MIN:
        return "loop"
    if fs.get("dup_dense_last_frame") is not None:
        return "timewarp"
    if (fs.get("echo_best_score") or 0) > ECHO_MIN:
        return "reverse"
    return None


def deterministic_verdict(st: Mapping[str, Any]) -> dict:
    """The pack's typing guide, implemented in code.

    Precedence follows the guide text exactly:
      recur_frac>=0.9 => loop; dup_dense_last_frame non-null => timewarp;
      echo>10 => reverse; >=2 confirmed discontinuities => room_swap;
      1 => splice; 0 + quiet signals => none.
    """
    fs = st["free_signals"]
    ndis = len(discontinuous_frames(st))
    if (fs.get("recur_frac_max") or 0) >= RECUR_LOOP_MIN:
        choice = "loop"
    elif fs.get("dup_dense_last_frame") is not None:
        choice = "timewarp"
    elif (fs.get("echo_best_score") or 0) > ECHO_MIN:
        choice = "reverse"
    elif ndis >= 2:
        choice = "room_swap"
    elif ndis == 1:
        choice = "splice"
    else:
        choice = "none"
    corrupted = 0.9 if choice != "none" else 0.1
    fan = {t: (0.9 if t == choice else 0.02) for t in TYPES}
    return {
        "corrupted": {"noul": corrupted},
        "break_type": {"choice": choice},
        "fan": fan,
    }


# ---------------------------------------------------------------- bounded arm

def run_bounded(
    st: dict,
    *,
    video_id: str = "clip",
    max_iters: int = MAX_ITERS,
    rubric=COHERENCE_RUBRIC,
    trace_path: "str | None" = None,
) -> dict:
    """Drive the bounded evidence policy over one clip state.

    Each step the state is adapted to `EvidenceState` and
    `choose_next_request` decides: gather (reveal that candidate's cached
    verdict), select (conclude -- the evidence floor is met), or review
    (abstain).  The shared arbitration owns the final type.
    """
    transcript: list[dict] = []
    selected: "str | None" = None
    stop_status = "review"
    stop_reason = "loop exited"

    for it in range(max_iters):
        remaining = max_iters - len(judged_candidates(st))
        state = to_evidence_state(
            st, video_id=video_id, budget=remaining, rubric=rubric
        )
        # High-precision free signals can terminate after one revealed
        # candidate. This mirrors the parent typing guide and avoids spending
        # the full budget when the signal itself is decisive.
        fast_type = signal_type(st)
        if fast_type and len(judged_candidates(st)) >= 1:
            decision = StopDecision(
                "select", None,
                f"free signal supports {fast_type} after one reveal",
            )
        elif remaining > 0 and not conclusion_determined(st):
            # The arbitrated type is still undecided: one confirmed seam
            # could be splice or room_swap, zero could still hide an edit.
            # Keep inspecting -- seam-target first (the parent backstop),
            # frame order otherwise.  Unjudged candidates always exist in
            # this branch, so seam_target_index never returns None.
            decision = StopDecision(
                "gather", None,
                "type undetermined; inspect candidate nearest a predicted seam",
                state.candidates[seam_target_index(st)],
            )
        else:
            decision = choose_next_request(state)
            floor_id = floor_candidate_id(st, state.rubric.min_evidence_confidence)
            if floor_id is not None:
                # Terminal state with at least one confirmed seam: conclude
                # with the best-effort type instead of abstaining.  This
                # reconciles the generic "multiple candidates meet the
                # rubric" review with this task -- >=2 confirmed seams IS the
                # room_swap signature, not unresolved ambiguity.
                decision = StopDecision(
                    "select",
                    decision.candidate_id
                    if decision.status == "select" else floor_id,
                    "evidence floor met; arbitrated type determined",
                )
            elif decision.status != "review":
                decision = StopDecision(
                    "review", None, "no candidate meets the evidence floor"
                )
        if trace_path is not None:
            record_trace(trace_path, state, decision)

        row = {"iter": it, "status": decision.status, "reason": decision.reason}
        if decision.status == "gather":
            index = int(decision.next_request.candidate_id)
            reveal(st, index)
            row["revealed"] = index
            transcript.append(row)
            continue
        if decision.status == "select":
            selected = decision.candidate_id
            stop_status, stop_reason = "select", decision.reason
        else:
            stop_status, stop_reason = "review", decision.reason
        transcript.append(row)
        break
    else:
        floor_id = floor_candidate_id(st, rubric.min_evidence_confidence)
        if floor_id is not None:
            # Budget spent but the floor is met: emit the best-effort typed
            # answer rather than withholding it.
            stop_status = "select"
            stop_reason = "iteration cap reached; evidence floor met"
            selected = floor_id
        else:
            stop_status, stop_reason = "review", "iteration cap reached"

    abstained = stop_status == "review"
    verdict = deterministic_verdict(st)
    arbitrated = arbitrate(
        verdict["break_type"]["choice"],
        verdict["corrupted"]["noul"],
        verdict["fan"],
        st,
        abstained=False,
    )
    # Abstain withholds the type claim; detection is still arbitrated.
    if abstained:
        composite = "none" if arbitrated == "none" else "corrupt_untyped"
    else:
        composite = arbitrated
    judged = judged_candidates(st)
    return {
        "policy": "bounded",
        "composite_type": composite,
        "arbitrated_type": arbitrated,
        "abstained": abstained,
        "stop_status": stop_status,
        "stop_reason": stop_reason,
        "selected_candidate": selected,
        "judge_calls": len(judged),
        "n_candidates": len(st["candidates"]),
        "discontinuous_at": discontinuous_frames(st),
        "verdict": verdict,
        "transcript": transcript,
    }


# ---------------------------------------------------------------- existing arm

def replay_existing(st: dict, record: Mapping[str, Any]) -> dict:
    """Re-simulate a recorded jevloop run on a fresh clip state.

    Applies the transcript's reveal order (`_revealed` marks every actual
    reveal, including forced/seam-look backstops), then re-runs the shared
    arbitration with the recorded Jev answers.  `replay_matches_record`
    flags whether the re-simulated composite equals the recorded one --
    it validates both the replay harness and the arbitration copy.
    """
    revealed: list[int] = []
    for row in record.get("transcript") or []:
        idx = row.get("_revealed")
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < len(st["candidates"]) and not st["candidates"][idx].get(
            "revealed"
        ):
            reveal(st, idx)
            revealed.append(idx)

    final = record.get("final") or {}
    fan = record.get("fan_probs") or {}
    abstained = bool(record.get("abstained"))
    composite = arbitrate(
        (final.get("break_type") or {}).get("choice"),
        (final.get("corrupted") or {}).get("noul", 0) or 0,
        fan,
        st,
        abstained=abstained,
    )
    recorded = record.get("composite_type") or "none"
    return {
        "policy": "existing",
        "composite_type": composite,
        "recorded_composite_type": recorded,
        "replay_matches_record": composite == recorded,
        "abstained": abstained,
        "judge_calls": len(judged_candidates(st)),
        "recorded_judge_calls": record.get("judge_calls"),
        "jev_calls": record.get("jev_calls"),
        "n_candidates": len(st["candidates"]),
        "recorded_n_candidates": record.get("n_candidates"),
        "discontinuous_at": discontinuous_frames(st),
        "reveal_order": revealed,
    }


def fresh_state(loader, cid: str) -> dict:
    """One fresh clip state per arm so reveal mutations can't leak across."""
    return copy.deepcopy(loader(cid))
