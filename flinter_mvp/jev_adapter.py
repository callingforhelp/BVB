"""Adapter: existing jev_loop clip state -> Flinter EvidenceState.

The existing `jev_loop.load_clip_state()` produces, per clip::

    {
      "source": ...,
      "free_signals": {recur_frac_max, dup_dense_last_frame, ...},
      "candidates": [{"frame", "gate", "verdict", "revealed"}, ...],
    }

`verdict` is the CACHED pair-judge outcome (hidden evidence: it exists in
the record before the policy sees it).  `revealed` is what the loop has
actually shown the policy.  This adapter converts that record into an
`EvidenceState` the bounded `choose_next_request` policy can act on:

- each judgeable candidate becomes a `CandidateInterval` (frame +/- 0.5,
  so a verdict attaches to exactly its own candidate);
- each REVEALED verdict becomes an `EvidenceItem` whose kind is
  `vlm_discontinuous` / `vlm_continuous` and whose provenance records the
  gate that nominated the candidate;
- each free signal becomes a `signal:*` evidence item (provenance only --
  never satisfies a rubric requirement);
- UNREVEALED verdicts never produce evidence items.  Manifest labels
  (operator, breaks_s, path, source) are not fields of `EvidenceState` and
  are never copied.

Units are frames throughout; the core is unit-agnostic.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .core import (
    CandidateInterval,
    EvidenceItem,
    EvidenceState,
    Rubric,
)

KIND_DISCONTINUOUS = "vlm_discontinuous"
KIND_CONTINUOUS = "vlm_continuous"
SIGNAL_PREFIX = "signal:"

#: Bounded policy rubric for the temporal-edit task: an interval is only
#: acceptable once a confident VLM discontinuity has been observed inside
#: it.  This is the adapter-level analog of the existing loop's evidence
#: floor; reconciliation of stopping semantics is a later milestone.
COHERENCE_RUBRIC = Rubric(
    name="coherence-seam-v1",
    required_observations=(KIND_DISCONTINUOUS,),
    min_evidence_confidence=0.55,
)

CANDIDATE_HALF_WIDTH = 0.5  # frames; verdict time lands only in its own interval


def candidate_interval(index: int, candidate: Mapping[str, Any]) -> CandidateInterval:
    """Map one jev_loop candidate record to a CandidateInterval.

    The id is the candidate's positional index -- the same index the
    existing loop uses for its `judge_<i>` actions -- so a gather request
    maps back to the record without a lookup table.
    """
    frame = float(candidate["frame"])
    return CandidateInterval(
        candidate_id=str(index),
        start=frame - CANDIDATE_HALF_WIDTH,
        end=frame + CANDIDATE_HALF_WIDTH,
    )


def verdict_evidence(index: int, candidate: Mapping[str, Any]) -> EvidenceItem:
    """Map a REVEALED cached verdict to a provenance-preserving item.

    Call only for candidates whose `revealed` payload is set.  The kind
    encodes verdict polarity because that is what the coherence rubric
    requires; the raw verdict fields are kept verbatim in `text`.
    """
    verdict = candidate["revealed"]
    discontinuous = verdict.get("continuous") is False
    return EvidenceItem(
        observation_id=f"vlm_{index}",
        time=float(candidate["frame"]),
        kind=KIND_DISCONTINUOUS if discontinuous else KIND_CONTINUOUS,
        text=json.dumps(
            {
                "continuous": verdict.get("continuous"),
                "same_scene": verdict.get("same_scene"),
                "break_kind": verdict.get("break_kind"),
            },
            sort_keys=True,
        ),
        confidence=float(verdict.get("confidence") or 0.0),
        frame_refs=(int(candidate["frame"]),),
        source=f"pairjudge:{candidate.get('gate', 'unknown')}",
    )


def free_signal_evidence(free_signals: Mapping[str, Any]) -> list[EvidenceItem]:
    """Expose the free deterministic signals as provenance items.

    They are clip-level evidence the existing loop reveals up front, so the
    bounded state carries them too.  Their kinds are `signal:*`, which no
    rubric requires -- they inform a learned scorer without affecting the
    deterministic floor.  Signals with a natural frame land at that frame;
    scalar-only signals sit at time -1, outside every candidate interval.
    """
    items: list[EvidenceItem] = []

    def add(name: str, value: Any, frame: float) -> None:
        items.append(
            EvidenceItem(
                observation_id=f"sig_{name}",
                time=frame,
                kind=f"{SIGNAL_PREFIX}{name}",
                text=json.dumps({name: value}, sort_keys=True, default=str),
                confidence=1.0,
                frame_refs=(int(frame),) if frame >= 0 else (),
                source="free_signal",
            )
        )

    recur_frame = free_signals.get("recur_argmax_frame")
    dup_frame = free_signals.get("dup_dense_last_frame")
    add("recur_frac_max", free_signals.get("recur_frac_max"),
        float(recur_frame) if recur_frame is not None else -1.0)
    add("dup_dense_last_frame", dup_frame,
        float(dup_frame) if dup_frame is not None else -1.0)
    add("dup_trailing_run", free_signals.get("dup_trailing_run"), -1.0)
    pair = free_signals.get("echo_best_pair") or []
    add("echo_best_score", free_signals.get("echo_best_score"),
        float(pair[0]) if pair else -1.0)
    for i, f in enumerate(free_signals.get("scenecut_iframe_frames") or []):
        add(f"scenecut_iframe_{i}", f, float(f))
    for i, peak in enumerate(free_signals.get("photo_jump_top3") or []):
        add(f"photo_jump_top_{i}", peak.get("z"), float(peak.get("frame", -1)))
    return items


def to_evidence_state(
    st: Mapping[str, Any],
    *,
    video_id: str,
    budget: int,
    rubric: Rubric = COHERENCE_RUBRIC,
    clip_frames: float = 900.0,
) -> EvidenceState:
    """Convert a jev_loop clip-state dict into an EvidenceState.

    `budget` is the number of further reveals the caller will still honor;
    the driver recomputes it each step.  Only revealed verdicts become
    `vlm_*` evidence -- an unrevealed candidate carries no verdict into the
    policy's view.
    """
    candidates: Sequence[Mapping[str, Any]] = st["candidates"]
    intervals = [candidate_interval(i, c) for i, c in enumerate(candidates)]

    judged_ids: set[str] = set()
    evidence: list[EvidenceItem] = []
    for i, c in enumerate(candidates):
        if c.get("revealed"):
            judged_ids.add(str(i))
            evidence.append(verdict_evidence(i, c))
    evidence.extend(free_signal_evidence(st.get("free_signals") or {}))

    return EvidenceState(
        video_id=video_id,
        rough_start=0.0,
        rough_end=clip_frames,
        rubric=rubric,
        candidates=intervals,
        evidence=evidence,
        judged_candidate_ids=judged_ids,
        budget=budget,
    )
