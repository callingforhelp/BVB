"""Small, deterministic core for the Flinter evidence-gathering MVP.

The model is deliberately not asked to emit the final verdict.  It receives
candidate evidence requests; code owns candidate construction, stopping
floors, abstention, and trace recording.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
import json
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Rubric:
    """Requirements that must be visible for an interval to be complete."""

    name: str
    required_observations: tuple[str, ...]
    # An interval with this many unresolved requirements must not be accepted.
    min_evidence_confidence: float = 0.55


@dataclass(frozen=True)
class EvidenceItem:
    """One timestamped observation, preserving provenance."""

    observation_id: str
    time: float
    kind: str
    text: str
    confidence: float
    frame_refs: tuple[int, ...] = ()
    source: str = "vlm"


@dataclass(frozen=True)
class CandidateInterval:
    candidate_id: str
    start: float
    end: float
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("candidate end must be greater than start")


@dataclass
class EvidenceState:
    video_id: str
    rough_start: float
    rough_end: float
    rubric: Rubric
    candidates: list[CandidateInterval] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    judged_candidate_ids: set[str] = field(default_factory=set)
    budget: int = 3

    def evidence_for(self, candidate: CandidateInterval) -> list[EvidenceItem]:
        return [
            item
            for item in self.evidence
            if candidate.start <= item.time <= candidate.end
        ]

    def observed_kinds(self, candidate: CandidateInterval) -> set[str]:
        return {item.kind for item in self.evidence_for(candidate)}

    def unresolved(self, candidate: CandidateInterval) -> tuple[str, ...]:
        present = self.observed_kinds(candidate)
        return tuple(
            kind for kind in self.rubric.required_observations if kind not in present
        )


@dataclass(frozen=True)
class StopDecision:
    status: str  # select, review, gather
    candidate_id: str | None
    reason: str
    next_request: CandidateInterval | None = None


def _dedupe(values: Iterable[float]) -> list[float]:
    return sorted({round(float(value), 3) for value in values})


def generate_candidates(
    starts: Sequence[float],
    ends: Sequence[float],
    *,
    rough_start: float,
    rough_end: float,
    max_candidates: int = 12,
) -> list[CandidateInterval]:
    """Generate a small, deterministic candidate set from cheap landmarks.

    This is intentionally proposal-only.  It never claims that a candidate is
    correct and it never averages timestamps into a new, unvalidated cut.
    """

    starts = _dedupe(x for x in starts if rough_start <= x < rough_end)
    ends = _dedupe(x for x in ends if rough_start < x <= rough_end)
    intervals = [
        CandidateInterval(
            candidate_id=f"c{i:02d}",
            start=start,
            end=end,
        )
        for i, (start, end) in enumerate(product(starts, ends))
        if end > start
    ]
    # Prefer intervals closest to the rough window before truncating.
    intervals.sort(key=lambda c: (abs(c.start - rough_start) + abs(c.end - rough_end), c.start, c.end))
    return intervals[:max_candidates]


def choose_next_request(state: EvidenceState) -> StopDecision:
    """Choose the next information request using fixed, auditable rules.

    The policy requests evidence for the candidate with the largest unresolved
    requirement set.  It stops only when one candidate satisfies every rubric
    requirement at the evidence floor and competing candidates are judged.
    """

    if not state.candidates:
        return StopDecision("review", None, "no candidates were proposed")
    if state.budget <= 0:
        return StopDecision("review", None, "evidence budget exhausted")

    complete: list[CandidateInterval] = []
    unresolved: list[tuple[int, CandidateInterval]] = []
    for candidate in state.candidates:
        items = state.evidence_for(candidate)
        if all(
            any(item.kind == required and item.confidence >= state.rubric.min_evidence_confidence for item in items)
            for required in state.rubric.required_observations
        ):
            complete.append(candidate)
        elif candidate.candidate_id not in state.judged_candidate_ids:
            # A judged candidate is not an admissible next request.  It may
            # remain unresolved; that is precisely when we should abstain
            # after exhausting the other candidates.
            unresolved.append((len(state.unresolved(candidate)), candidate))

    judged = len(state.judged_candidate_ids)
    if len(complete) == 1 and judged >= min(2, len(state.candidates)):
        return StopDecision("select", complete[0].candidate_id, "one candidate meets the evidence floor")

    if not unresolved:
        if len(complete) > 1:
            return StopDecision("review", None, "multiple candidates meet the rubric")
        return StopDecision("review", None, "no candidate meets the evidence floor")

    _, next_candidate = max(
        unresolved,
        key=lambda pair: (pair[0], -pair[1].start, -pair[1].end),
    )
    if next_candidate.candidate_id in state.judged_candidate_ids:
        return StopDecision("review", None, "remaining ambiguity is not resolved by an unjudged candidate")
    return StopDecision(
        "gather",
        None,
        "inspect the candidate with the most unresolved rubric requirements",
        next_candidate,
    )


def record_trace(path: str | Path, state: EvidenceState, decision: StopDecision) -> None:
    """Append one JSONL trace row, including exact evidence and candidates."""

    row: dict[str, Any] = {
        "video_id": state.video_id,
        "rough_window": [state.rough_start, state.rough_end],
        "rubric": asdict(state.rubric),
        "candidates": [asdict(candidate) for candidate in state.candidates],
        "evidence": [asdict(item) for item in state.evidence],
        "judged_candidate_ids": sorted(state.judged_candidate_ids),
        "budget": state.budget,
        "decision": asdict(decision),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
