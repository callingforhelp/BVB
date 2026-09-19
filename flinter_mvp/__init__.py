"""Bounded evidence-gathering primitives for Flinter."""

from .core import (
    CandidateInterval,
    EvidenceItem,
    EvidenceState,
    Rubric,
    StopDecision,
    choose_next_request,
    generate_candidates,
    record_trace,
)

__all__ = [
    "CandidateInterval",
    "EvidenceItem",
    "EvidenceState",
    "Rubric",
    "StopDecision",
    "choose_next_request",
    "generate_candidates",
    "record_trace",
]
