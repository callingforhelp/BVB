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
from .jev_adapter import (
    COHERENCE_RUBRIC,
    KIND_CONTINUOUS,
    KIND_DISCONTINUOUS,
    candidate_interval,
    free_signal_evidence,
    to_evidence_state,
    verdict_evidence,
)
from .jev_replay import (
    arbitrate,
    deterministic_verdict,
    discontinuous_frames,
    signal_type,
    replay_existing,
    run_bounded,
)

__all__ = [
    "COHERENCE_RUBRIC",
    "CandidateInterval",
    "EvidenceItem",
    "EvidenceState",
    "KIND_CONTINUOUS",
    "KIND_DISCONTINUOUS",
    "Rubric",
    "StopDecision",
    "arbitrate",
    "candidate_interval",
    "choose_next_request",
    "deterministic_verdict",
    "discontinuous_frames",
    "free_signal_evidence",
    "generate_candidates",
    "record_trace",
    "replay_existing",
    "run_bounded",
    "to_evidence_state",
    "verdict_evidence",
]
