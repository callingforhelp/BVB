from pathlib import Path

from flinter_mvp import (
    CandidateInterval,
    EvidenceItem,
    EvidenceState,
    Rubric,
    choose_next_request,
    generate_candidates,
    record_trace,
)


def rubric():
    return Rubric("place-object-v1", ("movement", "contact", "release", "resting"))


def test_candidate_generation_is_small_and_never_averages():
    candidates = generate_candidates(
        [10.0, 10.4, 10.8],
        [12.0, 12.4, 12.8],
        rough_start=9.5,
        rough_end=13.5,
        max_candidates=5,
    )
    assert len(candidates) == 5
    assert all(c.end > c.start for c in candidates)
    assert all(c.start in {10.0, 10.4, 10.8} for c in candidates)


def test_policy_requests_missing_release_before_selecting():
    candidates = [
        CandidateInterval("a", 10.0, 12.0),
        CandidateInterval("b", 10.0, 12.8),
    ]
    state = EvidenceState("v1", 9.5, 13.5, rubric(), candidates, [
        EvidenceItem("m", 10.5, "movement", "moving", .9),
        EvidenceItem("c", 11.8, "contact", "contact", .9),
        EvidenceItem("r", 12.5, "resting", "resting", .9),
    ], {"a"}, 2)
    decision = choose_next_request(state)
    assert decision.status == "gather"
    assert decision.next_request is not None
    assert decision.next_request.candidate_id == "b"


def test_policy_selects_only_after_evidence_floor_and_comparison():
    candidates = [CandidateInterval("a", 10.0, 12.0), CandidateInterval("b", 10.0, 12.8)]
    evidence = [
        EvidenceItem("m", 10.5, "movement", "moving", .9),
        EvidenceItem("c", 11.8, "contact", "contact", .9),
        EvidenceItem("r", 12.5, "release", "release", .9),
        EvidenceItem("s", 12.7, "resting", "resting", .9),
    ]
    state = EvidenceState("v1", 9.5, 13.5, rubric(), candidates, evidence, {"a", "b"}, 1)
    decision = choose_next_request(state)
    assert decision.status == "select"
    assert decision.candidate_id == "b"


def test_multiple_complete_candidates_review_and_trace(tmp_path: Path):
    candidates = [CandidateInterval("a", 10.0, 12.8), CandidateInterval("b", 10.1, 12.9)]
    evidence = [EvidenceItem(str(i), t, kind, kind, .9) for i, (t, kind) in enumerate([
        (10.5, "movement"), (11.8, "contact"), (12.5, "release"), (12.7, "resting")
    ])]
    state = EvidenceState("v1", 9.5, 13.5, rubric(), candidates, evidence, {"a", "b"}, 1)
    decision = choose_next_request(state)
    assert decision.status == "review"
    trace = tmp_path / "trace.jsonl"
    record_trace(trace, state, decision)
    assert trace.read_text(encoding="utf-8").count("\n") == 1
