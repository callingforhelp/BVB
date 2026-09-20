"""Milestone-1 tests: jev_loop->EvidenceState adapter, bounded-policy
stopping rules, shared arbitration, and deterministic replay.

All state fixtures mirror `load_clip_state` output: candidates carry a
cached `verdict` (hidden evidence) and a `revealed` payload (what the
policy has actually seen).  No manifest labels ever enter the state.
"""
import json
from dataclasses import asdict

from flinter_mvp import choose_next_request
from flinter_mvp.jev_adapter import (
    COHERENCE_RUBRIC,
    KIND_CONTINUOUS,
    KIND_DISCONTINUOUS,
    to_evidence_state,
)
from flinter_mvp.jev_replay import (
    arbitrate,
    deterministic_verdict,
    replay_existing,
    run_bounded,
)

QUIET = {
    "recur_frac_max": 0.12,
    "recur_argmax_frame": 100,
    "dup_dense_last_frame": None,
    "dup_trailing_run": 0,
    "echo_best_score": 1.5,
    "echo_best_pair": None,
    "scenecut_iframe_frames": [],
    "photo_jump_top3": [],
}


def cont(conf=0.95):
    return {"continuous": True, "same_scene": True,
            "break_kind": "none", "confidence": conf}


def discont(conf=0.95, kind="scene_change"):
    return {"continuous": False, "same_scene": False,
            "break_kind": kind, "confidence": conf}


def make_st(candidates, free_signals=None):
    """candidates: list of (frame, verdict, judged_bool)."""
    return {
        "source": "src",
        "free_signals": dict(free_signals or QUIET),
        "candidates": [
            {"frame": f, "gate": gate, "verdict": v,
             "revealed": v if judged else None}
            for (f, v, judged), gate in zip(
                candidates, ["pixel_gate"] * len(candidates))
        ],
    }


def state_for(st, budget=8):
    return to_evidence_state(st, video_id="v", budget=budget)


# ---------------------------------------------------------- required eight

def test_no_candidates_reviews():
    st = make_st([])
    decision = choose_next_request(state_for(st))
    assert decision.status == "review"
    assert decision.reason == "no candidates were proposed"
    out = run_bounded(make_st([]))
    assert out["abstained"] and out["composite_type"] == "none"
    assert out["judge_calls"] == 0


def test_budget_exhausted_reviews():
    st = make_st([(10, cont(), False), (20, cont(), False)])
    decision = choose_next_request(state_for(st, budget=0))
    assert decision.status == "review"
    assert decision.reason == "evidence budget exhausted"
    # driver-level: pre-judged candidates consume the same reveal budget
    st2 = make_st([(i * 100, cont(), i < 3) for i in range(1, 12)])
    out = run_bounded(st2)
    assert out["abstained"]
    assert out["stop_reason"] == "evidence budget exhausted"
    assert out["judge_calls"] == 10


def test_missing_required_observation_gathers():
    st = make_st([(10, cont(), True), (20, cont(), False), (30, cont(), False)])
    decision = choose_next_request(state_for(st))
    assert decision.status == "gather"
    assert decision.next_request is not None
    assert decision.next_request.candidate_id == "1"  # earliest unjudged


def test_evidence_floor_met_selects():
    st = make_st([(10, cont(), True), (20, discont(), True), (30, cont(), False)])
    decision = choose_next_request(state_for(st))
    assert decision.status == "select"
    assert decision.candidate_id == "1"
    out = run_bounded(st)
    assert not out["abstained"]
    assert out["stop_status"] == "select"
    assert out["selected_candidate"] == "1"
    assert out["composite_type"] == "splice"  # one confirmed discontinuity
    # the last candidate is still judged before selecting: one confirmed seam
    # does not distinguish splice from room_swap while a reveal could still
    # produce a second one
    assert out["judge_calls"] == 3


def test_multiple_valid_candidates_review():
    st = make_st([(10, discont(), True), (20, discont(), True), (30, cont(), True)])
    decision = choose_next_request(state_for(st))
    assert decision.status == "review"
    assert decision.reason == "multiple candidates meet the rubric"


def test_judged_candidates_not_requested_again():
    st = make_st([(10, cont(), True), (20, cont(), False), (30, cont(), False)])
    seen = set()
    for _ in range(4):
        decision = choose_next_request(state_for(st))
        if decision.status != "gather":
            break
        cid = decision.next_request.candidate_id
        assert cid not in seen
        seen.add(cid)
        idx = int(cid)
        st["candidates"][idx]["revealed"] = st["candidates"][idx]["verdict"]
    assert seen == {"1", "2"}  # candidate 0 was already judged
    decision = choose_next_request(state_for(st))
    assert decision.status == "review"  # nothing unjudged left to request


def test_policy_state_has_no_hidden_fields():
    marker = {"continuous": "MARKER_UNREVEALED", "same_scene": "x",
              "break_kind": "x", "confidence": 0.9}
    st = make_st([(10, cont(), True), (20, marker, False), (30, cont(), False)])
    st["operator"] = "splice"          # manifest fields must not propagate
    st["breaks_s"] = [0.67]
    state = state_for(st)
    blob = json.dumps(asdict(state), sort_keys=True,
                      default=lambda o: sorted(o))
    assert "MARKER_UNREVEALED" not in blob
    assert '"splice"' not in blob
    assert "operator" not in asdict(state)
    assert "breaks_s" not in asdict(state)
    for cand in asdict(state)["candidates"]:
        assert "verdict" not in cand and "revealed" not in cand
    judged = {item.observation_id for item in state.evidence
              if item.kind.startswith("vlm_")}
    assert judged == {"vlm_0"}  # only the revealed verdict


def test_identical_evidence_deterministic_replay():
    cands = [(10, cont(), False), (20, discont(), False), (30, cont(), False)]
    a = run_bounded(make_st(cands))
    b = run_bounded(make_st(cands))
    assert a == b
    assert a["composite_type"] == "splice"


# ---------------------------------------------------------- adapter checks

def test_adapter_maps_verdicts_and_provenance():
    st = make_st([(42, discont(0.97, "cut"), True), (77, cont(0.88), True)])
    state = state_for(st)
    by_id = {i.observation_id: i for i in state.evidence}
    assert by_id["vlm_0"].kind == KIND_DISCONTINUOUS
    assert by_id["vlm_0"].confidence == 0.97
    assert by_id["vlm_0"].frame_refs == (42,)
    assert by_id["vlm_0"].source == "pairjudge:pixel_gate"
    assert '"cut"' in by_id["vlm_0"].text
    assert by_id["vlm_1"].kind == KIND_CONTINUOUS
    assert state.judged_candidate_ids == {"0", "1"}


def test_free_signals_are_informational_only():
    fs = dict(QUIET, recur_frac_max=0.95, dup_dense_last_frame=500)
    st = make_st([(500, cont(), False)], free_signals=fs)
    state = state_for(st)
    kinds = {i.kind for i in state.evidence}
    assert "signal:recur_frac_max" in kinds
    assert "signal:dup_dense_last_frame" in kinds
    # signal evidence never satisfies the rubric -> still gathers
    decision = choose_next_request(state)
    assert decision.status == "gather"


def test_candidate_interval_isolates_verdict():
    st = make_st([(10, discont(), True), (11, cont(), False)])
    state = state_for(st)
    cand10, cand11 = state.candidates
    assert [i.observation_id for i in state.evidence_for(cand10)
            if i.kind.startswith("vlm_")] == ["vlm_0"]
    assert state.evidence_for(cand11) == []


# ---------------------------------------------------------- arbitration

def test_arbitration_seam_count_remap():
    st = make_st([(10, discont(), True), (500, discont(), True)])
    assert arbitrate("splice", 0.9, {"splice": 0.9}, st, abstained=False) == "room_swap"
    st1 = make_st([(10, discont(), True)])
    assert arbitrate("room_swap", 0.9, {"room_swap": 0.9}, st1, abstained=False) == "splice"


def test_arbitration_dup_swap_signature():
    fs = dict(QUIET, dup_trailing_run=200)
    st = make_st([(10, discont(kind="scene_change"), True)], free_signals=fs)
    assert arbitrate("loop", 0.9, {"loop": 0.9}, st, abstained=False) == "room_swap"
    assert arbitrate("splice", 0.9, {"splice": 0.9}, st, abstained=False) == "room_swap"
    st_no_tail = make_st([(10, discont(), True)])
    assert arbitrate("splice", 0.9, {"splice": 0.9},
                     st_no_tail, abstained=False) == "splice"


def test_arbitration_abstain_is_untyped():
    st = make_st([(10, discont(), True)])
    assert arbitrate("splice", 0.9, {"splice": 0.9}, st, abstained=True) \
        == "corrupt_untyped"


def test_deterministic_verdict_guide_precedence():
    st = make_st([(10, discont(), True)],
                 free_signals=dict(QUIET, recur_frac_max=0.95))
    assert deterministic_verdict(st)["break_type"]["choice"] == "loop"
    st = make_st([(10, discont(), True)],
                 free_signals=dict(QUIET, dup_dense_last_frame=700))
    assert deterministic_verdict(st)["break_type"]["choice"] == "timewarp"
    st = make_st([(10, discont(), True)],
                 free_signals=dict(QUIET, echo_best_score=15.0))
    assert deterministic_verdict(st)["break_type"]["choice"] == "reverse"
    st = make_st([(10, discont(), True), (500, discont(), True)])
    assert deterministic_verdict(st)["break_type"]["choice"] == "room_swap"
    st = make_st([(10, cont(), True), (20, cont(), True)])
    assert deterministic_verdict(st)["break_type"]["choice"] == "none"


# ---------------------------------------------------------- replay arms

def test_replay_existing_reproduces_record():
    st = make_st([(10, cont(), False), (20, discont(), False), (30, cont(), False)])
    record = {
        "composite_type": "splice",
        "abstained": False,
        "judge_calls": 2,
        "jev_calls": 3,
        "n_candidates": 3,
        "final": {"corrupted": {"noul": 0.8},
                  "break_type": {"choice": "splice"}},
        "fan_probs": {"splice": 0.8, "none": 0.1},
        "transcript": [{"iter": 0, "_revealed": 0},
                       {"iter": 1, "_revealed": 1},
                       {"iter": 2, "action": "conclude"}],
    }
    out = replay_existing(st, record)
    assert out["composite_type"] == "splice"
    assert out["replay_matches_record"]
    assert out["judge_calls"] == 2
    assert out["reveal_order"] == [0, 1]
    assert out["discontinuous_at"] == [20]


def test_replay_existing_respects_recorded_abstain():
    st = make_st([(10, discont(), False)])
    record = {"composite_type": "corrupt_untyped", "abstained": True,
              "final": {"corrupted": {"noul": 0.9},
                        "break_type": {"choice": "splice"}},
              "fan_probs": {"splice": 0.9},
              "transcript": [{"iter": 0, "_revealed": 0}]}
    out = replay_existing(st, record)
    assert out["composite_type"] == "corrupt_untyped"
    assert out["replay_matches_record"]


def test_bounded_clean_clip_abstains_but_types_none():
    st = make_st([(10, cont(), False), (20, cont(), False)])
    out = run_bounded(st)
    assert out["abstained"]
    assert out["composite_type"] == "none"
    assert out["stop_reason"] == "no candidate meets the evidence floor"


def test_free_signal_prior_stops_after_one_reveal():
    fs = dict(QUIET, recur_frac_max=0.95)
    st = make_st([(10, cont(), False), (20, cont(), False)], free_signals=fs)
    out = run_bounded(st)
    assert out["stop_status"] == "select"
    assert out["composite_type"] == "loop"
    assert out["judge_calls"] == 1


def test_bounded_selects_first_confirmed_seam():
    st = make_st([(10, cont(), False), (20, discont(), False), (30, cont(), False)])
    out = run_bounded(st)
    # one confirmed seam leaves the type undetermined (splice vs room_swap),
    # so the remaining candidate is judged before selecting
    assert out["judge_calls"] == 3
    assert out["composite_type"] == "splice"
    assert out["stop_status"] == "select"
    reveals = [r["revealed"] for r in out["transcript"] if "revealed" in r]
    assert reveals == [0, 1, 2]               # frame order, deterministic


def test_bounded_gathers_until_type_determined():
    # Two confirmed seams are the room_swap signature, not ambiguity: the
    # policy must keep gathering past the first seam and must not abstain on
    # "multiple candidates meet the rubric".
    st = make_st([(10, cont(), False), (20, discont(), False),
                  (30, cont(), False), (40, discont(), False)])
    out = run_bounded(st)
    assert not out["abstained"]
    assert out["stop_status"] == "select"
    assert out["composite_type"] == "room_swap"
    assert out["judge_calls"] == 4


def test_iteration_cap_with_floor_selects_best_effort():
    # Cap reached with one confirmed seam: emit the typed answer (splice)
    # rather than abstaining -- the evidence floor is met even though a
    # hypothetical second seam could not be ruled out inside the budget.
    st = make_st([(i * 100, discont() if i == 7 else cont(), False)
                  for i in range(1, 12)])
    out = run_bounded(st)
    assert not out["abstained"]
    assert out["stop_status"] == "select"
    assert out["composite_type"] == "splice"
    assert out["judge_calls"] == 10
