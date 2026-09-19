# Flinter evidence-gathering MVP

This worktree adds a narrow, deterministic core for the next Flinter experiment.
It does **not** replace the existing coherence detector and does not claim that
Jev understands raw video.

## Contract

```text
cheap signals → small candidate set → timestamped evidence requests
→ policy chooses the next request or abstains → code validates the evidence floor
→ reviewed result is recorded
```

The learned/remote scorer should be interchangeable. It may later be Jev, the
existing S1 policy, or a local NanoJev-style scorer. The model must not own
candidate construction, evidence floors, stopping safety, or final arbitration.

## Added module

`flinter_mvp/core.py` provides:

- provenance-preserving `EvidenceItem` records;
- `CandidateInterval` proposals;
- rubric requirements;
- deterministic candidate generation;
- evidence-floor stopping and abstention;
- append-only JSONL trace records.

The module is intentionally stdlib-only so it can be embedded in the existing
`.verify_tmp/coherence_mvp` pipeline without changing its API or credentials.

## Next integration step

Adapt `jev_loop.py` behind an adapter that converts its current state into
`EvidenceState`. Preserve the existing arbitration and OOD protocol. Then add a
single action-boundary rubric and compare:

1. current proposer;
2. deterministic selection;
3. Jev/S1 policy;
4. reviewed human outcome.

Report candidate coverage separately from candidate ranking. A scorer cannot
recover an interval that was never proposed.
