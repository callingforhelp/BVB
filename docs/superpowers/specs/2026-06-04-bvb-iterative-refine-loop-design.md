# BVB Human-in-the-Loop Iterative Refinement — Design

Date: 2026-06-04
Status: Approved (design), pending implementation plan

## 1. Purpose

Make the human ↔ Claude iterative refinement of BVB Blender scenes **fast, safe, and
repeatable across all scenes**. Today each scene restarts from scratch: manually pull QA,
eyeball the video, edit in Blender, hope it matches. We want a reusable loop where:

- the **mechanical work** (QA lookup, reference-frame extraction, unit-test lookup,
  checkpointing, the final official gate) is wrapped in one CLI tool, and
- the **collaboration protocol** (loop steps, when to stop for the human, verification
  rules, accumulated heuristics) lives in a skill so any Claude session behaves
  consistently and gets faster over time.

Scope is one scene at a time, interactive. Batch is out of scope (YAGNI) but the design
must not preclude it.

## 2. Division of Responsibility

| Actor | Owns |
|---|---|
| **CLI `bvbrefine.py`** (no live Blender) | pull QA from CSV tracker; extract N reference frames via ffmpeg; look up the scene's unit-tests; scaffold `blend/refine_<id>/`; run the final official gate (export bpy + `unit_test_metric.py`) |
| **Claude via BlenderMCP** (live Blender) | scene inspection (`get_scene_info`, `get_object_info`); edits (`execute_blender_code`); screenshot verification (`get_viewport_screenshot`); save `vNN` checkpoints (`save_as_mainfile(..., copy=True)`) |
| **Skill** (`bvb-refine-loop`) | the collaboration state machine; gate timing; checkpoint naming convention; the hybrid verification rule; the growing heuristic "skill set" |
| **Human** | approve structural changes; final pre-push verification; provide judge API key (optional) |

Rationale: the CLI cannot touch the Blender session Claude is editing (that is MCP's
job), so the CLI is restricted to filesystem/dataset/eval mechanics. This keeps each unit
independently testable.

## 3. Collaboration Loop

```
bvbrefine start <id>
    → prints QA + focus hints + the scene's unit-tests
    → extracts reference frames to blend/refine_<id>/frames/
    → (Claude) saves v00 baseline via MCP  (copy=True, live file untouched)

Claude: inventory scene vs frames vs QA; classify EACH qa axis as local | structural;
        build an ordered edit plan.

[GATE: structural]  Stop ONLY for structural changes (delete/add major objects,
                    resize room, change topology). Present plan, get human OK.
                    Local patches proceed autonomously.

for each edit:
    MCP execute → screenshot verify → live function-check (compute area / positions)
                → save vNN checkpoint (descriptive suffix)

bvbrefine gate <id>
    → export current blend → bpy, run unit_test_metric.py filtered to this scene
    → report official pass/fail per QA axis

[GATE: pre-push]    Show official results + final screenshots. Human approves →
                    copy final checkpoint to blend/<id>.blend, commit, push,
                    update Notion tracker.
```

The only two hard human gates are **structural changes** and **pre-push** (matches the
user's "只管大决策" preference and the repo guidance that final refinement is
human-verified).

## 4. Verification Rule (hybrid)

Driven by the unit-test `evaluator` field already present in `eval/unit_tests.jsonl`:

**Important constraint discovered during planning:** the stock `unit_test_metric.py`
sends BOTH `function` and `proposition` tests to the LLM judge — `function` tests use the
judge only to *ground* which object group is the floor/target (`grounded_params`), then a
deterministic function (`evaluate_function_test_with_params`) computes pass/fail. Without
an API key the stock runner marks judged tests as `error`. The deterministic functions
themselves (`room_area = floor_group.bbox.x * .y`, etc.) need no LLM.

The hybrid rule therefore is:

- **`function` tests** (room_area, count_objects, longest_dimension, closest_distance):
  during iteration Claude computes the quantity **directly in Blender** for instant
  feedback (e.g. floor-group area). At the final gate, `bvbrefine gate` runs an **offline
  evaluator** that reuses the eval harness's deterministic pieces (`parse_bpy_scene`,
  `build_groups`, `evaluate_function_test_with_params`) and supplies `grounded_params`
  itself (Claude/heuristic picks the floor group), so the function-test gate runs with **no
  LLM API**. Passing `--judge` instead defers to the stock harness (LLM grounding) for a
  fully benchmark-faithful result.
- **`proposition` tests** (route_planning and other subjective statements): during
  iteration Claude reasons from geometry + reference frames. At the final gate a real LLM
  judge decides true/false. The judge is **optional**: with `--judge` + an API key the gate
  is official; without it, Claude's reasoning is the fallback and the gate marks the
  proposition test "self-judged" rather than "official".

## 5. CLI: `bvbrefine.py`

Single Python entry point (`refine/bvbrefine.py`), stdlib + ffmpeg only, with subcommands:

| Subcommand | Behavior |
|---|---|
| `start <id> [--frames N]` | runs `qa` + `frames`, scaffolds `blend/refine_<id>/`, prints a baseline-save snippet for Claude to run via MCP |
| `qa <id>` | reads `Private & Shared/…Scene Tracker….csv`, prints Source, QA axes, QA text, GT answers, per-axis focus hints (reuse the hint table already in `start.sh`); also lists the scene's rows from `eval/unit_tests.jsonl` (statement, evaluator, expected) |
| `frames <id> [--n 12]` | locates `VSI-Bench/{arkitscenes,scannet,scannetpp}/<id>.mp4`, extracts N evenly-spaced frames to `blend/refine_<id>/frames/frame_%02d.jpg` |
| `checkpoints <id>` | lists saved `vNN` checkpoints with their descriptive suffixes |
| `gate <id> [--judge] [--floor-group KEY]` | (1) export `blend/<id>.blend` → bpy via `eval/batch_export_blend_to_bpy.py` (background Blender + `addons/export_bpy_code.py`) into `blend/refine_<id>/bpy/<id>.py`; (2) load the scene's `function`/`proposition` tests; (3) **offline path (default):** parse the bpy with the harness helpers, ground function tests (use `--floor-group`, else heuristic: the largest flat MESH group whose name contains "floor"/"room"), call `evaluate_function_test_with_params`, print pass/fail; proposition tests printed as "self-judged (needs --judge)"; (4) **`--judge` path:** write a pairs jsonl `{"id": "<id>", "pred_bpy_path": "bpy/<id>.py"}` and shell out to `eval/unit_test_metric.py --pairs … --unit-tests eval/unit_tests.jsonl` (needs `OPENAI_API_KEY` + `BVB_JUDGE_MODEL`) for fully official results. |

All per-scene state lives under `blend/refine_<id>/` (frames, checkpoints, gate output),
which is git-ignored via `blend/refine_*/`.

## 6. Checkpoint Convention

- Directory: `blend/refine_<id>/`  (git-ignored)
- Naming: `<id>_vNN_<short-desc>.blend`, NN zero-padded, starting `v00_baseline`
- Saved with `bpy.ops.wm.save_as_mainfile(filepath=..., copy=True)` so the **live file
  stays `blend/<id>.blend`** and is the only thing ever pushed.
- Rollback = re-open a `vNN` (or `bpy.ops.wm.open_mainfile`) and continue.
- The final approved state is copied to `blend/<id>.blend` for commit.

## 7. Skill: `bvb-refine-loop`

A new skill (sibling to `bvb-scene-builder`; the builder stays for from-scratch builds).
It encodes:

- the §3 loop state machine and the two hard gates;
- the §4 hybrid verification rule;
- the §6 checkpoint convention (with the `copy=True` snippet);
- viewpoint / left-right anchoring discipline (carried over from `bvb-scene-builder`);
- a **growing "Heuristics" section** appended after each scene, e.g.:
  - arkitscenes bedrooms with two beds usually have one spurious bed → keep the bed that
    clusters with the route objects (nightstand + wardrobe);
  - window in a sloped ceiling = skylight, not a vertical-wall window;
  - room_size too large → shrink Floor + the four walls to the target area, apply scale=1;
  - never infer left/right from world axes — anchor to the reference viewpoint first.

## 8. Propagation

Any new scene enters the loop with `bvbrefine start <next_id>`. The skill keeps Claude's
behavior identical across sessions; the Heuristics section makes each subsequent scene
faster because recurring failure modes become one-line rules instead of rediscovery.

## 9. Out of Scope (YAGNI)

- Batch / parallel multi-scene refinement.
- A GUI. The loop is CLI + MCP + chat.
- Auto-push without the pre-push human gate.
- Defining a hallucination metric (the eval README defers this too).

## 10. Dependencies / Prerequisites

- BlenderMCP connected (addon running, `uvx blender-mcp` registered) — already set up.
- ffmpeg/ffprobe on PATH — present.
- `VSI-Bench/` cloned for reference videos — present.
- Optional for the official proposition gate: `OPENAI_API_KEY` + `BVB_JUDGE_MODEL`
  (+ `OPENAI_BASE_URL` for non-OpenAI endpoints). Absent → self-judged fallback.
