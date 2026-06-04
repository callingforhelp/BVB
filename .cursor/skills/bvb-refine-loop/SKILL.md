---
name: bvb-refine-loop
description: Human-in-the-loop iterative refinement of an existing BVB Blender scene against its QA via BlenderMCP. Use when refining/optimizing a .blend so the evaluation model answers the scene's QA correctly, when the user runs ./start.sh or bvbrefine, or mentions BVB refinement, checkpoints, or the refinement tracker. Pairs with the bvbrefine CLI (qa/frames/start/checkpoints/gate). For building a scene from scratch use bvb-scene-builder instead.
---

# BVB Refine Loop

Drive the human↔Claude refinement of ONE existing BVB scene. Mechanical work is the
`refine/bvbrefine.py` CLI; this skill is the collaboration protocol Claude follows.

## Loop

1. `python refine/bvbrefine.py start <id>` — prints QA + focus hints + the scene's unit
   tests, extracts reference frames to `blend/refine_<id>/frames/`, and prints a baseline
   snapshot snippet. Run that snippet via BlenderMCP `execute_blender_code` to save
   `blend/refine_<id>/<id>_v00_baseline.blend` (`save_as_mainfile(..., copy=True)`).
2. Inventory: `get_scene_info` + read the frames. Classify EACH QA axis as **local** (color,
   small position, lighting, missing detail) or **structural** (room size, topology, major
   furniture, scene identity). Build an ordered edit plan.
3. **GATE — structural changes:** stop and get the human's OK before any structural edit.
   Local patches proceed autonomously.
4. Per edit: `execute_blender_code` (one object group, radians via `math.radians`, deselect
   first, set `dimensions` then apply scale) → `get_viewport_screenshot` to verify → live
   function-check (compute floor area / relative positions directly) → snapshot a new
   `<id>_vNN_<desc>.blend` checkpoint (`copy=True`, so the live file stays `blend/<id>.blend`).
5. `python refine/bvbrefine.py gate <id>` — offline deterministic scoring of `function`
   tests; add `--judge` (needs `OPENAI_API_KEY` + `BVB_JUDGE_MODEL`) for the official
   proposition judge.
6. **GATE — pre-push:** show official results + final screenshots. On approval, the final
   checkpoint is already the live `blend/<id>.blend`; verify exportability (the gate's bpy
   export is the same path), then commit, push, and update the Notion tracker's Refiner column.

## Verification rule (hybrid)

- `function` tests (room_area, counts, distances): compute live in Blender each step;
  `bvbrefine gate` re-scores offline at the end. room_area = floor-group bbox `X*Y`.
- `proposition` tests (route_planning): reason from geometry + frames live; `--judge` for
  the official check.

## Checkpoint convention

- Dir `blend/refine_<id>/` (git-ignored). Names `<id>_vNN_<desc>.blend`, `v00_baseline` first.
- Always `save_as_mainfile(filepath=..., copy=True)` — never switch the live file off
  `blend/<id>.blend`. Roll back by `open_mainfile` on a `vNN`.

## Viewpoint / left-right discipline

Never infer left/right/front/back from Blender world axes. Anchor to the reference
viewpoint first (doorways, sight lines); define the viewer-facing vector; verify
left/right placements from that viewpoint. Do not add doors/openings not seen in the video.

## Heuristics (append one line per scene as you learn them)

- arkitscenes bedrooms with two beds usually have one spurious bed → keep the bed that
  clusters with the route objects (nightstand + wardrobe); delete the other.
- A window in a sloped ceiling is a skylight (roof window), not a vertical-wall window.
- room_size too large → shrink Floor + the four walls to the target footprint, apply scale=1
  so mesh scale ends (1,1,1). Target check: floor bbox X*Y within the test tolerance.
- "wardrobe with two mirrors" means two distinct mirror panels on the door — model both.
