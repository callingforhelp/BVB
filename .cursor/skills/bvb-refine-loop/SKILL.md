---
name: bvb-refine-loop
description: Use when refining or optimizing an existing BVB Blender scene (.blend) so an evaluation model answers the scene's QA correctly, when the user runs ./start.sh or bvbrefine, mentions BVB refinement / checkpoints / the refinement tracker, or asks to fix a reconstructed room against its source video. For building a scene from scratch use bvb-scene-builder instead.
---

# BVB Refine Loop

Drive the **human-led** refinement of ONE existing BVB scene via BlenderMCP. The user
compares the source video against Blender and decides what is wrong; you ask, edit, and
verify. Mechanical setup is the `refine/bvbrefine.py` CLI. Reusable Blender snippets are in
`blender-mcp-recipes.md` — read it before editing geometry.

## Interaction protocol (this is the point — don't skip it)

The user looks at the video; you don't see what they see. So:

1. **The user names the problem.** Do NOT open with "here are the N things wrong and my
   plan." Your read of the video is a *hypothesis to verify with them*, not a fact. (We once
   "knew" a bedroom had one spurious bed — it had two real beds. Verifying saved a bad edit.)
2. **Ask before structural edits.** Show *measured* geometry (bounding boxes, the slope line,
   the floor footprint) and confirm direction/amount before moving walls or deleting groups.
3. **For deletions, list candidates with reasons and let the user veto specific items**
   before removing anything.
4. **Hiding / isolating ≠ deleting.** Say so every time, or the user thinks you wiped the
   scene. Prefer wireframe over hiding.
5. **One coordinated edit → screenshot → checkpoint.** Then hand back for the next call.

## Loop

1. `python refine/bvbrefine.py start <id>` — prints QA + focus hints + the scene's unit
   tests, extracts reference frames to `blend/refine_<id>/frames/`. Read the frames. Save a
   `v00_baseline` snapshot via MCP (`save_as_mainfile(..., copy=True)`).
2. Wait for the user to state a problem. Ask clarifying questions. Edit one group at a time
   (recipes file), `get_viewport_screenshot` to verify, then snapshot `<id>_vNN_<desc>.blend`.
3. `python refine/bvbrefine.py gate <id>` — offline-deterministic `function` tests (room_area
   etc.); add `--judge` (needs `OPENAI_API_KEY` + `BVB_JUDGE_MODEL`) for `proposition` tests
   (route_planning).
4. On the user's approval: `save_mainfile()` to write `blend/<id>.blend`, then their git flow
   (e.g. a personal branch + a scene-only PR of just the `.blend` to main).

## Verification

- `function` tests: compute the quantity live in Blender for instant feedback, then
  `bvbrefine gate` re-scores offline. `room_area` = the **floor group's** bbox X·Y.
- `proposition` tests (route_planning): reason from geometry + frames; `--judge` for official.

## Common problems → quick fix (all snippets in `blender-mcp-recipes.md`)

| Symptom | Fix |
|---|---|
| Top view snaps back to perspective | set `region_3d.view_rotation` quaternion + `view_perspective='ORTHO'` |
| Sloped ceiling blocks the top view | wireframe shading (preferred), or hide ceiling + restore — **tell the user** |
| bbox / `matrix_world` reads the old value after a move | `bpy.context.view_layer.update()` before reading |
| Multi-part object (wardrobe) drifts when scaled/rotated | transform the whole group about the 3D cursor at its bbox center |
| A wall can't follow the sloped ceiling | rebuild it as a pentagon prism via bmesh (`z(y)` from the ceiling verts) |
| Floating outlets / bed feet / a door behind the bed | stray-object scan → confirm → delete |
| Baseboard / crown left behind after moving a wall | move/trim the trim with its wall (separate objects!) |
| Need a doorway in a wall | split the wall into two segments (no booleans) |

## Checkpoint convention

`blend/refine_<id>/` (git-ignored). Names `<id>_vNN_<desc>.blend`, `v00_baseline` first.
Always `save_as_mainfile(..., copy=True)` so the live `blend/<id>.blend` is untouched until
final approval. Roll back by opening any `vNN`.

## Heuristics (append one line per scene as you learn them)

- Two beds can BOTH be real — verify against the video; don't assume one is a hallucination.
- Attic room: side walls should be pentagons whose top follows the sloped ceiling; trim the
  crown molding to the flat-top section so it doesn't float over the slope.
- A window in a sloped ceiling is a skylight, not a vertical-wall window.
- "Wardrobe with two mirrors" → two distinct mirror panels on the door that **faces the room**
  (rotate the wardrobe so its mirrored face points inward, then re-seat it against the wall).
- room_size too large → resize the Floor and shift the back wall toward the beds by the same
  Δy (recipe), hitting the target area; confirm with `gate`. Floor bbox X·Y is what's scored.
- An indented ("凹进") corner = remove the floor there and wall off the adjacent room edge;
  the leftover region reads as outside.
- Never infer left/right/front/back from world axes — anchor to the video viewpoint first
  (doorways, sight lines). Don't add doors/openings not seen in the video.
