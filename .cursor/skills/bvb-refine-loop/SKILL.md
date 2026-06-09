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

1. **Follow the user's lead on who drives diagnosis.** Default: the user names the problem.
   But some users will ask *you* to propose your read first, then discuss high-level before
   executing — do that when asked. Either way your read is a *hypothesis to verify*, not a
   fact. (We once "knew" a bedroom had one spurious bed — it had two real beds.)
2. **Ask before structural edits.** Show *measured* geometry (bounding boxes, the slope line,
   the floor footprint) and confirm direction/amount before moving walls or deleting groups.
3. **For deletions, list candidates with reasons and let the user veto specific items**
   before removing anything.
4. **Split a batch of instructions into "do-now" vs "flag-first."** When the user dumps
   several edits at once, execute the unambiguous ones, but STOP and flag anything that
   changes what a QA test measures — especially **renaming/reclassifying an object** (e.g.
   "this is a water heater, not a washer"). Renaming can orphan the QA's named object; check
   whether a test still has its target before you proceed (see Heuristics).
5. **Hiding / isolating ≠ deleting.** Say so every time, or the user thinks you wiped the
   scene. Prefer wireframe over hiding.
6. **One coordinated edit → screenshot → checkpoint.** Then hand back for the next call.

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
  `bvbrefine gate` re-scores offline. It auto-grounds **`room_area`** (floor bbox X·Y),
  **`longest_dimension`** (object_size — matches `object_ref`, picking the LARGEST matching
  group so it grabs the body not a sub-part), and **`closest_distance`** (two objects from
  `params.object_refs`). Only `count_objects` still needs `--judge`. A `function` test that
  prints `actual=None` means grounding failed, NOT that the scene is wrong.
- **Size tests measure `obj.scale`, so a size-QA object MUST be a scaled cube.** The harness
  `object_extent` reads each object's `scale`, not its mesh bbox — a complex/cylinder mesh (or
  a cube with applied scale that exports without scale) reads as **1.0 m** and the test prints
  `actual=100`. If an object the QA measures isn't a `primitive_cube_add(size=1)` + `obj.scale`,
  rebuild it from scaled cubes at the target size (this is also the "primitives only" rule).
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
| Room too small/narrow — widen but keep one wall fixed | anchored axis-scale (cursor at the fixed wall, resize on that axis); then move the opposite wall + its furniture group |
| Appliance should be wall-mounted / under-counter | translate the part group in Z (mount) or place under the counter; counter/cabinet extend to it |
| `object_size` gate prints `actual=100` (or ~1 m) | the object is a complex/cylinder mesh — rebuild it from `primitive_cube_add(size=1)` + `obj.scale` at the target size |
| QA needs a whole missing room (combined space) | analyze the video first (Workflow over many frames), then build the second room south of the existing one through the door |

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
- room_area scores ONE floor's bbox X·Y — so a multi-room / "combined space" needs a SINGLE
  rectangular Floor whose bbox = the target area (two separate floors or an L-shape fail: the
  bbox is one floor or the L's outer rectangle). When the same scene also fixes a distance
  (e.g. sofa↔toilet 3.9 m), the area + distance constraints can force the proportions (we got
  a long 2×7.8 m rectangle); flag that trade-off to the user rather than guessing.
- "Combined space" QA but only one room modeled → the other room is through the door; build it
  south of the existing room (door wall becomes the shared partition, cut its opening), and
  analyze the video with a Workflow first since you can't see it.
- An indented ("凹进") corner = remove the floor there and wall off the adjacent room edge;
  the leftover region reads as outside.
- Never infer left/right/front/back from world axes — anchor to the video viewpoint first
  (doorways, sight lines). Don't add doors/openings not seen in the video.
- Reclassifying an object can orphan a QA target: when the user says the wall box is a water
  heater (not a washer), the "washer" object_size + route tests lose their object. Add the
  real appliance the QA names (e.g. a front-load washer under the counter, sized to the GT)
  AND keep the reclassified one. Name it so the judge finds it (`Washer_Body`).
- Galley / utility room too narrow: widen by anchored X-scale of the floor + back/front
  walls (cursor at the fixed wall), move the opposite wall out, and shift the counter +
  cabinets + sink + appliances as one group to sit against that wall, clear of the doors.
- Match appliances to the video: under-counter front-load washer (round door facing the
  room), wall-mounted water heater near the ceiling; a "table" with a solid base, not legs.

## Delivery (git) — keep tooling personal, ship only the .blend

- Work on a personal branch (`andy`); the `refine/` tooling, skill, docs, and checkpoints
  live there and never go to `main`.
- Stay current: `git fetch origin` then `git merge origin/main` into your branch periodically
  (teammates push scenes + export-script fixes to main).
- Ship a scene: branch clean off `origin/main`, take ONLY the one file, push to main:
  `git checkout -b deliver-<id> origin/main` → `git checkout <yourbranch> -- blend/<id>.blend`
  → commit `Refine Blender Scene <id>` → push `deliver-<id>:main` → delete the temp branch.
- SSH (port 22) may time out here; prefix fetch/push with the gh credential helper:
  `git -c url."https://github.com/".insteadOf="git@github.com:" -c credential.helper='!gh auth git-credential' <fetch|push> …`
- After the push lands, the user updates the Notion tracker row: `Refiner` = their name,
  `Spatial` = `Done` (those two columns only).
