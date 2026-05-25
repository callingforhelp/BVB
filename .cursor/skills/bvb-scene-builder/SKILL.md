---
name: bvb-scene-builder
description: Build 3D scenes in Blender via BlenderMCP for the BVB (Blender-VideoBench) benchmark. Use this skill whenever the user asks to reconstruct a 3D scene from video keyframes, build a Blender scene from images or JSON metadata, create ground truth (GT) data for video understanding benchmarks, or mentions BVB, BlenderMCP scene building, or inverse graphics. Also trigger when the user uploads video frames and asks to recreate the scene in Blender, or when they provide scene metadata and want it turned into a Blender scene. This skill should be used even if the user just says "build this scene" with attached images — if BlenderMCP is connected, this is likely a BVB task.
---

# BVB Scene Builder

Build reproducible 3D scenes in Blender via BlenderMCP for the BVB benchmark. Scenes must use only basic primitives and solid colors so they can be cleanly exported as `bpy` Python code.

## Quick Reference

Before starting, make sure BlenderMCP is connected. Test with `get_scene_info`.

### Forbidden

- Imported models (.obj, .fbx, .glTF)
- Subdivision surfaces, sculpted geometry, booleans
- Curves, NURBS, text objects
- Geometry Nodes, modifiers
- Any node graph beyond a single Principled BSDF

---
### Grouped Parts

After building a multi-part object (e.g., a chair = seat + back + 4 legs), add all children to the main body part using an `Collection` folder. This ensures the object moves as a unit and prevents accidental drift.

For example:

- Chair
  - Chair_Seat
  - Chair_Back
  - Chair_Leg_1
  - Chair_Leg_2
  - Chair_Leg_3
  - Chair_Leg_4
- Sofa
  - Sofa_Seat
  - Sofa_Back
  - Sofa_Arm_1
  - Sofa_Arm_2
...
---

## Workflow

### Step 1: Analyze Input

The user provides **keyframes** (images) and optionally **JSON metadata**.

From keyframes, identify:
- Room layout and dimensions
- Object types and approximate positions
- Color palette
- Light sources and shadows
- Camera angle

From JSON metadata, extract:
- Exact positions, scales, rotations
- Material colors (RGB values)
- Light types and energies
- Camera lens and position
- Animation keyframes (if any)

### Refining Existing Scenes

When the user asks to optimize or refine an existing `.blend` or exported `.py`, do **not** assume the current scene is mostly correct. First compare several video/keyframe timestamps against the current Blender scene and classify the mismatch:

- **Local mismatch**: colors, small object positions, lighting, or missing details are wrong. Patch incrementally.
- **Structural mismatch**: room layout, camera path, doors/windows, major furniture positions, or scene identity are wrong. Do not keep layering patches on the old scene.

For structural mismatch:
1. Identify 1-3 anchor objects that are already correct, such as a bed, chandelier, or fixed architectural feature.
2. Preserve only those anchors, useful lights/camera, and reusable solid materials.
3. Delete or hide incorrect major object groups before rebuilding them.
4. Rebuild wrong regions directly at their final target positions instead of moving/rotating/scaling old groups through multiple edits.
5. Use `dimensions` to size primitives, then immediately apply scale so mesh transforms end with `scale=(1, 1, 1)`.
6. After each rebuild step, take a screenshot and verify spatial relationships, not just whether the requested objects exist.

Before editing an existing scene, audit:
- [ ] Compare current scene against multiple video/keyframe timestamps.
- [ ] List correct anchor objects to preserve.
- [ ] List incorrect major groups to remove or rebuild.
- [ ] Decide whether the task is local refinement or structural rebuild.
- [ ] Avoid accumulating hidden old objects unless explicitly needed.
- [ ] Verify multi-part objects remain coherent after transforms.

### Viewpoint and Left/Right Verification

When refining an existing scene, do not infer "left", "right", "front", or "back" from Blender world axes alone. First anchor the scene to a reference viewpoint from the video.

Before moving major objects or connected rooms:
1. Identify the camera/viewer position in the reference frame, especially doorways, hallway views, mirrors, and room-to-room sight lines.
2. Define the viewer-facing direction as a vector in Blender coordinates. Write down which world axis corresponds to viewer-left and viewer-right.
3. For every object described as being on the left/right wall, in front of/behind another object, or visible through a doorway, verify it from that same viewpoint, not only from top-down view.
4. If there is exactly one visible door/opening in the reference, do not add a second opening or side room to satisfy QA geometry. Preserve the observed connectivity first, then adjust object positions within that topology.
5. After changing room topology or large furniture positions, create or move a temporary validation camera to the reference viewpoint and take a screenshot from that camera. Compare against the video frame before saving.

Common failure mode: placing an object on Blender `+X` because it seems like "right" in top-down coordinates, when the viewer in the video is facing `-Y`, making viewer-right equal to Blender `-X`.

### Step 2: Build Scene (in this order)

Execute code via `blender:execute_blender_code` in **small, incremental steps**. Take a screenshot after each major step to verify before proceeding.

```
1.  Clear scene
2.  Set render engine (EEVEE) and world environment
3.  Create ALL materials
4.  Build room structure (floor and walls; do not add a ceiling unless it is explicitly visible and important in the reference)
5.  Add large furniture
6.  Add medium objects
7.  Add small objects
8.  Set up visible light fixtures and light sources (chandeliers, ceiling lights, lamps, Sun, Area, Point, etc)
...
```

**Important execution rules:**
- Each `execute_blender_code` call should contain at most ONE complete object group (e.g., one chair with all its parts, or all four walls). Do not try to build the entire scene in one code block.
- Always use `import math` and `math.radians()` for any rotation — NEVER pass raw degree numbers to `rotation_euler`.
- Start each code block with `bpy.ops.object.select_all(action='DESELECT')` to avoid leftover selection state.
- For primitive dimensions, prefer setting `obj.dimensions = (...)` and then `bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)` so final mesh scale is `(1, 1, 1)`.
- Avoid repeated incremental move/rotate/scale edits on complex multi-part objects. If parts drift or the object feels fragmented, rebuild that object group cleanly at the target location.
- Do not omit visible lamps or chandeliers just because the ceiling is omitted. If a fixture is visible in the reference, model the fixture body and add an appropriate light source.

### Step 3: Verify and Adjust

After building, take a viewport screenshot and compare with keyframes. Adjust:
- Object positions and sizes
- Material colors
- Light energy and positions


### Step 4: Final Check

Before telling the user the scene is ready:
1. Verify NO complex meshes exist (all objects should be basic primitives)
2. Verify NO linked material nodes (all Base Color should be solid values)
3. Verify visibility states are correct
4. **Verify structural integrity: zoom into joints/corners to confirm no floating parts. Hanging lights may float only when the reference clearly shows a suspended fixture.**

---
### Naming Convention

Use descriptive names with prefixes for grouped parts:
- `Wall_Front`, `Wall_Back`, `Wall_Left`, `Wall_Right`
- `Sofa_Seat`, `Sofa_Back`, `Sofa_Arm_1`, `Sofa_Arm_2`
- `Chair_1_Seat`, `Chair_1_Back`, `Chair_1_Leg_1`
- `Ceiling_Light`, `Floor_Lamp_Light`, ...

Collection folder use the object name without part suffix: `Chair_1`, `Table_1`, `Sofa_1`... etc.

---

## Common Pitfalls Checklist

Before finishing any scene, verify:

- [ ] **No orphan objects**: Every multi-part object should be add to a `Collection` (except the light sources and camera)
- [ ] **Rotations in radians**: All `rotation_euler` values use `math.radians()`, never raw degrees
- [ ] **Visible lights included**: If the reference shows a chandelier, ceiling light, wall lamp, or floor lamp, the fixture body and light source are both present.
- [ ] **Viewpoint left/right checked**: For doorway or room-to-room views, define the viewer-facing vector and verify left/right wall placement from that viewpoint, not from Blender world axes or top-down view.
- [ ] **Connectivity preserved**: Do not create extra doors, openings, or side rooms unless they are visible in the reference video.
- [ ] if you notice that the walls that are supposed to closure but not, the scale value of the walls should be checked (usually set scale=1 can fix many problems).

---

## Export

When the user is satisfied, remind them to export using the **Export BPY Code** addon:

`File > Export > Blender Python Script (.py)`

This addon exports the scene as reproducible `bpy` Python code. It will warn about:
- Complex meshes that can't be exactly reconstructed (should be zero if built correctly)
- Materials with linked nodes (should be zero if using only solid colors)
