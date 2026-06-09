# BlenderMCP Recipes for BVB Refinement

Battle-tested `execute_blender_code` snippets. Each one fixes a problem that actually bit
us during refinement. Copy and adapt.

---

## Reliable orthographic views (top/side)

`bpy.ops.view3d.view_axis(...)` via MCP often snaps back to a tilted "User Perspective".
Set the view matrix directly instead — this sticks:

```python
import bpy, mathutils
QUATS = {'TOP': (1, 0, 0, 0), 'FRONT': (0.7071, 0.7071, 0, 0), 'RIGHT': (0.5, 0.5, 0.5, 0.5)}
def set_ortho(view='TOP'):
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            rv = area.spaces.active.region_3d
            rv.view_rotation = mathutils.Quaternion(QUATS[view])
            rv.view_perspective = 'ORTHO'
            area.spaces.active.shading.type = 'SOLID'
set_ortho('TOP')
```

- TOP = floor plan (verify layout, room footprint, object placement).
- RIGHT = look down -X: side walls show their Y-Z silhouette → check a sloped-wall top
  against the ceiling line in one shot.

## Seeing through a sloped ceiling

A top view is blocked by `Ceiling_Sloped`. Two non-destructive options — **prefer wireframe**:

```python
# Option A (preferred): wireframe — nothing "disappears", whole scene stays visible
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces.active.shading.type = 'WIREFRAME'

# Option B: hide ceiling for a solid top view, then ALWAYS restore
for n in ['Ceiling_Sloped', 'Ceiling_Light']:
    o = bpy.data.objects.get(n)
    if o: o.hide_set(True)
# ... screenshot ...  then: o.hide_set(False)
```

**Always tell the user before hiding or isolating.** Hidden / local-view objects look
deleted and will alarm them. Say "temporarily hiding X for this screenshot, restoring after".

## Depsgraph is stale right after a transform

Setting `.location` / `.scale` does NOT immediately update `matrix_world` / `bound_box`.
Reading them back gives the OLD value. Force an update first:

```python
o.location.x += 0.30
bpy.context.view_layer.update()      # without this, the bbox read below is stale
xs = [(o.matrix_world @ mathutils.Vector(c)).x for c in o.bound_box]
```

## Transform a multi-part object coherently (scale / rotate a group)

Scaling or rotating each part independently makes a multi-part object (e.g. a wardrobe =
body + mirror + handles + divider) drift apart. Pivot the whole group about its shared
center via the 3D cursor:

```python
import bpy, mathutils, math
parts = [o for o in bpy.context.scene.objects if o.name.startswith('Wardrobe')]
pts = [o.matrix_world @ mathutils.Vector(c) for o in parts for c in o.bound_box]
center = (sum(p.x for p in pts)/len(pts), sum(p.y for p in pts)/len(pts), sum(p.z for p in pts)/len(pts))
# use bbox center for axis-true scaling:
center = ((min(p.x for p in pts)+max(p.x for p in pts))/2,
          (min(p.y for p in pts)+max(p.y for p in pts))/2,
          (min(p.z for p in pts)+max(p.z for p in pts))/2)
sc = bpy.context.scene
sc.cursor.location = center
old = sc.tool_settings.transform_pivot_point
sc.tool_settings.transform_pivot_point = 'CURSOR'
bpy.ops.object.select_all(action='DESELECT')
for o in parts: o.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for region in area.regions:
            if region.type == 'WINDOW':
                with bpy.context.temp_override(area=area, region=region):
                    bpy.ops.transform.resize(value=(1, 1.5, 1), orient_type='GLOBAL')   # widen in Y
                    # bpy.ops.transform.rotate(value=math.radians(90), orient_axis='Z')  # or rotate
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
sc.tool_settings.transform_pivot_point = old
```

Note: rotating a group about its bbox center can pull a wall-hugging object off the wall
(its bbox changes). Re-seat it against the wall afterward with a plain translate.

## Checkpoint save (never touch the live file mid-refinement)

```python
bpy.ops.wm.save_as_mainfile(filepath='/abs/blend/refine_<id>/<id>_vNN_<desc>.blend', copy=True)
```

`copy=True` writes a snapshot WITHOUT switching the active file off `blend/<id>.blend`. Only
at final approval do a normal `bpy.ops.wm.save_mainfile()` to write the deliverable.

## Resize one edge of a wall/floor, keep the opposite edge fixed

Moving a back wall toward the beds = raise the floor/wall's south edge while the north edge
(front) stays put. Works for any axis; assumes a centered origin (true for primitive cubes
and after `transform_apply`).

```python
def resize_edge_y(name, new_y_min):
    o = bpy.data.objects[name]
    ys = [(o.matrix_world @ mathutils.Vector(c)).y for c in o.bound_box]
    y0, y1 = min(ys), max(ys)
    o.location.y = (new_y_min + y1) / 2
    d = o.dimensions.copy(); d.y = (y1 - new_y_min); o.dimensions = d
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = o; o.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
```

**Coordinated room-depth change:** rigid-translate the back-wall + entry assembly by Δy,
then `resize_edge_y` the floor and both side walls' south edges by the same Δy. Don't forget
the back baseboard/crown — they're separate objects and get left behind (see Common Mistakes
in SKILL.md).

## Open a doorway: split a wall (no booleans)

Booleans are forbidden. To put a gap in a wall, shrink it to one side and add the other side
as a new primitive, leaving the opening between them:

```python
def cube(name, loc, scale, mat=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object; o.name = name; o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat: o.data.materials.append(mat)
    return o
# Wall spanning x[-2.1,4.1] at y, opening x[0.55,1.45]:
#   shrink existing wall to left segment x[-2.1,0.55]; cube() a right segment x[1.45,4.1].
#   optional frame = two thin jamb cubes + a header cube around the opening.
```

## Rebuild a side wall to follow a sloped ceiling (pentagon prism)

A box wall can't follow the roof slope. Rebuild it as a pentagon prism whose top edge =
the ceiling line. First read the ceiling's two profile points to get `z(y)`:

```python
import bpy, bmesh, mathutils
# Ceiling line from its two extreme verts: (y_hi, z_hi)=back/high, (y_lo, z_lo)=front/low
# slope m = (z_lo - z_hi)/(y_lo - y_hi);  z(y) = z_hi + m*(y - y_hi);  cap at wall_top.
def rebuild_sloped_wall(name, x0, x1, y_back, y_front, wall_top, z_of_y):
    old = bpy.data.objects[name]
    mat = old.data.materials[0] if old.data.materials else None
    coll = old.users_collection[0] if old.users_collection else bpy.context.scene.collection
    bpy.data.objects.remove(old, do_unlink=True)
    zf = min(wall_top, z_of_y(y_front))
    y_knee = next(y for y in [y_back + 0.01*i for i in range(int((y_front-y_back)/0.01))]
                  if z_of_y(y) <= wall_top)        # where slope meets the flat top
    prof = [(y_back, 0.0), (y_front, 0.0), (y_front, zf), (y_knee, wall_top), (y_back, wall_top)]
    mesh = bpy.data.meshes.new(name + '_mesh')
    obj = bpy.data.objects.new(name, mesh); coll.objects.link(obj)
    bm = bmesh.new()
    v0 = [bm.verts.new((x0, y, z)) for (y, z) in prof]
    v1 = [bm.verts.new((x1, y, z)) for (y, z) in prof]
    bm.faces.new(v0); bm.faces.new(list(reversed(v1)))
    for i in range(len(prof)):
        j = (i + 1) % len(prof)
        bm.faces.new((v0[i], v0[j], v1[j], v1[i]))
    bm.normal_update(); bm.to_mesh(mesh); bm.free()
    if mat: obj.data.materials.append(mat)
```

Then trim the wall's crown molding to the flat-top region only (`resize_edge_y`), or it
floats above the now-lower sloped section. Custom-mesh walls still export fine through the
Export BPY Code addon (verified), and `room_area` only reads the floor group, so the gate
still passes.

## Find stray / hallucination objects

These scenes ship with floating artifacts: orphan outlets at the old floor edge, bed feet
scattered off their bed, a "Bathroom_Light" in a bedroom, doors behind the headboard. List
small objects sitting outside every furniture/wall footprint, then confirm with the user
before deleting:

```python
import bpy, mathutils
def center_xy(o):
    pts = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
    return ((min(p.x for p in pts)+max(p.x for p in pts))/2,
            (min(p.y for p in pts)+max(p.y for p in pts))/2)
# Sanity heuristics, not auto-delete:
#  - a Bed*_Foot_* whose XY is outside its Bed*_Base footprint  -> stray
#  - an Outlet_*/Light_* sitting below a removed floor / past a wall  -> stray
#  - a name semantically wrong for the room (Bathroom_* in a bedroom) -> stray
for o in bpy.context.scene.objects:
    if 'foot' in o.name.lower() or 'outlet' in o.name.lower():
        print(o.name, [round(v, 2) for v in center_xy(o)])
```

## Widen a room, anchoring one wall (axis-scale at the cursor)

To make a too-narrow room wider while keeping the left wall at x=0 fixed: scale the floor and
the perpendicular walls along X about a cursor placed at the fixed wall. This anchors one edge
and grows the other regardless of each object's origin. Then translate the opposite wall out,
and move the counter/cabinet/appliance group by the same delta to sit against it.

```python
import bpy, mathutils
def scale_axis_anchored(names, target_max, axis=0, anchor=0.0):
    sc = bpy.context.scene
    sc.cursor.location = (anchor, 0, 0) if axis == 0 else (0, anchor, 0)
    old = sc.tool_settings.transform_pivot_point
    sc.tool_settings.transform_pivot_point = 'CURSOR'
    for n in names:
        o = bpy.data.objects[n]
        cur = [(o.matrix_world @ mathutils.Vector(c))[axis] for c in o.bound_box]
        f = (target_max - anchor) / (max(cur) - anchor)
        val = [1, 1, 1]; val[axis] = f
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = o; o.select_set(True)
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with bpy.context.temp_override(area=area, region=region):
                            bpy.ops.transform.resize(value=val, orient_type='GLOBAL')
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    sc.tool_settings.transform_pivot_point = old
# scale_axis_anchored(['Floor'], 2.58); scale_axis_anchored(['Wall_Back','Wall_Front'], 2.68)
# then: Wall_Right.location.x += 0.98; and shift the counter group +0.98 in X.
```

## Mount an appliance on the wall / under a counter

Move a multi-part appliance (body + door + panel) as a group. Wall-mount = scale to the GT
longest dimension about its base, then lift in Z and keep it flush to the wall. Under-counter
washer = a body box on the floor with a round front door (a cylinder turned to face the room).

```python
import bpy, mathutils, math
# --- wall-mount: size to 91cm longest (scale Z about the base), then raise to z=0.9 ---
parts = [bpy.data.objects[n] for n in ['Water_Heater', 'Water_Heater_Panel']]
pts = [o.matrix_world @ mathutils.Vector(c) for o in parts for c in o.bound_box]
cx = (min(p.x for p in pts)+max(p.x for p in pts))/2
cy = (min(p.y for p in pts)+max(p.y for p in pts))/2
sc = bpy.context.scene; sc.cursor.location = (cx, cy, min(p.z for p in pts))
sc.tool_settings.transform_pivot_point = 'CURSOR'
# ... select parts, bpy.ops.transform.resize(value=(1,1, 0.91/0.85)), transform_apply ...
for o in parts: o.location.z += 0.9            # lift onto the wall above the counter

# --- under-counter front-load washer (round door faces the room, -X) ---
bpy.ops.mesh.primitive_cube_add(size=1, location=(2.28, 0.6, 0.425))
body = bpy.context.active_object; body.name = 'Washer_Body'
body.scale = (0.60, 0.60, 0.85); bpy.ops.object.transform_apply(scale=True)
bpy.ops.mesh.primitive_cylinder_add(radius=0.17, depth=0.04, location=(1.975, 0.6, 0.45))
door = bpy.context.active_object; door.name = 'Washer_Door'
door.rotation_euler = (0, math.radians(90), 0)   # disc faces ±X (the room)
bpy.ops.object.transform_apply(rotation=True)
```

## Solid-base furniture (not legs)

When the user wants a table/cabinet "solid underneath," delete the legs and add one base box
from the floor to just under the top.

```python
# delete legs: for o in [x for x in bpy.context.scene.objects if x.name.startswith('Table_Leg')]: bpy.data.objects.remove(o, do_unlink=True)
# Table_Base = a cube spanning the top's XY footprint, z from 0 to (top_z - top_thickness).
```
