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

## self_verify — run after EVERY edit (stragglers + floaters; check clipping by hand)

Set `ROOMS` / `WALLS_*` to the current scene's world bounds, then run this every time you move
or reshape anything. Anything it prints gets re-seated, dropped, or whitelisted — never ignored.

```python
import bpy, mathutils
ROOMS  = [(-2.1, 0.3, -1.05, 0.6), (-0.9, 0.3, 0.6, 5.1)]  # (x0,x1,y0,y1) per room (bath ∪ hall)
WALLS_X = (-2.1, 0.3); WALLS_Y = (-1.05, 0.6)              # inner faces; bbox touching one = on-wall
STRUCT  = ('Wall','Floor','Border','Downlight','Hallway','Vent','Ceiling','Light','Camera')
def bb(o):
    p=[o.matrix_world@mathutils.Vector(c) for c in o.bound_box]
    return (min(q.x for q in p),max(q.x for q in p),min(q.y for q in p),
            max(q.y for q in p),min(q.z for q in p),max(q.z for q in p))
def in_room(cx,cy):
    return any(x0-0.2<cx<x1+0.2 and y0-0.2<cy<y1+0.2 for (x0,x1,y0,y1) in ROOMS)
def on_wall(x0,x1,y0,y1):
    return x0<WALLS_X[0]+0.15 or x1>WALLS_X[1]-0.15 or y0<WALLS_Y[0]+0.15 or y1>WALLS_Y[1]-0.15
for o in bpy.context.scene.objects:
    if o.type!='MESH' or o.hide_get() or any(o.name.startswith(s) for s in STRUCT): continue
    x0,x1,y0,y1,z0,z1=bb(o); cx,cy=(x0+x1)/2,(y0+y1)/2
    if not in_room(cx,cy):                     print("OUT-OF-ROOM:", o.name, round(cx,2),round(cy,2))
    elif z0>0.1 and not on_wall(x0,x1,y0,y1):   print("MAYBE-FLOATING:", o.name, "z_min",round(z0,2),(round(cx,2),round(cy,2)))
# Whitelist real surfaces by hand (an item resting on the vanity top / cistern / tub rim is fine).
# Clipping: for the objects you just MOVED, check their bbox vs each wall's inner face and vs
# each other (overlap on all 3 axes). NB the gate's size/distance use location ± obj.scale/2.
```

---

## overlap_scan — pairwise 穿模 + wall-poke (run after EVERY edit)

Automated replacement for "check clipping by hand." Builds each non-structural mesh's world
AABB and flags any CROSS-SET pair overlapping on all 3 axes beyond `TOL_PEN` (positive depth),
plus any object poking past a wall's inner face. Same-unit siblings are NOT inferred from the
first underscore token — they are skipped only when BOTH names are members of a DECLARED set
(reuse `connected_scan`'s SETS), so `Toilet` vs `Toilet_Paper_Roll` is NOT silently exempted.
Named accessories you list are scanned even against their host. Call `view_layer.update()` first.
`bound_box` ignores UNAPPLIED Mirror/Solidify modifiers — apply them or read the evaluated AABB.

```python
import bpy, mathutils
TOL_PEN = 0.01                       # +1cm: flush faces read ~0 and stay clean
# EDIT PER SCENE: the explicit structural-object names to skip (exact match on the leading
# underscore-delimited TOKEN, so 'Wall' skips Wall_01 but NOT Wallpaper/Wall_Cabinet):
STRUCT = {'Wall','Floor','Ceiling','Border','Downlight','Vent','Camera','Light','Hallway'}
# DECLARE the connected sets (same list as connected_scan); only pairs BOTH inside one set skip:
SETS = []   # e.g. [('Vanity',), ('Shower',), ('Water_Heater','Water_Heater_Panel')]
def aabb(o):
    p=[o.matrix_world@mathutils.Vector(c) for c in o.bound_box]   # apply modifiers first
    return [min(q[i] for q in p) for i in range(3)],[max(q[i] for q in p) for i in range(3)]
def set_of(n):                                   # the declared set whose prefix n startswith
    for g in SETS:
        if any(n.startswith(p) for p in g): return g
    return None
bpy.context.view_layer.update()
def is_struct(n): return n.split('_')[0] in STRUCT
allm=[o for o in bpy.context.scene.objects if o.type=='MESH' and not o.hide_get()]
ms=[o for o in allm if not is_struct(o.name)]
print("SCANNED",len(ms),[o.name for o in ms])
print("EXCLUDED",[o.name for o in allm if is_struct(o.name)])   # eyeball: no real fixture here
# derive the wall-poke band from THIS scene's Floor (fallback: the union of Wall_* AABBs):
flr=bpy.data.objects.get('Floor')
if flr:
    (fx0,fy0,_),(fx1,fy1,_)=aabb(flr); WALLS_X=(fx0,fx1); WALLS_Y=(fy0,fy1)
else:
    walls=[o for o in allm if o.name.split('_')[0]=='Wall']
    bs=[aabb(o) for o in walls]
    WALLS_X=(min(b[0][0] for b in bs),max(b[1][0] for b in bs))
    WALLS_Y=(min(b[0][1] for b in bs),max(b[1][1] for b in bs))
print("wall-poke band: X",[round(v,2) for v in WALLS_X],"Y",[round(v,2) for v in WALLS_Y])
B={o.name:aabb(o) for o in ms}
for i in range(len(ms)):
    for j in range(i+1,len(ms)):
        a,b=ms[i].name,ms[j].name
        ga,gb=set_of(a),set_of(b)
        if ga is not None and ga is gb: continue   # both in one declared set — see connected_scan
        lo_a,hi_a=B[a]; lo_b,hi_b=B[b]
        ov=[min(hi_a[k],hi_b[k])-max(lo_a[k],lo_b[k]) for k in range(3)]   # per-axis overlap
        pen=min(ov)                              # interpenetration depth (positive = into volume)
        if pen>TOL_PEN: print("穿模:",a,b,"depth",round(pen,3),[round(v,3) for v in ov])
for o in ms:                                     # wall-poke: AABB crosses a wall's inner face
    (x0,y0,z0),(x1,y1,z1)=B[o.name]
    if x0<WALLS_X[0]-TOL_PEN or x1>WALLS_X[1]+TOL_PEN or \
       y0<WALLS_Y[0]-TOL_PEN or y1>WALLS_Y[1]+TOL_PEN:
        print("WALL-POKE:",o.name)
```

If any fixture name starts with a STRUCT token it is silently skipped — rename it or drop the
token from STRUCT; the EXCLUDED print is there to catch exactly that. A child poking THROUGH its
own body (chaise into sofa body) is still 穿模 even within one declared set — measure those
sub-part relations directly after a move.

---

## connected_scan — declared-set connectivity + gap scan (run after EVERY edit)

Automated detach check. You DECLARE which object-name prefixes must touch (a fixture's parts, an
enclosure/frame's pieces, two walls at a corner) and, optionally, a wall an object must be flush
to (wall-normal axis only). For each set it builds an adjacency edge wherever two members' gap
≤ `TOL_GAP`, then asserts the whole set is ONE connected component (a frame split into two
internally-touching halves prints SET SPLIT — a per-member nearness test would miss this). It also
prints a RESIDUAL line for any member sitting at a sub-tolerance gap so the fix target (≈ 0) is
enforced, not silently permitted. Call `view_layer.update()` first; apply Mirror/Solidify
modifiers (or read the evaluated AABB) — `bound_box` ignores unapplied modifiers.

```python
import bpy, mathutils
TOL_GAP = 0.02                       # 2cm to DECIDE detached; loosen to 0.05 for hand-built frames
def aabb(o):
    p=[o.matrix_world@mathutils.Vector(c) for c in o.bound_box]   # apply modifiers first
    return [min(q[i] for q in p) for i in range(3)],[max(q[i] for q in p) for i in range(3)]
def gap(A,B,axes=(0,1,2)):           # euclidean of per-axis POSITIVE separations; 0 => touching
    (la,ha),(lb,hb)=A,B
    seps=[max(la[k]-hb[k], lb[k]-ha[k], 0.0) for k in axes]
    return sum(s*s for s in seps)**0.5
bpy.context.view_layer.update()
def members(pfx): return [o for o in bpy.context.scene.objects
                          if o.type=='MESH' and not o.hide_get() and o.name.startswith(pfx)]
# DECLARE this scene's units (prefixes whose members must form one connected component):
SETS = []   # e.g. [('Vanity',), ('Shower',), ('Water_Heater','Water_Heater_Panel')]
for grp in SETS:
    objs=[o for p in grp for o in members(p)]
    if not objs: print("EMPTY SET",grp,"— nothing grounded; check prefixes"); continue
    print("SET",grp,"scanned",[o.name for o in objs])
    B={o.name:aabb(o) for o in objs}
    # connected-components over edges where gap <= TOL_GAP (note: boundary inclusive):
    comp={o.name:i for i,o in enumerate(objs)}
    for a in objs:
        for b in objs:
            if a is not b and gap(B[a.name],B[b.name])<=TOL_GAP:
                r=comp[b.name]; o=comp[a.name]
                for k in comp:
                    if comp[k]==r: comp[k]=o
    roots=set(comp.values())
    if len(roots)>1:
        for r in roots: print("SET SPLIT",grp,"component",[k for k in comp if comp[k]==r])
    for o in objs:                                   # surface residual sub-tolerance gaps
        nn=min((gap(B[o.name],B[s.name]) for s in objs if s is not o), default=0.0)
        if nn>TOL_GAP:    print("GAP:",o.name,"nearest-sibling",round(nn,3))
        elif nn>0.0:      print("RESIDUAL:",o.name,"gap",round(nn,3),"— extend to ~0")
# flush-against-wall: only the wall-normal axis must touch (axis index 0=X,1=Y)
FLUSH = []   # e.g. [('Vanity', 'Wall_Back', 1)]  -> Vanity must touch Wall_Back on Y
for opfx,wall,nax in FLUSH:
    w=bpy.data.objects.get(wall)
    if not w: print("MISSING WALL",wall); continue
    for o in members(opfx):
        g=gap(aabb(o),aabb(w),axes=(nax,))
        if g>TOL_GAP:  print("DETACHED:",o.name,"from",wall,"axis",nax,round(g,3))
        elif g>0.0:    print("RESIDUAL:",o.name,"from",wall,"gap",round(g,3),"— extend to ~0")
```

SET SPLIT / GAP / RESIDUAL all mean re-seat to gap ≈ 0 (or −0.5cm overlap). After deepening a
room, add the perpendicular walls' corner pairs to `SETS`; after a shell-scale, add every
wall-mounted fixture↔wall to `FLUSH` — the scale moved the wall but not the fixture.
