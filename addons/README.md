# Blender Export BPY Code

A Blender addon that exports your scene as a reproducible Python (`bpy`) script. Run the exported script in any Blender instance to reconstruct the exact same scene.

![Blender 3.6+](https://img.shields.io/badge/Blender-3.6%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Why?

Blender can export to FBX, OBJ, glTF, USD — but none of these export as **Python code**. This addon fills that gap.

Use cases:

- **Benchmarking**: Compare AI-generated scene code against human-authored ground truth using the same `bpy` API representation
- **Reproducibility**: Share scenes as self-contained Python scripts, no `.blend` files needed
- **Education**: See exactly what `bpy` calls produce a given scene
- **Version control**: Track scene changes as readable Python diffs
- **Automation**: Generate base scripts to modify programmatically

## Installation

1. Use the `export_bpy_code.py` file in this `addons/` folder.
2. In Blender: `Edit → Preferences → Add-ons → Install...`
3. Select the downloaded file
4. Enable the checkbox next to "Export Scene as BPY Code"

## Usage

### Export

`File → Export → Blender Python Script (.py)`

### Import (run the exported script)

There are three ways to run an exported `.py` file:

**In Blender GUI:**
1. Switch to the **Scripting** workspace (tab at the top of the window, far right)
2. In the Text Editor, click **Open** (📂 icon) and select the `.py` file
3. Click **▶ Run Script** (or press `Alt+P`)

**From command line:**
```bash
blender --python my_scene.py
```

**Headless (no GUI, useful for batch benchmarking):**
```bash
blender --background --python my_scene.py
```

### Export options

| Option | Default | Description |
|--------|---------|-------------|
| Include Clear Scene | ✅ | Prepend code to wipe the scene before rebuilding |
| Include Camera | ✅ | Export camera objects |
| Include Lights | ✅ | Export light objects |
| Include Render Settings | ✅ | Export render engine, resolution, world environment |
| Skip Hidden Objects | ❌ | Don't export objects hidden in viewport (eye icon off) |
| Precision | 6 digits | Decimal places for coordinates (4 / 6 / 8) |

## What gets exported

| Element | How it's exported |
|---------|-------------------|
| Cubes (8 verts, 6 faces) | `bpy.ops.mesh.primitive_cube_add()` |
| Planes (4 verts, 1 face) | `bpy.ops.mesh.primitive_plane_add()` |
| Cylinders | `bpy.ops.mesh.primitive_cylinder_add()` — detected by geometry, with estimated side count, axis, depth, radius, and non-uniform cross-section scale |
| Cones | `bpy.ops.mesh.primitive_cone_add()` — same detection path as cylinders, with estimated side count and end radii |
| UV Spheres | `bpy.ops.mesh.primitive_uv_sphere_add()` — reconstructed as a scaled sphere/ellipsoid |
| Tori | `bpy.ops.mesh.primitive_torus_add()` — preserves approximate major/minor radius and non-uniform XY scale |
| Complex meshes | ⚠️ Cube fallback + warning (see below) |
| Materials | Solid diffuse colors and Principled BSDF scalar/color inputs, including base color, roughness, metallic, emission, alpha, blend mode |
| Material slots | All object material slots are appended; per-face material indices are not reconstructed for primitive fallbacks |
| Supported modifiers | Common lightweight procedural modifiers are recreated by parameter (see below) |
| Lights | Type, energy, color, size, spot angle |
| Cameras | Lens, clip range, sensor width |
| Empties | Display type and size |
| Parent relationships | `obj.parent = ...` |
| Object display state | Visibility, display type, show name, show in front, object color, shadow visibility |
| Render engine | EEVEE / Cycles + engine-specific settings |
| World environment | Background color and strength |

### Scale handling

The addon correctly handles objects where scale has been applied (`Ctrl+A → Apply Scale`). It reads the actual mesh vertex bounding box and multiplies by `obj.scale` to compute the correct export scale, so the reconstructed scene matches regardless of how the original was built.

### Export warnings

When exporting, the addon checks for issues and shows warnings in Blender's status bar:

**Complex meshes** — Any mesh that doesn't match a known primitive (cube, plane, cylinder, cone, sphere, torus) is approximated as a cube. The warning lists affected objects with their vertex/face counts.

**Linked materials** — If a material's Base Color uses textures, Color Ramps, or other nodes instead of a plain color value, only the fallback solid color is exported. The warning lists affected materials and what node type is connected.

**Unsupported or partial modifiers** — Supported lightweight modifiers are exported by parameter. Unsupported modifiers are skipped and listed in the warning.

> **Tip for benchmarking**: If you see warnings, replace arbitrary complex meshes with primitives or supported procedural modifiers, and use solid colors for materials before exporting GT scenes.

## Example output

```python
"""
Blender Scene: my_room.blend
Objects: 42
Materials: 12
Generated by Export BPY Code addon v1.3.0
"""

import bpy
import math

# Clear Scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for m in list(bpy.data.materials): bpy.data.materials.remove(m)
# ...

# Render Settings
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# World Environment
world = bpy.context.scene.world
# ...

# Materials
mat_Wall = bpy.data.materials.new(name='Wall')
mat_Wall.use_nodes = True
_bsdf = mat_Wall.node_tree.nodes['Principled BSDF']
_bsdf.inputs['Base Color'].default_value = (0.95, 0.92, 0.87, 1.0)
_bsdf.inputs['Roughness'].default_value = 0.9
_bsdf.inputs['Metallic'].default_value = 0.0

# Objects
bpy.ops.mesh.primitive_cube_add(size=1, location=(-2.5, -2.0, 0.4))
obj = bpy.context.active_object
obj.name = 'Wall_Front'
obj.rotation_euler = (0.0, 0.0, 0.0)
obj.scale = (6.0, 0.08, 2.6)
obj.data.materials.append(bpy.data.materials['Wall'])
```

## Limitations

- **Complex meshes**: Meshes that don't match a known primitive are approximated as cubes. A warning popup lists affected objects. This keeps the export as a compact single `.py` file instead of embedding large vertex/face arrays.
- **Textures / Image nodes**: Only solid Principled BSDF values are exported. Materials using image textures or procedural nodes fall back to the socket's default color. A warning lists affected materials.
- **Modifiers**: Common lightweight procedural modifiers are exported by parameter: `ARRAY`, `BEVEL`, `BOOLEAN`, `MIRROR`, `SCREW`, `SIMPLE_DEFORM`, `SOLIDIFY`, `SUBSURF`, `TRIANGULATE`, and `WEIGHTED_NORMAL`. Other modifiers are skipped with a warning. Applying modifiers into arbitrary dense meshes can reduce export fidelity because the result may fall back to a cube.
- **Node graphs**: Only the Principled BSDF node is read. Complex shader setups are simplified to base color, roughness, metallic, emission, and alpha.
- **Constraints & Drivers**: Not exported.
- **Shape keys / Armatures**: Not exported.
- **Animation keyframes**: Not exported. Only the current frame's transforms are used.
- **Collections**: Object hierarchy via parenting is exported, but Blender collections are not recreated.
- **Viewport shading preferences**: Material Preview studio light settings are per-workspace, not per-file, and cannot be exported. Use Rendered mode (`F12`) to compare scenes accurately.

## Contributing

PRs welcome! Some ideas:

- [ ] Export animation keyframes
- [ ] Expand modifier parameter coverage and round-trip tests
- [ ] Support Geometry Nodes parameters
- [ ] Better compact approximations for complex meshes without embedding full vertex/face arrays
- [ ] Export collections and their visibility states
- [ ] Round-trip validation tool (export → import → compare)

## License

MIT