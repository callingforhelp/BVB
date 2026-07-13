"""
Blender Scene: scene0686_00.blend
Rebuilt from VSI-Bench/scannet/scene0686_00.mp4 keyframes.

Coordinate anchor:
- Looking from the sink or toilet entrance toward the rear tiled wall, image-left is Blender -X and image-right is Blender +X.
- The sink alcove is on the left side of the restroom; the toilet stall is on the right side.
"""

import bpy
import math

# ============================================================
# Clear Scene
# ============================================================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for m in list(bpy.data.materials):
    bpy.data.materials.remove(m)
for m in list(bpy.data.meshes):
    bpy.data.meshes.remove(m)
for m in list(bpy.data.lights):
    bpy.data.lights.remove(m)
for m in list(bpy.data.cameras):
    bpy.data.cameras.remove(m)

# ============================================================
# Render Settings
# ============================================================
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.eevee.taa_render_samples = 64

world = bpy.context.scene.world or bpy.data.worlds.new('World')
bpy.context.scene.world = world
world.use_nodes = True
_bg = world.node_tree.nodes.get('Background')
if _bg:
    _bg.inputs['Color'].default_value = (0.78, 0.74, 0.64, 1.0)
    _bg.inputs['Strength'].default_value = 0.35

# ============================================================
# Helpers
# ============================================================
def make_mat(name, color, roughness=0.6, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Alpha'].default_value = alpha
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
    return mat

def cube(name, location, dimensions, mat_name, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(bpy.data.materials[mat_name])
    return obj

def cyl(name, location, radius, depth, mat_name, vertices=32, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(bpy.data.materials[mat_name])
    return obj

# ============================================================
# Materials
# ============================================================
make_mat('Mat_Wall_Tile', (0.66, 0.49, 0.22, 1.0), 0.68)
make_mat('Mat_Wall_Grout', (0.86, 0.76, 0.55, 1.0), 0.75)
make_mat('Mat_Floor_Tile', (0.78, 0.69, 0.46, 1.0), 0.72)
make_mat('Mat_Floor_Grout', (0.24, 0.21, 0.18, 1.0), 0.8)
make_mat('Mat_Partition', (0.82, 0.69, 0.32, 1.0), 0.55)
make_mat('Mat_Partition_Edge', (0.62, 0.49, 0.24, 1.0), 0.5)
make_mat('Mat_Ceramic', (0.96, 0.95, 0.9, 1.0), 0.22)
make_mat('Mat_Basin_Water_Shadow', (0.72, 0.78, 0.76, 1.0), 0.25)
make_mat('Mat_Metal', (0.62, 0.62, 0.58, 1.0), 0.18, 0.85)
make_mat('Mat_Mirror', (0.73, 0.78, 0.78, 0.55), 0.04, 0.5, 0.55)
make_mat('Mat_Dispenser_White', (0.9, 0.88, 0.82, 1.0), 0.42)
make_mat('Mat_Dispenser_Black', (0.03, 0.03, 0.03, 1.0), 0.5)
make_mat('Mat_Outlet', (0.55, 0.48, 0.38, 1.0), 0.45)
make_mat('Mat_Dark_Debris', (0.08, 0.04, 0.025, 1.0), 0.75)

# ============================================================
# Room Shell and Tile Grids
# ============================================================
# Floor footprint includes the sink alcove, toilet stall, and the foreground corridor seen near 40s.
cube('Floor_Tile_Base', (0.95, -2.15, -0.035), (5.7, 5.25, 0.07), 'Mat_Floor_Tile')

# Small square floor tiles: dark grout lines in the same orientation as the video.
for i, x in enumerate([round(-1.8 + n * 0.22, 2) for n in range(27)]):
    cube(f'Floor_Grout_X_{i:02d}', (x, -2.15, 0.006), (0.012, 5.25, 0.012), 'Mat_Floor_Grout')
for i, y in enumerate([round(-4.65 + n * 0.22, 2) for n in range(25)]):
    cube(f'Floor_Grout_Y_{i:02d}', (0.95, y, 0.008), (5.7, 0.012, 0.012), 'Mat_Floor_Grout')

# Continuous yellow tiled rear wall, plus alcove/stall side walls and foreground corridor corner.
cube('Wall_Back_Tiled', (0.95, 0.08, 1.25), (5.7, 0.12, 2.5), 'Mat_Wall_Tile')
cube('Wall_Sink_Left', (-1.9, -1.05, 1.25), (0.12, 2.25, 2.5), 'Mat_Wall_Tile')
cube('Wall_Sink_Right_Divider', (1.15, -1.05, 1.25), (0.12, 2.25, 2.5), 'Mat_Wall_Tile')
cube('Wall_Stall_Right', (3.8, -1.05, 1.25), (0.12, 2.25, 2.5), 'Mat_Wall_Tile')
cube('Wall_Corridor_Right', (3.8, -3.35, 1.15), (0.12, 2.45, 2.3), 'Mat_Wall_Tile')
cube('Column_Corridor_Corner', (-1.25, -3.15, 1.15), (0.42, 0.42, 2.3), 'Mat_Dispenser_White')
cube('Column_Yellow_Base', (-1.25, -3.15, 0.12), (0.58, 0.58, 0.24), 'Mat_Wall_Tile')

# Larger wall tiles and light grout, placed just off visible wall surfaces.
for i, z in enumerate([0.42, 0.78, 1.14, 1.5, 1.86, 2.22]):
    cube(f'Back_Wall_Grout_H_{i}', (0.95, -0.002, z), (5.72, 0.018, 0.012), 'Mat_Wall_Grout')
for i, x in enumerate([round(-1.55 + n * 0.36, 2) for n in range(15)]):
    cube(f'Back_Wall_Grout_V_{i:02d}', (x, -0.004, 1.25), (0.012, 0.018, 2.5), 'Mat_Wall_Grout')
for i, z in enumerate([0.42, 0.78, 1.14, 1.5, 1.86, 2.22]):
    cube(f'Sink_Left_Wall_Grout_H_{i}', (-1.834, -1.05, z), (0.018, 2.25, 0.012), 'Mat_Wall_Grout')
    cube(f'Sink_Right_Wall_Grout_H_{i}', (1.084, -1.05, z), (0.018, 2.25, 0.012), 'Mat_Wall_Grout')
    cube(f'Stall_Right_Wall_Grout_H_{i}', (3.734, -1.05, z), (0.018, 2.25, 0.012), 'Mat_Wall_Grout')
for i, y in enumerate([round(-2.0 + n * 0.36, 2) for n in range(6)]):
    cube(f'Sink_Left_Wall_Grout_V_{i}', (-1.832, y, 1.25), (0.018, 0.012, 2.5), 'Mat_Wall_Grout')
    cube(f'Sink_Right_Wall_Grout_V_{i}', (1.082, y, 1.25), (0.018, 0.012, 2.5), 'Mat_Wall_Grout')
    cube(f'Stall_Right_Wall_Grout_V_{i}', (3.732, y, 1.25), (0.018, 0.012, 2.5), 'Mat_Wall_Grout')

# ============================================================
# Sink Alcove: left side in the reference layout
# ============================================================
cube('Sink_Basin_Block', (-0.35, -0.35, 0.88), (1.05, 0.48, 0.18), 'Mat_Ceramic')
cube('Sink_Back_Lip', (-0.35, -0.09, 0.99), (1.05, 0.11, 0.22), 'Mat_Ceramic')
cube('Sink_Bowl_Recess', (-0.35, -0.43, 0.99), (0.56, 0.26, 0.035), 'Mat_Basin_Water_Shadow')
cyl('Sink_Drain', (-0.35, -0.43, 1.015), 0.035, 0.01, 'Mat_Metal')
cyl('Sink_PTrap_Vertical', (-0.35, -0.16, 0.58), 0.045, 0.38, 'Mat_Metal')
cyl('Sink_PTrap_Horizontal', (-0.35, -0.08, 0.52), 0.035, 0.34, 'Mat_Metal', rotation=(math.radians(90), 0.0, 0.0))

cyl('Faucet_Spout_Base', (-0.35, -0.2, 1.12), 0.045, 0.08, 'Mat_Metal')
cube('Faucet_Spout_Neck', (-0.35, -0.31, 1.16), (0.06, 0.22, 0.04), 'Mat_Metal')
cyl('Faucet_Left_Handle', (-0.6, -0.18, 1.11), 0.04, 0.05, 'Mat_Metal')
cyl('Faucet_Right_Handle', (-0.1, -0.18, 1.11), 0.04, 0.05, 'Mat_Metal')
cube('Sink_Access_Panel', (-0.35, -0.006, 0.42), (0.42, 0.018, 0.34), 'Mat_Partition_Edge')

cube('Mirror_Above_Sink', (-0.35, -0.012, 1.75), (1.45, 0.018, 0.62), 'Mat_Mirror')
cube('Mirror_Ledge_Tissue_Box', (0.15, -0.06, 1.42), (0.42, 0.18, 0.12), 'Mat_Dispenser_White')
cube('White_Dispenser_Left_Wall', (-1.835, -0.42, 1.65), (0.08, 0.24, 0.38), 'Mat_Dispenser_White')
cube('Black_Paper_Towel_Right_Wall', (1.085, -0.72, 1.74), (0.08, 0.46, 0.32), 'Mat_Dispenser_Black')
cube('Outlet_Right_Of_Sink', (1.08, -0.9, 1.1), (0.018, 0.16, 0.22), 'Mat_Outlet')

# ============================================================
# Toilet Stall: right side, with right-wall dispenser and open left door
# ============================================================
cube('Stall_Left_Partition', (1.5, -1.05, 1.08), (0.09, 2.2, 2.05), 'Mat_Partition')
cube('Stall_Right_Partition', (3.35, -1.05, 1.08), (0.09, 2.2, 2.05), 'Mat_Partition')
cube('Stall_Front_Right_Post', (3.35, -2.05, 1.08), (0.12, 0.16, 2.05), 'Mat_Partition_Edge')
cube('Stall_Front_Left_Post', (1.5, -2.05, 1.08), (0.12, 0.16, 2.05), 'Mat_Partition_Edge')
cube('Stall_Open_Door_Left', (1.05, -1.7, 1.05), (0.08, 1.05, 2.0), 'Mat_Partition', rotation=(0.0, 0.0, math.radians(-8)))
cube('Stall_Door_Latch_Right', (3.285, -1.76, 1.24), (0.06, 0.2, 0.08), 'Mat_Metal')

cyl('Toilet_Bowl_Round', (2.42, -0.5, 0.42), 0.36, 0.22, 'Mat_Ceramic', vertices=48)
cube('Toilet_Bowl_Front_Cutout', (2.42, -0.78, 0.47), (0.28, 0.18, 0.08), 'Mat_Basin_Water_Shadow')
cube('Toilet_Wall_Mount_Base', (2.42, -0.08, 0.42), (0.58, 0.18, 0.36), 'Mat_Ceramic')
cube('Toilet_Open_Seat_Left', (2.24, -0.52, 0.65), (0.12, 0.52, 0.05), 'Mat_Ceramic', rotation=(0.0, math.radians(-18), 0.0))
cube('Toilet_Open_Seat_Right', (2.6, -0.52, 0.65), (0.12, 0.52, 0.05), 'Mat_Ceramic', rotation=(0.0, math.radians(18), 0.0))
cube('Toilet_Flush_Valve_Box', (2.42, -0.006, 1.06), (0.26, 0.018, 0.18), 'Mat_Metal')
cyl('Toilet_Flush_Pipe', (2.42, -0.08, 0.82), 0.025, 0.45, 'Mat_Metal')
cube('Toilet_Back_Access_Panel', (2.02, -0.004, 0.88), (0.38, 0.018, 0.3), 'Mat_Partition_Edge')
cube('Toilet_Paper_Dispenser_Right_Wall', (3.29, -0.68, 1.18), (0.12, 0.42, 0.38), 'Mat_Dispenser_White')
cube('Toilet_Paper_Black_Cover', (3.22, -0.68, 1.35), (0.08, 0.42, 0.14), 'Mat_Dispenser_Black')

# ============================================================
# Foreground Corridor Details
# ============================================================
cyl('Round_Floor_Drain', (2.65, -3.55, 0.025), 0.16, 0.018, 'Mat_Metal', vertices=40)
for i, off in enumerate([-0.08, -0.04, 0.0, 0.04, 0.08]):
    cube(f'Round_Floor_Drain_Slot_{i}', (2.65 + off, -3.55, 0.045), (0.012, 0.22, 0.01), 'Mat_Floor_Grout')
cube('Small_Dark_Object_On_Floor', (-0.95, -3.75, 0.045), (0.34, 0.08, 0.035), 'Mat_Dark_Debris', rotation=(0.0, 0.0, math.radians(18)))

# ============================================================
# Lighting and Cameras
# ============================================================
bpy.ops.object.light_add(type='AREA', location=(0.2, -1.2, 2.45))
obj = bpy.context.active_object
obj.name = 'Light_Sink_Area'
obj.data.energy = 700.0
obj.data.size = 1.4

bpy.ops.object.light_add(type='AREA', location=(2.5, -1.2, 2.45))
obj = bpy.context.active_object
obj.name = 'Light_Stall_Area'
obj.data.energy = 620.0
obj.data.size = 1.2

bpy.ops.object.light_add(type='AREA', location=(1.0, -3.35, 2.35))
obj = bpy.context.active_object
obj.name = 'Light_Corridor_Area'
obj.data.energy = 380.0
obj.data.size = 1.6

# Main camera matches the sink-alcove reference: facing +Y, image-left = -X, image-right = +X.
bpy.ops.object.camera_add(location=(-0.35, -2.9, 1.45), rotation=(math.radians(72), 0.0, 0.0))
cam = bpy.context.active_object
cam.name = 'Camera_Main_Sink_View'
cam.data.lens = 20.0
bpy.context.scene.camera = cam

# Validation cameras for the other video viewpoints.
bpy.ops.object.camera_add(location=(2.42, -2.85, 1.35), rotation=(math.radians(72), 0.0, 0.0))
cam = bpy.context.active_object
cam.name = 'Camera_Stall_View'
cam.data.lens = 22.0

bpy.ops.object.camera_add(location=(0.4, -4.25, 1.25), rotation=(math.radians(62), 0.0, math.radians(-18)))
cam = bpy.context.active_object
cam.name = 'Camera_Corridor_View'
cam.data.lens = 22.0

# ============================================================
# Collections
# ============================================================
# Keep object names explicit for BVB evaluation/export readability.
