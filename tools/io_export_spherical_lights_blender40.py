import bpy
import os
from struct import pack

# To define light sources, create a mesh for the unit sphere named
# "spherical_light" in Blender. Give it a material called _emission. Then place
# suitably scaled instances of this mesh in your scene wherever you want a
# spherical light. All of them have equal brightness. Run this script to create
# the *.lights file and io_export_vulkan_blender28.py to export a *.vks file
# that includes the geometry for the spherical lights.

# Get the directory where your .blend file is saved
# Note: You must save your .blend file first for this to work!

# Ensure the filename matches your renderer's expectations
FILENAME = "testscene.lights"

blend_dir = os.path.dirname(bpy.data.filepath)
output_path = os.path.join(blend_dir, FILENAME)

# Find all objects using the "spherical_light" mesh data
lights = [obj for obj in bpy.data.objects 
          if obj.data and obj.data.name == "spherical_light"]

if not blend_dir:
    print("Error: Save your .blend file first!")
else:
    with open(output_path, "wb") as file:
        # Write count (Unsigned Int)
        file.write(pack("I", len(lights)))
        
        for obj in lights:
            pos = obj.matrix_world.translation
            # We must write 4 floats to match the renderer's fread(..., sizeof(float) * 4, ...)
            radius = obj.dimensions.x / 2.0
            file.write(pack("ffff", pos.x, pos.y, pos.z, radius))
    
    print(f"Successfully exported {len(lights)} lights to: {output_path}")