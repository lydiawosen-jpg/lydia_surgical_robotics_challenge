# This should be run in blender's python console, not in a separate python environment. It will export to a .json file the control 
# points world coordinates of the wire using backup_curve in blender and pair them in segments in a list. Then Calculating_ring_wire_error.py script
# will read the .json file.

import bpy
import json
import os

def export_blender_curve_to_json():
    try:
        curve_obj = bpy.data.objects['backup_curve']
    except KeyError:
        print("Error: Could not find 'backup_curve'.")
        return

    spline = curve_obj.data.splines[0] # grabs the first and only spline in the curve object
    blender_points = spline.bezier_points
    
    
    my_total_curve = []

    # The loop stops at len-1 so i+1 never goes out of bounds
    for i in range(len(blender_points) - 1):
        p_start = blender_points[i]
        p_end = blender_points[i+1]
        
        
        p0_world = p_start.co
        p1_world = p_start.handle_right
        p2_world = p_end.handle_left
        p3_world = p_end.co
        
        # Convert the math vectors to standard JSON-friendly lists
        segment = [
            list(p0_world), 
            list(p1_world),
            list(p2_world),
            list(p3_world)  
        ]
        
        my_total_curve.append(segment)

    output_path = os.path.expanduser("~/my_bezier_curve.json")

    with open(output_path, 'w') as f:
        json.dump(my_total_curve, f, indent=4)

    print(f"SUCCESS: Exported {len(my_total_curve)} world-space segments to: {output_path}")

export_blender_curve_to_json()