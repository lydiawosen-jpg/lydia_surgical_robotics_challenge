import numpy as np
from scipy.optimize import minimize_scalar, OptimizeResult
import json
import os


# Uses bezier equation and predfined P values to know coordinates of that t value
def get_bezier_point(t, P0, P1, P2, P3):
    # This is the standard cubic Bezier formula applied to 3D vectors
    wire_xyz = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3
    return wire_xyz


# Subtracts the ring COM coordinates from that t value's coordinates to get distance, objective function f(t) = ||C(t) - P_ring||
def distance_objective(t, ring_com, P0, P1, P2, P3):
    wire_xyz = get_bezier_point(t, P0, P1, P2, P3)
    distance = np.linalg.norm(wire_xyz - ring_com)  # np.linalg.norm comutes magnitude from this vector subtraction
    return distance


# look in homedirectory to fine file, read the file and load it back into a standard Python list
json_path = os.path.expanduser("~/my_bezier_curve.json")

with open(json_path, 'r') as f:
    my_total_curve = json.load(f)

ring_com = np.array([0.077795, 0.165143, 0.710670])

# Initialize with an infinitely large distance
final_result = OptimizeResult(fun=float('inf'))
for segment in my_total_curve:
    # convert raw list sequence into numpy array and assign to P0, P1, P2, P3 for readability in the distance_objective function
    P0 = np.array(segment[0])
    P1 = np.array(segment[1])
    P2 = np.array(segment[2])
    P3 = np.array(segment[3])

    result = minimize_scalar(
        distance_objective,
        bounds=(0.0, 1.0),
        method='bounded',
        args=(ring_com, P0, P1, P2, P3)
    )
    if result.fun < final_result.fun: # this strategy is called bubble sort 
        final_result = result
        winning_segment_points = (P0, P1, P2, P3)

# present answers
closest_t = final_result.x
closest_wire_point = get_bezier_point(closest_t, *winning_segment_points)
min_distance = final_result.fun

print(f"The closest parameter is t = {closest_t:.6f}")
print(f"The exact 3D coordinates on the wire are: {closest_wire_point}")
print(f"The absolute shortest distance to the ring is: {min_distance} meters")










        

