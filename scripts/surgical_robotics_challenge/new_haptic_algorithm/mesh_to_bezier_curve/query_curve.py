import argparse
import math
import sys

from find_mesh_centerline import basis_values, extract_bspline_from_obj


def squared_distance(first, second):
    return sum((first[axis] - second[axis]) ** 2 for axis in range(3))


def evaluate_bspline(control_points, knots, degree, parameter):
    weights = basis_values(parameter, degree, knots, len(control_points))
    return tuple(
        sum(weight * point[axis] for weight, point in zip(weights, control_points))
        for axis in range(3)
    )


def closest_curve_parameter(point, control_points, knots, degree, samples=2048):
    def objective(parameter):
        return squared_distance(point, evaluate_bspline(control_points, knots, degree, parameter))

    grid = [index / samples for index in range(samples + 1)]
    candidates = [(0.0, 0.0), (1.0, 1.0)]
    for index in range(1, samples):
        current = objective(grid[index])
        if current <= objective(grid[index - 1]) and current <= objective(grid[index + 1]):
            candidates.append((grid[index - 1], grid[index + 1]))

    best_parameter = 0.0
    best_distance = math.inf
    for left, right in candidates:
        for _ in range(48):
            first = left + (right - left) / 3
            second = right - (right - left) / 3
            if objective(first) <= objective(second):
                right = second
            else:
                left = first
        parameter = (left + right) / 2
        current_distance = objective(parameter)
        if current_distance < best_distance:
            best_parameter, best_distance = parameter, current_distance
    return best_parameter


def main():
    parser = argparse.ArgumentParser(
        description="Find the closest B-spline traversal position and control points for a 3D query point."
    )
    parser.add_argument("x", type=float, help="Query point X coordinate.")
    parser.add_argument("y", type=float, help="Query point Y coordinate.")
    parser.add_argument("z", type=float, help="Query point Z coordinate.")
    parser.add_argument("mesh_path", nargs="?", default="mesh/wire_visual.OBJ", help="Path to the input OBJ mesh.")
    parser.add_argument("-s", "--slices", type=int, default=60, help="Number of centerline samples (default: 60).")
    parser.add_argument("--smooth", type=float, default=0.0, help="Centerline simplification tolerance (default: 0.0).")
    args = parser.parse_args()

    try:
        control_points, knots, degree = extract_bspline_from_obj(
            args.mesh_path, num_slices=args.slices, smoothing=args.smooth
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    query_point = (args.x, args.y, args.z)
    parameter = closest_curve_parameter(query_point, control_points, knots, degree)
    nearest = sorted(
        enumerate(control_points),
        key=lambda indexed_point: squared_distance(query_point, indexed_point[1]),
    )[:2]

    print(f"t: {parameter:.9f}")
    for index, point in nearest:
        print(f"control_point[{index}]: {point[0]:.9f}, {point[1]:.9f}, {point[2]:.9f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
