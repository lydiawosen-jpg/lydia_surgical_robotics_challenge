import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_obj_mesh(obj_path):
    vertices = []
    faces = []

    try:
        lines = Path(obj_path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read mesh file: {error}") from error

    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            try:
                vertices.append(tuple(float(value) for value in fields[1:4]))
            except ValueError as error:
                raise ValueError(f"invalid vertex record: {line}") from error
        elif fields[0] == "f" and len(fields) >= 4:
            face = []
            for field in fields[1:]:
                index_text = field.split("/", 1)[0]
                try:
                    index = int(index_text)
                except ValueError as error:
                    raise ValueError(f"invalid face record: {line}") from error
                face.append(index - 1 if index > 0 else len(vertices) + index)
            if len(face) != 4:
                raise ValueError("only quad-faced tube meshes are supported")
            if any(index < 0 or index >= len(vertices) for index in face):
                raise ValueError(f"face references a missing vertex: {line}")
            faces.append(face)

    if not vertices:
        raise ValueError("mesh contains no vertices")
    if not faces:
        raise ValueError("mesh contains no faces")
    return vertices, faces


def edge_key(first, second):
    return (first, second) if first < second else (second, first)


def ordered_boundary_loop(boundary_adjacency):
    if not boundary_adjacency or any(len(neighbors) != 2 for neighbors in boundary_adjacency.values()):
        raise ValueError("mesh does not have a closed open-tube boundary")

    start = min(boundary_adjacency)
    loop = [start]
    previous = start
    current = boundary_adjacency[start][0]
    while current != start:
        if current in loop:
            raise ValueError("mesh boundary is not a single simple loop")
        loop.append(current)
        next_vertices = [vertex for vertex in boundary_adjacency[current] if vertex != previous]
        if len(next_vertices) != 1:
            raise ValueError("mesh boundary is not a single simple loop")
        previous, current = current, next_vertices[0]
    return loop


def opposite_vertices(face, first, second):
    first_index = face.index(first)
    second_index = face.index(second)
    if face[(first_index + 1) % 4] == second:
        direction = 1
    elif face[(first_index - 1) % 4] == second:
        direction = -1
    else:
        raise ValueError("mesh contains a malformed quad")
    return face[(first_index - direction) % 4], face[(second_index + direction) % 4]


def extract_centerline(vertices, faces):
    edge_faces = defaultdict(list)
    for face in faces:
        for index, vertex in enumerate(face):
            edge_faces[edge_key(vertex, face[(index + 1) % 4])].append(face)

    boundary_adjacency = defaultdict(list)
    for (first, second), adjacent_faces in edge_faces.items():
        if len(adjacent_faces) == 1:
            boundary_adjacency[first].append(second)
            boundary_adjacency[second].append(first)
        elif len(adjacent_faces) != 2:
            raise ValueError("mesh is not a two-manifold tube")

    ring = ordered_boundary_loop(boundary_adjacency)
    rings = []
    previous_ring = set()

    while True:
        rings.append(ring)
        current_ring = set(ring)
        next_vertices = {}

        for index, first in enumerate(ring):
            second = ring[(index + 1) % len(ring)]
            candidates = []
            for face in edge_faces[edge_key(first, second)]:
                first_next, second_next = opposite_vertices(face, first, second)
                if (
                    first_next not in current_ring
                    and second_next not in current_ring
                    and first_next not in previous_ring
                    and second_next not in previous_ring
                ):
                    candidates.append((first_next, second_next))

            if len(candidates) != 1:
                break
            first_next, second_next = candidates[0]
            for vertex, next_vertex in ((first, first_next), (second, second_next)):
                if vertex in next_vertices and next_vertices[vertex] != next_vertex:
                    raise ValueError("mesh does not contain consistently connected tube rings")
                next_vertices[vertex] = next_vertex
        else:
            if len(next_vertices) != len(ring) or len(set(next_vertices.values())) != len(ring):
                raise ValueError("mesh does not contain consistently connected tube rings")
            previous_ring, ring = current_ring, [next_vertices[vertex] for vertex in ring]
            continue
        break

    if len(rings) < 2:
        raise ValueError("could not trace enough cross-section rings")
    return [
        tuple(sum(vertices[vertex][axis] for vertex in ring) / len(ring) for axis in range(3))
        for ring in rings
    ]


def distance(first, second):
    return math.sqrt(sum((first[axis] - second[axis]) ** 2 for axis in range(3)))


def resample_polyline(points, count):
    if count < 2:
        raise ValueError("--slices must be at least 2")

    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + distance(first, second))
    if cumulative[-1] == 0:
        raise ValueError("centerline has zero length")

    result = []
    segment = 0
    for sample in range(count):
        target = cumulative[-1] * sample / (count - 1)
        while segment < len(points) - 2 and cumulative[segment + 1] < target:
            segment += 1
        span = cumulative[segment + 1] - cumulative[segment]
        fraction = 0.0 if span == 0 else (target - cumulative[segment]) / span
        result.append(
            tuple(
                points[segment][axis] + fraction * (points[segment + 1][axis] - points[segment][axis])
                for axis in range(3)
            )
        )
    return result


def point_line_distance(point, first, second):
    direction = tuple(second[axis] - first[axis] for axis in range(3))
    length_squared = sum(component * component for component in direction)
    if length_squared == 0:
        return distance(point, first)
    projection = sum((point[axis] - first[axis]) * direction[axis] for axis in range(3)) / length_squared
    closest = tuple(first[axis] + projection * direction[axis] for axis in range(3))
    return distance(point, closest)


def simplify_polyline(points, tolerance):
    if tolerance <= 0 or len(points) <= 2:
        return points

    greatest_distance = 0.0
    split_index = 0
    for index in range(1, len(points) - 1):
        current_distance = point_line_distance(points[index], points[0], points[-1])
        if current_distance > greatest_distance:
            greatest_distance = current_distance
            split_index = index
    if greatest_distance <= tolerance:
        return [points[0], points[-1]]
    return simplify_polyline(points[: split_index + 1], tolerance)[:-1] + simplify_polyline(points[split_index:], tolerance)


def basis_values(parameter, degree, knots, count):
    values = [0.0] * count
    if parameter == 1.0:
        values[-1] = 1.0
        return values
    for index in range(count):
        values[index] = 1.0 if knots[index] <= parameter < knots[index + 1] else 0.0
    for current_degree in range(1, degree + 1):
        previous = values
        values = [0.0] * count
        for index in range(count):
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = knots[index + current_degree + 1] - knots[index + 1]
            if left_denominator:
                values[index] += (parameter - knots[index]) * previous[index] / left_denominator
            if right_denominator and index + 1 < count:
                values[index] += (knots[index + current_degree + 1] - parameter) * previous[index + 1] / right_denominator
    return values


def solve_linear_system(coefficients, values):
    size = len(values)
    augmented = [row[:] + [values[index]] for index, row in enumerate(coefficients)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("could not fit a spline to the centerline")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[-1] for row in augmented]


def interpolation_parameters(points):
    lengths = [0.0]
    for first, second in zip(points, points[1:]):
        lengths.append(lengths[-1] + distance(first, second))
    if lengths[-1] == 0:
        raise ValueError("centerline has zero length")
    return [length / lengths[-1] for length in lengths]


def interpolation_knots(parameters, degree):
    count = len(parameters)
    return (
        [0.0] * (degree + 1)
        + [sum(parameters[index : index + degree]) / degree for index in range(1, count - degree)]
        + [1.0] * (degree + 1)
    )


def interpolate_bspline(points, degree):
    if len(points) < degree + 1:
        raise ValueError(f"at least {degree + 1} centerline points are required")

    parameters = interpolation_parameters(points)
    count = len(points)
    knots = interpolation_knots(parameters, degree)
    coefficients = [basis_values(parameter, degree, knots, count) for parameter in parameters]
    return [
        tuple(solve_linear_system(coefficients, [point[axis] for point in points])[index] for axis in range(3))
        for index in range(count)
    ]


def extract_bspline_from_obj(obj_path, num_slices=50, degree=3, smoothing=0.0):
    """Trace an open quad tube and return its controls, knots, and spline degree."""
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if smoothing < 0:
        raise ValueError("--smooth must not be negative")

    vertices, faces = load_obj_mesh(obj_path)
    centerline = resample_polyline(extract_centerline(vertices, faces), num_slices)
    centerline = simplify_polyline(centerline, smoothing)
    return (
        interpolate_bspline(centerline, degree),
        interpolation_knots(interpolation_parameters(centerline), degree),
        degree,
    )


def extract_control_points_from_obj(obj_path, num_slices=50, degree=3, smoothing=0.0):
    """Trace an open quad tube's centerline and return interpolating B-spline controls."""
    control_points, _, _ = extract_bspline_from_obj(obj_path, num_slices, degree, smoothing)
    return control_points


def visualize_centerline(vertices, faces, control_points, knots, degree):
    """Display the source mesh and the B-spline centerline in an interactive 3D plot."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as error:
        raise ValueError(
            "visualization requires matplotlib; install it with 'python3 -m pip install matplotlib'"
        ) from error

    curve = [
        tuple(
            sum(
                weight * control_point[axis]
                for weight, control_point in zip(
                    basis_values(parameter, degree, knots, len(control_points)), control_points
                )
            )
            for axis in range(3)
        )
        for parameter in (index / 300 for index in range(301))
    ]
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.add_collection3d(
        Poly3DCollection(
            [[vertices[index] for index in face] for face in faces],
            alpha=0.2,
            facecolor="steelblue",
            edgecolor="gray",
            linewidth=0.2,
        )
    )
    axes.plot(*zip(*curve), color="crimson", linewidth=2.5, label="B-spline centerline")
    axes.scatter(*zip(*control_points), color="darkorange", s=12, label="Control points")
    axes.set(xlabel="X", ylabel="Y", zlabel="Z", title="Mesh centerline")
    axes.legend()

    minimums = [min(point[axis] for point in vertices) for axis in range(3)]
    maximums = [max(point[axis] for point in vertices) for axis in range(3)]
    center = [(minimum + maximum) / 2 for minimum, maximum in zip(minimums, maximums)]
    radius = max(maximum - minimum for minimum, maximum in zip(minimums, maximums)) / 2
    for axis, value in enumerate(center):
        getattr(axes, f"set_{'xyz'[axis]}lim")(value - radius, value + radius)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract B-spline control points from an open quad tube OBJ mesh.")
    parser.add_argument("mesh_path", nargs="?", default="mesh/wire_visual.OBJ", help="Path to the input OBJ mesh.")
    parser.add_argument("-s", "--slices", type=int, default=60, help="Number of centerline samples (default: 60).")
    parser.add_argument("-o", "--output", help="Path to save the control points as CSV.")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Maximum centerline deviation when simplifying before fitting (default: 0.0).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open an interactive 3D view of the mesh, fitted centerline, and control points.",
    )
    args = parser.parse_args()

    try:
        control_points, knots, degree = extract_bspline_from_obj(
            args.mesh_path, num_slices=args.slices, smoothing=args.smooth
        )
        if args.visualize:
            vertices, faces = load_obj_mesh(args.mesh_path)
            visualize_centerline(vertices, faces, control_points, knots, degree)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully calculated {len(control_points)} control points:\n")
    for point in control_points:
        print(f"{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}")
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as output:
            for point in control_points:
                output.write(f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}\n")
        print(f"\nSaved control points to: {args.output}")
