import argparse
import math
import re
import sys
from pathlib import Path

from find_mesh_centerline import (
    distance,
    extract_centerline,
    load_obj_mesh,
    resample_polyline,
    simplify_polyline,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ADF = REPO_ROOT / "ADF/Phantoms/ring_wire_env/default.yaml"
DEFAULT_MESH = REPO_ROOT / "ADF/Phantoms/ring_wire_env/high_res/wire_visual.OBJ"
RADIUS_ANCHOR = "wire_capsule_radius"


def point_segment_distance(point, first, second):
    direction = [second[axis] - first[axis] for axis in range(3)]
    length_squared = sum(component * component for component in direction)
    fraction = 0.0
    if length_squared:
        fraction = sum((point[axis] - first[axis]) * direction[axis] for axis in range(3)) / length_squared
        fraction = max(0.0, min(1.0, fraction))
    return distance(point, [first[axis] + fraction * direction[axis] for axis in range(3)])


def estimate_radius(vertices, centerline):
    segments = list(zip(centerline, centerline[1:]))
    distances = [min(point_segment_distance(vertex, a, b) for a, b in segments) for vertex in vertices]
    return sum(distances) / len(distances)


def z_axis_to_rpy(direction):
    """Return AMBF (r, p, y), with R = Rz(y) Ry(p) Rx(r), mapping +Z onto the unit vector."""
    ux, uy, uz = direction
    kx, ky = -uy, ux  # k = z cross u
    s_squared = kx * kx + ky * ky
    if s_squared < 1e-18:
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] if uz > 0 else [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    else:
        skew = [[0, 0, ky], [0, 0, -kx], [-ky, kx, 0]]
        factor = (1 - uz) / s_squared
        matrix = [
            [
                (1 if i == j else 0)
                + skew[i][j]
                + factor * sum(skew[i][m] * skew[m][j] for m in range(3))
                for j in range(3)
            ]
            for i in range(3)
        ]
    pitch = -math.asin(max(-1.0, min(1.0, matrix[2][0])))
    if abs(math.cos(pitch)) < 1e-9:
        return 0.0, pitch, math.atan2(-matrix[0][1], matrix[1][1])
    return (
        math.atan2(matrix[2][1], matrix[2][2]),
        pitch,
        math.atan2(matrix[1][0], matrix[0][0]),
    )


def build_capsules(points, radius_text, radius):
    """AMBF capsule height includes both end caps, so add 2 * radius to the segment length."""
    lines = ["  compound collision shape:"]
    for index, (first, second) in enumerate(zip(points, points[1:])):
        length = distance(first, second)
        if length == 0:
            continue
        unit = [(second[axis] - first[axis]) / length for axis in range(3)]
        middle = [(first[axis] + second[axis]) / 2 for axis in range(3)]
        roll, pitch, yaw = z_axis_to_rpy(unit)
        lines += [
            f"  - name: capsule_{index}",
            "    shape: CAPSULE",
            "    geometry:",
            "      axis: Z",
            f"      height: {length + 2 * radius:.6f}",
            f"      radius: {radius_text}",
            "    offset:",
            "      position:",
            f"        x: {middle[0]:.6f}",
            f"        y: {middle[1]:.6f}",
            f"        z: {middle[2]:.6f}",
            "      orientation:",
            f"        r: {roll:.6f}",
            f"        p: {pitch:.6f}",
            f"        y: {yaw:.6f}",
        ]
    return lines


def update_adf(adf_path, body_name, capsule_lines):
    lines = adf_path.read_text(encoding="utf-8").splitlines()
    header = re.compile(rf"^BODY {re.escape(body_name)}:\s*$")
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        raise ValueError(f"'BODY {body_name}' not found in {adf_path}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace() and lines[i][0] != "#"), len(lines))

    body = lines[start + 1 : end]
    existing = next((i for i, line in enumerate(body) if line.startswith("  compound collision shape:")), None)
    if existing is not None:
        stop = existing + 1
        while stop < len(body) and (body[stop].startswith("  - ") or body[stop].startswith("    ")):
            stop += 1
        del body[existing:stop]
        insert_at = existing
    else:
        insert_at = len(body)
        while insert_at > 0 and not body[insert_at - 1].strip():
            insert_at -= 1
    body[insert_at:insert_at] = capsule_lines

    lines[start + 1 : end] = body
    adf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_radius_anchor(adf_path):
    match = re.search(rf"^.*&{RADIUS_ANCHOR}\s+([0-9.eE+-]+)\s*$", adf_path.read_text(encoding="utf-8"), re.MULTILINE)
    return float(match.group(1)) if match else None


def set_radius_anchor(adf_path, radius):
    text = adf_path.read_text(encoding="utf-8")
    text = re.sub(rf"(&{RADIUS_ANCHOR}\s+)[0-9.eE+-]+", lambda m: f"{m.group(1)}{radius}", text, count=1)
    adf_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Approximate a tube mesh with a compound of capsule collision shapes and write it into an ADF body."
    )
    parser.add_argument("mesh_path", nargs="?", default=str(DEFAULT_MESH), help="Path to the wire OBJ mesh.")
    parser.add_argument("-a", "--adf", default=str(DEFAULT_ADF), help="ADF file to update.")
    parser.add_argument("-b", "--body", default="wire_visual", help="Body name in the ADF (default: wire_visual).")
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        help=f"Capsule radius. Defaults to the '&{RADIUS_ANCHOR}' anchor in the ADF, else the radius estimated from the mesh.",
    )
    parser.add_argument("-t", "--tolerance", type=float, default=0.002, help="Max centerline deviation per capsule (default: 0.002).")
    parser.add_argument("-s", "--slices", type=int, default=400, help="Centerline samples before simplifying (default: 400).")
    args = parser.parse_args()

    adf_path = Path(args.adf)
    try:
        vertices, faces = load_obj_mesh(args.mesh_path)
        centerline = extract_centerline(vertices, faces)
        points = simplify_polyline(resample_polyline(centerline, args.slices), args.tolerance)

        anchor_radius = read_radius_anchor(adf_path)
        if args.radius is not None:
            radius = args.radius
            if anchor_radius is not None:
                set_radius_anchor(adf_path, radius)
        else:
            radius = anchor_radius if anchor_radius is not None else round(estimate_radius(vertices, centerline), 4)
        radius_text = f"*{RADIUS_ANCHOR}" if anchor_radius is not None else str(radius)

        update_adf(adf_path, args.body, build_capsules(points, radius_text, radius))
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(points) - 1} capsules (radius {radius}) to '{args.body}' in {adf_path}")
