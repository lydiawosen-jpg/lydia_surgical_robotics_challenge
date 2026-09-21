#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ambf_msgs.msg import RigidBodyCmd, RigidBodyState
import numpy as np
import PyKDL
import time
import sys
import os
from argparse import ArgumentParser

# import the centerline tools from mentor's repo
sys.path.append(os.path.expanduser('~/mesh_to_bezier_curve'))
from find_mesh_centerline import extract_bspline_from_obj, basis_values


class RingMoverNode(Node):
    def __init__(self, enable_perturbations=False):
        super().__init__('ring_mover')

        self.enable_perturbations = enable_perturbations
        self.get_logger().info(f"Perturbations enabled: {enable_perturbations}")

        # Load centerline from wire mesh OBJ
        mesh_path = '/mnt/c/Users/lydia/Documents/Hopkins/lydia_surgical_robotics_challenge/ADF/Phantoms/ring_wire_env/high_res/wire_visual.OBJ'
        self.control_points, self.knots, self.degree = extract_bspline_from_obj(
            mesh_path, num_slices=100
        )
        self.get_logger().info(f"Loaded centerline with {len(self.control_points)} control points")

        # Publisher to command ring pose
        self.ring_cmd_pub = self.create_publisher(
            RigidBodyCmd, '/ambf/env/phantom/ring_visual/Command', 1)

        # Subscribe to wire pose to get transform to world frame
        self.T_wire_world = None
        self.wire_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/wire_visual/State',
            self.wire_callback, 1)

        # Subscribe to ring pose to find starting t
        self.current_ring_pose = None
        self.ring_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/ring_visual/State',
            self.ring_callback, 1)

        self.get_logger().info("Waiting for wire and ring poses...")

    def wire_callback(self, msg):
        if self.T_wire_world is not None:
            return
        wire_pos = PyKDL.Vector(
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z)
        wire_rot = PyKDL.Rotation.Quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w)
        self.T_wire_world = PyKDL.Frame(wire_rot, wire_pos)
        self.get_logger().info(f"Wire pose locked in: ({wire_pos.x():.4f}, {wire_pos.y():.4f}, {wire_pos.z():.4f})")

    def ring_callback(self, msg):
        self.current_ring_pose = msg

    def evaluate_bspline(self, t):
        """Get point on B-spline centerline at parameter t (0 to 1)"""
        weights = basis_values(t, self.degree, self.knots, len(self.control_points))
        return np.array([
            sum(w * cp[axis] for w, cp in zip(weights, self.control_points))
            for axis in range(3)
        ])

    def get_tangent(self, t, epsilon=0.001):
        """Numerical tangent at parameter t — points in direction of increasing t"""
        t_clamped = min(max(t, epsilon), 1.0 - epsilon)
        p_forward = self.evaluate_bspline(t_clamped + epsilon)
        p_backward = self.evaluate_bspline(t_clamped - epsilon)
        tangent = p_forward - p_backward
        return tangent / np.linalg.norm(tangent)

    def find_closest_t(self, world_pos):
        """Find t value on B-spline closest to a given world position"""
        # convert world position to wire local frame
        world_kdl = PyKDL.Vector(world_pos[0], world_pos[1], world_pos[2])
        local_kdl = self.T_wire_world.Inverse() * world_kdl
        local_pos = np.array([local_kdl.x(), local_kdl.y(), local_kdl.z()])

        # coarse search across 100 t values
        best_t = 0.0
        best_dist = float('inf')
        for i in range(101):
            t = i / 100.0
            p = self.evaluate_bspline(t)
            dist = np.linalg.norm(p - local_pos)
            if dist < best_dist:
                best_dist = dist
                best_t = t

        # fine search around best_t
        search_range = 0.05
        for i in range(100):
            t = best_t - search_range + (2 * search_range * i / 100.0)
            t = min(max(t, 0.0), 1.0)
            p = self.evaluate_bspline(t)
            dist = np.linalg.norm(p - local_pos)
            if dist < best_dist:
                best_dist = dist
                best_t = t

        return best_t, best_dist

    def command_ring_pose(self, position_local, orientation_local):
        """Send pose command to ring — converts from wire local frame to world frame"""
        if self.T_wire_world is None:
            return

        # transform position from wire local frame to world frame
        pos_kdl = PyKDL.Vector(position_local[0], position_local[1], position_local[2])
        pos_world = self.T_wire_world * pos_kdl

        # transform orientation from wire local frame to world frame
        rot_world = self.T_wire_world.M * orientation_local

        msg = RigidBodyCmd()
        msg.pose.position.x = pos_world.x()
        msg.pose.position.y = pos_world.y()
        msg.pose.position.z = pos_world.z()

        q = rot_world.GetQuaternion()
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        msg.cartesian_cmd_type = 1
        self.ring_cmd_pub.publish(msg)

    def run_test(self):
        # wait for wire pose
        while self.T_wire_world is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        # wait for ring pose
        while self.current_ring_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        # diagnostic prints
        r, p, y = self.T_wire_world.M.GetRPY()
        print(f"Wire body rotation RPY (deg): ({np.degrees(r):.2f}, {np.degrees(p):.2f}, {np.degrees(y):.2f})")
        print(f"\nWire origin in AMBF world: ({self.T_wire_world.p.x():.4f}, {self.T_wire_world.p.y():.4f}, {self.T_wire_world.p.z():.4f})")

        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p_local = self.evaluate_bspline(t_val)
            p_kdl = PyKDL.Vector(p_local[0], p_local[1], p_local[2])
            p_world = self.T_wire_world * p_kdl
            print(f"t={t_val:.2f} local: ({p_local[0]:.4f}, {p_local[1]:.4f}, {p_local[2]:.4f}) | world: ({p_world.x():.4f}, {p_world.y():.4f}, {p_world.z():.4f})")

        # find where ring currently sits on the centerline
        ring_world = np.array([
            self.current_ring_pose.pose.position.x,
            self.current_ring_pose.pose.position.y,
            self.current_ring_pose.pose.position.z
        ])
        start_t, start_dist = self.find_closest_t(ring_world)
        print(f"\nRing current position in world: ({ring_world[0]:.4f}, {ring_world[1]:.4f}, {ring_world[2]:.4f})")
        print(f"Closest t on centerline: {start_t:.3f} (distance: {start_dist*1000:.1f}mm)")

        self.get_logger().info(f"Starting movement from t={start_t:.3f} toward t=0.05")

        # -------------------------------------------------------
        # PERTURBATION PARAMETERS — tune these t ranges after
        # watching terminal output to find flat section and peak-
        # to-trough section for your specific wire
        # -------------------------------------------------------
        # translational perturbation — before first peak (flat section near left base)
        trans_perturb_t_start = start_t       # start of flat section
        trans_perturb_t_end   = 0.85          # end before peak begins
        trans_perturb_amplitude = 0.005        # 5mm sideways

        # rotational perturbation — between first peak and first trough
        rot_perturb_t_start = 0.75            # just after first peak
        rot_perturb_t_end   = 0.55            # just before first trough
        rot_perturb_amplitude = np.radians(20) # 20 degrees tilt
        # -------------------------------------------------------

        rate_hz = 50
        period = 1.0 / rate_hz
        travel_speed = 0.02  # t units per second

        t = start_t

        # initialize parallel transport frame from starting tangent
        initial_tangent = -self.get_tangent(t)
        initial_up = np.array([0.0, 0.0, 1.0])  # world z as up
        if abs(np.dot(initial_tangent, initial_up)) > 0.9:
            initial_up = np.array([1.0, 0.0, 0.0])
        initial_x = np.cross(initial_up, initial_tangent)
        initial_x /= np.linalg.norm(initial_x)
        prev_x_axis = initial_x

        while rclpy.ok() and t >= 0.05:
            centerline_pos = self.evaluate_bspline(t)
            tangent = -self.get_tangent(t)

            # parallel transport — smoothly evolve frame from previous iteration
            x_axis = prev_x_axis - np.dot(prev_x_axis, tangent) * tangent
            if np.linalg.norm(x_axis) < 1e-6:
                x_axis = np.cross(np.array([0.0, 0.0, 1.0]), tangent)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(tangent, x_axis)
            y_axis /= np.linalg.norm(y_axis)

            # base orientation — ring Z axis along tangent
            base_rot = PyKDL.Rotation(
                x_axis[0], y_axis[0], tangent[0],
                x_axis[1], y_axis[1], tangent[1],
                x_axis[2], y_axis[2], tangent[2]
            )

            # default — no perturbation
            final_pos = centerline_pos
            final_rot = base_rot

            if self.enable_perturbations:

                # TRANSLATIONAL PERTURBATION — flat section before first peak
                
                if 0.90 < t <= 0.93:   # short window — tune these values
                    phase = (0.93 - t) / (0.93 - 0.90)  # 0 to 1 over this window
                    perturb_pos = trans_perturb_amplitude * np.sin(2 * np.pi * 1.0 * phase) * x_axis
                    # 2.0 * phase means 2 full sine cycles in this window = 2 side-to-side oscillations
                    final_pos = centerline_pos + perturb_pos
                    print(f"t={t:.3f} | TRANSLATIONAL perturb: ({perturb_pos[0]*1000:.1f}, {perturb_pos[1]*1000:.1f}, {perturb_pos[2]*1000:.1f})mm")

                # ROTATIONAL PERTURBATION — short window between first peak and first trough
                elif 0.55 < t <= 0.60:   # short window — tune these values
                    phase = (0.60 - t) / (0.60 - 0.55)  # 0 to 1 over this window
                    tilt_angle = rot_perturb_amplitude * np.sin(2 * np.pi * 1.0 * phase)
                    perturb_rot = PyKDL.Rotation.RotX(tilt_angle)
                    final_rot = base_rot * perturb_rot
                    print(f"t={t:.3f} | ROTATIONAL perturb: {np.degrees(tilt_angle):.1f} deg")

            self.command_ring_pose(final_pos, final_rot)
            prev_x_axis = x_axis

            t -= travel_speed * period
            time.sleep(period)
            rclpy.spin_once(self, timeout_sec=0)

        self.get_logger().info("Test complete — ring reached right end of wire")


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument('--perturb', action='store_true',
                        help='Enable translational and rotational perturbations')
    parsed_args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    node = RingMoverNode(enable_perturbations=parsed_args.perturb)
    node.run_test()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()