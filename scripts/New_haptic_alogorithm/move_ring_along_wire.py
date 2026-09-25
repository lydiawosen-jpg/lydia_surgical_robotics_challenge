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
from std_msgs.msg import Empty
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
# import the centerline tools from mentor's repo
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh_to_bezier_curve'))
from find_mesh_centerline import extract_bspline_from_obj, basis_values
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from surgical_robotics_challenge.utils.utilities import frame_to_pose, pose_to_frame


class RingMoverNode(Node):
    MTM_TRANSLATION_SCALE = 0.2

    def __init__(self, enable_perturbations=False,
                 enable_mtm_perturbations=False, pedal_traversal=False):
        super().__init__('ring_mover')

        self.enable_perturbations = enable_perturbations
        self.enable_mtm_perturbations = enable_mtm_perturbations
        self.pedal_traversal = pedal_traversal
        self.get_logger().info(f"Perturbations enabled: {enable_perturbations}")
        self.get_logger().info(
            f"Traversal mode: {'cam pedals' if pedal_traversal else 'automatic'}")
        self.cam_plus_pressed = False
        self.cam_minus_pressed = False
        self.clutch_pressed = False
        self.mtm_pose = None
        self.mtm_pose_at_clutch = None
        self.camera_pose = None

        # Load centerline from wire mesh OBJ
        mesh_path = os.path.join(os.path.dirname(__file__), 'mesh_to_bezier_curve', 'mesh', 'wire_visual.OBJ')
        self.control_points, self.knots, self.degree = extract_bspline_from_obj(
            mesh_path, num_slices=100
        )
        self.get_logger().info(f"Loaded centerline with {len(self.control_points)} control points")

        # Publisher to command ring pose
        self.ring_cmd_pub = self.create_publisher(
            RigidBodyCmd, '/ambf/env/phantom/ring_visual/Command', 1)
        
        self.reset_pub = self.create_publisher(
            Empty, '/ambf/env/World/Command/Reset/Bodies', 1)

        self.cam_plus_sub = self.create_subscription(
            Joy, '/console1/focus_plus', self.cam_plus_callback, 1)
        self.cam_minus_sub = self.create_subscription(
            Joy, '/console1/focus_minus', self.cam_minus_callback, 1)
        self.clutch_sub = self.create_subscription(
            Joy, '/console1/clutch', self.clutch_callback, 1)
        self.mtm_pose_sub = self.create_subscription(
            PoseStamped, '/MTML/measured_cp', self.mtm_pose_callback, 1)
        self.camera_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/CameraFrame/State',
            self.camera_callback, 1)

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
        self.T_wire_world = pose_to_frame(msg.pose)
        self.get_logger().info(
            f"Wire pose locked in: ({self.T_wire_world.p.x():.4f}, "
            f"{self.T_wire_world.p.y():.4f}, {self.T_wire_world.p.z():.4f})")

    def cam_plus_callback(self, msg):
        self.cam_plus_pressed = bool(msg.buttons[0])

    def cam_minus_callback(self, msg):
        self.cam_minus_pressed = bool(msg.buttons[0])

    def clutch_callback(self, msg):
        pressed = bool(msg.buttons[0])
        if pressed and not self.clutch_pressed and self.enable_mtm_perturbations:
            self.mtm_pose_at_clutch = self.mtm_pose
        elif not pressed:
            self.mtm_pose_at_clutch = None
        self.clutch_pressed = pressed

    def mtm_pose_callback(self, msg):
        self.mtm_pose = pose_to_frame(msg.pose)
        if (self.clutch_pressed and self.enable_mtm_perturbations and
            self.mtm_pose_at_clutch is None):
            self.mtm_pose_at_clutch = self.mtm_pose

    def camera_callback(self, msg):
        self.camera_pose = pose_to_frame(msg.pose)

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
        msg.pose = frame_to_pose(PyKDL.Frame(rot_world, pos_world))

        msg.cartesian_cmd_type = 1
        self.ring_cmd_pub.publish(msg)

    def get_traversal_direction(self):
        if not self.pedal_traversal:
            return -1.0
        if self.cam_plus_pressed and not self.cam_minus_pressed:
            return -1.0
        if self.cam_minus_pressed and not self.cam_plus_pressed:
            return 1.0
        return 0.0

    def get_clutch_perturbation(self):
        if (not self.enable_mtm_perturbations or not self.clutch_pressed or
            self.mtm_pose_at_clutch is None or self.mtm_pose is None or
            self.camera_pose is None):
            return np.zeros(3), PyKDL.Rotation.Identity()

        mtm_base_to_camera = PyKDL.Rotation.RotX(-0.865)
        translation_mtm = self.mtm_pose.p - self.mtm_pose_at_clutch.p
        translation_camera = (self.MTM_TRANSLATION_SCALE *
                      (mtm_base_to_camera * translation_mtm))
        translation_world = self.camera_pose.M * translation_camera
        translation_local = self.T_wire_world.M.Inverse() * translation_world
        rotation_mtm = self.mtm_pose_at_clutch.M.Inverse() * self.mtm_pose.M
        rotation_camera = mtm_base_to_camera * rotation_mtm * mtm_base_to_camera.Inverse()
        rotation_world = self.camera_pose.M * rotation_camera * self.camera_pose.M.Inverse()
        rotation_local = self.T_wire_world.M.Inverse() * rotation_world * self.T_wire_world.M
        return np.array([
            translation_local.x(),
            translation_local.y(),
            translation_local.z()
        ]), rotation_local

    def reset_bodies(self):
        self.get_logger().info("Resetting all bodies before moving the wire")
        self.reset_pub.publish(Empty())
        time.sleep(0.5)  # give AMBF time to reset

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

        while rclpy.ok() and 0.05 <= t <= 1.0:
            rclpy.spin_once(self, timeout_sec=0.0)
            traversal_direction = self.get_traversal_direction()
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

            clutch_translation, clutch_rotation = self.get_clutch_perturbation()
            final_pos = final_pos + clutch_translation
            final_rot = final_rot * clutch_rotation

            # Keep publishing while paused so the ring holds its pose.
            self.command_ring_pose(final_pos, final_rot)
            prev_x_axis = x_axis

            t += traversal_direction * travel_speed * period
            t = min(max(t, 0.05), 1.0)
            time.sleep(period)

        self.get_logger().info("Test complete — ring reached right end of wire")


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument('--perturb', action='store_true',
                        help='Enable translational and rotational perturbations')
    parser.add_argument('--mtm-perturb', action='store_true',
                        help='Enable clutch-controlled MTML pose perturbations')
    parser.add_argument('--pedal-traverse', action='store_true',
                        help='Use cam+ and cam- pedals instead of automatic traversal')
    parsed_args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    node = RingMoverNode(
        enable_perturbations=parsed_args.perturb,
        enable_mtm_perturbations=parsed_args.mtm_perturb,
        pedal_traversal=parsed_args.pedal_traverse)
    node.reset_bodies()  # reset before starting the test
    node.run_test()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()