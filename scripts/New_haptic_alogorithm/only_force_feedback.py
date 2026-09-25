#!/usr/bin/env python3
import os
import sys
import numpy as np
import PyKDL
import rclpy
import threading
import time
from rclpy.node import Node
from ambf_msgs.msg import RigidBodyState, RigidBodyCmd
from scipy.optimize import minimize_scalar, minimize
from geometry_msgs.msg import WrenchStamped, TwistStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Joy
from argparse import ArgumentParser
# import centerline tools from mentor's repo
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh_to_bezier_curve'))
from find_mesh_centerline import extract_bspline_from_obj, basis_values
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from surgical_robotics_challenge.utils.utilities import pose_to_frame


class WireTrackerNode(Node):
    MTM_TRANSLATION_SCALE = 0.2

    def __init__(self, simulation_mode=False, enable_mtm_perturbations=False):
        super().__init__('wire_distance_tracker')

        self.simulation_mode = simulation_mode
        self.enable_mtm_perturbations = enable_mtm_perturbations
        self.get_logger().info(f"Simulation mode: {self.simulation_mode}")

        # LOAD CENTERLINE FROM WIRE MESH OBJ
        mesh_path = os.path.join(os.path.dirname(__file__), 'mesh_to_bezier_curve', 'mesh', 'wire_visual.OBJ')
        self.control_points, self.knots, self.degree = extract_bspline_from_obj(
            mesh_path, num_slices=100)
        self.get_logger().info(
            f"Loaded B-spline centerline with {len(self.control_points)} control points")

        # STATE VARIABLES
        self.latest_T_wire_world = None
        self.latest_T_camera_world = None
        self.latest_ring_msg = None
        self.latest_twist_L = None
        self.latest_twist_R = None
        self.coag_pressed = False
        self.prev_u_tangent = None
        self.last_ring_cmd_time = None
        self.mover_timeout_reported = False
        self.clutch_pressed = False
        self.mtm_pose = None
        self.mtm_pose_at_clutch = None

        # SUBSCRIBERS
        self.coag_sub = self.create_subscription(
            Joy, '/console1/operator_present', self.coag_callback, 1)
        self.clutch_sub = self.create_subscription(
            Joy, '/console1/clutch', self.clutch_callback, 1)
        self.mtm_pose_sub = self.create_subscription(
            PoseStamped, '/MTML/measured_cp', self.mtm_pose_callback, 1)
        self.wire_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/wire_visual/State',
            self.wire_pose_callback, 1)
        self.ring_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/ring_visual/State',
            self.ring_pose_callback, 1)
        # heartbeat from move_ring_along_wire.py — used to zero wrench if it stops
        self.ring_cmd_sub = self.create_subscription(
            RigidBodyCmd, '/ambf/env/phantom/ring_visual/Command',
            self.ring_cmd_callback, 1)
        self.camera_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/CameraFrame/State',
            self.camera_pose_callback, 1)
        self.twist_sub_L = self.create_subscription(
            TwistStamped, '/MTML/measured_cv', self.twist_callback_L, 1)
        self.twist_sub_R = self.create_subscription(
            TwistStamped, '/MTMR/measured_cv', self.twist_callback_R, 1)

        # PUBLISHERS — wrench output to MTMs
        self.wrench_pub_L = self.create_publisher(
            WrenchStamped, '/MTML/body/servo_cf', 1)
        self.wrench_pub_R = self.create_publisher(
            WrenchStamped, '/MTMR/body/servo_cf', 1)

        # PUBLISHERS — orientation absolute flag
        self.orientation_abs_pub_L = self.create_publisher(
            Bool, '/MTML/body/set_cf_orientation_absolute', 1)
        self.orientation_abs_pub_R = self.create_publisher(
            Bool, '/MTMR/body/set_cf_orientation_absolute', 1)

        # PUBLISHERS — debug topics for PlotJuggler
        self.debug_error_pub = self.create_publisher(
            Float32, '/debug/radial_error', 1)
        self.debug_force_mag_pub = self.create_publisher(
            Float32, '/debug/force_magnitude', 1)
        self.debug_angular_error_pub = self.create_publisher(
            Float32, '/debug/angular_error', 1)
        self.debug_torque_x_pub = self.create_publisher(
            Float32, '/debug/torque_x', 1)
        self.debug_torque_y_pub = self.create_publisher(
            Float32, '/debug/torque_y', 1)
        self.debug_torque_z_pub = self.create_publisher(
            Float32, '/debug/torque_z', 1)
        self.debug_ring_pos_x_pub = self.create_publisher(
            Float32, '/debug/ring_pos_x', 1)
        self.debug_ring_pos_y_pub = self.create_publisher(
            Float32, '/debug/ring_pos_y', 1)
        self.debug_ring_pos_z_pub = self.create_publisher(
            Float32, '/debug/ring_pos_z', 1)

        # publish absolute orientation flag once at startup
        abs_flag = Bool()
        abs_flag.data = True
        self.orientation_abs_pub_L.publish(abs_flag)
        self.orientation_abs_pub_R.publish(abs_flag)

        # START CONTROL LOOP BACKGROUND THREAD
        self.control_thread = threading.Thread(
            target=self.run_control_loop, daemon=True)
        self.control_thread.start()

        self.get_logger().info("WireTrackerNode ready.")

    # ---------------------- CALLBACKS ----------------------

    def coag_callback(self, msg):
        self.coag_pressed = msg.buttons[0]

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

    def ring_cmd_callback(self, msg):
        self.last_ring_cmd_time = time.time()
        self.mover_timeout_reported = False

    def wire_pose_callback(self, msg_wire):
        self.latest_T_wire_world = pose_to_frame(msg_wire.pose)

    def ring_pose_callback(self, msg_ring):
        self.latest_ring_msg = msg_ring

    def camera_pose_callback(self, msg_camera):
        self.latest_T_camera_world = pose_to_frame(msg_camera.pose)

    def twist_callback_L(self, msg):
        self.latest_twist_L = msg

    def twist_callback_R(self, msg):
        self.latest_twist_R = msg

    # ---------------------- B-SPLINE EVALUATION ----------------------

    def evaluate_bspline(self, t):
        """Evaluate B-spline centerline at parameter t (0 to 1)"""
        weights = basis_values(t, self.degree, self.knots,
                               len(self.control_points))
        return np.array([
            sum(w * cp[axis] for w, cp in zip(weights, self.control_points))
            for axis in range(3)
        ])

    def get_bspline_tangent(self, t, epsilon=0.001):
        """Numerical tangent of B-spline at parameter t"""
        t_clamped = min(max(t, epsilon), 1.0 - epsilon)
        p_forward = self.evaluate_bspline(t_clamped + epsilon)
        p_backward = self.evaluate_bspline(t_clamped - epsilon)
        tangent = p_forward - p_backward
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            print("Warning: B-spline tangent norm is near zero at t =", t)
            return np.array([1.0, 0.0, 0.0])  # fallback
        return tangent / norm

    # ---------------------- GEOMETRY ----------------------

    def get_ring_frame_in_wire(self, msg_ring):
        T_ring_world = pose_to_frame(msg_ring.pose)
        if (self.enable_mtm_perturbations and self.clutch_pressed and
            self.mtm_pose_at_clutch is not None and self.mtm_pose is not None and
            self.latest_T_camera_world is not None):
            mtm_base_to_camera = PyKDL.Rotation.RotX(-0.865)
            translation_mtm = self.mtm_pose.p - self.mtm_pose_at_clutch.p
            translation_camera = (self.MTM_TRANSLATION_SCALE *
                                  (mtm_base_to_camera * translation_mtm))
            translation_delta = self.latest_T_camera_world.M * translation_camera
            rotation_mtm = self.mtm_pose_at_clutch.M.Inverse() * self.mtm_pose.M
            rotation_camera = (mtm_base_to_camera * rotation_mtm *
                               mtm_base_to_camera.Inverse())
            rotation_delta = (self.latest_T_camera_world.M * rotation_camera *
                              self.latest_T_camera_world.M.Inverse())
            T_ring_world = PyKDL.Frame(
                rotation_delta * T_ring_world.M,
                T_ring_world.p + translation_delta)
        return self.latest_T_wire_world.Inverse() * T_ring_world

    def get_closest_wire_point(self, ring_com):
        """Find the closest point on the B-spline centerline to ring_com"""

        def distance_to_bspline(t_array):
            t = float(t_array[0])
            t = min(max(t, 0.0), 1.0)
            p = self.evaluate_bspline(t)
            return np.linalg.norm(p - ring_com)

        # coarse search — evaluate at 100 evenly spaced t values
        best_t = 0.0
        best_dist = float('inf')
        for i in range(101):
            t = i / 100.0
            p = self.evaluate_bspline(t)
            dist = np.linalg.norm(p - ring_com)
            if dist < best_dist:
                best_dist = dist
                best_t = t

        # fine search — minimize around best coarse t
        result = minimize(
            distance_to_bspline,
            x0=[best_t],
            bounds=[(max(0.0, best_t - 0.1), min(1.0, best_t + 0.1))],
            method='L-BFGS-B')

        closest_t = float(np.clip(result.x[0], 0.0, 1.0))
        closest_wire_point = self.evaluate_bspline(closest_t)
        min_distance = np.linalg.norm(closest_wire_point - ring_com)

        return closest_t, min_distance, closest_wire_point

    # ---------------------- FORCE FEEDBACK ----------------------

    def compute_linear_force(self, min_distance, closest_wire_point,
                              ring_com, kp_pos, linear_deadband):
        f_linear = np.zeros(3)
        u_vector_ring_to_wire = np.zeros(3)
        if min_distance > linear_deadband:
            vector_ring_to_wire = closest_wire_point - ring_com
            u_vector_ring_to_wire = (vector_ring_to_wire /
                                      np.linalg.norm(vector_ring_to_wire))
            effective_error = min_distance - linear_deadband
            f_linear = kp_pos * effective_error * u_vector_ring_to_wire
        return f_linear, u_vector_ring_to_wire

    def compute_linear_damping_L(self, kd_pos, u_vector_ring_to_wire):
        vel = np.array([self.latest_twist_L.twist.linear.x,
                        self.latest_twist_L.twist.linear.y,
                        self.latest_twist_L.twist.linear.z])
        return kd_pos * np.dot(vel, u_vector_ring_to_wire) * u_vector_ring_to_wire

    def compute_linear_damping_R(self, kd_pos, u_vector_ring_to_wire):
        vel = np.array([self.latest_twist_R.twist.linear.x,
                        self.latest_twist_R.twist.linear.y,
                        self.latest_twist_R.twist.linear.z])
        return kd_pos * np.dot(vel, u_vector_ring_to_wire) * u_vector_ring_to_wire

    def compute_rotational_error(self, closest_t, T_ring_wire):
        """Compute angular error between ring Z axis and wire tangent"""
        # The mover drives the ring along decreasing B-spline t.
        u_tangent = -self.get_bspline_tangent(closest_t)
        u_ring_z = np.array([T_ring_wire.M.UnitZ().x(),
                              T_ring_wire.M.UnitZ().y(),
                              T_ring_wire.M.UnitZ().z()])
        dot_product = float(np.dot(u_tangent, u_ring_z))
        # A ring axis has no meaningful forward/backward direction here.
        dot_product = abs(np.clip(dot_product, -1.0, 1.0))
        angular_error_rad = np.arccos(dot_product)
        angular_error_deg = np.degrees(angular_error_rad)
        return angular_error_deg, u_tangent, u_ring_z, dot_product

    def limit_tangent_rate(self, u_tangent, max_angle_change_rad=0.05):
        if self.prev_u_tangent is None:
            self.prev_u_tangent = u_tangent
            return u_tangent
        dot = np.clip(np.dot(self.prev_u_tangent, u_tangent), -1.0, 1.0)
        angle = np.arccos(dot)
        if angle > max_angle_change_rad and angle > 1e-6:
            alpha = max_angle_change_rad / angle
            filtered = (1 - alpha) * self.prev_u_tangent + alpha * u_tangent
            filtered /= np.linalg.norm(filtered)
        else:
            filtered = u_tangent
        self.prev_u_tangent = filtered
        return filtered

    def compute_torque(self, angular_error_deg, filtered_tangent,
                        u_ring_z, dot_product, kp_rot, angular_deadband):
        if angular_error_deg < angular_deadband:
            return np.zeros(3)
        u_ring_z_aligned = u_ring_z
        if np.dot(u_ring_z_aligned, filtered_tangent) < 0:
            u_ring_z_aligned = -u_ring_z_aligned
        rotation_axis = np.cross(u_ring_z_aligned, filtered_tangent)
        norm_axis = np.linalg.norm(rotation_axis)
        if norm_axis < 1e-6:
            return np.zeros(3)
        u_rotation_axis = rotation_axis / norm_axis
        effective_error_rad = np.radians(angular_error_deg - angular_deadband)
        return kp_rot * effective_error_rad * u_rotation_axis

    def compute_torque_damping_L(self, kd_rot):
        ang_vel = np.array([self.latest_twist_L.twist.angular.x,
                            self.latest_twist_L.twist.angular.y,
                            self.latest_twist_L.twist.angular.z])
        return kd_rot * ang_vel

    def compute_torque_damping_R(self, kd_rot):
        ang_vel = np.array([self.latest_twist_R.twist.angular.x,
                            self.latest_twist_R.twist.angular.y,
                            self.latest_twist_R.twist.angular.z])
        return kd_rot * ang_vel

    def transform_and_publish_wrench(self, max_force, max_torque,
                                      f_total_L, f_total_R,
                                      torque_total_L, torque_total_R):
        # on dVRK — zero the wrench when coag is not pressed
        # in simulation mode — always publish for testing
        if not self.simulation_mode and not self.coag_pressed:
            self.zero_wrench()
            return

        R_wire_to_camera = (self.latest_T_camera_world.M.Inverse() *
                            self.latest_T_wire_world.M)
        T_baseoffset = PyKDL.Frame(
            # PyKDL.Rotation.RPY((3.14 - 0.8) / 2, 0, 0),
            PyKDL.Rotation.RPY(0.865, 0, 0),
            PyKDL.Vector(0, 0, 0))

        def to_camera_frame(f_np):
            f_vec = PyKDL.Vector(f_np[0], f_np[1], f_np[2])
            return T_baseoffset.M * (R_wire_to_camera * f_vec)

        def build_wrench(f_cam, t_cam):
            msg = WrenchStamped()
            msg.wrench.force.x = float(np.clip(f_cam.x(), -max_force, max_force))
            msg.wrench.force.y = float(np.clip(f_cam.y(), -max_force, max_force))
            msg.wrench.force.z = float(np.clip(f_cam.z(), -max_force, max_force))
            msg.wrench.torque.x = float(np.clip(t_cam.x(), -max_torque, max_torque))
            msg.wrench.torque.y = float(np.clip(t_cam.y(), -max_torque, max_torque))
            msg.wrench.torque.z = float(np.clip(t_cam.z(), -max_torque, max_torque))
            return msg

        f_L_cam = to_camera_frame(f_total_L)
        f_R_cam = to_camera_frame(f_total_R)
        t_L_cam = to_camera_frame(torque_total_L)
        t_R_cam = to_camera_frame(torque_total_R)

        # always publish to both arms — user just holds MTM handles
        self.wrench_pub_L.publish(build_wrench(f_L_cam, t_L_cam))
        self.wrench_pub_R.publish(build_wrench(f_R_cam, t_R_cam))

    # ---------------------- CONTROL LOOP ----------------------

    def control_loop(self):
        # PARAMETERS — tune these
        max_force        = 3.0    # N
        max_torque       = 0.5    # N·m
        kp_pos           = 1000    # N/m
        kd_pos           = 1.0    # N/(m/s)
        kp_rot           = 0.05    # N·m/rad
        kd_rot           = 0.0    # N·m/(rad/s)
        linear_deadband  = 0.0005  # m
        angular_deadband = 2.0    # deg; ignore tracking noise near tangent alignment
        mover_timeout    = 0.5    # s — max silence from move_ring_along_wire.py before zeroing

        # SAFETY GUARD
        if (self.latest_ring_msg is None or
                self.latest_T_wire_world is None or
                self.latest_T_camera_world is None or
                (not self.simulation_mode and self.latest_twist_L is None) or
                (not self.simulation_mode and self.latest_twist_R is None)):
            return

        # MOVER HEARTBEAT — zero wrench if move_ring_along_wire.py has stopped publishing
        if (self.last_ring_cmd_time is not None and
                time.time() - self.last_ring_cmd_time > mover_timeout):
            if not self.mover_timeout_reported:
                print("Move ring command timeout - zeroing wrench", flush=True)
                self.mover_timeout_reported = True
            self.zero_wrench()
            return

        # SNAPSHOT
        ring_msg = self.latest_ring_msg

        # GEOMETRY
        T_ring_wire = self.get_ring_frame_in_wire(ring_msg)
        ring_com = np.array([T_ring_wire.p.x(),
                              T_ring_wire.p.y(),
                              T_ring_wire.p.z()])
        closest_t, min_distance, closest_wire_point = \
            self.get_closest_wire_point(ring_com)

        # RING CAME OFF SAFETY CHECK
        if min_distance > 0.015:
            zero = np.zeros(3)
            self.transform_and_publish_wrench(
                max_force, max_torque, zero, zero, zero, zero)
            return

        # DEBUG — ring position and radial error
        self.debug_ring_pos_x_pub.publish(Float32(data=float(ring_com[0])))
        self.debug_ring_pos_y_pub.publish(Float32(data=float(ring_com[1])))
        self.debug_ring_pos_z_pub.publish(Float32(data=float(ring_com[2])))
        self.debug_error_pub.publish(Float32(data=float(min_distance)))

        # LINEAR FORCE
        f_linear, u_vector_ring_to_wire = self.compute_linear_force(
            min_distance, closest_wire_point, ring_com, kp_pos, linear_deadband)

        if self.simulation_mode:
            f_damping_L = np.zeros(3)
            f_damping_R = np.zeros(3)
        else:
            f_damping_L = self.compute_linear_damping_L(kd_pos, u_vector_ring_to_wire)
            f_damping_R = self.compute_linear_damping_R(kd_pos, u_vector_ring_to_wire)

        f_total_L = f_linear - f_damping_L
        f_total_R = f_linear - f_damping_R

        # DEBUG — force magnitude
        self.debug_force_mag_pub.publish(
            Float32(data=float(np.linalg.norm(f_total_L))))

        # ROTATIONAL TORQUE
        angular_error_deg, u_tangent, u_ring_z, dot_product = \
            self.compute_rotational_error(closest_t, T_ring_wire)
        filtered_tangent = u_tangent
        torque_angular = self.compute_torque(
            angular_error_deg, filtered_tangent, u_ring_z,
            dot_product, kp_rot, angular_deadband)

        if self.simulation_mode:
            torque_damping_L = np.zeros(3)
            torque_damping_R = np.zeros(3)
        else:
            torque_damping_L = self.compute_torque_damping_L(kd_rot)
            torque_damping_R = self.compute_torque_damping_R(kd_rot)

        torque_total_L = torque_angular - torque_damping_L
        torque_total_R = torque_angular - torque_damping_R

        # DEBUG — angular error and torque
        self.debug_angular_error_pub.publish(
            Float32(data=float(angular_error_deg)))
        self.debug_torque_x_pub.publish(Float32(data=float(torque_angular[0])))
        self.debug_torque_y_pub.publish(Float32(data=float(torque_angular[1])))
        self.debug_torque_z_pub.publish(Float32(data=float(torque_angular[2])))

        # PUBLISH WRENCH
        self.transform_and_publish_wrench(
            max_force, max_torque,
            f_total_L, f_total_R,
            torque_total_L, torque_total_R)

    def run_control_loop(self):
        rate_hz = 200
        period = 1.0 / rate_hz
        while rclpy.ok():
            t_start = time.perf_counter()
            self.control_loop()
            elapsed = time.perf_counter() - t_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def zero_wrench(self):
        """Publish zero force/torque to both MTMs, bypassing coag gating"""
        zero_msg = WrenchStamped()
        self.wrench_pub_L.publish(zero_msg)
        self.wrench_pub_R.publish(WrenchStamped())


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        '--sim', action='store_true',
        help='Simulation mode — bypasses dVRK requirements for laptop testing')
    parser.add_argument(
        '--mtm-perturb', action='store_true',
        help='Enable clutch-controlled MTML pose perturbations')
    parsed_args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    tracker = WireTrackerNode(
        simulation_mode=parsed_args.sim,
        enable_mtm_perturbations=parsed_args.mtm_perturb)
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.zero_wrench()
        tracker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()