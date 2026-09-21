#!/usr/bin/env python3
import json
import os
import numpy as np
import PyKDL
import rclpy
import threading
import time
from rclpy.node import Node
from ambf_msgs.msg import RigidBodyState, RigidBodyCmd
from scipy.optimize import minimize_scalar, OptimizeResult
from geometry_msgs.msg import WrenchStamped, TwistStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Joy
from argparse import ArgumentParser


def get_bezier_point(t, P0, P1, P2, P3):
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3

def distance_objective(t, ring_com, P0, P1, P2, P3):
    wire_xyz = get_bezier_point(t, P0, P1, P2, P3)
    return np.linalg.norm(wire_xyz - ring_com)


class WireTrackerNode(Node):
    def __init__(self, simulation_mode=False):
        super().__init__('wire_distance_tracker')

        self.simulation_mode = simulation_mode
        self.get_logger().info(f"Simulation mode: {self.simulation_mode}")

        # LOAD BEZIER CURVE DATA
        json_path = os.path.join(os.path.dirname(__file__), "my_bezier_curve.json")
        with open(json_path, 'r') as f:
            self.my_total_curve = json.load(f)
        self.get_logger().info("Bezier curve loaded.")

        # STATE VARIABLES
        self.latest_T_wire_world = None
        self.latest_T_camera_world = None
        self.latest_ring_msg = None
        self.latest_twist_L = None
        self.latest_twist_R = None
        self.coag_pressed = False
        self.prev_u_tangent = None

        # SUBSCRIBERS
        self.coag_sub = self.create_subscription(
            Joy, '/console1/operator_present', self.coag_callback, 1)
        self.wire_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/wire_visual/State',
            self.wire_pose_callback, 1)
        self.ring_sub = self.create_subscription(
            RigidBodyState, '/ambf/env/phantom/ring_visual/State',
            self.ring_pose_callback, 1)
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

    def wire_pose_callback(self, msg_wire):
        wire_pos = PyKDL.Vector(
            msg_wire.pose.position.x,
            msg_wire.pose.position.y,
            msg_wire.pose.position.z)
        wire_rot = PyKDL.Rotation.Quaternion(
            msg_wire.pose.orientation.x,
            msg_wire.pose.orientation.y,
            msg_wire.pose.orientation.z,
            msg_wire.pose.orientation.w)
        self.latest_T_wire_world = PyKDL.Frame(wire_rot, wire_pos)

    def ring_pose_callback(self, msg_ring):
        self.latest_ring_msg = msg_ring

    def camera_pose_callback(self, msg_camera):
        camera_pos = PyKDL.Vector(
            msg_camera.pose.position.x,
            msg_camera.pose.position.y,
            msg_camera.pose.position.z)
        camera_rot = PyKDL.Rotation.Quaternion(
            msg_camera.pose.orientation.x,
            msg_camera.pose.orientation.y,
            msg_camera.pose.orientation.z,
            msg_camera.pose.orientation.w)
        self.latest_T_camera_world = PyKDL.Frame(camera_rot, camera_pos)

    def twist_callback_L(self, msg):
        self.latest_twist_L = msg

    def twist_callback_R(self, msg):
        self.latest_twist_R = msg

    # ---------------------- GEOMETRY ----------------------

    def get_ring_frame_in_wire(self, msg_ring):
        ring_pos = PyKDL.Vector(
            msg_ring.pose.position.x,
            msg_ring.pose.position.y,
            msg_ring.pose.position.z)
        ring_rot = PyKDL.Rotation.Quaternion(
            msg_ring.pose.orientation.x,
            msg_ring.pose.orientation.y,
            msg_ring.pose.orientation.z,
            msg_ring.pose.orientation.w)
        T_ring_world = PyKDL.Frame(ring_rot, ring_pos)
        return self.latest_T_wire_world.Inverse() * T_ring_world

    def get_closest_wire_point(self, ring_com):
        scored_segments = []
        for segment in self.my_total_curve:
            dists = [np.linalg.norm(np.array(segment[i]) - ring_com)
                     for i in range(4)]
            scored_segments.append((min(dists), segment))
        scored_segments.sort(key=lambda x: x[0])
        top_candidates = scored_segments[:3]

        final_result = OptimizeResult(fun=float('inf'))
        winning_segment_points = None

        for _, segment in top_candidates:
            P0, P1, P2, P3 = map(np.array, segment)
            result = minimize_scalar(
                distance_objective,
                bounds=(0.0, 1.0),
                method='bounded',
                args=(ring_com, P0, P1, P2, P3))
            if result.fun < final_result.fun:
                final_result = result
                winning_segment_points = (P0, P1, P2, P3)

        closest_t = final_result.x
        min_distance = final_result.fun
        closest_wire_point = get_bezier_point(closest_t, *winning_segment_points)
        return closest_t, min_distance, closest_wire_point, winning_segment_points

    def tangent_vector(self, closest_t, P0, P1, P2, P3):
        return (3 * (1 - closest_t)**2 * (P1 - P0) +
                6 * (1 - closest_t) * closest_t * (P2 - P1) +
                3 * closest_t**2 * (P3 - P2))

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

    def compute_rotational_error(self, closest_t, T_ring_wire,
                                  winning_segment_points):
        P0, P1, P2, P3 = winning_segment_points
        tangent = self.tangent_vector(closest_t, P0, P1, P2, P3)
        u_tangent = tangent / np.linalg.norm(tangent)
        u_ring_z = np.array([T_ring_wire.M.UnitZ().x(),
                              T_ring_wire.M.UnitZ().y(),
                              T_ring_wire.M.UnitZ().z()])
        dot_product = np.dot(u_tangent, u_ring_z)
        clipped_dot = np.clip(dot_product, -1.0, 1.0)
        angular_error_rad = np.arccos(clipped_dot)
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
        u_ring_z_aligned = -u_ring_z if dot_product < 0 else u_ring_z
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
        # on dVRK — only publish when coag is pressed
        # in simulation mode — always publish for testing
        if not self.simulation_mode and not self.coag_pressed:
            return

        R_wire_to_camera = (self.latest_T_camera_world.M.Inverse() *
                            self.latest_T_wire_world.M)
        T_baseoffset = PyKDL.Frame(
            PyKDL.Rotation.RPY((3.14 - 0.8) / 2, 0, 0),
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

        # always publish to both arms — no grasp check needed
        # user just holds MTM handles to feel forces as ring moves on its own
        self.wrench_pub_L.publish(build_wrench(f_L_cam, t_L_cam))
        self.wrench_pub_R.publish(build_wrench(f_R_cam, t_R_cam))

    # ---------------------- CONTROL LOOP ----------------------

    def control_loop(self):
        # PARAMETERS — tune these
        max_force        = 2.0    # N
        max_torque       = 0.5    # N·m
        kp_pos           = 50     # N/m
        kd_pos           = 1.0    # N/(m/s)
        kp_rot           = 0.1    # N·m/rad
        kd_rot           = 0.0    # N·m/(rad/s)
        linear_deadband  = 0.001  # m
        angular_deadband = 0.0    # deg

        # SAFETY GUARD
        # in simulation mode skip twist check — no dVRK connected
        if (self.latest_ring_msg is None or
                self.latest_T_wire_world is None or
                self.latest_T_camera_world is None or
                (not self.simulation_mode and self.latest_twist_L is None) or
                (not self.simulation_mode and self.latest_twist_R is None)):
            return

        # SNAPSHOT — grab once for consistent computation this cycle
        ring_msg = self.latest_ring_msg

        # GEOMETRY
        T_ring_wire = self.get_ring_frame_in_wire(ring_msg)
        ring_com = np.array([T_ring_wire.p.x(),
                              T_ring_wire.p.y(),
                              T_ring_wire.p.z()])
        closest_t, min_distance, closest_wire_point, winning_segment_points = \
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

        # skip damping in simulation mode — no twist data available
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
            self.compute_rotational_error(
                closest_t, T_ring_wire, winning_segment_points)
        filtered_tangent = self.limit_tangent_rate(u_tangent)
        torque_angular = self.compute_torque(
            angular_error_deg, filtered_tangent, u_ring_z,
            dot_product, kp_rot, angular_deadband)

        # skip torque damping in simulation mode too
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

        # PUBLISH WRENCH TO MTMs
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


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        '--sim', action='store_true',
        help='Simulation mode — bypasses dVRK twist/coag/grasp requirements for laptop testing')
    parsed_args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    tracker = WireTrackerNode(simulation_mode=parsed_args.sim)
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()