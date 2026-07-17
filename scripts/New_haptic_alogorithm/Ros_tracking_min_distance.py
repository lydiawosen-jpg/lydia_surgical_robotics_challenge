#!/usr/bin/env python3
import json
import os
import numpy as np
import PyKDL
import rclpy
import threading
import time
from rclpy.node import Node
from ambf_msgs.msg import RigidBodyState, RigidBodyCmd, ActuatorCmd, GhostObjectState, ContactSensorState
from scipy.optimize import minimize_scalar, OptimizeResult
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import WrenchStamped, TwistStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from datetime import datetime
import socket

# ==========================================
# 1. CORE BEZIER MATH
# ==========================================
def get_bezier_point(t, P0, P1, P2, P3):
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3

def distance_objective(t, ring_com, P0, P1, P2, P3):
    wire_xyz = get_bezier_point(t, P0, P1, P2, P3)
    return np.linalg.norm(wire_xyz - ring_com)


class WireTrackerNode(Node):
    def __init__(self):
        super().__init__('wire_distance_tracker')
        
        # LOAD THE WIRE DATA ONCE (Saves CPU cycles)
        # Using the absolute path so it runs from any workspace folder safely
        json_path = os.path.join(os.path.dirname(__file__), "my_bezier_curve.json")
        with open(json_path, 'r') as f:
            self.my_total_curve = json.load(f)
        
        self.get_logger().info("Waiting for static wire pose from AMBF...")

        print("0 - Task started\n1 - Checkpoint 1 Passed\n2 - Checkpoint 2 Passed\n3 - Checkpoint 3 Passed\n4 - Checkpoint 4 Passed\n5 - Checkpoint 5 Passed\n6 - Wire Touched\n7 - Ring Dropped\n8 - Task Ended")
        #----------------------------------------------
        # Initialize Socket placeholders
        self.server_socket = None
        self.client_socket = None
        self.cobi_connected = False

        # Start background thread to handle socket connections without blocking ROS
        self.socket_thread = threading.Thread(target=self.setup_socket_connection, daemon=True)
        self.socket_thread.start()
        #--------------------------------
        self.count = 0
        # Initialize the wire's world frame as None until we receive it
        self.latest_T_wire_world = None  
        self.latest_T_camera_world = None
        self.latest_ring_msg = None
        self.latest_twist_L = None
        self.latest_twist_R = None
        self.coag_pressed = False
        self.ring_grasped_psm1= False
        self.ring_grasped_psm2= False

        # Start a background thread that calls control_loop() repeatedly
        self.control_thread = threading.Thread(target=self.run_control_loop, daemon=True)
        self.control_thread.start()

        self.start_flag_sent = False 
        self.start_trigger = []
        self.checkpoint1 = []
        self.checkpoint1_sent = False
        self.checkpoint2 = []
        self.checkpoint2_sent = False
        self.checkpoint3 = []
        self.checkpoint3_sent = False
        self.checkpoint4 = []
        self.checkpoint4_sent = False
        self.checkpoint5 = []
        self.checkpoint5_sent = False
        self.ring_was_held = False
        self.end_trigger = []
        self.end_flag_sent = False
        self.previous_not_touched = False
        self.start_trigger_sub = self.create_subscription(GhostObjectState, '/ambf/env/phantom/start_trigger/State', self.start_trigger_callback, 1)
        self.checkpoint1_sub = self.create_subscription(GhostObjectState, '/ambf/env/phantom/checkpoint1/State', self.checkpoint1_callback, 1) # can use partial to consolodate callbacks
        self.checkpoint2_sub = self.create_subscription(GhostObjectState, '/ambf/env/phantom/checkpoint2/State', self.checkpoint2_callback, 1)
        self.checkpoint3_sub = self.create_subscription(GhostObjectState, '/ambf/env/phantom/checkpoint3/State', self.checkpoint3_callback, 1)
        self.checkpoint4_sub = self.create_subscription(GhostObjectState, '/ambf/env/phantom/checkpoint4/State', self.checkpoint4_callback, 1)
        self.checkpoint5_sub = self.create_subscription(GhostObjectState, '/ambf/env/phantom/checkpoint5/State', self.checkpoint5_callback, 1)
        self.end_trigger_sub = self.create_subscription(GhostObjectState, '/ambf/env/phantom/end_trigger/State', self.end_trigger_callback, 1)

        self.ring_contact_sensor = []
        self.ring_contact_sensor_sub = self.create_subscription(ContactSensorState, '/ambf/env/phantom/ring_contact_sensor/State', self.ring_contact_sensor_callback, 1)

        self.psm1_grasp_sub = self.create_subscription( ActuatorCmd, "/ambf/env/ghosts/psm1/Actuator0/Command", self.grasp_psm1_callback, 1)
        self.psm2_grasp_sub = self.create_subscription( ActuatorCmd, "/ambf/env/ghosts/psm2/Actuator0/Command", self.grasp_psm2_callback, 1)

        # Subscribe to the coagulation pedal topic
        self.coag_sub = self.create_subscription(Joy, '/console1/operator_present', self.coag_callback, 1)

        # Subscribe to the wire and ring state topics
        self.wire_sub = self.create_subscription(RigidBodyState,'/ambf/env/phantom/wire_visual/State',self.wire_pose_callback,1)
        self.ring_sub = self.create_subscription(RigidBodyState,'/ambf/env/phantom/ring_visual/State',self.ring_pose_callback,1)
        self.camera_sub = self.create_subscription(RigidBodyState,'/ambf/env/phantom/CameraFrame/State',self.camera_pose_callback,1)
        
        # Create publishers to mtmL and mtmR servo channels
        self.wrench_pub_L = self.create_publisher(WrenchStamped, '/MTML/body/servo_cf', 1)
        self.wrench_pub_R = self.create_publisher(WrenchStamped, '/MTMR/body/servo_cf', 1)

        # Create orientation-absolute flag publishers
        self.orientation_abs_pub_L = self.create_publisher(Bool, '/MTML/body/set_cf_orientation_absolute', 1) 
        self.orientation_abs_pub_R = self.create_publisher(Bool, '/MTMR/body/set_cf_orientation_absolute', 1)

        self.twist_sub_L = self.create_subscription(TwistStamped, '/MTML/measured_cv', self.twist_callback_L, 1)
        self.twist_sub_R = self.create_subscription(TwistStamped, '/MTMR/measured_cv', self.twist_callback_R, 1)

        # Publish the absolute oriention flag once
        abs_flag = Bool()
        abs_flag.data = True
        self.orientation_abs_pub_L.publish(abs_flag)
        self.orientation_abs_pub_R.publish(abs_flag)   

    def setup_socket_connection(self):
        host_ip = "10.162.34.171" # Linux desktop IP address
        port = 6400  # COBI Studio default port

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # allows port to be reused immediately after the program exits
        server_socket.bind((host_ip, port))
        server_socket.listen(1)
        self.client_socket, client_address = server_socket.accept()
    
    def ring_contact_sensor_callback(self, msg):
        # Initialize an empty list to store the names
        sensed_objects = []
        
        # Check if contact_events has any items and loop through them
        if msg.contact_events:
            for event in msg.contact_events:
                # Extract the name as a clean string and add it to our list
                object_name_str = event.object_name.data
                sensed_objects.append(object_name_str)     
        # Save the final list of strings to your class variable
        self.ring_contact_sensor = sensed_objects
       
    def start_trigger_callback(self, msg):
        self.start_trigger = msg.sensed_objects
    
    def checkpoint1_callback(self, msg):
        self.checkpoint1  = msg.sensed_objects
    
    def checkpoint2_callback(self, msg):
        self.checkpoint2  = msg.sensed_objects
    
    def checkpoint3_callback(self, msg):
        self.checkpoint3  = msg.sensed_objects
    
    def checkpoint4_callback(self, msg):
        self.checkpoint4  = msg.sensed_objects
    
    def checkpoint5_callback(self, msg):
        self.checkpoint5  = msg.sensed_objects
    
    def end_trigger_callback(self, msg):
        self.end_trigger = msg.sensed_objects

    def grasp_psm1_callback(self, msg):
        self.ring_grasped_psm1 = msg.actuate 

    def grasp_psm2_callback(self, msg):
        self.ring_grasped_psm2 = msg.actuate  

    def twist_callback_L(self, msg_twist_L):
        self.latest_twist_L = msg_twist_L

    def twist_callback_R(self, msg_twist_R):
        self.latest_twist_R = msg_twist_R

    def coag_callback(self, msg):
        # Updates if button is pressed or not
        self.coag_pressed = msg.buttons[0] # index 0 is specific button that maps to caog control
    
    def camera_pose_callback(self, msg_camera):
        camera_pos = PyKDL.Vector(msg_camera.pose.position.x, msg_camera.pose.position.y, msg_camera.pose.position.z)
        camera_rot = PyKDL.Rotation.Quaternion(msg_camera.pose.orientation.x, msg_camera.pose.orientation.y, msg_camera.pose.orientation.z, msg_camera.pose.orientation.w)
        self.latest_T_camera_world = PyKDL.Frame(camera_rot, camera_pos) # utils src can import function
    
    def wire_pose_callback(self, msg_wire):
        #Transform matrix for wire relative to ambf world
        wire_pos=PyKDL.Vector(msg_wire.pose.position.x, msg_wire.pose.position.y, msg_wire.pose.position.z)
        wire_rot=PyKDL.Rotation.Quaternion(msg_wire.pose.orientation.x, msg_wire.pose.orientation.y, msg_wire.pose.orientation.z, msg_wire.pose.orientation.w) # PyKDL uses xyzw for quaternion
        self.latest_T_wire_world=PyKDL.Frame(wire_rot, wire_pos)
    
    def ring_pose_callback(self, msg_ring):
        self.latest_ring_msg = msg_ring
    

    def get_ring_frame_in_wire(self, msg_ring):
        # grab latest ring msg
        msg_ring = self.latest_ring_msg
        
        # Extract the live ring center of mass relative to the world frame from the incoming ROS message
        ring_pos = PyKDL.Vector(msg_ring.pose.position.x, msg_ring.pose.position.y, msg_ring.pose.position.z)
        ring_rot = PyKDL.Rotation.Quaternion(msg_ring.pose.orientation.x, msg_ring.pose.orientation.y, msg_ring.pose.orientation.z, msg_ring.pose.orientation.w)
        T_ring_world = PyKDL.Frame(ring_rot, ring_pos)
        
        # Transform matrix for ring rlative to wire
        T_ring_wire = self.latest_T_wire_world.Inverse() * T_ring_world
        return T_ring_wire


    def get_closest_wire_point(self, ring_com): 
        # USE MINIMIZATION ON THE 3 CLOSEST SEGMENTS TO SAVE CPU TIME
        # Score every segment based on its closest individual control point
        scored_segments = []
        for segment in self.my_total_curve:
            # Find the absolute closest control point in this specific segment
            p0_dist = np.linalg.norm(np.array(segment[0]) - ring_com)
            p1_dist = np.linalg.norm(np.array(segment[1]) - ring_com)
            p2_dist = np.linalg.norm(np.array(segment[2]) - ring_com)
            p3_dist = np.linalg.norm(np.array(segment[3]) - ring_com)
    
            min_cp_dist = min(p0_dist, p1_dist, p2_dist, p3_dist)
            scored_segments.append((min_cp_dist, segment))

        # Sort them so the segments with the closest control points are first
        scored_segments.sort(key=lambda x: x[0])

        # Take only the top 3 best candidate segments to test with SciPy
        top_candidates = scored_segments[:3]

        final_result = OptimizeResult(fun=float('inf'))
        winning_segment_points = None

        # Run the optimizer only on these 3 candidate segments
        for min_dist, segment in top_candidates:
            P0, P1, P2, P3 = map(np.array, segment)
            
            result = minimize_scalar(
                distance_objective,
                bounds=(0.0, 1.0),
                method='bounded',
                args=(ring_com, P0, P1, P2, P3)
            )
            #print(f"result in loop: {result}")
            if result.fun < final_result.fun:
                final_result = result
                winning_segment_points = (P0, P1, P2, P3)
        
                
        # EXTRACT LIVE RESULTS
        closest_t = final_result.x
        #print(f"t value: {final_result.x}")
        min_distance = final_result.fun
        closest_wire_point = get_bezier_point(closest_t, *winning_segment_points)
        
        # Print out your live tracking coordinates to the terminal window
        #self.get_logger().info(f"Shortest Distance: {min_distance:.4f}m | Wire XYZ: {closest_wire_point}")
        #print(f"Shortest Distance: {min_distance:.4f}m | Wire XYZ: {closest_wire_point}")

        return closest_t, min_distance, closest_wire_point, winning_segment_points
        
    def compute_rotational_error(self,closest_t, T_ring_wire, winning_segment_points):
        # Calculate the tangent vector (derivative) at your closest_t
        P0, P1, P2, P3 = winning_segment_points
         
        # Calculate the tangent vector (derivative) at your closest_t
        P0, P1, P2, P3 = winning_segment_points
        tangent = (3 * (1 - closest_t)**2 * (P1 - P0) + 
                6 * (1 - closest_t) * closest_t * (P2 - P1) + 
                3 * closest_t**2 * (P3 - P2)) # this should be a function since its the same excpet for one variable, to make it go faster

        # Convert tangent into a unit vector
        u_tangent = tangent / np.linalg.norm(tangent)

        # Extract the Ring's Z-axis unit vector from its KDL rotation matrix
        # In PyKDL, Frame.M.UnitZ() gives the local Z axis vector relative to the wire frame
        u_ring_z = np.array([T_ring_wire.M.UnitZ().x(), 
                            T_ring_wire.M.UnitZ().y(), 
                            T_ring_wire.M.UnitZ().z()])

        # Calculate the angular error
        dot_product = np.dot(u_tangent, u_ring_z)

        # Use absolute value if direction/flipping doesn't matter
        #dot_product_val = abs(dot_product) 

        clipped_dot = np.clip(dot_product, -1.0, 1.0) # Clipping to avoid floating-point math errors outside [-1, 1]
        angular_error_rad = np.arccos(clipped_dot)
        angular_error_deg = np.degrees(angular_error_rad)

        # Log new error metrics
        #self.get_logger().info(f"Rotational Erro$ T_camera_worldr: {angular_error_deg:.2f}°")

        # break down rotaional into which direction
        return angular_error_deg, u_tangent, u_ring_z, dot_product  


    def compute_linear_force(self, min_distance, closest_wire_point, ring_com, kp_pos, linear_deadband):
        linear_error = min_distance
        # Initialize force and unit vector to zero in case the error is within the deadband
        f_linear = np.array([0.0, 0.0, 0.0])
        u_vector_ring_to_wire = np.array([0.0, 0.0, 0.0])
        
        if linear_error > linear_deadband:
            vector_ring_to_wire = closest_wire_point - ring_com  
            u_vector_ring_to_wire = vector_ring_to_wire / np.linalg.norm(vector_ring_to_wire)
            
            effective_linear_error = linear_error - linear_deadband
            f_linear = kp_pos * effective_linear_error * u_vector_ring_to_wire # keeping u_vector positive causes convergent force
        return f_linear, u_vector_ring_to_wire  # f_linear is a numpy array realtive to wire
    
    def compute_linear_damping_L(self, kd_pos,u_vector_ring_to_wire): 
        mtmL_vel = np.array([self.latest_twist_L.twist.linear.x, self.latest_twist_L.twist.linear.y, self.latest_twist_L.twist.linear.z])
        # Project MTML velocity onto the pull direction
        vel_into_wall = np.dot(mtmL_vel, u_vector_ring_to_wire)
        f_linear_damping_L = kd_pos * vel_into_wall * u_vector_ring_to_wire
        return f_linear_damping_L # is a numpy array realtive to wire
    
    def compute_linear_damping_R(self, kd_pos, u_vector_ring_to_wire):
        mtmR_vel = np.array([self.latest_twist_R.twist.linear.x, self.latest_twist_R.twist.linear.y, self.latest_twist_R.twist.linear.z])
        # Project MTMR velocity onto the pull direction
        vel_into_wall = np.dot(mtmR_vel, u_vector_ring_to_wire)
        f_linear_damping_R = kd_pos * vel_into_wall * u_vector_ring_to_wire
        return f_linear_damping_R # is a numpy array realtive to wire

    def compute_torque(self, angular_error_deg, u_tangent, u_ring_z, dot_product, kp_rot, angular_deadband):
        if angular_error_deg < angular_deadband:
            torque_angular = np.array([0.0, 0.0, 0.0])
        else:
            # Calculate the axis of rotation for the angular error
            u_ring_z_aligned = -u_ring_z if dot_product < 0 else u_ring_z
            rotation_axis = np.cross(u_tangent, u_ring_z_aligned)
            if np.linalg.norm(rotation_axis) > 1e-6:  # safety check - cross product with zero angle is zero vector, norm of this would cause dividing by zero
                u_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            else:
                u_rotation_axis = np.array([0.0, 0.0, 0.0])  # No meaningful rotation axis

            effective_angular_error_rad = np.radians(angular_error_deg - angular_deadband)
            torque_angular = kp_rot * effective_angular_error_rad * u_rotation_axis # torque is in direction of error
        return torque_angular # is a numpy array realtive to wire

    def compute_torque_damping_L(self, kd_rot):
        mtmL_ang_vel = np.array([self.latest_twist_L.twist.angular.x, self.latest_twist_L.twist.angular.y, self.latest_twist_L.twist.angular.z])
        torque_damping_L = kd_rot * mtmL_ang_vel # torque in same direction as mtmL
        return torque_damping_L # is a numpy array realtive to wire

    def compute_torque_damping_R(self, kd_rot):
        mtmR_ang_vel = np.array([self.latest_twist_R.twist.angular.x, self.latest_twist_R.twist.angular.y, self.latest_twist_R.twist.angular.z])
        torque_damping_R = kd_rot * mtmR_ang_vel # torque in same direction as mtmR
        return torque_damping_R # is a numpy array realtive to wire
   
    def transform_and_publish_wrench(self, max_force, max_torque, f_total_linear_L, f_total_linear_R, torque_total_L, torque_total_R): 
        # if coag is pressed then continue
        if not self.coag_pressed:
            return

        # Rotation from wire frame to camera frame: wire -> world -> camera
        R_wire_to_camera = self.latest_T_camera_world.M.Inverse() * self.latest_T_wire_world.M
        # Include correction for mtm console tilt
        T_baseoffset = PyKDL.Frame(PyKDL.Rotation.RPY((3.14 - 0.8) / 2, 0, 0), PyKDL.Vector(0, 0, 0))
        
        def to_camera_frame(f_L_or_R):
            f_vec = PyKDL.Vector(f_L_or_R[0], f_L_or_R[1], f_L_or_R[2])
            f_cam = T_baseoffset.M * (R_wire_to_camera * f_vec)
            return f_cam
        
        f_total_L_cam = to_camera_frame(f_total_linear_L)
        f_total_R_cam = to_camera_frame(f_total_linear_R)

        torque_total_L_cam = to_camera_frame(torque_total_L)
        torque_total_R_cam = to_camera_frame(torque_total_R)
        
        def build_wrench(f_cam, t_cam):
            wrench_msg = WrenchStamped()
            wrench_msg.wrench.force.x = float(np.clip(f_cam.x(), -max_force, max_force)) # WrenchStamped expects float, but np.clip returns a numpy scalar
            wrench_msg.wrench.force.y = float(np.clip(f_cam.y(), -max_force, max_force))
            wrench_msg.wrench.force.z = float(np.clip(f_cam.z(), -max_force, max_force))
            wrench_msg.wrench.torque.x = float(np.clip(t_cam.x(), -max_torque, max_torque))
            wrench_msg.wrench.torque.y = float(np.clip(t_cam.y(), -max_torque, max_torque))
            wrench_msg.wrench.torque.z = float(np.clip(t_cam.z(), -max_torque, max_torque))
            return wrench_msg

        # Send your wrenchstamped to servo channels
        zero_wrench = build_wrench(PyKDL.Vector(0.0, 0.0, 0.0), PyKDL.Vector(0.0, 0.0, 0.0))
        
        if self.ring_grasped_psm1:
            self.wrench_pub_L.publish(build_wrench(f_total_L_cam, torque_total_L_cam))
        else:
            self.wrench_pub_L.publish(zero_wrench)

        # if self.ring_grasped_psm2:
        #     self.wrench_pub_R.publish(build_wrench(f_total_R_cam, torque_total_R_cam))
        # else:
        #     self.wrench_pub_R.publish(zero_wrench)
         

   # have another yaml file for contactsensor on ring, look at email adnan sent u, can make the cobi start running and stop      
# ---------------------- COBI Flags ----------------------
    def task_started(self, min_distance): # call this in control loop after the arguments are defined
        start_ring_sensed = any("ring" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.start_trigger)
        if start_ring_sensed and (self.ring_grasped_psm1 or self.ring_grasped_psm2): #and min_distance < 0.005 # check thresholds
            return True
        return False  

    def checkpoint1_passed(self, min_distance):
        checkpoint1_sensed = any("ring" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.checkpoint1)
        if checkpoint1_sensed and self.start_flag_sent: #and min_distance < 0.005: 
            return True
        return False

    def checkpoint2_passed(self, min_distance):
        checkpoint2_sensed = any("ring" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.checkpoint2)
        if checkpoint2_sensed and self.start_flag_sent: #and min_distance < 0.005: 
            return True
        return False

    def checkpoint3_passed(self, min_distance):
        checkpoint3_sensed = any("ring" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.checkpoint3)
        if checkpoint3_sensed and self.start_flag_sent: #and min_distance < 0.005: 
            return True
        return False
    
    def checkpoint4_passed(self, min_distance):
        checkpoint4_sensed = any("ring" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.checkpoint4)
        if checkpoint4_sensed and self.start_flag_sent: # and min_distance < 0.005: 
            return True
        return False

    def checkpoint5_passed(self, min_distance):
        checkpoint5_sensed = any("ring" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.checkpoint5)
        if checkpoint5_sensed and self.start_flag_sent: #and min_distance < 0.005:
            return True
        return False

    def wire_touched(self, min_distance):
        touching_wire = any("wire" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.ring_contact_sensor)
        if not touching_wire:
            self.previous_not_touched = True
        if self.previous_not_touched and touching_wire and (self.ring_grasped_psm1 or self.ring_grasped_psm2) and self.start_flag_sent and not self.end_flag_sent:
            self.previous_not_touched = False
            return True
        return False  
    
    def ring_dropped(self, min_distance):
        if self.ring_grasped_psm1 or self.ring_grasped_psm2:
            self.ring_was_held = True
        if self.ring_was_held and not (self.ring_grasped_psm1 or self.ring_grasped_psm2) and self.start_flag_sent and not self.end_flag_sent:
            self.ring_was_held = False
            return True
        return False  
    
    def task_ended(self, min_distance):
        end_ring_sensed = any("ring" in (obj.data if hasattr(obj, 'data') else str(obj)) for obj in self.end_trigger)
        if end_ring_sensed and self.start_flag_sent: #and min_distance < 0.005 # check thresholds, should they have to hold ring as passing ending, defin threshold at top of class and make capital
            return True
        return False

# should go in control loop:
# if self.task_started() and self.start_flag_sent:
#     client_socket.sendall(bytes(0))
#     self.start_flag_sent = False  # command to permannetly set to true
#     print("Task started")
# if self.wire_touched():
#     client_socket.sendall(bytes(1))
#     print("Wire touched")
# if self.ring_dropped():
#     client_socket.sendall(bytes(2))
#     print("Ring dropped")
# if self.task_ended():
#     client_socket.sendall(bytes(3))
#     print("Task ended")

# ---------------------- Control Loop ----------------------
    def control_loop(self):
        max_force = 2.0 # N
        max_torque = 0.05 # N·m
        kp_pos = 120  # Spring constant for position (N/m)
        kd_pos = 3  # Damping constant for velocity (N/(m/s))
        kp_rot = 0.1  # Spring constant for rotation (N·m/rad)
        kd_rot = 0.0  # Damping constant for angular velocity (N·m/(rad/s))
        linear_deadband = 0.0001 # meters, distance from wire centerline where no force is applied
        angular_deadband = 0  # degrees, angle from wire tangent where no force is applied
        
        if (self.latest_ring_msg is None or
            self.latest_T_wire_world is None or
            self.latest_T_camera_world is None or
            self.latest_twist_L is None or
            self.latest_twist_R is None):
            return
        ring_msg = self.latest_ring_msg
        T_ring_wire = self.get_ring_frame_in_wire(ring_msg)
        ring_com = np.array([T_ring_wire.p.x(), T_ring_wire.p.y(), T_ring_wire.p.z()])
        closest_t, min_distance, closest_wire_point, winning_segment_points = self.get_closest_wire_point(ring_com)
        if min_distance > 0.015:
            #print("Force feedback disabled, ring has come off the wire")
            return
        f_linear, u_vector_ring_to_wire = self.compute_linear_force(min_distance, closest_wire_point, ring_com, kp_pos, linear_deadband)
        f_linear_damping_L = self.compute_linear_damping_L(kd_pos, u_vector_ring_to_wire)
        f_linear_damping_R = self.compute_linear_damping_R(kd_pos, u_vector_ring_to_wire)
        f_total_linear_L = f_linear - f_linear_damping_L
        f_total_linear_R = f_linear - f_linear_damping_R
        
        angular_error_deg, u_tangent, u_ring_z, dot_product = self.compute_rotational_error(closest_t, T_ring_wire, winning_segment_points)
        torque_angular = self.compute_torque(angular_error_deg, u_tangent, u_ring_z, dot_product, kp_rot, angular_deadband)
        torque_damping_L = self.compute_torque_damping_L(kd_rot)
        torque_damping_R = self.compute_torque_damping_R(kd_rot)
        torque_total_L = -torque_angular - torque_damping_L # negative torque angular to resist error rotation, negative dmaping roates in opposite direction of mtm
        torque_total_R = -torque_angular - torque_damping_R # negative torque angular to resist error rotation

        #self.transform_and_publish_wrench(max_force, max_torque, f_total_linear_L, f_total_linear_R, torque_total_L, torque_total_R)  # Publish the total linear force to both MTMs
    
        if self.task_started(min_distance) and not self.start_flag_sent:
            self.start_flag_sent = True
            self.client_socket.sendall(bytes([0])) 
            print("Task Started")
        if self.checkpoint1_passed(min_distance) and not self.checkpoint1_sent: # can make list of these checkpoint and run through them
            self.checkpoint1_sent = True
            self.client_socket.sendall(bytes([1]))
            print("Passed checkpoint 1")
        if self.checkpoint2_passed(min_distance) and not self.checkpoint2_sent:
            self.checkpoint2_sent = True
            self.client_socket.sendall(bytes([2]))
            print("Passed checkpoint 2")
        if self.checkpoint3_passed(min_distance) and not self.checkpoint3_sent:
            self.checkpoint3_sent = True
            self.client_socket.sendall(bytes([3]))
            print("Passed checkpoint 3")
        if self.checkpoint4_passed(min_distance) and not self.checkpoint4_sent:
            self.checkpoint4_sent = True
            self.client_socket.sendall(bytes([4]))
            print("Passed checkpoint 4")
        if self.checkpoint5_passed(min_distance) and not self.checkpoint5_sent:
            self.checkpoint5_sent = True
            self.client_socket.sendall(bytes([5]))
            print("Passed checkpoint 5")
        if self.wire_touched(min_distance):
            self.count += 1
            self.client_socket.sendall(bytes([6]))
            print(f"Wire Touched {self.count}")
        if self.ring_dropped(min_distance): # test on dvrk if this sends multiple messages during one action of dropping
            self.client_socket.sendall(bytes([7]))
            print("Ring Dropped")
        if self.task_ended(min_distance) and not self.end_flag_sent:
            self.end_flag_sent = True
            self.client_socket.sendall(bytes([8]))
            print("Task Ended")

    def run_control_loop(self):
        t1 = datetime.now()
        rate_hz = 200 # number of updates per second
        period = 1.0 / rate_hz # wait this many seconds before each update
        while rclpy.ok(): # returns true as long as ros2 is still running and hasn't been interrupted by ctrl c
            self.control_loop()
            time.sleep(period) # limits updates to realistically set frequency
            t2 = datetime.now()
            #print("\t ***Delta T (seconds):", (t2 - t1).total_seconds())
            t1 = t2 


def main(args=None):
    rclpy.init(args=args)
    tracker = WireTrackerNode()
    
    try:
        rclpy.spin(tracker) # create a single thread executor, add this to it, and spin it to keep the node alive
    except KeyboardInterrupt:
        pass
    finally:
        tracker.destroy_node()
        rclpy.shutdown()
        self.client_socket.close()
        self.server_socket.close()


if __name__ == '__main__':
    main()



# # --------Calculating force feedback resistive--------- should go inside ring callback
# ring_inner_radius = 0.0  # meters
# wire_radius = 0.0  # meters
# threshold_distance = ring_inner_radius - wire_radius  # meters
# kp_pos = 0  # Spring constant for position (N/m)
# kd_pos = 0    # Damping constant for velocity (N/(m/s))


# if min_distance < threshold_distance:
#     f_spring = np.array([0.0, 0.0, 0.0])
# else:
#     # find direction for force feedback
#     vector_ring_to_wire = wire_xyz - ring_com  # check if this direction is correct
#     u_vector_ring_to_wire = vector_ring_to_wire / np.linalg.norm(vector_ring_to_wire)

#     # how much force to apply based on mtm and psm position difference
#     mtm_xyz =  
#     psm_xyz = # read from teleop script
#     displacement_mag = np.linalg.norm(mtm_xyz - psm_xyz)

#     # Spring force based on displacement
#     f_spring = kp_pos * displacement_mag * u_vector_ring_to_wire 


#     # Damping force based on velocity difference
#     mtm_vel =
#     psm_vel = # read from teleop script
#     velocity_diff = mtm_vel - psm_vel
#     vel_into_wall = np.dot(velocity_diff, u_vector_ring_to_wire)
#     if vel_into_wall > 0:  # Only apply damping if moving into the wall
#         f_damping = kd_pos * vel_into_wall * u_vector_ring_to_wire
#     else:
#         f_damping = np.array([0.0, 0.0, 0.0])
    


    # add force max safety






    
   # maybe do any deviation is penalized but ask micahel if doing this or just normal rigid body constraint
   # put sensor on ring to know when ring com deviates from wire centerline, first make ghost object
   # make sure to convert from psm frame to camera frame then to mtm frame before applying force, look in teleop scripts wrench
   # ask if fnirs matlab can run on linux
   


   # 127 in mtm_teleop_comm - camera to mtm frame before sending wrench
   # could write script to move ring known distance and check your results, could have another class in same script 
   
   # transform from ambf world frame to camera (subscribe to camer coordinates) then send to mtm (already given)
    # make max buffer zone if ring fals off
    # remember only force when ring is held and when coag pedal is pressed

    # publish to this once, line 211 in mtm_device_crtk: set_wrench_orientation_absolute_topic_name = name + 'body/set_cf_orientation_absolute'
    # line 210 in mtm_device_crtk: wrench_pub_topic_name = name + 'body/servo_cf'
    # import from geometry.msg get WrenchStamped
    # use body/servo_cf and not spatial
    # publish to mtmR/servo_cf and mtmL/servo_cf

    # need to make force feedback only act on psm ring is touching

   
   # put calculation stuff outisde of callback in diff thread and set que size to 1, callbacks should be small, in beginning of method make a copy to store and use most recent ring info once
   # find way tp qauntify complexity of task like more sin waves in different planes, could have multiple trials of making it more complex
   # could first train on less complex curve then final test on more complex curve, could also quantify by number of sharp turns and number of flips
   # find way to measure complexity
   # ros interface on matlab if fnirs run on matlab then can use diff computer for fnirs and synchronize the ros collecting between both compouters
   # can cobi studio data format be in same structure so can avoid synchronizing
   





