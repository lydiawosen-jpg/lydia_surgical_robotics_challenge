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
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy

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
        json_path = os.path.expanduser("/mnt/c/Users/lydia/my_bezier_curve.json")
        with open(json_path, 'r') as f:
            self.my_total_curve = json.load(f)
        
        self.get_logger().info("Waiting for static wire pose from AMBF...")

        self.coag_pressed = False
        self.coag_sub = self.create_subscription(Joy, '/console1/operator_present', self.coag_callback, 1)
        
        # Start a background thread that calls control_loop() repeatedly
        self.control_thread = threading.Thread(ros2 run dvrk_robot dvrk_system -j system-MTML-MTMR.jsontarget=self.run_control_loop, daemon=True)
        self.control_thread.start()
        
        # Initialize the wire's world frame as None until we receive it
        self.latest.T_wire_world = None  
        self.latest.T_camera_world = None

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

        # Publish the absolute oriention flag once
        abs_flag = Bool()
        abs_flag.data = True
        self.orientation_abs_pub_L.publish(abs_flag)
        self.orientation_abs_pub_R.publish(abs_flag)
        
        self.get_logger().info("Successfully subscribed to wire and ring topics.")     


    def coag_callback(self):
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
        T_ring_wire = self.T_wire_world.Inverse() * T_ring_world
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
        print(f"t value: {final_result.x}")
        min_distance = final_result.fun
        closest_wire_point = get_bezier_point(closest_t, *winning_segment_points)
        
        # Print out your live tracking coordinates to the terminal window
        #self.get_logger().info(f"Shortest Distance: {min_distance:.4f}m | Wire XYZ: {closest_wire_point}")
        print(f"Shortest Distance: {min_distance:.4f}m | Wire XYZ: {closest_wire_point}")

        return closest_t, min_distance, closest_wire_point, winning_segment_points
        
    # def compute_rotational_error(self,closest_t, T_ring_wire, winning_segment_points):
        
    #     # 1. Calculate the tangent vector (derivative) at your closest_t
    #     P0, P1, P2, P3 = winning_segment_points
    #     tangent = (3 * (1 - closest_t)**2 * (P1 - P0) + 
    #             6 * (1 - closest_t) * closest_t * (P2 - P1) + 
    #             3 * closest_t**2 * (P3 - P2)) # this should be a function since its the same excpet for one variable, to make it go faster

    #     # 2. Convert tangent into a unit vector
    #     u_tangent = tangent / np.linalg.norm(tangent)

    #     # 3. Extract the Ring's Z-axis unit vector from its KDL rotation matrix
    #     # In PyKDL, Frame.M.UnitZ() gives the local Z axis vector relative to the wire frame
    #     u_ring_z = np.array([T_ring_wire.M.UnitZ().x(), 
    #                         T_ring_wire.M.UnitZ().y(), 
    #                         T_ring_wire.M.UnitZ().z()])

    #     # 4. Calculate the angular error
    #     dot_product = np.dot(u_tangent, u_ring_z)

    #     # Use absolute value if direction/flipping doesn't matter
    #     dot_product_val = abs(dot_product) 

    #     clipped_dot = np.clip(dot_product_val, -1.0, 1.0) # Clipping to avoid floating-point math errors outside [-1, 1]
    #     angular_error_rad = np.arccos(clipped_dot)
    #     angular_error_deg = np.degrees(angular_error_rad)

    #     # 5. Log your new error metrics
    #     self.get_logger().info(f"Rotational Error: {angular_error_deg:.2f}°")

    #     # break down rotaional into which direction
    #     return angular_error_deg, u_tangent, u_ring_z


    def compute_radial_force(self, min_distance, closest_wire_point, ring_com):
        kp_pos = 50  # Spring constant for position (N/m)
        kd_pos = 0  # Damping constant for velocity (N/(m/s))
        kp_rot = 0  # Spring constant for rotation (N·m/°) # is this normally in radians?
        kd_rot = 0  # Damping constant for angular velocity (N·m/(°/s))

        radial_deadband = 0.005 # meters, distance from wire centerline where no force is applied
        radial_error = min_distance

        angular_deadband = 5.0 # degrees, angle from wire tangent where no force is applied

        # Translational force feedback calculation
        if radial_error <= radial_deadband:
            f_radial = np.array([0.0, 0.0, 0.0])
        else:
            vector_ring_to_wire = closest_wire_point - ring_com  
            u_vector_ring_to_wire = vector_ring_to_wire / np.linalg.norm(vector_ring_to_wire)
            
            effective_radial_error = radial_error - radial_deadband
            f_radial = kp_pos * effective_radial_error * u_vector_ring_to_wire
        return f_radial # this is a numpy array

    def transform_and_publish_wrench(self, f_radial): 
        # if coag is pressed then continue
        if not self.coag_pressed:
            return
        
        max_force = 2.0
        
        # Build a PyKDL Vector from radial force, only rotate (don't translate) a force vector
        f_radial_vec = PyKDL.Vector(f_radial[0], f_radial[1], f_radial[2])

        # Rotation from wire frame to camera frame: wire -> world -> camera
        R_wire_to_camera = self.T_camera_world.M.Inverse() * self.T_wire_world.M

        f_radial_camera = R_wire_to_camera * f_radial_vec
        
        # Include correction for mtm console tilt
        T_baseoffset = PyKDL.Frame(PyKDL.Rotation.RPY((3.14 - 0.8) / 2, 0, 0), PyKDL.Vector(0, 0, 0))
        f_radial_camera = T_baseoffset.M * f_radial_camera
        

        # Create wrenchstamped message type and fill it in
        msg_f_radial = WrenchStamped()
        msg_f_radial.wrench.force.x = np.clip(f_radial_camera.x(), -max_force, max_force)
        msg_f_radial.wrench.force.y = np.clip(f_radial_camera.y(), -max_force, max_force)
        msg_f_radial.wrench.force.z = np.clip(f_radial_camera.z(), -max_force, max_force)
        msg_f_radial.wrench.torque.x = 0.0
        msg_f_radial.wrench.torque.y = 0.0
        msg_f_radial.wrench.torque.z = 0.0

        # Send your wrenchstamped to servo channels
        self.wrench_pub_L.publish(msg_f_radial)
        self.wrench_pub_R.publish(msg_f_radial)

    def control_loop(self):
        if self.latest_ring_msg is None or self.latest_T_wire_world is None or self.latest_T_camera_world is None:
            return
        msg_ring = self.latest_ring_msg
        T_ring_wire = self.get_ring_frame_in_wire(msg_ring)
        ring_com = np.array([T_ring_wire.p.x(), T_ring_wire.p.y(), T_ring_wire.p.z()])
        closest_t, min_distance, closest_wire_point, winning_segment_points = self.get_closest_wire_point(ring_com)
        if min_distance > 0.015:
            print("Force feedback disabled, ring has come off the wire")
            return
        f_radial = self.compute_radial_force(min_distance, closest_wire_point, ring_com)
        self.transform_and_publish_wrench(f_radial)

    def run_control_loop(self):
        rate_hz = 200
        period = 1.0 / rate_hz
        while rclpy.ok():
            self.control_loop()
            time.sleep(period)


def main(args=None):
    rclpy.init(args=args)
    tracker = WireTrackerNode()
    
    # Use a MultiThreadedExecutor so callbacks can run independently
    executor = MultiThreadedExecutor()
    executor.add_node(tracker)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.destroy_node()
        rclpy.shutdown()


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

   
   # put calculation stuff outisde of callback in diff thread and set que size to 1, callbacks should be small, in beginning of method make a copy to store and use most recent ring info once
   # find way tp qauntify complexity of task like more sin waves in different planes, could have multiple trials of making it more complex
   # could first train on less complex curve then final test on more complex curve, could also quantify by number of sharp turns and number of flips
   # find way to measure complexity
   # ros interface on matlab if fnirs run on matlab then can use diff computer for fnirs and synchronize the ros collecting between both compouters
   # can cobi studio data format be in same structure so can avoid synchronizing
   





    # --------Rotational force feedback calculation---------
        # if angular_error_deg < angular_deadband:
        #     torque_angular = np.array([0.0, 0.0, 0.0])
        # else:
        #     # Calculate the axis of rotation for the angular error
        #     u_ring_z_aligned = -u_ring_z if dot_product < 0 else u_ring_z
        #     rotation_axis = np.cross(u_tangent, u_ring_z_aligned)
        #     if np.linalg.norm(rotation_axis) > 1e-6:  # safety check - cross product with zero angle is zero vector, norm of this would cause dividing by zero
        #         u_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        #     else:
        #         u_rotation_axis = np.array([0.0, 0.0, 0.0])  # No meaningful rotation axis

        #     effective_angular_error_rad = np.radians(angular_error_deg - angular_deadband)
        #     torque_angular = kp_rot * effective_angular_error_rad * u_rotation_axis