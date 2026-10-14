#!/usr/bin/env python3

class WireTrackerNode(Node):
    def __init__(self):
        super().__init__('wire_distance_tracker')
        self.wrench_pub_L = self.create_publisher(WrenchStamped, '/MTML/body/servo_cf', 1)


    def wrench(self):
        # Publish a zero wrench message to the MTML
        wrench_msg = WrenchStamped()
        wrench_msg.wrench.force.x = 0.0
        wrench_msg.wrench.force.y = 0.0
        wrench_msg.wrench.force.z = 0.0
        wrench_msg.wrench.torque.x = 0.0
        wrench_msg.wrench.torque.y = 0.0
        wrench_msg.wrench.torque.z = 0.0

        self.wrench_pub_L.publish(wrench_msg)
    
    def control_loop(self):
        self.wrench()

    def run_control_loop(self):
        t1 = datetime.now()
        rate_hz = 600 # number of updates per second
        period = 1.0 / rate_hz # wait this many seconds before each update
        while rclpy.ok(): # returns true as long as ros2 is still running and hasn't been interrupted by ctrl c
            self.control_loop()
            time.sleep(period)

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
        #self.client_socket.close()
        #self.server_socket.close()


if __name__ == '__main__':
    main()


