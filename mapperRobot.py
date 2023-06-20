# Name: Phuc Tran
# Date: May 9th, 2023 

# Python Libraries
import math
import numpy

# ROS Libraries
import rospy 
import tf 
from geometry_msgs.msg import Twist, Pose, Quaternion, PoseArray # message type for cmd_vel
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid # message type  

# Constant 
# Topic Names
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel' 
DEFAULT_LASER_SCAN_TOPIC = 'base_scan'
DEFAULT_MAP_TOPIC = 'map'
DEFAULT_POSE_SEQUENCE_TOPIC = 'pose_sequence'

# Publishing Frequency 
FREQUENCY = 10 # Hz 

# Mapping Constants 
MINIMUM_DIST_THRESHOLD = 1 # units
WIDTH = 200 
HEIGHT = 200
RESOLUTION = 0.05 

# Transform Frames 
MAP_FRAME = 'map'
ODOM_FRAME = 'odom'
BASE_LINK_FRAME = 'base_link'

# Velocities that will be used 
LINEAR_VELOCITY = 0.2 # m/s 
ANGULAR_VELOCITY = 0  # m/s 
# PD Constants for Ziegler-Nichols heuristics
KU = 10
PROPORTIONAL_GAIN = 0.8 * KU
DERIVATIVE_GAIN = 60
INTEGRAL_GAIN = 0

# Field of view for the right side of the robot and the front
IN_FRONT_RAD_LEFT = math.pi / 16
IN_FRONT_RAD_RIGHT = -1 * math.pi / 16

MIN_SCAN_ANGLE_RAD = -1 * math.pi/4
MAX_SCAN_ANGLE_RAD = -1 * 3 * math.pi/4
TARGET = 0.3

class MapperRobot:
    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity = ANGULAR_VELOCITY, proportional_gain = PROPORTIONAL_GAIN,
                 scan_angle = [MIN_SCAN_ANGLE_RAD, MAX_SCAN_ANGLE_RAD],
                 obstacle_angle = [IN_FRONT_RAD_LEFT, IN_FRONT_RAD_RIGHT], target = TARGET,
                 derivative_gain = DERIVATIVE_GAIN, integral_gain = INTEGRAL_GAIN):
        """Constructor."""

        # Initializing publisher/subscriber
        self._cmd_pub = rospy.Publisher(DEFAULT_CMD_VEL_TOPIC, Twist, queue_size = 1)
        self._poses_pub = rospy.Publisher(DEFAULT_POSE_SEQUENCE_TOPIC, PoseArray, queue_size = 1)
        self._laser_sub = rospy.Subscriber(DEFAULT_LASER_SCAN_TOPIC, LaserScan, self._laser_callback, queue_size = 1)
        self._map_pub = rospy.Publisher(DEFAULT_MAP_TOPIC, OccupancyGrid, queue_size = 1)
        self.transformListener  = tf.TransformListener()

        # LaserScan variables 
        self.angle_increment = None
        self.ranges = None
        self.map_msg = OccupancyGrid
        self.rescaledPositions = None

        # Ray Tracing variables
        self.robot_point = None

        # PID Controller Constants 
        # Parameters
        self.linear_velocity = linear_velocity 
        self.angular_velocity = angular_velocity

        # Controller constants
        self.proportional_gain = float(proportional_gain)
        self.derivative_gain = float(derivative_gain)
        self.integral_gain = float(integral_gain)
        self.Tu = 0.0

        # LaserScan constants
        self.scan_angle = scan_angle
        self.obstacle_angle = obstacle_angle
        self.in_front = float('inf')
        self.right = float('inf')
        self._in_front_flag = False
        self.max_range = None

        # Error calculations
        self.target = target                               # meters from the wall
        self.dt = 0.1                                      # 10Hz sleep = 0.1 seconds between each calculation
        self.errors = []
        self.periods = []
        self.time_increment = 0

    def move(self):
        """Send a velocity command according to the error."""
        twist_msg = Twist()
        twist_msg.linear.x = self.linear_velocity
        angular_velocity = 0
        
        # PID error accumulation
        if len(self.errors) >= 2 and self.Tu != 0:
            # the integrator and derivator time constant
            Ti = self.Tu / 2
            Td = self.Tu / 8

            proportional_error = self.proportional_gain * self.errors[-1] 
            derivative_error = self.derivative_gain * (self.errors[-1] - self.errors[-2]) / Td
            integrative_error = self.integral_gain * Ti * sum(self.errors)
            angular_velocity += proportional_error + derivative_error + integrative_error

        else:
            error = self.errors[-1] if len(self.errors) >= 1 else 0 
            angular_velocity += self.proportional_gain * error

        # Publish message
        twist_msg.angular.z = angular_velocity
        self._cmd_pub.publish(twist_msg)

    def rotate(self):
        """Send an angular velocity message to avoid front wall collisions."""
        rate = rospy.Rate(FREQUENCY)
        twist_msg = Twist()
        angular_velocity = 0.2 # rad/s 
        twist_msg.angular.z = angular_velocity
        while not rospy.is_shutdown():
            self._cmd_pub.publish(twist_msg)
            rate.sleep()
            if self.in_front > 0.4:
                break
        self.stop()
        self._in_front_flag = False
        rate.sleep()

    def _second_laser_callback(self, msg):
        """Measure the closest distance on the front and right side of the robot."""

        # Find the range of distances on the righr side of the robot
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        max_index = int((self.scan_angle[0] - angle_min) / angle_increment)
        min_index = int((self.scan_angle[1] - angle_min) / angle_increment)
        front_index_left = int((self.obstacle_angle[0] - angle_min) / angle_increment)
        front_index_right = int((self.obstacle_angle[1] - angle_min) / angle_increment)

        in_front_distances = msg.ranges[front_index_right:front_index_left]
        right_side_distances = msg.ranges[min_index:max_index]

        # Find the minimum of the distances as well as the front, which will be useful for corner cases
        self.right = min(right_side_distances)
        self.in_front = min(in_front_distances)
        self.errors.append(self.target - self.right)

        if (self.in_front < 0.4):
            self._in_front_flag = True 

        # To find the period between each oscillation by calculating whenever it's at the sweet spot
        if(abs(self.errors[-1]) < 0.01):
            if rospy.get_rostime().secs not in self.periods:
                self.periods.append(rospy.get_rostime().secs)
        if len(self.periods) > 1:
            self.Tu = float(2.0 * (self.periods[-1] - self.periods[-2]))
    
    def stop(self):
        """Stops the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def _laser_callback(self, msg):
        """Save the angle increment and ranges to index for potential obstacles."""
        self.angle_increment = msg.angle_increment 
        self.ranges = msg.ranges
        self.max_range = msg.range_max
        self._second_laser_callback(msg)

    def make_empty_map(self):
        """Creates an Empty Occupancy Grid to Update."""
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = MAP_FRAME

        # map's dimension
        map_msg.info.resolution = RESOLUTION
        map_msg.info.width = WIDTH
        map_msg.info.height = HEIGHT

        # IMPORTANT: The map's data is a 1-D Array
        map_msg.data = numpy.ones(WIDTH*HEIGHT) * -1
        self.map_msg = map_msg

    def publish_map(self):
        """Permanent loop to publish the map."""
        self._map_pub.publish(self.map_msg)

    def compute_ray_tracing(self, p1, p2):
        """Refer to: https://medium.com/geekculture/bresenhams-line-drawing-algorithm-2e0e953901b3."""
        """All pseudocode is referred to the medium article."""
        reverse = False
        negSlope = False 
        greaterThanOneSlope = False

        # If x1 > x2, swap values
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
            reverse = True

        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]

        # Short circuit
        if dx == 0:
            rayList = [p1]
            minimum = int(min(p1[1], dy + p1[1]))
            maximum = int(max(p1[1], dy + p1[1]))
            for i in range(minimum, maximum):
                rayList.append([p1[0], i])
            return rayList

        slope = float(dy)/float(dx)

        # Negative slope, and swap the values at the end
        if slope < 0:
            p1[1] *= -1 
            p2[1] *= -1 
            negSlope = True

        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        slope = float(dy)/float(dx)

        # Slope > 1, swap values
        if slope > 1:
            p1[0], p1[1] = p1[1], p1[0]
            p2[0], p2[1] = p2[1], p2[0]
            greaterThanOneSlope = True

        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        slope = float(dy)/float(dx)

        rayList = [p1]

        if slope >= 0 and 1 >= slope and dx > 0:
            cur_x = p1[0]
            cur_y = p1[1]

            while cur_x < p2[0]:
                
                # Initial decision variable
                if cur_x == p1[0]:
                    decision_variable = 2 * dy - dx 

                # The change in the E and NE direction
                deltaEast = 2*dy 
                deltaNorthEast = 2*(dy - dx)

                # If decision variable favors east, then we iterate east
                if decision_variable <= 0:
                    decision_variable += deltaEast
                else:
                    decision_variable += deltaNorthEast
                    cur_y += 1   

                cur_x += 1
                rayList.append([cur_x, cur_y])

        # Swap the x and y values back
        if greaterThanOneSlope:
            for ray in rayList:
                ray[0], ray[1] = ray[1], ray[0]

        # Swap the negative values of y back
        if negSlope:
            for ray in rayList:
                ray[1] *= -1

        if reverse: 
            rayList = list(reversed(rayList))

        return rayList
    
    def transform_local_to_global(self):
        """Transforms the local base_scan points to the global frame."""
        self.transformListener.waitForTransform(BASE_LINK_FRAME, MAP_FRAME, rospy.Time(0), rospy.Duration(0.2))

        self.rescaledPositions = [] 
        if self.localPositionList:
            for pose in self.localPositionList:
                if pose[0] == 0 and pose[1] == 0:
                    self.rescaledPositions.append([pose[0], pose[1]])
                else:
                    # Receive the transformation matrix from map frame to base link
                    (translation, quaternion) = self.transformListener.lookupTransform(BASE_LINK_FRAME, MAP_FRAME, rospy.Time(0))
                    t = tf.transformations.translation_matrix(translation)
                    r = tf.transformations.quaternion_matrix(quaternion)
                    map_T_bl =  t.dot(r)
                    bl_T_map = numpy.linalg.inv(map_T_bl)
                    bl_point = pose
                    map_point = bl_T_map.dot(numpy.array(bl_point))

                    # Check if the map has the data already occupied, ignore if the point has been scanned before
                    temp_map = numpy.reshape(self.map_msg.data.copy(), (HEIGHT, WIDTH))
                    if temp_map[int(map_point[1]), int(map_point[0])] == 100:
                        self.rescaledPositions.append([0, 0])
                    else:
                        self.rescaledPositions.append([map_point[0], map_point[1]])

        # Saves the points as PoseArray for RViz visualization purposes
        poses_msg = PoseArray()
        poses_msg.header.frame_id = MAP_FRAME
        poses = []
        for rescaledPosition in self.rescaledPositions:
            pose = Pose()
            pose.position.x = rescaledPosition[0]
            pose.position.y = rescaledPosition[1]
            pose.orientation.z = 0
            poses.append(pose)
        
        poses_msg.poses = poses
        self._poses_pub.publish(poses_msg)

    def mark_ray(self, reshaped_map, ray):
        """Mark the points between the LiDAR wall scan and the robot"""
        # Mark the wall
        if reshaped_map[int(ray[-1][1]) - 1, int(ray[-1][0]) - 1] != -1:
            reshaped_map[int(ray[-1][1]), int(ray[-1][0])] = 100

        # Mark the points between the wall and robot as free space
        for point in ray[:-1]:
            x, y = int(point[0]), int(point[1])
            reshaped_map[y, x] = 0
        
        # Reshape it back into a 1D array after the update
        self.map_msg.data = numpy.reshape(reshaped_map, (HEIGHT*WIDTH, 1))

    def update_map(self):
        """Update the Occupancy Grid."""
        self.transformListener.waitForTransform(BASE_LINK_FRAME, MAP_FRAME, rospy.Time(0), rospy.Duration(0.2))
        reshaped_map = numpy.reshape(self.map_msg.data, (HEIGHT, WIDTH))

        (translation, quaternion) = self.transformListener.lookupTransform(BASE_LINK_FRAME, MAP_FRAME, rospy.Time(0))
        t = tf.transformations.translation_matrix(translation)
        r = tf.transformations.quaternion_matrix(quaternion)
        map_T_bl =  t.dot(r)
        bl_T_map = numpy.linalg.inv(map_T_bl)
        while not rospy.is_shutdown():
            # points that the LiDAR gives as occupied space
            for point in self.rescaledPositions:
                if point[0] == 0 and point[1] == 0:
                    continue
                else:
                    # compute the ray using Bresenham's algorithm and mark the ray
                    new_point = numpy.array(point) // RESOLUTION
                    robot_pos = bl_T_map.dot(numpy.array([0, 0, 0, 1])) // RESOLUTION
                    robot_pos = [robot_pos[0], robot_pos[1]]
                    ray = self.compute_ray_tracing(robot_pos, new_point)
                    if len(ray) > 2:
                        self.mark_ray(reshaped_map, ray, new_point)
            break

    def find_local_position(self):
        """Finds the local position of the LiDAR message w.r.t. the robot's orientation."""
        self.localPositionList = [] 
        starting_theta = -math.pi
        curr = starting_theta
        for distance in self.ranges:
            x = distance * math.cos(curr)
            y = distance * math.sin(curr)
            if distance < MINIMUM_DIST_THRESHOLD:
                self.localPositionList.append([x, y, 0, 1])
            else: 
                self.localPositionList.append([0, 0, 0, 1])
            curr += self.angle_increment
        
    def spin(self):
        """Permanent loop."""
        rate = rospy.Rate(FREQUENCY)

        self.make_empty_map()
        while not rospy.is_shutdown():
            if self.ranges and self.angle_increment:
                self.find_local_position()
                self.transform_local_to_global()
                self.update_map()
                self.publish_map()
            if self._in_front_flag:
                self.rotate()
            else:
                self.move() 
            rate.sleep()


def main():
    # 1. initialize node
    rospy.init_node("MapperRobotNode")

    # sleep for some time 
    rospy.sleep(2)

    # 2. initialize class
    mapperRobot = MapperRobot()

    # If interrupted, send a stop command 
    rospy.on_shutdown(mapperRobot.stop)

    try:
        mapperRobot.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("what")
        rospy.logerr("ROS node interrupted")

if __name__ == "__main__":
    """Run the main function."""
    main()
