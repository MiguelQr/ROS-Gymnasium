import os
import time
import math
import cv2
import rclpy
import numpy as np

import matplotlib.pyplot as plt
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu

from cv_bridge import CvBridge

from rosgym.utils.env_utils import quat_to_euler, tf_decompose, euler_from_quaternion

import yaml


class LaserScanSensor:
    def __init__(self, max_dist, num_points, robot_type, robot_radius=None, robot_size=None, collision_tolerance=0.05):
        self.max_dist = max_dist
        self.num_points = num_points
        self.robot_type = robot_type
        self.robot_radius = robot_radius
        self.robot_size = robot_size
        self.collision_tolerance = collision_tolerance

        # Validate input dimensions
        if self.robot_type == "circular" and self.robot_radius is None:
            raise ValueError("Circular robot type requires a 'robot_radius' parameter.")
        if self.robot_type == "rectangular" and self.robot_size is None:
            raise ValueError("Rectangular robot type requires a 'robot_size' parameter as (length, width).")

        self.collision_vector = self.get_collision_vector()
        self.collision_threshold = 3

    def get_collision_vector(self):

        if self.robot_type == "circular":
            return np.full(360, self.robot_radius + self.collision_tolerance)

        elif self.robot_type == "rectangular":
            sL = self.robot_size[0] / 2 + self.collision_tolerance  # Length semi-axis
            sW = self.robot_size[1] / 2 + self.collision_tolerance  # Width semi-axis

            degrees = np.linspace(0, 2 * math.pi, 360, endpoint=False)
            vec1 = sL / np.maximum(np.cos(degrees), 0.0001)
            vec2 = sW / np.maximum(np.sin(degrees + 0.0001), 0.0001)

            return np.minimum(np.abs(vec1), np.abs(vec2))

        else:
            return np.full(360, 0.5)

    def process_data(self, points):

        points = np.nan_to_num(points, nan=self.max_dist, posinf=self.max_dist, neginf=self.max_dist)

        collision_points = np.sum(points < self.collision_vector)
        collision_threshold = 17
        collision = collision_points >= collision_threshold

        # Add Gaussian noise and clip to valid range
        points = self.add_noise(points)
        points = np.clip(points, 0.2, self.max_dist)

        # Downsample the laser scan points to reduce processing
        div = int(360 / self.num_points)
        scan_range = np.minimum.reduceat(points, np.arange(0, len(points), div))

        # Get minimum distance and corresponding angle
        min_obstacle_distance = np.min(points)

        return scan_range, min_obstacle_distance, collision

    def add_noise(self, points):
        noise = np.random.normal(loc=0.0, scale=0.05, size=points.shape)
        return points + noise


class ImuSensor:
    def __init__(self):
        pass

    def process_data(self, data):
        _, _, yaw = euler_from_quaternion(data.orientation)

        return yaw


class OdomSensor:
    def __init__(self):
        pass

    def process_data(self, data, vel=False):
        pos_x = data.pose.pose.position.x
        pos_y = data.pose.pose.position.y
        zqr = data.pose.pose.orientation.z
        wqr = data.pose.pose.orientation.w
        zr = quat_to_euler(zqr, wqr)
        # _,_,yaw = euler_from_quaternion(data.pose.pose.orientation)

        if vel:
            vx = data.twist.twist.linear.x
            wz = data.twist.twist.angular.z

            return [pos_x, pos_y, zr], [vx, wz]

        else:
            return [pos_x, pos_y, zr]


class DepthCamera:
    def __init__(self, width, height, cutoff_dist, show=False):
        self.width = width
        self.height = height
        self.cutoff_dist = cutoff_dist
        self.show = show
        self.random_noise = False

    def process_data(self, frame):
        np.seterr(all="raise")
        # IF SIMULATION
        max_depth = self.cutoff_dist  # [m]
        # IF REAL CAMERA
        # max_depth = self.cutoff*1000 [mm]

        depth_frame = np.nan_to_num(frame, nan=0.0, posinf=max_depth, neginf=0.0)
        depth_frame[depth_frame == 0.0] = max_depth
        depth_frame = np.minimum(depth_frame, max_depth)

        if self.random_noise:
            noise1 = np.random.normal(loc=0.0, scale=0.2, size=depth_frame.shape)
            noise2 = np.random.normal(loc=0.0, scale=1.0, size=depth_frame.shape) * depth_frame / 10
            depth_frame = np.clip(depth_frame + noise1 + noise2, 0.0, max_depth)

        depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        depth_frame = depth_frame.astype(np.float32)

        if self.show:
            depth_frame = (
                depth_frame / max_depth
                if np.amax(depth_frame) > 1.0 and np.amax(depth_frame) <= max_depth
                else depth_frame
            )
            depth_frame = (
                depth_frame *
                255.0 if np.amax(depth_frame) <= 1.0 else depth_frame
            )
            self.show_image(depth_frame)

        # add a channel to have dims = 3
        depth_frame = np.expand_dims(depth_frame, axis=-1)

        # if 3 channels are needed by the backbone copy over 3
        # depth_frame = np.tile(depth_frame, (1, 1, 3))

        return depth_frame

    def show_image(self, image):
        colormap = np.asarray(image, dtype=np.uint8)
        cv2.namedWindow("Depth Image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Depth Image", colormap)
        cv2.waitKey(1)


class RGBCamera:
    def __init__(self, width, height, show=False):
        self.width = width
        self.height = height
        self.show = show

    def process_data(self, rgb):
        img = np.array(rgb)
        img_resized = cv2.resize(img, (self.height, self.width))

        return np.array(img_resized)

    def show_image(self, image):
        colormap = np.asarray(image, dtype=np.uint8)
        cv2.namedWindow("Depth Image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Depth Image", colormap)
        cv2.waitKey(1)


class Sensors:
    def __init__(self, node):
        # super().__init__("generic_sensors_node")
        self.param = self.get_param(node)

        self.odom_data = None
        self.laser_data = None
        self.depth_data = None
        self.rgb_data = None
        self.imu_data = None

        self.imu_sub = None
        self.depth_sub = None
        self.rgb_sub = None
        self.laser_sub = None
        self.odom_sub = None

        self.imu_process = None
        self.depth_process = None
        self.rgb_process = None
        self.laser_process = None
        self.odom_process = None
        self.node = node
        self.sensor_msg = {}
        self.sensors = self.activate_sensors()
        self.bridge = CvBridge()

    def get_param(self, node):
        configFilepath = node.main_params_path

        # Load the topic parameters
        with open(configFilepath, "r") as file:
            configParams = yaml.safe_load(file)["main_node"]["ros__parameters"]

        return configParams

    def activate_sensors(self):
        if self.param["imu_enabled"] == "true":
            self.node.get_logger().debug("IMU subscription done")
            self.imu_sub = self.node.create_subscription(
                Imu, self.param["sensors_topic"]["imu_topic"], self.imu_cb, 1
            )

            self.imu_process = ImuSensor()
            self.sensor_msg["imu"] = "None"

        if self.param["camera_enabled"] == "true":
            self.node.get_logger().debug("Depth subscription done")
            self.depth_sub = self.node.create_subscription(
                Image,
                self.param["sensors_topic"]["depth_topic"],
                self.depth_camera_cb,
                1,
            )
            self.depth_process = DepthCamera(
                self.param["depth_param"]["width"],
                self.param["depth_param"]["height"],
                self.param["depth_param"]["dist_cutoff"],
                self.param["depth_param"]["show_image"],
            )
            self.sensor_msg["depth"] = "None"

        if self.param["lidar_enabled"] == "true":
            self.node.get_logger().debug("Laser scan subscription done")
            self.laser_sub = self.node.create_subscription(
                LaserScan,
                self.param["sensors_topic"]["laser_topic"],
                self.laser_scan_cb,
                1,
            )
            self.laser_process = LaserScanSensor(
                self.param["laser_param"]["max_distance"],
                self.param["laser_param"]["num_points"],
                self.param["robot_type"],
                self.param["robot_radius"],
                self.param["robot_size"],
                self.param["collision_tolerance"],
            )
            self.sensor_msg["scan"] = "None"

        self.node.get_logger().debug("Odometry subscription done")
        self.odom_sub = self.node.create_subscription(
            Odometry, self.param["sensors_topic"]["odom_topic"], self.odometry_cb, 1
        )
        self.odom_process = OdomSensor()
        self.sensor_msg["odom"] = "None"

    def imu_cb(self, msg):
        self.imu_data = msg
        self.sensor_msg["imu"] = msg

    def depth_camera_cb(self, msg):
        self.depth_data = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.sensor_msg["depth"] = msg

    def rgb_camera_cb(self, msg):
        self.rgb_data = self.bridge.imgmsg_to_cv2(msg)
        self.sensor_msg["rgb"] = msg

    def laser_scan_cb(self, msg):
        self.laser_data = msg.ranges
        self.sensor_msg["scan"] = msg

    def odometry_cb(self, msg):
        self.odom_data = msg
        self.sensor_msg["odom"] = msg

    def get_odom(self, vel=False):
        if self.odom_sub is None:
            self.node.get_logger().warn("NO Odometry subscription")
            return None
        if self.odom_data is None:
            self.node.get_logger().warn("NO Odometry data")
            return None

        if vel:
            data, velocities = self.odom_process.process_data(
                self.odom_data, vel)
            return data, velocities
        else:
            data = self.odom_process.process_data(self.odom_data)
            return data

    def get_depth(self):
        if self.depth_sub is None:
            self.node.get_logger().warn("NO Depth subscription")
            return None
        if self.depth_data is None:
            self.node.get_logger().warn("NO depth image")
            return None
        data = self.depth_process.process_data(self.depth_data)
        return data

    def get_rgb(self):
        if self.rgb_sub is None:
            self.node.get_logger().warn("NO RGB subscription")
            return None
        if self.rgb_data is None:
            self.node.get_logger().warn("NO RGB image")
            return None

        data = self.rgb_process.process_data(self.rgb_data)
        return data

    def get_imu(self):
        if self.imu_sub is None:
            self.node.get_logger().warn("NO IMU subscription")
            return None
        elif self.imu_data is None:
            self.node.get_logger().warn("NO IMU data")
        else:
            data = self.imu_process.process_data(self.imu_data)
            return data

    def get_laser(self, min_obstacle_distance=False):
        if self.laser_sub is None:
            self.node.get_logger().warn("NO laser subscription")
            return None, False
        if self.laser_data is None:
            self.node.get_logger().warn("NO laser data")
            return None, False
        processed_data, min_obstacle_distance_v, collision = self.laser_process.process_data(
            self.laser_data
        )
        if min_obstacle_distance:
            return processed_data, min_obstacle_distance_v, collision
        return processed_data, collision
