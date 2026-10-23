#!/usr/bin/env python3

import os
import numpy as np
import math
import time
import json

import rclpy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from rosgym.utils.env_utils import process_odom
from rosgym.ros_gym_env import ROSGymEnv

from gymnasium import spaces


class DepthNavEnv(ROSGymEnv):
    def __init__(self, render_mode=None):
        """ """
        super().__init__("depth_camera_env")
        if render_mode not in [None, "human", "rgb_array"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        print("DepthNavEnv: Initializing environment")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("data_path", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ("depth_param.width", rclpy.Parameter.Type.INTEGER),
                ("depth_param.height", rclpy.Parameter.Type.INTEGER),
                ("depth_param.dist_cutoff", rclpy.Parameter.Type.DOUBLE),
                ("depth_param.stacked_images", rclpy.Parameter.Type.INTEGER),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
            ],
        )

        goals_path = os.path.join(get_package_share_directory(self.package_name), "goals_and_poses")
        goals_path = os.path.join(goals_path, self.mode)
        self.data_path = os.path.join(goals_path, self.get_parameter("data_path").get_parameter_value().string_value)

        self.goal_tolerance = self._param("goal_tolerance", "double_value")
        
        self.image_width = self._param("depth_param.width", "integer_value")
        self.image_height = self._param("depth_param.height", "integer_value")
        self.stacked_images = self._param("depth_param.stacked_images", "integer_value")
        self.max_depth = self._param("depth_param.dist_cutoff", "double_value")

        self.lidar_distance = self._param("laser_param.max_distance", "double_value")
        self.lidar_points = self._param("laser_param.num_points", "integer_value")
        
        # Gymnasium spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=self.max_depth, 
            shape=(self.stacked_images, self.image_height, self.image_width), 
            dtype=np.float32
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # Initialize image buffer
        self.image_buffer = np.zeros(
            (self.stacked_images, self.image_height, self.image_width),
            dtype=np.float32
        )

        # Initialize environment
        self.initial_pose, self.goals, self.poses = self._load_goals_and_poses()
        self.goal_pose = self.goals[0]
        self.previous_twist = Twist()
        self.previous_goal_info = None
        self.t0 = 0.0
        self.goal_count = 0
        self.collision_count = 0
        self.evaluate = False
        self.starting_episodes = 0
        self.timeout_steps = 500

        self._spin_sensors_callbacks()

        print("Environment Initialized")

        self.get_logger().debug("ROSGYM_Environment: Starting process")

    def _new_episode(self):
        """Reset the environment for a new episode."""
        self.get_logger().info(f"Starting new episode {self.episode}.")
        self.collision_count = 0
        # Progress reward must not compare against the previous episode's goal
        self.previous_goal_info = None

        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_twist)

        # Do not leak frames from the previous episode into the new one
        self.image_buffer = np.zeros_like(self.image_buffer)

        self._reset_simulation()
        self._respawn_robot()
        self._respawn_goal()
        time.sleep(0.25)

    def _check_events(self, sensor_data):
        """Check for termination events (collision, goal). Timeouts are handled as truncation by the base class."""
        collision = sensor_data["collision"]
        goal_info = sensor_data["goal_info"]

        if collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info(f"COLLISION DETECTED.")
                return True, "collision"
        else:
            self.collision_count = 0
        
        if goal_info[0] < self.goal_tolerance:
            self.goal_count += 1
            return True, "goal"

        return False, "none"

    def _get_observation(self, sensor_data):
        """Get the current observation."""
        depth_image = sensor_data["depth_image"]
        
        if len(depth_image.shape) == 3 and depth_image.shape[2] == 1:
            depth_image = depth_image.squeeze(2)
        
        self.image_buffer = np.roll(self.image_buffer, shift=1, axis=0)
        self.image_buffer[0] = depth_image
        
        return self.image_buffer

    def _get_reward(self, sensor_data):
        """Calculate the reward based on the event and goal distance."""
        goal_info = sensor_data["goal_info"]
        event = sensor_data["event"]
        collision = sensor_data["collision"]

        if self.previous_goal_info:
            reward = (self.previous_goal_info[0] - goal_info[0]) * 30
        else:
            reward = 0
        yaw_reward = (1 - 2 * math.sqrt(abs(goal_info[1] / math.pi))) * 0.6
        reward += yaw_reward

        self.previous_goal_info = goal_info

        if collision:
            reward -= 10

        if event == "goal":
            reward += 1000
        elif event == "collision":
            reward += -200
        return reward

    def _get_sensor_data(self):
        """Retrieve sensor data from the robot."""
        sensor_data = {
            "scan": self.sensors.get_laser(),
            "odom": self.sensors.get_odom(),
            "depth": self.sensors.get_depth(),
        }

        if sensor_data["scan"] is not None:
            lidar_measurements, collision = sensor_data["scan"]
        else:
            lidar_measurements = np.ones(self.lidar_points) * self.lidar_distance
            collision = False

        depth_image = sensor_data["depth"] if sensor_data["depth"] is not None else np.ones((self.image_height, self.image_width, 1)) * self.max_depth
        odom = sensor_data["odom"] if sensor_data["odom"] is not None else [0.0, 0.0, 0.0]

        goal_info, _ = process_odom(self.goal_pose, odom)

        depth_image_normalized = depth_image / self.max_depth

        sensor_data_dic = {
            "scan": lidar_measurements,
            "depth_image": depth_image_normalized,
            "goal_info": goal_info,
            "collision": collision,
        }

        return sensor_data_dic

    def _log_results(self):
        self.get_logger().debug(f"Goals reached: {self.goal_count}")

    def _reset_simulation(self):
        """Reset the Gazebo simulation."""
        req = Empty.Request()
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Service not available, waiting again...")
        # Use synchronous call instead of async
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        #time.sleep(0.5)  # Allow time for reset to complete

    def _respawn_goal(self):
        """Respawn the goal in a new position."""
        if self.episode <= self.starting_episodes:
            x = self.np_random.uniform(-2.9, 2.9) + self.initial_pose[0]
            y = self.np_random.uniform(-2.9, 2.9) + self.initial_pose[1]
            self.goal_pose = [x, y]
        else:
            index = self.np_random.integers(len(self.goals))
            self.goal_pose = self.goals[index]

        self.get_logger().info(f"New goal pose: {self.goal_pose}")

    def _respawn_robot(self):
        """Respawn the robot in a new position."""
        if self.episode <= self.starting_episodes:
            x, y, yaw = self.initial_pose
        else:
            index = self.np_random.integers(len(self.poses))
            x, y, yaw = self.poses[index]

        qz = np.sin(yaw / 2)
        qw = np.cos(yaw / 2)
        state = (
            f"{{\"state\": {{\"name\": \"{self.robot_name}\", "
            f"\"pose\": {{\"position\": {{\"x\": {x}, \"y\": {y}, \"z\": 0.02}}, "
            f"\"orientation\": {{\"z\": {qz}, \"w\": {qw}}}}}}}}}"
        )
        os.system(f"ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState {state}  2>/dev/null")
        #time.sleep(0.25)

    def _send_action(self, action):

        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist)

    def _load_goals_and_poses(self):
        """Load goals and poses from a JSON file."""
        data_path = os.path.join(
            get_package_share_directory(self.package_name),
            "goals_and_poses",
            #self.mode,
            self.data_path,
        )
        with open(data_path, "r") as file:
            data = json.load(file)
        return data["initial_pose"], data["goals"], data["poses"]