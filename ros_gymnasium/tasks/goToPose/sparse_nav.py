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
from ros_gymnasium.utils.env_utils import depth_to_rgb, process_odom
from ros_gymnasium.ros_gymnasium_env import ROSGymnasiumEnv

from gymnasium import spaces

ACTION_TYPES = ("discrete", "continuous")


class SparseNavEnv(ROSGymnasiumEnv):
    """
    Go-to-goal navigation from stacked depth images with a sparse reward.

    Args:
        action_type: "discrete" (10 fixed velocity commands) or "continuous" ([linear, angular] velocity)
        timeout_steps: Steps before the episode is truncated
        render_mode: None or "rgb_array"
    """

    def __init__(self, action_type="discrete", timeout_steps=500, render_mode=None):
        super().__init__("depth_camera_env")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if action_type not in ACTION_TYPES:
            raise ValueError(f"Unsupported action_type: {action_type}, expected one of {ACTION_TYPES}")
        self.render_mode = render_mode
        self.action_type = action_type

        self.declare_parameters(
            namespace="",
            parameters=[
                ("data_path", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ("grid_size", rclpy.Parameter.Type.DOUBLE),
                ("robot_spawn_z", rclpy.Parameter.Type.DOUBLE),
                ("goal_spawn_z", rclpy.Parameter.Type.DOUBLE),
                ("sparse_reward.goal", rclpy.Parameter.Type.DOUBLE),
                ("sparse_reward.collision", rclpy.Parameter.Type.DOUBLE),
                ("sparse_reward.timeout", rclpy.Parameter.Type.DOUBLE),
                ("depth_param.width", rclpy.Parameter.Type.INTEGER),
                ("depth_param.height", rclpy.Parameter.Type.INTEGER),
                ("depth_param.dist_cutoff", rclpy.Parameter.Type.DOUBLE),
                ("depth_param.stacked_images", rclpy.Parameter.Type.INTEGER),
                ("laser_param.max_distance", rclpy.Parameter.Type.DOUBLE),
                ("laser_param.num_points", rclpy.Parameter.Type.INTEGER),
            ],
        )

        self.MIN_LINEAR_VELOCITY = 0.01

        self.data_path = self._param("data_path", "string_value")

        self.goal_tolerance = self._param("goal_tolerance", "double_value")
        self.grid_size = self._param("grid_size", "double_value")
        self.robot_spawn_z = self._param("robot_spawn_z", "double_value")
        self.goal_spawn_z = self._param("goal_spawn_z", "double_value")
        self.rewards = {
            event: self._param(f"sparse_reward.{event}", "double_value")
            for event in ("goal", "collision", "timeout")
        }

        self.image_width = self._param("depth_param.width", "integer_value")
        self.image_height = self._param("depth_param.height", "integer_value")
        self.stacked_images = self._param("depth_param.stacked_images", "integer_value")
        self.max_depth = self._param("depth_param.dist_cutoff", "double_value")

        self.lidar_distance = self._param("laser_param.max_distance", "double_value")
        self.lidar_points = self._param("laser_param.num_points", "integer_value")

        # Gymnasium spaces
        if self.action_type == "discrete":
            self.LINEAR_VELOCITIES = [0.2, 0.4]  # two linear velocities
            self.ANGULAR_VELOCITIES = [
                math.pi/6,   # 30 degrees/s right turn
                math.pi/12,  # 15 degrees/s right turn
                0.0,         # straight
                -math.pi/12, # 15 degrees/s left turn
                -math.pi/6   # 30 degrees/s left turn
            ]

            # All combinations [0.2, π/6], [0.2, π/12], ..., [0.4, -π/6]
            self.actions = [
                (lin_vel, ang_vel)
                for lin_vel in self.LINEAR_VELOCITIES
                for ang_vel in self.ANGULAR_VELOCITIES
            ]
            self.action_space = spaces.Discrete(len(self.actions))
        else:
            self.action_space = spaces.Box(
                low=np.array([self.MIN_LINEAR_VELOCITY, -1.0], dtype=np.float32),  # Linear velocity can only be positive
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )

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
        self.timeout_steps = timeout_steps

        self._spin_sensors_callbacks()

        self.get_logger().debug("ROSGymnasiumEnv: Starting process")

    def _new_episode(self):
        """Reset the environment for a new episode."""
        self.collision_count = 0

        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_twist)

        # Do not leak frames from the previous episode into the new one
        self.image_buffer = np.zeros_like(self.image_buffer)

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
                self.get_logger().info(f"Collision Detected.")
                return True, "collision"
        else:
            self.collision_count = 0

        if goal_info[0] < self.goal_tolerance:
            self.goal_count += 1
            self.get_logger().info(f"Goal Reached.")
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

    def _render_frame(self):
        """Latest depth frame as an RGB image."""
        return depth_to_rgb(self.image_buffer[0], self.max_depth)

    def _get_reward(self, sensor_data):
        """Sparse reward: sparse_reward.<event> on goal, collision or timeout, 0 otherwise."""
        return self.rewards.get(sensor_data["event"], 0.0)

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

        sensor_data_dic = {
            "scan": lidar_measurements,
            "depth_image": depth_image,
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

        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        time.sleep(0.1)

    def _sample_position(self):
        """Sample an (x, y) position in a random quadrant, at least 1 m away from both axes."""
        sign_x, sign_y = self.np_random.choice([-1.0, 1.0], size=2)
        x, y = self.np_random.uniform(1, self.grid_size, size=2)
        return float(sign_x * x), float(sign_y * y)

    def _respawn_goal(self):
        """Respawn the goal in a new position."""
        x, y = self._sample_position()
        self.goal_pose = [x, y]

        if not self._set_entity_state("goal", x, y, z=self.goal_spawn_z):
            self.get_logger().warn("Using logical goal position even though teleportation failed")
            return False
        return True

    def _respawn_robot(self):
        """Respawn the robot in a new position"""
        x, y = self._sample_position()
        yaw = self.np_random.uniform(0, 2 * math.pi)

        if self._set_entity_state(self.robot_name, x, y, z=self.robot_spawn_z, yaw=yaw):
            return True
        return self._reset_simulation()

    def _send_action(self, action):
        """Publish the action as a velocity command."""
        twist = Twist()

        if self.action_type == "discrete":
            linear_x, angular_z = self.actions[action]
        else:
            linear_x = max(float(action[0]), self.MIN_LINEAR_VELOCITY)
            angular_z = float(action[1])

        twist.linear.x = linear_x
        twist.angular.z = angular_z

        self.cmd_vel_pub.publish(twist)

    def _load_goals_and_poses(self):
        """Load goals and poses from a JSON file."""
        data_path = os.path.join(
            get_package_share_directory(self.package_name),
            "goals_and_poses",
            self.data_path,
        )
        with open(data_path, "r") as file:
            data = json.load(file)
        return data["initial_pose"], data["goals"], data["poses"]
