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
from ros_gymnasium.utils.env_utils import goal_observation_space, process_odom
from ros_gymnasium.ros_gymnasium_env import ROSGymnasiumEnv

from gymnasium import spaces

ACTION_TYPES = ("discrete", "continuous")


class SparseNavEnv(ROSGymnasiumEnv):
    """Go-to-goal navigation with a sparse reward.

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
                ("goal_observation", rclpy.Parameter.Type.BOOL),
                ("grid_size", rclpy.Parameter.Type.DOUBLE),
                ("robot_spawn_z", rclpy.Parameter.Type.DOUBLE),
                ("goal_spawn_z", rclpy.Parameter.Type.DOUBLE),
                ("sparse_reward.goal", rclpy.Parameter.Type.DOUBLE),
                ("sparse_reward.collision", rclpy.Parameter.Type.DOUBLE),
                ("sparse_reward.timeout", rclpy.Parameter.Type.DOUBLE),
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

        if self.action_type == "discrete":
            self.LINEAR_VELOCITIES = [0.2, 0.4]
            self.ANGULAR_VELOCITIES = [
                math.pi/6,
                math.pi/12,
                0.0,
                -math.pi/12,
                -math.pi/6
            ]

            self.actions = [
                (lin_vel, ang_vel)
                for lin_vel in self.LINEAR_VELOCITIES
                for ang_vel in self.ANGULAR_VELOCITIES
            ]
            self.action_space = spaces.Discrete(len(self.actions))
        else:
            self.action_space = spaces.Box(
                low=np.array([self.MIN_LINEAR_VELOCITY, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )

        if self._param("goal_observation", "bool_value"):
            self._add_task_observation("goal", goal_observation_space())
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        _, goals, _ = self._load_goals_and_poses()
        self.goal_pose = goals[0]
        self.goal_count = 0
        self.collision_count = 0
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

        self._respawn_robot()
        self._respawn_goal()
        time.sleep(0.25)

    def _check_events(self, sensor_data):
        """Return (terminated, event) for collisions and reaching the goal."""
        collision = sensor_data["collision"]
        goal_info = sensor_data["goal"]

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

    def _get_reward(self, sensor_data):
        """Sparse reward: sparse_reward.<event> on goal, collision or timeout, 0 otherwise."""
        return self.rewards.get(sensor_data["event"], 0.0)

    def _get_sensor_data(self):
        """Sensor readings plus "goal": the goal distance and angle relative to the robot."""
        sensor_data = super()._get_sensor_data()
        sensor_data["goal"], _ = process_odom(self.goal_pose, sensor_data["odom"])
        return sensor_data

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
        """Respawn the robot in a new position."""
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
