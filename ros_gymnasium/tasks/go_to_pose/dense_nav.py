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
from ros_gymnasium.utils.env_utils import process_odom
from ros_gymnasium.ros_gymnasium_env import ROSGymnasiumEnv

from gymnasium import spaces


class DenseNavEnv(ROSGymnasiumEnv):
    def __init__(self, render_mode=None):
        """ """
        super().__init__("depth_camera_env")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        print("DenseNavEnv: Initializing environment")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("data_path", rclpy.Parameter.Type.STRING),
                ("goal_tolerance", rclpy.Parameter.Type.DOUBLE),
                ("robot_spawn_z", rclpy.Parameter.Type.DOUBLE),
                ("goal_spawn_z", rclpy.Parameter.Type.DOUBLE),
                ("dense_reward.progress_scale", rclpy.Parameter.Type.DOUBLE),
                ("dense_reward.heading_scale", rclpy.Parameter.Type.DOUBLE),
                ("dense_reward.collision_step", rclpy.Parameter.Type.DOUBLE),
                ("dense_reward.goal", rclpy.Parameter.Type.DOUBLE),
                ("dense_reward.collision", rclpy.Parameter.Type.DOUBLE),
            ],
        )

        self.data_path = self._param("data_path", "string_value")

        self.goal_tolerance = self._param("goal_tolerance", "double_value")
        self.robot_spawn_z = self._param("robot_spawn_z", "double_value")
        self.goal_spawn_z = self._param("goal_spawn_z", "double_value")
        self.rewards = {
            key: self._param(f"dense_reward.{key}", "double_value")
            for key in ("progress_scale", "heading_scale", "collision_step", "goal", "collision")
        }
        
        # Gymnasium spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # Initialize environment
        self.initial_pose, self.goals, self.poses = self._load_goals_and_poses()
        self.goal_pose = self.goals[0]
        self.previous_goal_info = None
        self.goal_count = 0
        self.collision_count = 0
        self.starting_episodes = 0
        self.timeout_steps = 500

        self._spin_sensors_callbacks()

        print("Environment Initialized")

        self.get_logger().debug("ROSGymnasiumEnv: Starting process")

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

    def _get_reward(self, sensor_data):
        """Dense reward: progress towards the goal, heading to the goal, and dense_reward.<event> bonuses."""
        goal_info = sensor_data["goal_info"]
        event = sensor_data["event"]
        collision = sensor_data["collision"]

        if self.previous_goal_info:
            reward = (self.previous_goal_info[0] - goal_info[0]) * self.rewards["progress_scale"]
        else:
            reward = 0
        yaw_reward = (1 - 2 * math.sqrt(abs(goal_info[1] / math.pi))) * self.rewards["heading_scale"]
        reward += yaw_reward

        self.previous_goal_info = goal_info

        if collision:
            reward += self.rewards["collision_step"]

        if event == "goal":
            reward += self.rewards["goal"]
        elif event == "collision":
            reward += self.rewards["collision"]
        return reward

    def _get_sensor_data(self):
        """Sensor readings plus the goal distance and angle relative to the robot."""
        sensor_data = super()._get_sensor_data()
        sensor_data["goal_info"], _ = process_odom(self.goal_pose, sensor_data["odom"])
        return sensor_data

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

        if not self._set_entity_state("goal", self.goal_pose[0], self.goal_pose[1], z=self.goal_spawn_z):
            self.get_logger().warn("Using logical goal position even though teleportation failed")
            return False
        return True

    def _respawn_robot(self):
        """Respawn the robot in a new position."""
        if self.episode <= self.starting_episodes:
            x, y, yaw = self.initial_pose
        else:
            index = self.np_random.integers(len(self.poses))
            x, y, yaw = self.poses[index]

        if not self._set_entity_state(self.robot_name, x, y, z=self.robot_spawn_z, yaw=yaw):
            self.get_logger().warn("Robot teleportation failed")
            return False
        return True

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
            self.data_path,
        )
        with open(data_path, "r") as file:
            data = json.load(file)
        return data["initial_pose"], data["goals"], data["poses"]