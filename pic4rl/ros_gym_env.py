#!/usr/bin/env python3

import os
import numpy as np
import time
import yaml
import logging
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.sensors import Sensors
from pic4rl.utils.env_utils import *
import gymnasium as gym
from gymnasium import spaces

from abc import ABC, abstractmethod


class ROSGymEnv(Node, gym.Env, ABC):
    def __init__(self, node_name="rl_base_env"):
        """ """
        super().__init__(node_name="rl_base_env")

        # ROS parameters
        self.declare_parameter("package_name", "pic4rl")
        self.declare_parameter("training_params_path", rclpy.Parameter.Type.STRING)
        train_params_path = self.get_parameter("training_params_path").get_parameter_value().string_value
        with open(train_params_path, "r") as train_param_file:
            self.train_params = yaml.safe_load(train_param_file)["training_params"]

        self.package_name = self.get_parameter("package_name").get_parameter_value().string_value
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.sensor_type = self.get_parameter("sensor").get_parameter_value().string_value
        self.evaluate = False

        # ROS 2 setup
        #qos = QoSProfile(depth=10)
        #todo self.pause_physics_client = self.create_client(Empty, "pause_physics")
        #todo self.unpause_physics_client = self.create_client(Empty, "unpause_physics")

        self.reset_world_client = self.create_client(Empty, "reset_world")
        self.sensors = Sensors(self)

        # Training variables
        self.episode_step = 0
        self.episode = 0

        # Logging and metrics
        self.logdir = self._create_logdir()
        #todo if self.mode == "testing":
            #todo self.nav_metrics = Navigation_Metrics(self.logdir)

        self.get_logger().debug(f"{node_name}: Starting process")

    def step(self, action):
        """ """
        self._send_action(action)
        self._spin_sensors_callbacks()

        sensor_data = self._get_sensor_data()

        next_observation = _get_observation(sensor_data)
        terminated, event = self._check_events(sensor_data)
        reward = self._get_reward(sensor_data)
        truncated = self.episode_step >= self.timeout_steps
        
        #todo if self.mode == "testing":
            #todo self.nav_metrics.get_metrics_data(lidar_measurements, self.episode_step)

        self.previous_goal_info = goal_info
        self.episode_step += 1
        return next_observation, reward, terminated, truncated, {}

    def reset(self, seed=None):
        """ """
        super().reset(seed=seed)
        self.total_steps += self.episode_step
        self.episode += 1
        self.episode_step = 0

        #todo if self.mode == "testing":
            #todo self.nav_metrics.calc_metrics(self.episode, self.initial_pose, self.goal_pose)
            #todo self.nav_metrics.log_metrics_results(self.episode)
            #todo self.nav_metrics.save_metrics_results(self.episode)

        
        logging.info(f"Total_episodes: {self.episode}{' evaluation episode' if self.evaluate else ''}, Total_steps: {self.total_steps}, episode_steps: {self.episode_step+1}\n")
        
        self.new_episode()

        # Reset the buffer with four copies of the first depth image
        self._spin_sensors_callbacks()

        sensor_data = self._get_sensor_data()
        next_observation = self._get_observation(sensor_data)
        
        return next_observation, {}

    def close(self):
        self.get_logger().info("Closing environment...")
        rclpy.shutdown()

    def _create_logdir(self):
        """Create a log directory for training."""
        log_path = os.path.join(
            get_package_share_directory(self.package_name),
            "../../../../",
            self.train_params["--logdir"],
        )
        return create_logdir(self.train_params["--policy"], self.sensor_type, log_path)

    def _spin_sensors_callbacks(self):
        """ """
        self.get_logger().debug("spinning node...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            self.get_logger().debug(f"empty_measurements")
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
    
    @abstractmethod
    def _send_action(self, twist):
        """ """
        pass

    @abstractmethod
    def _get_sensor_data(self):
        """Retrieve sensor data from the robot."""
        pass

    @abstractmethod
    def _get_observation(self, sensor_data):
        """Retrieve next observation"""
        pass

    @abstractmethod
    def _check_events(self, sensor_data):
        """Check for termination events (collision, goal, timeout)."""
        pass

    @abstractmethod
    def _get_reward(self, sensor_data):
        """Calculate the reward based on the event and goal distance."""
        pass

    @abstractmethod
    def new_episode(self):
        """ """
        pass