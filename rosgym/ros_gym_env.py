#!/usr/bin/env python3

import os
import yaml
import logging

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from ament_index_python.packages import get_package_share_directory
from rosgym.sensors import Sensors
from rosgym.utils.env_utils import create_logdir
import gymnasium as gym
import time
from datetime import timedelta 

from abc import ABC, abstractmethod


class ROSGymEnv(Node, gym.Env, ABC):
    
    def __init__(self, node_name: str = "rl_base_env"):
        """ """
        super().__init__(node_name=node_name)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("package_name", rclpy.Parameter.Type.STRING),
                ("training_params_path", rclpy.Parameter.Type.STRING),
                ("main_params_path", rclpy.Parameter.Type.STRING),
                ("mode", rclpy.Parameter.Type.STRING),
                ("world_name", rclpy.Parameter.Type.STRING),
                ("robot_name", rclpy.Parameter.Type.STRING),
                ("sensor", rclpy.Parameter.Type.STRING),
            ],
        )

        # ROS parameters
        train_params_path = self.get_parameter("training_params_path").get_parameter_value().string_value
        with open(train_params_path, "r") as train_param_file:
            self.train_params = yaml.safe_load(train_param_file)["training_params"]

        self.main_params_path = self.get_parameter("main_params_path").get_parameter_value().string_value
        
        self.package_name = self.get_parameter("package_name").get_parameter_value().string_value
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.world_name = self.get_parameter("world_name").get_parameter_value().string_value
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.sensor_type = self.get_parameter("sensor").get_parameter_value().string_value
        self.evaluate = False

        # ROS 2 setup
        self.reset_world_client = self.create_client(Empty, "reset_world")
        self.sensors = Sensors(self)

        # Training variables
        self.episode_step = 0
        self.episode = 0
        self.total_steps = 0 

        # Logging and metrics
        self.logdir = self._create_logdir()

        # Gazebo process
        self.gazebo_process = None

        self.start_time = time.time()

        self.get_logger().debug(f"{node_name}: Starting process")

    def step(self, action):
        """ """
        self._send_action(action)
        self._spin_sensors_callbacks()

        sensor_data = self._get_sensor_data()

        next_observation = self._get_observation(sensor_data)
        terminated, event = self._check_events(sensor_data)
        sensor_data["event"] = event

        reward = self._get_reward(sensor_data)
        truncated = self.episode_step >= self.timeout_steps
        
        self.episode_step += 1
        return next_observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """ """
        super().reset(seed=seed)
        self.total_steps += self.episode_step
        self.episode += 1
        self.episode_step = 0

        episode_time = time.time() - self.start_time
        formatted_time = str(timedelta(seconds=int(episode_time)))

        print(f"Total steps: {self.total_steps}, total training time: {formatted_time}")
        logging.info(f"Total_episodes: {self.episode}{' evaluation episode' if self.evaluate else ''}, Total_steps: {self.total_steps}, episode_steps: {self.episode_step+1}\n")
        
        self.new_episode()
        self._spin_sensors_callbacks()

        sensor_data = self._get_sensor_data()
        next_observation = self._get_observation(sensor_data)
        
        return next_observation, {}

    def render(self, mode="none"):
        """ """
        world_path = os.path.join(
            get_package_share_directory(self.package_name),
            "worlds",
            self.world_name,
        )

        if mode == "human":
            self.get_logger().info("Launching Gazebo with GUI...")
            self._launch_gazebo(world_path, headless=False)
        elif mode == "none":
            self.get_logger().info("Launching Gazebo in headless mode...")
            self._launch_gazebo(world_path, headless=True)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

        # Spawn the robot in the world
        self._spawn_robot()

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
        self.get_logger().debug("spinning node...")
        #print("spinning node...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            #print("measurements not ready")
            self.get_logger().debug(f"empty_measurements")
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
        #print("measurements ready")

    def _launch_gazebo(self, world_path, headless=True):
        """
        WIP
        Launch Gazebo with the specified world file.

        Args:
            world_path (str): Path to the Gazebo world file.
            headless (bool): If True, run Gazebo in headless mode (no GUI).
        """
        gazebo_cmd = ["gazebo", world_path]
        if headless:
            gazebo_cmd.append("--headless-rendering")

        # Launch Gazebo as a subprocess
        self.gazebo_process = subprocess.Popen(gazebo_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # Wait for Gazebo to fully launch

    def _spawn_robot(self):
        """
        WIP
        Spawn the robot in the Gazebo world.
        """
        # Example robot spawn logic (replace with actual implementation)
        x, y, z = 0.0, 0.0, 0.1  # Robot spawn position
        qz, qw = 0.0, 1.0  # Robot orientation (quaternion)

        state = (
            f"{{"
            f"state: {{"
            f"name: '{self.robot_name}', "
            f"pose: {{"
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{z: {qz}, w: {qw}}}"
            f"}}"
            f"}}"
            f"}}"
        )
        os.system(f"ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState '{state}'")
        self.get_logger().info(f"Spawned robot '{self.robot_name}' at position ({x}, {y}, {z}).")    
    
    @abstractmethod
    def _send_action(self, twist):
        """Transform RL action to ROS command."""
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