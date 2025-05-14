#!/usr/bin/env python3

import os
import yaml
import logging
import subprocess
import time
from datetime import timedelta
from abc import ABC, abstractmethod

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from ament_index_python.packages import get_package_share_directory

from rosgym.sensors import Sensors
from rosgym.utils.env_utils import create_logdir
import gymnasium as gym

def _param(self, name: str, type_: str):
    return getattr(self.get_parameter(name).get_parameter_value(), type_)

class ROSGymEnv(Node, gym.Env, ABC):
    
    def __init__(self, node_name: str = "rl_base_env"):
        super().__init__(node_name=node_name)
        self.get_logger().debug(f"{node_name}: Starting process")

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
        train_params_path = self._param("training_params_path", "string_value")
        self.main_params_path = self._param("main_params_path", "string_value")
        
        if not os.path.isfile(train_params_path):
            raise FileNotFoundError(f"Training config not found at {train_params_path}")

        with open(train_params_path, "r") as train_param_file:
            self.train_params = yaml.safe_load(train_param_file)["training_params"]

        
        self.package_name = self._param("package_name", "string_value")
        self.mode = self._param("mode", "string_value")
        self.world_name = self._param("world_name", "string_value")
        self.robot_name = self._param("robot_name", "string_value")
        self.sensor_type = self._param("sensor", "string_value")
        self.evaluate = False

        # ROS 2 setup
        self.reset_world_client = self.create_client(Empty, "reset_world")
        self.sensors = Sensors(self)

        # Training variables
        self.episode_step = 0
        self.episode = 0
        self.total_steps = 0 
        self.start_time = time.time()
        self.gazebo_process = None
        self.logdir = self._create_logdir()

        self.get_logger().debug(f"{node_name}: Starting process")

    def step(self, action):
        self._send_action(action)
        self._spin_sensors_callbacks()

        sensor_data = self._get_sensor_data()

        next_obs = self._get_observation(sensor_data)
        terminated, event = self._check_events(sensor_data)
        sensor_data["event"] = event

        reward = self._get_reward(sensor_data)
        truncated = self.episode_step >= self.timeout_steps
        self.episode_step += 1

        return next_obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_steps += self.episode_step
        self.episode += 1
        self.episode_step = 0

        elapsed_time = str(timedelta(seconds=int(time.time() - self.start_time)))
        self.get_logger().info(
            f"[Episode {self.episode} {'(eval)' if self.evaluate else ''}] Total steps: {self.total_steps}, "
            f"Training time: {elapsed_time}"
        )
        logging.info(
            f"Total_episodes: {self.episode}, Total_steps: {self.total_steps}, episode_steps: {self.episode_step + 1}\n"
        )

        self._new_episode()
        self._spin_sensors_callbacks()

        sensor_data = self._get_sensor_data()
        next_obs = self._get_observation(sensor_data)
        
        return next_obs, {}

    def render(self, mode="none"):
        world_path = os.path.join(
            get_package_share_directory(self.package_name),
            "worlds",
            self.world_name,
        )

        headless = mode != "human"
        self.get_logger().info(f"Launching Gazebo {'in headless mode' if headless else 'with GUI'}...")
        self._launch_gazebo(world_path, headless=headless)
        self._spawn_robot()

    def close(self):
        self.get_logger().info("Closing environment...")
        #self.gazebo_process.terminate()
        #self.gazebo_process.wait(timeout=5)
        rclpy.shutdown()

    def _create_logdir(self):
        log_path = os.path.join(
            get_package_share_directory(self.package_name),
            "../../../../",
            self.train_params["--logdir"],
        )
        return create_logdir(self.train_params["--policy"], self.sensor_type, log_path)
    
    def _launch_gazebo(self, world_path, headless=True):
        cmd = ["gazebo", world_path]
        if headless:
            cmd.append("--headless-rendering")
        self.gazebo_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(5)
    
    def _param(self, name, dtype):
        return self.get_parameter(name).get_parameter_value().__getattribute__(dtype)

    def _spin_sensors_callbacks(self):
        self.get_logger().debug("Spinning node until sensors ready...")
        rclpy.spin_once(self)
        retries = 0
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self, timeout_sec=0.1)
            retries += 1
            if retries > 100:
                raise TimeoutError("Sensor data not received after 100 attempts.")
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
        
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
    def _check_events(self, sensor_data):
        """Check for termination events (collision, goal, timeout)."""
        pass

    @abstractmethod
    def _new_episode(self):
        """ """
        pass

    @abstractmethod
    def _get_observation(self, sensor_data):
        """Retrieve next observation"""
        pass

    @abstractmethod
    def _get_reward(self, sensor_data):
        """Calculate the reward based on the event and goal distance."""
        pass

    @abstractmethod
    def _get_sensor_data(self):
        """Retrieve sensor data from the robot."""
        pass  
    
    @abstractmethod
    def _send_action(self, twist):
        """Transform RL action to ROS command."""
        pass