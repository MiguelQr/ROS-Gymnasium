#!/usr/bin/env python3

import os
import math
import yaml
import logging
import time
from datetime import timedelta
from abc import ABC, abstractmethod

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from ament_index_python.packages import get_package_share_directory

from ros_gymnasium.sensors import Sensors
from ros_gymnasium.utils.env_utils import create_logdir
import gymnasium as gym

def _param(self, name: str, type_: str):
    return getattr(self.get_parameter(name).get_parameter_value(), type_)

class ROSGymnasiumEnv(Node, gym.Env, ABC):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, node_name: str = "rl_base_env"):
        super().__init__(node_name=node_name)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("package_name", rclpy.Parameter.Type.STRING),
                ("training_params_path", rclpy.Parameter.Type.STRING),
                ("main_params_path", rclpy.Parameter.Type.STRING),
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
        self.world_name = self._param("world_name", "string_value")
        self.robot_name = self._param("robot_name", "string_value")
        self.sensor_type = self._param("sensor", "string_value")
        self.evaluate = False

        # ROS 2 setup
        self.reset_world_client = self.create_client(Empty, "reset_world")
        self.set_entity_state_client = self.create_client(SetEntityState, "/set_entity_state")
        self.sensors = Sensors(self)

        # Training variables
        self.episode_step = 0
        self.episode = 0
        self.total_steps = 0 
        self.start_time = time.time()
        self.logdir = self._create_logdir()

        self.get_logger().debug(f"{node_name}: Starting process")

    def step(self, action):
        self._send_action(action)
        self._spin_sensors_callbacks()
        self.episode_step += 1

        sensor_data = self._get_sensor_data()

        next_obs = self._get_observation(sensor_data)
        terminated, event = self._check_events(sensor_data)

        # Hitting the time limit is a truncation, not a terminal state: the
        # robot could have kept going, so the agent must still bootstrap.
        truncated = not terminated and self.episode_step >= self.timeout_steps
        if truncated:
            event = "timeout"
        sensor_data["event"] = event

        reward = self._get_reward(sensor_data)

        return next_obs, reward, terminated, truncated, {"event": event}

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

    def render(self):
        """Return the latest camera frame as an (H, W, 3) uint8 array when render_mode is "rgb_array"."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self):
        self.get_logger().info("Closing environment...")
        self._log_results()
        rclpy.shutdown()

    def _create_logdir(self):
        log_path = os.path.join(
            get_package_share_directory(self.package_name),
            "../../../../",
            self.train_params["logdir"],
        )
        return create_logdir(self.train_params["algorithm"], self.sensor_type, log_path)
    
    def _emergency_reset(self):
        """
        Emergency reset when sensors stop responding.
        Attempts to reset the simulation to recover from potential physics issues.
        
        Returns:
            bool: True if reset was successful
        """
        try:
            self.get_logger().warn("Attempting emergency robot reset...")
            
            if hasattr(self, 'cmd_vel_pub'):
                from geometry_msgs.msg import Twist
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            
            if not self.reset_world_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("Reset world service not available")
                return False
                
            req = Empty.Request()
            future = self.reset_world_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            time.sleep(0.2)
            
            self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
            recovery_attempts = 0
            while None in self.sensors.sensor_msg.values() and recovery_attempts < 20:
                rclpy.spin_once(self, timeout_sec=0.1)
                recovery_attempts += 1
            
            missing = [k for k, v in self.sensors.sensor_msg.items() if v is None]
            if len(missing) < len(self.sensors.sensor_msg):
                self.get_logger().info(f"Received some sensor data, still missing: {missing}")
                return True
            else:
                self.get_logger().error("No sensor data received after reset")
                return False
            
        except Exception as e:
            self.get_logger().error(f"Emergency reset failed: {e}")
            return False
    
    def _set_entity_state(self, name, x, y, z=0.05, yaw=0.0):
        """
        Teleport a Gazebo entity to (x, y, z) with heading yaw and zero velocity.

        Returns:
            bool: True if Gazebo confirmed the new state
        """
        request = SetEntityState.Request()
        request.state.name = name
        request.state.reference_frame = "world"
        request.state.pose.position.x = float(x)
        request.state.pose.position.y = float(y)
        request.state.pose.position.z = float(z)
        request.state.pose.orientation.z = math.sin(yaw / 2.0)
        request.state.pose.orientation.w = math.cos(yaw / 2.0)

        future = self.set_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if future.result() is None:
            self.get_logger().error(f"set_entity_state call timed out for '{name}'")
            return False
        return future.result().success

    def _param(self, name, dtype):
        return self.get_parameter(name).get_parameter_value().__getattribute__(dtype)

    def _spin_sensors_callbacks(self, max_attempts=100):
        #self.get_logger().debug("Spinning node until sensors ready...")
        rclpy.spin_once(self)
        retries = 0
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self, timeout_sec=0.1)
            retries += 1
        
            # Log which sensors we're waiting for if we've waited a while
            if retries == 50:
                missing = [k for k, v in self.sensors.sensor_msg.items() if v is None]
                self.get_logger().warn(f"Waiting for sensors: {missing}")
        
            # Try recovery if we timeout
            if retries > max_attempts:
                missing = [k for k, v in self.sensors.sensor_msg.items() if v is None]
                self.get_logger().error(f"Sensor timeout: missing {missing} after {max_attempts} attempts.")
                
                # Attempt emergency recovery
                if self._emergency_reset():
                    self.get_logger().info("Emergency recovery successful")
                    break
                else:
                    self.get_logger().error("Emergency recovery failed")
                    raise TimeoutError(f"Sensor data not received after {max_attempts} attempts.")

        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
        
    def _render_frame(self):
        """Latest camera frame as an (H, W, 3) uint8 array, or None if the task has no camera."""
        return None

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
    def _log_results(self):
        """Log results to file."""
        pass
    
    @abstractmethod
    def _send_action(self, twist):
        """Transform RL action to ROS command."""
        pass
