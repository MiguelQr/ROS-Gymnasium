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
        self._log_results()
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
    def _log_results(self):
        """Log results to file."""
        pass
    
    @abstractmethod
    def _send_action(self, twist):
        """Transform RL action to ROS command."""
        pass
    
    def _spin_all_sensors_callbacks(self, max_attempts=100):
        """Spin node to receive callbacks for all sensors."""
        # First, check if the robots are actually spawned in Gazebo
        try:
            from gazebo_msgs.msg import ModelStates
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            
            # Create a temporary subscription to check model states
            qos = QoSProfile(depth=1)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            
            model_states = None
            def model_callback(msg):
                nonlocal model_states
                model_states = msg.name
            
            model_sub = self.node.create_subscription(
                ModelStates, 
                '/model_states', 
                model_callback,
                qos
            )
            
            # Try to get model states to see if robots exist
            attempt = 0
            while model_states is None and attempt < 10:
                rclpy.spin_once(self.node, timeout_sec=0.5)
                attempt += 1
            
            if model_states:
                existing_robots = [name for name in self.robot_namespaces if name in model_states]
                missing_robots = [name for name in self.robot_namespaces if name not in model_states]
                
                if missing_robots:
                    self._logger.error(f"Missing robots in Gazebo: {missing_robots}")
                    self._logger.error(f"Available models in Gazebo: {model_states}")
                else:
                    self._logger.info(f"All robots successfully spawned in Gazebo: {existing_robots}")
            else:
                self._logger.error("Could not get model states from Gazebo")
                
            # Clean up temporary subscription
            self.node.destroy_subscription(model_sub)
        
        except Exception as e:
            self._logger.error(f"Error checking robot existence: {str(e)}")
        
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