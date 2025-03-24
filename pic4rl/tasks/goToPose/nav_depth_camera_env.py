#!/usr/bin/env python3

import os
import numpy as np
import math
import subprocess
import jsonimport random
import time
import yaml
import logging
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.sensors import Sensors
from pic4rl.utils.env_utils import *
from pic4rl.testing.nav_metrics import Navigation_Metrics

import gymnasium as gym
from gymnasium import spaces


class Pic4rlEnvironmentCamera(Node, gym.Env):
    def __init__(self):
        """ """
        super().__init__("pic4rl_training_camera")

        # ROS parameters
        self.declare_parameter("package_name", "pic4rl")
        self.declare_parameter("training_params_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("main_params_path", rclpy.Parameter.Type.STRING)


        train_params_path = self.get_parameter("training_params_path").get_parameter_value().string_value
        
        with open(train_params_path, "r") as train_param_file:
            train_params = yaml.safe_load(train_param_file)["training_params"]

        self.package_name = self.get_parameter("package_name").get_parameter_value().string_value
        goals_path = os.path.join(get_package_share_directory(self.package_name), "goals_and_poses")
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        goals_path = os.path.join(goals_path, self.mode)
        self.data_path = os.path.join(goals_path, self.get_parameter("data_path").get_parameter_value().string_value)

        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self.visual_data = self.get_parameter("visual_data").get_parameter_value().string_value
        self.features = self.get_parameter("features").get_parameter_value().integer_value
        self.channels = self.get_parameter("channels").get_parameter_value().integer_value
        self.image_width = self.get_parameter("depth_param.width").get_parameter_value().integer_value
        self.image_height = self.get_parameter("depth_param.height").get_parameter_value().integer_value
        self.max_depth = self.get_parameter("depth_param.dist_cutoff").get_parameter_value().double_value
        self.lidar_distance = self.get_parameter("laser_param.max_distance").get_parameter_value().double_value
        self.lidar_points = self.get_parameter("laser_param.num_points").get_parameter_value().integer_value
        self.params_update_freq = self.get_parameter("update_frequency").get_parameter_value().double_value
        self.sensor_type = self.get_parameter("sensor").get_parameter_value().string_value

        # ROS 2 setup
        qos = QoSProfile(depth=10)
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", qos)
        self.reset_world_client = self.create_client(Empty, "reset_world")
        self.pause_physics_client = self.create_client(Empty, "pause_physics")
        self.unpause_physics_client = self.create_client(Empty, "unpause_physics")
        self.sensors = Sensors(self)

        # Gymnasium spaces
        self.image_height = 64   # Example size, you can adjust
        self.image_width = 64    # Example size, you can adjust
        self.num_stacked_images = 4  # Stacking 4 depth images

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.num_stacked_images, self.image_height, self.image_width), 
            dtype=np.float32
        )

        # Initialize image buffer
        self.image_buffer = np.zeros(
            (self.num_stacked_images, self.image_height, self.image_width),
            dtype=np.float32
        )


        # Training variables
        self.episode_step = 0
        self.episode = 0
        self.collision_count = 0
        self.goal_count
        self.timeout_steps = int(self.train_params["--episode-max-steps"])
        self.change_episode = int(self.train_params["--change_goal_and_pose"])
        self.starting_episodes = int(self.train_params["--starting_episodes"])

        
        # Logging and metrics
        self.logdir = self._create_logdir()
        if self.get_parameter("mode").value == "testing":
            self.nav_metrics = Navigation_Metrics(self.logdir)

        # Initialize environment
        self.initial_pose, self.goals, self.poses = self._load_goals_and_poses()
        self.goal_pose = self.goals[0]
        self.previous_twist = Twist()
        self.t0 = 0.0
        self.evaluate = False


        #if "--model-dir" in train_params:
        #    self.model_path = os.path.join(get_package_share_directory(self.package_name),'../../../../', train_params["--model-dir"])

        #if "--rb-path-load" in train_params:
        #    self.rb_path_load = os.path.join(get_package_share_directory(self.package_name),'../../../../', train_params["--rb-path-load"])

        self._spin_sensors_callbacks()

        #elf.get_logger().info(f"Gym mode: {self.mode}")
        #if self.mode == "testing":
        #    self.nav_metrics = Navigation_Metrics(self.logdir)
        self.get_logger().debug("PIC4RL_Environment: Starting process")

    def step(self, action, episode_step=0):
        """ """
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self._send_action(twist)
        self._spin_sensors_callbacks()
        lidar_measurements, depth_image, goal_info, robot_pose, collision = self.get_sensor_data()

        # Update the buffer: prepend the newest image and remove the oldest
        self.image_buffer = np.vstack((depth_image[np.newaxis, :], self.image_buffer[:-1]))

        next_observation = self.image_buffer# self.get_observation(goal_info, depth_image)
        done, event = self._check_events(goal_info, collision)
        reward = self._get_reward(goal_info, event)
        truncated = self.episode_step >= self.timeout_steps
        
        if self.mode == "testing":
            self.nav_metrics.get_metrics_data(lidar_measurements, self.episode_step)

        self.previous_goal_info = goal_info
        self.episode_step += 1
        return next_observation, reward, done, truncated, {}


    def reset(self, n_episode, tot_steps, evaluate=False):
        """ """
        super().reset(seed=seed)
        self.total_steps += self.episode_step
        self.n_episode += 1
        self.episode_step = 0
        self.collision_count = 0

        if self.mode == "testing":
            self.nav_metrics.calc_metrics(n_episode, self.initial_pose, self.goal_pose)
            self.nav_metrics.log_metrics_results(n_episode)
            self.nav_metrics.save_metrics_results(n_episode)

        self.evaluate = evaluate
        logging.info(
            f"Total_episodes: {n_episode}{' evaluation episode' if self.evaluate else ''}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n"
        )
        self._new_episode()

        # Reset the buffer with four copies of the first depth image
        self._spin_sensors_callbacks()
        lidar_measurements, depth_image, goal_info, robot_pose, collision = self._get_sensor_data()
        self.image_buffer = np.stack([depth_image] * self.num_stacked_images, axis=0)

        # Perform initial step to reset variables
        observation = self.image_buffer

        return observation, {}

    def close(self):
        self.get_logger().info("Closing environment...")
        rclpy.shutdown()

    def _create_logdir(self):
        """Create a log directory for training."""
        log_path = os.path.join(
            get_package_share_directory(self.get_parameter("package_name").value),
            "../../../../",
            self.train_params["--logdir"],
        )
        return create_logdir(self.train_params["--policy"], self.get_parameter("sensor").value, log_path)

    def _load_goals_and_poses(self):
        """Load goals and poses from a JSON file."""
        data_path = os.path.join(
            get_package_share_directory(self.get_parameter("package_name").value),
            "goals_and_poses",
            self.get_parameter("mode").value,
            self.get_parameter("data_path").value,
        )
        with open(data_path, "r") as file:
            data = json.load(file)
        return data["initial_pose"], data["goals"], data["poses"]

    def _spin_sensors_callbacks(self):
        """ """
        self.get_logger().debug("spinning node...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            self.get_logger().debug(f"empty_measurements")
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def _send_action(self, twist):
        """ """
        self.cmd_vel_pub.publish(twist)

    def _get_sensor_data(self):
        """Retrieve sensor data from the robot."""
        sensor_data = {
            "scan": self.sensors.get_laser(),
            "odom": self.sensors.get_odom(),
            "depth": self.sensors.get_depth(),
        }
        lidar_measurements = sensor_data["scan"] or np.ones(self.lidar_points) * self.lidar_distance
        depth_image = sensor_data["depth"] or np.ones((self.get_parameter("depth_param.height").value, self.get_parameter("depth_param.width").value, 1)) * self.get_parameter("depth_param.dist_cutoff").value
        odom = sensor_data["odom"] or [0.0, 0.0, 0.0]
        goal_info, _ = self.process_odom(self.goal_pose, odom)
        collision = sensor_data["scan"] is None  # Simplified collision detection

        depth_image_normalized = depth_image / self.max_depth

        return lidar_measurements, depth_image_normalized, goal_info, collision

    def _check_events(self, goal_info, collision):
        """Check for termination events (collision, goal, timeout)."""
        if collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                return True, "collision"
        elif goal_info[0] < self.goal_tolerance:
            self.goal_count += 1
            return True, "goal"
        elif self.episode_step >= self.timeout_steps:
            return True, "timeout"
        return False, "none"

    def _get_reward(self, goal_info, event):
        """Calculate the reward based on the event and goal distance."""
        reward = (self.previous_goal_info[0] - goal_info[0]) * 30
        yaw_reward = (1 - 2 * math.sqrt(abs(goal_info[1] / math.pi))) * 0.6
        reward += yaw_reward

        if event == "goal":
            reward += 1000
        elif event == "collision":
            reward += -200
        #self.get_logger().debug(str(reward))
        return reward

    def _new_episode(self):
        """Reset the environment for a new episode."""
        self._reset_simulation()
        self._respawn_robot()
        self._respawn_goal()

    def _reset_simulation(self):
        """Reset the Gazebo simulation."""
        req = Empty.Request()
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Service not available, waiting again...")
        self.reset_world_client.call_async(req)

    def _respawn_robot(self):
        """Respawn the robot in a new position."""
        if self.episode <= self.starting_episodes:
            x, y, yaw = self.initial_pose
        else:
            index = random.randint(0, len(self.poses) - 1)
            x, y, yaw = self.poses[index]

        qz = np.sin(yaw / 2)
        qw = np.cos(yaw / 2)
        state = f"'{'{state: {name: '{self.robot_name}', pose: {position: {x: {x}, y: {y}, z: 0.07}}, orientation: {z: {qz}, w: {qw}}}}'}'"
        subprocess.run(
            f"ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState {state}",
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        time.sleep(0.25)

    def _respawn_goal(self):
        """Respawn the goal in a new position."""
        if self.episode <= self.starting_episodes:
            x = random.uniform(-2.9, 2.9) + self.initial_pose[0]
            y = random.uniform(-2.9, 2.9) + self.initial_pose[1]
            self.goal_pose = [x, y]
        else:
            index = random.randint(0, len(self.goals) - 1)
            self.goal_pose = self.goals[index]

        self.get_logger().info(f"New goal pose: {self.goal_pose}")
"""
    def pause(self):
        """ """
        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again
            ...")
        future = self.pause_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def unpause(self):
        """ """
        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")
        future = self.unpause_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
"""

