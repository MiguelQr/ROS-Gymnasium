#!/usr/bin/env python3

import os
import numpy as np
import math
import random
import time
import json

import rclpy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from rosgym.utils.env_utils import process_odom
from rosgym.ros_gym_env import ROSGymEnv

from gymnasium import spaces

from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist

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
                ("visual_data", rclpy.Parameter.Type.STRING),
                ("num_features", rclpy.Parameter.Type.INTEGER),
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
        self.visual_data = self._param("visual_data", "string_value")
        self.num_features = self._param("num_features", "integer_value")
        
        self.image_width = self._param("depth_param.width", "integer_value")
        self.image_height = self._param("depth_param.height", "integer_value")
        self.stacked_images = self._param("depth_param.stacked_images", "integer_value")
        self.max_depth = self._param("depth_param.dist_cutoff", "double_value")

        self.lidar_distance = self._param("laser_param.max_distance", "double_value")
        self.lidar_points = self._param("laser_param.num_points", "integer_value")
        
        # Gymnasium spaces
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),  # Linear velocity can only be 0 or positive
            high=np.array([1.0, 1.0]), 
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

        self.reset_client = self.create_client(SetEntityState, '/set_entity_state')

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
        self.grid_size = 3

        self._spin_sensors_callbacks()

        print("Environment Initialized")

        self.get_logger().debug("ROSGYM_Environment: Starting process")

    def _new_episode(self):
        """Reset the environment for a new episode."""
        self.get_logger().info(f"Starting new episode {self.episode}.")
        self.collision_count = 0

        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_twist)

        self._reset_simulation()
        self._respawn_robot()
        self._respawn_goal()
        time.sleep(0.25)

    def _check_events(self, sensor_data):
        """Check for termination events (collision, goal, timeout)."""
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
        elif self.episode_step >= self.timeout_steps:
            return True, "timeout"
        
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
        """Calculate the reward based on the event and goal distance.
        Implements a sparse reward:
        - Success: 1 - 0.9 * (step_count / max_steps)
        - Failure or ongoing: 0
        """
        event = sensor_data["event"]
        
        #self.previous_goal_info = sensor_data["goal_info"]
        
        reward = 0.0
        
        if event == "goal":
            reward = 1.0 - 0.9 * (self.episode_step / self.timeout_steps)
        
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
    
    def _reset_simulation(self):
        """Reset the Gazebo simulation."""
        req = Empty.Request()
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Service not available, waiting again...")
        
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        time.sleep(0.1)  

    def _respawn_goal(self):
        """Respawn the goal in a new position."""

        quadrant = random.randint(1, 4)
        
        if quadrant == 1:  
            x = random.uniform(-self.grid_size, -1)
            y = random.uniform(-self.grid_size, -1)
        elif quadrant == 2:  
            x = random.uniform(1, self.grid_size)
            y = random.uniform(1, self.grid_size)
        elif quadrant == 3: 
            x = random.uniform(-self.grid_size, -1)
            y = random.uniform(1, self.grid_size)
        else: 
            x = random.uniform(1, self.grid_size)
            y = random.uniform(-self.grid_size, -1)
    
        self.goal_pose = [x, y]
    
        request = SetEntityState.Request()
        state = EntityState()
        state.name = 'goal' 
        state.reference_frame = "world"

        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = 0.05
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        request.state = state

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f"Goal teleported to [{x:.2f}, {y:.2f}]")
                return True
            else:
                self.get_logger().error("Goal teleportation failed (service returned False)")
        else:
            self.get_logger().error("Goal teleportation service call timed out")
        
        self.get_logger().warn("Using logical goal position even though teleportation failed")
        return False

    def _respawn_robot(self):
        """Respawn the robot in a new position"""

        quadrant = random.randint(1, 4)
        
        if quadrant == 1:
            x = random.uniform(-self.grid_size, -1)
            y = random.uniform(-self.grid_size, -1)
        elif quadrant == 2:
            x = random.uniform(1, self.grid_size)
            y = random.uniform(1, self.grid_size)
        elif quadrant == 3:
            x = random.uniform(-self.grid_size, -1)
            y = random.uniform(1, self.grid_size)
        else:
            x = random.uniform(1, self.grid_size)
            y = random.uniform(-self.grid_size, -1)

        yaw = random.uniform(0, 2 * math.pi)
    
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0) 
        
        request = SetEntityState.Request()
        state = EntityState()
        state.name = self.robot_name
        state.reference_frame = "world"

        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = 0.1
    
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = float(qz)
        state.pose.orientation.w = float(qw)

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        request.state = state

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f"Robot teleported to [{x:.2f}, {y:.2f}]")
                return True
            else:
                self.get_logger().error("Teleportation failed (service returned False)")
        else:
            self.get_logger().error("Service call timed out")
        
        return self._reset_simulation()

    def _send_action(self, action):

        twist = Twist()
        linear_x = float(action[0])
        if linear_x < 0:
            linear_x = 0.0  
        
        twist.linear.x = linear_x
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