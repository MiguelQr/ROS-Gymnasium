#!/usr/bin/env python3

import os
import yaml
import json
import launch
import re
import random
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml

def camel_to_snake(s):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def generate_random_position():
    """Generate a random position in one of the four quadrants."""
    quadrant = random.randint(1, 4)
    
    if quadrant == 1:  # Bottom-left quadrant
        x = random.uniform(-5, -1)
        y = random.uniform(-5, -1)
    elif quadrant == 2:  # Top-right quadrant
        x = random.uniform(1, 5)
        y = random.uniform(1, 5)
    elif quadrant == 3:  # Top-left quadrant
        x = random.uniform(-5, -1)
        y = random.uniform(1, 5)
    else:  # Bottom-right quadrant
        x = random.uniform(1, 5)
        y = random.uniform(-5, -1)
    
    # Random orientation (yaw)
    yaw = random.uniform(0, 6.28)
    
    return [x, y, yaw]

def prepare_launch(context, *args, **kwargs):
    # Get all configuration parameters
    sensor = LaunchConfiguration("sensor").perform(context=context)
    task = LaunchConfiguration("task").perform(context=context)
    mode = LaunchConfiguration("mode").perform(context=context)
    pkg_name = LaunchConfiguration("pkg_name").perform(context=context)
    main_params_path = LaunchConfiguration("main_params").perform(context=context)
    training_params_path = LaunchConfiguration("training_params").perform(context=context)
    world_name_arg = LaunchConfiguration("world_name").perform(context=context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context=context) == "true"
    
    with open(main_params_path, 'r') as file:
        config_params = yaml.safe_load(file)["main_node"]["ros__parameters"]

    # Process configuration
    if sensor or task or mode:
        # Create parameter rewrites dictionary
        param_rewrites = {}
        if sensor:
            param_rewrites["sensor"] = sensor
        if task:
            param_rewrites["task"] = task
        if mode:
            param_rewrites["mode"] = mode
        if world_name_arg and world_name_arg != "empty.world":
            param_rewrites["world_name"] = world_name_arg
            
        # Load original parameters
        with open(main_params_path, 'r') as file:
            config_params = yaml.safe_load(file)["main_node"]["ros__parameters"]
            
        # Apply the rewrites manually
        for key, value in param_rewrites.items():
            config_params[key] = value
            
        # Create a proper parameters dictionary for the node
        configured_params = config_params
    else:
        # If no substitutions, just load the file directly
        with open(main_params_path, 'r') as file:
            config_params = yaml.safe_load(file)["main_node"]["ros__parameters"]
        configured_params = config_params
    
    
    # Get the final world name from config_params (either from file or overridden)
    world_name = config_params["world_name"]
    

    # Set up for rosgym node
    sensor_name = config_params["sensor"]
    task_name = config_params["task"]
    executable_name = camel_to_snake(task_name) + "_" + camel_to_snake(sensor_name)
    
    # Fetch goals and poses
    goals_path = os.path.join(
        get_package_share_directory(pkg_name),
        'goals_and_poses',
        config_params['mode'],
        config_params['data_path']
    )
    with open(goals_path, 'r') as file:
        goal_and_poses = json.load(file)
    
    #robot_pose = goal_and_poses["initial_pose"]
    robot_pose = generate_random_position()
    goal_pose = goal_and_poses["goals"][0]
    
    # Prepare world, robot and goal paths
    world_path = os.path.join(
        get_package_share_directory(pkg_name),  # Keep reference to gazebo_sim for now
        'worlds', 
        config_params["world_name"]
    )
    
    print("\n=============================================")
    print(f"Loading world: {config_params['world_name']}")
    print(f"Full path: {world_path}")
    print("=============================================\n")
    
    robot_pkg = get_package_share_directory(config_params["robot_name"])
    
    goal_entity = os.path.join(
        get_package_share_directory(pkg_name), 
        'models', 
        'goal_box', 
        'model.sdf'
    )
    
    # Launch actions
    actions = []
    
    # Add robot description
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_pkg, 'launch', 'description.launch.py')
        )
    )
    actions.append(robot_description)
    
    # Add robot spawning
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', config_params["robot_name"], 
            '-x', str(robot_pose[0]),
            '-y', str(robot_pose[1]), 
            '-z', '0.1',
            '-Y', str(robot_pose[2]),
            '-topic', '/robot_description'
        ],
    )
    actions.append(spawn_robot)
    
    # Add goal spawning
    spawn_goal = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', 'goal', 
            '-file', goal_entity, 
            '-x', str(goal_pose[0]),
            '-y', str(goal_pose[1])
        ]
    )
    actions.append(spawn_goal)
    
    # Add Gazebo 
    gazebo = launch.actions.ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen',
    )

    actions.append(TimerAction(period=5.0, actions=[gazebo]))
    
    # Add rosgym node
    rosgym_node = Node(
        package=pkg_name,
        executable=executable_name,
        name="rosgym_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            # Pass the actual dictionary instead of RewrittenYaml
            configured_params,
            {"package_name": pkg_name},
            {"main_params_path": main_params_path},
            {"training_params_path": training_params_path},
            {"use_sim_time": use_sim_time}
        ],
    )
    actions.append(TimerAction(period=50.0, actions=[rosgym_node]))
    
    return actions

def generate_launch_description():
    # Declare all the launch arguments
    launch_args = [
        DeclareLaunchArgument(
            "sensor", 
            default_value="Camera", 
            description="Sensor type: Camera or Lidar"
        ),
        DeclareLaunchArgument(
            "task", 
            default_value="GoToPose", 
            description="Task type: GoToPose, Following, Vineyards"
        ),
        DeclareLaunchArgument(
            "mode", 
            default_value="training", 
            description="Mode: training or testing"
        ),
        DeclareLaunchArgument(
            "pkg_name", 
            default_value="rosgym", 
            description="Package name"
        ),
        DeclareLaunchArgument(
            "main_params", 
            default_value=os.path.join(
                get_package_share_directory("rosgym"), 
                "config", 
                "main_params.yaml"
            ),
            description="Path to main parameters file"
        ),
        DeclareLaunchArgument(
            "training_params", 
            default_value=os.path.join(
                get_package_share_directory("rosgym"), 
                "config", 
                "training_params.yaml"
            ),
            description="Path to training parameters file"
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="empty.world",
            description="Gazebo world file name"
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock if true"
        )
    ]

    # Create the launch description
    ld = LaunchDescription(launch_args)
    
    # Add the opaque function
    ld.add_action(OpaqueFunction(function=prepare_launch))
    
    return ld