#!/usr/bin/env python3

import os
import yaml
import json
import launch
import re
import random
import math
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml

TRAINING_EXECUTABLES = ["go_to_pose_camera"]  # Installed by CMakeLists.txt

def camel_to_snake(s):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def generate_random_position(grid_size):
    """Random [x, y, yaw] in one of the four quadrants, between 1 m and grid_size from both axes."""
    x = random.choice([-1.0, 1.0]) * random.uniform(1, grid_size)
    y = random.choice([-1.0, 1.0]) * random.uniform(1, grid_size)
    yaw = random.uniform(0, 2 * math.pi)
    return [x, y, yaw]

def prepare_launch(context, *args, **kwargs):
    # Get all configuration parameters
    sensor = LaunchConfiguration("sensor").perform(context=context)
    task = LaunchConfiguration("task").perform(context=context)
    pkg_name = LaunchConfiguration("pkg_name").perform(context=context)
    main_params_path = LaunchConfiguration("main_params").perform(context=context)
    training_params_path = LaunchConfiguration("training_params").perform(context=context)
    world_name_arg = LaunchConfiguration("world_name").perform(context=context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context=context) == "true"
    
    with open(main_params_path, 'r') as file:
        config_params = yaml.safe_load(file)["main_node"]["ros__parameters"]

    # Launch arguments override main_params.yaml when given
    for key, value in (("sensor", sensor), ("task", task), ("world_name", world_name_arg)):
        if value:
            config_params[key] = value
    configured_params = config_params

    # Get the final world name from config_params (either from file or overridden)
    world_name = config_params["world_name"]
    

    # Set up for ros_gymnasium node
    sensor_name = config_params["sensor"]
    task_name = config_params["task"]
    executable_name = camel_to_snake(task_name) + "_" + camel_to_snake(sensor_name)
    if executable_name not in TRAINING_EXECUTABLES:
        raise RuntimeError(
            f"No training executable '{executable_name}' for task '{task_name}' and sensor '{sensor_name}'. "
            f"Available: {TRAINING_EXECUTABLES}"
        )
    
    # Fetch goals and poses
    goals_path = os.path.join(
        get_package_share_directory(pkg_name),
        'goals_and_poses',
        config_params['data_path']
    )
    with open(goals_path, 'r') as file:
        goal_and_poses = json.load(file)
    
    #robot_pose = goal_and_poses["initial_pose"]
    robot_pose = generate_random_position(config_params["grid_size"])
    goal_pose = goal_and_poses["goals"][0]
    
    # Prepare world, robot and goal paths
    world_path = os.path.join(
        get_package_share_directory(pkg_name),  # Keep reference to gazebo_sim for now
        'worlds', 
        config_params["world_name"]
    )
    
    if not os.path.isfile(world_path):
        worlds = sorted(os.listdir(os.path.dirname(world_path)))
        raise RuntimeError(f"World '{world_name}' not found. Available: {worlds}")

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
            '-z', str(config_params['robot_spawn_z']),
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
            '-y', str(goal_pose[1]), 
            '-z', str(config_params['goal_spawn_z'])
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
    
    # Add ros_gymnasium node
    ros_gymnasium_node = Node(
        package=pkg_name,
        executable=executable_name,
        name="ros_gymnasium_node",
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
    actions.append(TimerAction(period=50.0, actions=[ros_gymnasium_node]))
    
    return actions

def generate_launch_description():
    # Declare all the launch arguments
    launch_args = [
        DeclareLaunchArgument(
            "sensor", 
            default_value="",
            description="Sensor type (default: sensor in main_params.yaml); only camera is implemented"
        ),
        DeclareLaunchArgument(
            "task", 
            default_value="",
            description="Task type (default: task in main_params.yaml); only goToPose is implemented"
        ),
        DeclareLaunchArgument(
            "pkg_name", 
            default_value="ros_gymnasium", 
            description="Package name"
        ),
        DeclareLaunchArgument(
            "main_params", 
            default_value=os.path.join(
                get_package_share_directory("ros_gymnasium"), 
                "config", 
                "main_params.yaml"
            ),
            description="Path to main parameters file"
        ),
        DeclareLaunchArgument(
            "training_params", 
            default_value=os.path.join(
                get_package_share_directory("ros_gymnasium"), 
                "config", 
                "training_params.yaml"
            ),
            description="Path to training parameters file"
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="",
            description="World file in worlds/ (default: world_name in main_params.yaml)"
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