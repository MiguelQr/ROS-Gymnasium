#!/usr/bin/env python3

"""Multi-robot training launch (work in progress)."""

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

def camel_to_snake(s):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def generate_non_overlapping_positions(num_positions, grid_size=10, min_distance=4.0):
    """Generate non-overlapping positions in the grid."""
    positions = []
    attempts = 0
    max_attempts = 100
    
    while len(positions) < num_positions and attempts < max_attempts:
        quadrant = random.randint(1, 4)
        
        if quadrant == 1:
            x = random.uniform(-grid_size, -2)
            y = random.uniform(-grid_size, -2)
        elif quadrant == 2:
            x = random.uniform(2, grid_size)
            y = random.uniform(2, grid_size)
        elif quadrant == 3:
            x = random.uniform(-grid_size, -2)
            y = random.uniform(2, grid_size)
        else:
            x = random.uniform(2, grid_size)
            y = random.uniform(-grid_size, -2)
            
        valid = True
        for px, py in positions:
            distance = ((x - px)**2 + (y - py)**2)**0.5
            if distance < min_distance:
                valid = False
                break
                
        if valid:
            positions.append((x, y))
        
        attempts += 1
        
    if len(positions) < num_positions:
        print("Could not find enough non-overlapping positions, using grid layout")
        
        grid_dim = int((num_positions)**0.5) + 1
        spacing = grid_size / (grid_dim + 1)
        
        positions = []
        for i in range(num_positions):
            row = i // grid_dim
            col = i % grid_dim
            x = -grid_size/2 + (col + 1) * spacing
            y = -grid_size/2 + (row + 1) * spacing
            positions.append((x, y))
            
    return positions

def prepare_launch(context, *args, **kwargs):
    sensor = LaunchConfiguration("sensor").perform(context=context)
    task = LaunchConfiguration("task").perform(context=context)
    pkg_name = LaunchConfiguration("pkg_name").perform(context=context)
    main_params_path = LaunchConfiguration("main_params").perform(context=context)
    training_params_path = LaunchConfiguration("training_params").perform(context=context)
    world_name_arg = LaunchConfiguration("world_name").perform(context=context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context=context) == "true"
    num_robots = int(LaunchConfiguration("num_robots").perform(context=context))
    
    with open(main_params_path, 'r') as file:
        config_params = yaml.safe_load(file)["main_node"]["ros__parameters"]

    for key, value in (("sensor", sensor), ("task", task), ("world_name", world_name_arg)):
        if value:
            config_params[key] = value
    configured_params = config_params

    world_name = config_params["world_name"]
    
    sensor_name = config_params["sensor"]
    task_name = config_params["task"]
    executable_name = "multi_agent_training"
    
    goals_path = os.path.join(
        get_package_share_directory(pkg_name),
        'goals_and_poses',
        config_params['data_path']
    )
    with open(goals_path, 'r') as file:
        goal_and_poses = json.load(file)
    
    robot_positions = generate_non_overlapping_positions(num_robots, grid_size=10, min_distance=4.0)
    goal_positions = generate_non_overlapping_positions(num_robots, grid_size=10, min_distance=4.0)
    
    world_path = os.path.join(
        get_package_share_directory(pkg_name),
        'worlds', 
        config_params["world_name"]
    )
    
    print("\n=============================================")
    print(f"Loading world: {config_params['world_name']} with {num_robots} robots")
    print(f"Full path: {world_path}")
    print("=============================================\n")
    
    robot_pkg = get_package_share_directory(config_params["robot_name"])
    
    goal_entity = os.path.join(
        get_package_share_directory(pkg_name), 
        'models', 
        'goal_box', 
        'model.sdf'
    )
    
    actions = []
    
    gazebo = launch.actions.ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen',
    )
    actions.append(gazebo)
    
    actions.append(TimerAction(period=5.0, actions=[]))
    
    for i in range(num_robots):
        namespace = f"robot{i}"
        x, y = robot_positions[i]
        yaw = random.uniform(0, 6.28)
        
        urdf_path = os.path.join(robot_pkg, 'urdf', 'tb2.urdf.xacro')

        from xacro import process_file, process_doc
        import xml.etree.ElementTree as ET

        xacro_args = {
            'name_suffix': namespace
        }

        urdf_str = ""
        if os.path.exists(urdf_path):
            urdf_str = process_file(urdf_path, mappings=xacro_args).toxml()
        else:
            print(f"[ERROR] URDF path not found: {urdf_path}")
        
        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=namespace,
            output='screen',
            parameters=[{
                'robot_description': urdf_str,
                'use_sim_time': use_sim_time
            }],
        )
        actions.append(robot_state_publisher)
        
        joint_state_publisher = Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            namespace=namespace,
            output='screen',
        )
        actions.append(joint_state_publisher)
        
        actions.append(TimerAction(period=4.0, actions=[]))
        
        spawn_robot = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            output='screen',
            arguments=[
                '-entity', namespace,
                '-topic', f'/{namespace}/robot_description',
                '-x', str(x),
                '-y', str(y), 
                '-z', '0.05',
                '-Y', str(yaw),
                '-robot_namespace', namespace
            ]
        )
        actions.append(spawn_robot)
        
        actions.append(TimerAction(period=2.0, actions=[]))
        
        gx, gy = goal_positions[i]
        spawn_goal = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            output='screen',
            arguments=[
                '-entity', f'goal_{i}', 
                '-file', goal_entity, 
                '-x', str(gx),
                '-y', str(gy), 
                '-z', '0.05'
            ]
        )
        actions.append(spawn_goal)
    
    print(f"Looking for URDF file at {urdf_path}")
    if not os.path.exists(urdf_path):
        print(f"WARNING: URDF file not found at {urdf_path}")
        alt_urdf_path = os.path.join(robot_pkg, 'urdf', 'tb2.urdf.xacro')
        if os.path.exists(alt_urdf_path):
            print(f"Found alternative URDF at {alt_urdf_path}")
            urdf_path = alt_urdf_path
        else:
            print("Searching for any URDF files in the package...")
            for root, _, files in os.walk(robot_pkg):
                for file in files:
                    if file.endswith('.urdf') or file.endswith('.xacro'):
                        print(f"Found potential URDF: {os.path.join(root, file)}")
    
    training_node = Node(
        package=pkg_name,
        executable=executable_name,
        name="multi_agent_training_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            configured_params,
            {"package_name": pkg_name},
            {"main_params_path": main_params_path},
            {"training_params_path": training_params_path},
            {"use_sim_time": use_sim_time},
            {"num_envs": num_robots}
        ],
        arguments=[
            '--num_envs', str(num_robots),
            '--run_name', f'multi_robot_nav_{num_robots}',
            '--total_timesteps', '1000000',
        ]
    )
    
    actions.append(TimerAction(period=60.0, actions=[training_node]))
    
    return actions

def generate_launch_description():
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
        ),
        DeclareLaunchArgument(
            "num_robots",
            default_value="4",
            description="Number of robots to spawn"
        )
    ]

    ld = LaunchDescription(launch_args)
    
    ld.add_action(OpaqueFunction(function=prepare_launch))
    
    return ld
