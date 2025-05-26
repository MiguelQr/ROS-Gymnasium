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

def generate_non_overlapping_positions(num_positions, grid_size=10, min_distance=4.0):
    """Generate non-overlapping positions in the grid."""
    positions = []
    attempts = 0
    max_attempts = 100
    
    while len(positions) < num_positions and attempts < max_attempts:
        # Generate a random position
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
            
        # Check if this position is far enough from all existing positions
        valid = True
        for px, py in positions:
            distance = ((x - px)**2 + (y - py)**2)**0.5
            if distance < min_distance:
                valid = False
                break
                
        if valid:
            positions.append((x, y))
        
        attempts += 1
        
    # If we couldn't find enough non-overlapping positions, just place them in a grid
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
    # Get all configuration parameters
    sensor = LaunchConfiguration("sensor").perform(context=context)
    task = LaunchConfiguration("task").perform(context=context)
    mode = LaunchConfiguration("mode").perform(context=context)
    pkg_name = LaunchConfiguration("pkg_name").perform(context=context)
    main_params_path = LaunchConfiguration("main_params").perform(context=context)
    training_params_path = LaunchConfiguration("training_params").perform(context=context)
    world_name_arg = LaunchConfiguration("world_name").perform(context=context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context=context) == "true"
    num_robots = int(LaunchConfiguration("num_robots").perform(context=context))
    
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
    executable_name = "multi_agent_training"  # Use our multi-agent training script
    
    # Fetch goals and poses
    goals_path = os.path.join(
        get_package_share_directory(pkg_name),
        'goals_and_poses',
        config_params['mode'],
        config_params['data_path']
    )
    with open(goals_path, 'r') as file:
        goal_and_poses = json.load(file)
    
    # Generate robot and goal positions
    robot_positions = generate_non_overlapping_positions(num_robots, grid_size=10, min_distance=4.0)
    goal_positions = generate_non_overlapping_positions(num_robots, grid_size=10, min_distance=4.0)
    
    # Prepare world, robot and goal paths
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
    
    # Launch actions
    actions = []
    
    # Add Gazebo first (like in training.launch.py)
    gazebo = launch.actions.ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen',
    )
    actions.append(gazebo)
    
    # Wait for Gazebo to initialize before adding robots
    actions.append(TimerAction(period=5.0, actions=[]))
    
    # Add robot descriptions and spawn robots
    for i in range(num_robots):
        namespace = f"robot{i}"
        x, y = robot_positions[i]
        yaw = random.uniform(0, 6.28)
        
        # Create an explicit robot state publisher for each robot with its description
        urdf_path = os.path.join(robot_pkg, 'urdf', 'tb2.urdf.xacro')

        from xacro import process_file, process_doc
        import xml.etree.ElementTree as ET

        # Process the xacro file with the name_suffix parameter set to the robot's namespace ID
        xacro_args = {
            'name_suffix': namespace
        }

        urdf_str = ""
        if os.path.exists(urdf_path):
            urdf_str = process_file(urdf_path, mappings=xacro_args).toxml()
        else:
            print(f"[ERROR] URDF path not found: {urdf_path}")
        
        # Use robot state publisher directly with the URDF content rather than a launch file
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
        
        # Add a joint state publisher for each robot
        joint_state_publisher = Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            namespace=namespace,
            output='screen',
        )
        actions.append(joint_state_publisher)
        
        # Wait for the robot description to be published
        actions.append(TimerAction(period=4.0, actions=[]))
        
        # Spawn robot - directly in the namespace without the leading slash in the topic name
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
        
        # Add a delay between each robot spawn
        actions.append(TimerAction(period=2.0, actions=[]))
        
        # Spawn goal for this robot
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
    
    # Try to locate the actual URDF file for debugging
    print(f"Looking for URDF file at {urdf_path}")
    if not os.path.exists(urdf_path):
        print(f"WARNING: URDF file not found at {urdf_path}")
        # Try alternative locations
        alt_urdf_path = os.path.join(robot_pkg, 'urdf', 'tb2.urdf.xacro')
        if os.path.exists(alt_urdf_path):
            print(f"Found alternative URDF at {alt_urdf_path}")
            urdf_path = alt_urdf_path
        else:
            # Find any URDF files in the package
            print("Searching for any URDF files in the package...")
            for root, _, files in os.walk(robot_pkg):
                for file in files:
                    if file.endswith('.urdf') or file.endswith('.xacro'):
                        print(f"Found potential URDF: {os.path.join(root, file)}")
    
    # Add the training script as a node with increased delay
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
    
    # Increase the delay to give enough time for all robots and sensors to initialize
    actions.append(TimerAction(period=60.0, actions=[training_node]))
    
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
            default_value="gridworld_4x4.world",
            description="Gazebo world file name"
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

    # Create the launch description
    ld = LaunchDescription(launch_args)
    
    # Add the opaque function
    ld.add_action(OpaqueFunction(function=prepare_launch))
    
    return ld
