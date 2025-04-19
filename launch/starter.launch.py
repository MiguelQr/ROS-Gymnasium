from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import yaml
from nav2_common.launch import RewrittenYaml

def camel_to_snake(s):
    if len(s) <= 1:
        return s.lower()
    # Changing the first character of the input string to lowercase
    # and calling the recursive function on the modified string
    return cameltosnake(s[0].lower() + s[1:])


def print_params(context, *args, **kwargs):
    sensor = LaunchConfiguration("sensor").perform(context=context)
    task = LaunchConfiguration("task").perform(context=context)
    mode = LaunchConfiguration("mode").perform(context=context)
    pkg_name = LaunchConfiguration("pkg_name").perform(context=context)
    main_params = LaunchConfiguration("main_params").perform(context=context)
    training_params = LaunchConfiguration("training_params").perform(context=context)

    if sensor or task or mode:
        configured_params = RewrittenYaml(
            source_file=main_params,
            param_rewrites={
                "sensor": sensor,
                "task": task,
                "mode": mode,
            },
            convert_types=True,
        )
        print("Parameters substituted")
    else:
        configured_params = main_params

    with open(configured_params, "r") as file:
        main_params_dict = yaml.safe_load(file)["main_node"]['ros__parameters']
        print(main_params_dict)

        sensor_name = main_params_dict["sensor"]
        task_name = main_params_dict["task"]
        executable_name = camel_to_snake(task_name) + "_" + camel_to_snake(sensor_name)
        print(executable_name)

    task_node = Node(
        package=pkg_name,
        executable=executable_name,
        name="pic4rl_starter",
        output="screen",
        emulate_tty=True,
        parameters=[
            main_params_dict,
            {"package_name": pkg_name},
            {"main_params_path": main_params},
            {"training_params_path": training_params},
        ],
    )
    return [task_node]

def generate_launch_description():
    pkg_name = LaunchConfiguration("pkg_name")

    # Declare the launch arguments
    launch_args = [
        DeclareLaunchArgument("sensor", default_value="", description="sensor type: camera or lidar"),
        DeclareLaunchArgument("task", default_value="", description="task type: goToPose, Following, Vineyards"),
        DeclareLaunchArgument("pkg_name", default_value="pic4rl", description="package name"),
        DeclareLaunchArgument("main_params", default_value=PathJoinSubstitution([FindPackageShare(pkg_name), "config", "main_params.yaml"]), description="main_params.yaml"),
        DeclareLaunchArgument("training_params", default_value=PathJoinSubstitution([FindPackageShare(pkg_name), "config", "training_params.yaml"]), description="training_params.yaml"),
        DeclareLaunchArgument("mode", default_value="", description="mode: training or testing"),
    ]

    # Create the launch description and add actions
    ld = LaunchDescription()
    for arg in launch_args:
        ld.add_action(arg)
    ld.add_action(OpaqueFunction(function=print_params))

    return ld