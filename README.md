# ROS-Gymnasium

![ROS 2](https://img.shields.io/badge/ROS%202-Foxy%20%7C%20Humble-blue) ![Gymnasium](https://img.shields.io/badge/Gymnasium-Latest-green) ![Gazebo](https://img.shields.io/badge/Gazebo-11-orange)

A framework for reinforcement learning in robotics that connects Gymnasium environments to Gazebo simulations through ROS 2.

## Overview

ROS-Gymnasium provides an interface for connecting the Gymnasium (formerly OpenAI Gym) reinforcement learning library, with the Gazebo physics simulator. This bridge allows reinforcement learning agents to be trained in simulated robotic environments.

## Features

- **Gymnasium Integration**: Compatibility with the Gymnasium API for reinforcement learning.
- **ROS 2 Based**: Built on ROS 2 for  robotics development.
- **Sensor Support**: Handles camera, laser and odom sensors.
- **Configurable Environments**: Allows to define new environments with different robots, sensors, and tasks.
- **Image Recording**: `render_mode="rgb_array"` returns the robot's depth view.

## Installation

### Prerequisites

- ROS 2 Foxy (Ubuntu 20.04) or ROS 2 Humble (Ubuntu 22.04), tested on both
- Gazebo 11
- Python 3.8+
- A robot description package named after `robot_name` in `config/main_params.yaml` (default `tb2_ugv`) that provides `launch/description.launch.py` and publishes `/robot_description`, with a depth camera on `/camera/depth/image_raw`, a laser on `/scan` and odometry on `/odom`

### Setup

1. **Create a ROS 2 workspace**:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/MiguelQr/ROS-Gymnasium.git
   ```

3. **Install dependencies**:
   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

4. **Install the Python packages**:
   ```bash
   pip install gymnasium "stable-baselines3[extra]"
   ```

5. **Build the workspace**:
   ```bash
   colcon build
   source install/setup.bash
   ```

## Usage

`training.launch.py` starts Gazebo with the selected world, spawns the robot and the goal, and runs the training script (`go_to_pose_camera`). The script trains the environment and algorithm set in `config/training_params.yaml` (default: PPO on `SparseNavEnvDiscrete-v0`).

Train in the default world (`gridworld_4x4.world` from `config/main_params.yaml`):
```bash
ros2 launch ros_gymnasium training.launch.py
```

Train in a larger world:
```bash
ros2 launch ros_gymnasium training.launch.py world_name:=gridworld_8x8.world
```

Available worlds: `gridworld_4x4.world`, `gridworld_6x6.world`, `gridworld_8x8.world`, `tesis_avoidance.world`, `tesis_circuit.world`, `tesis_obstacle.world`.

TensorBoard logs are written to `./results/<run_name>` and the trained model is saved as `<run_name>.zip`:
```bash
tensorboard --logdir results
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| world_name | `world_name` in `main_params.yaml` | World file from `worlds/` |
| sensor | `sensor` in `main_params.yaml` | Sensor type; only `camera` is implemented |
| task | `task` in `main_params.yaml` | Task type; only `goToPose` is implemented |
| main_params | `config/main_params.yaml` | Robot, sensor and world parameters |
| training_params | `config/training_params.yaml` | Training algorithm and hyperparameters |
| pkg_name | ros_gymnasium | Package name |
| use_sim_time | true | Use simulation clock |

## Creating Environments

Environments can be createdby extending the `ROSGymnasiumEnv` base class:

```python
from ros_gymnasium.ros_gymnasium_env import ROSGymnasiumEnv

class CustomEnv(ROSGymnasiumEnv):
    def __init__(self):
        super().__init__("my_custom_env")
        # Your initialization code
        
    def _send_action(self, action):
        # Implementation for sending actions
        
    def _get_sensor_data(self):
        # Implementation for retrieving sensor data
        
    def _get_observation(self, sensor_data):
        # Implementation for processing observations
        
    # Abstract methods...
```

## Configuration

Environment parameters can be configured in the YAML files located in the `config` directory:

- `main_params.yaml`: Main parameters for the environments.
- `training_params.yaml`: Parameters specific to training algorithms.

## Work in Progress

- **Multi-robot training** (`launch/multi_robot_training.launch.py`, `ros_gymnasium/multi_robot_sensors.py`, `scripts/multi_agent_training`): the environment class `MultiRobotNavEnv` is not included yet, so this pipeline is not registered or installed.

## Environments

| ID | Class | Actions | Reward |
|----|-------|---------|--------|
| `SparseNavEnvDiscrete-v0` | `SparseNavEnv(action_type="discrete")` | 10 fixed velocity commands | Sparse |
| `SparseNavEnv-v0` | `SparseNavEnv(action_type="continuous")` | [linear, angular] velocity | Sparse |
| `DenseNavEnv-v0` | `DenseNavEnv` | [linear, angular] velocity | Dense (distance and heading) |

All environments observe a stack of depth images and are registered by `ros_gymnasium.register_env.register_envs()`.

## Project Structure

```
ROS-Gymnasium/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── main_params.yaml          # Robot, sensor and world parameters
│   └── training_params.yaml      # SB3 algorithm and hyperparameters
├── goals_and_poses/             # Goal and spawn poses per world
├── launch/
│   ├── training.launch.py
│   └── multi_robot_training.launch.py   # Work in progress
├── models/goal_box/
├── worlds/
├── ros_gymnasium/
│   ├── ros_gymnasium_env.py            # ROSGymnasiumEnv base class
│   ├── register_env.py
│   ├── sensors.py
│   ├── multi_robot_sensors.py    # Work in progress
│   ├── tasks/goToPose/
│   │   ├── sparse_nav.py         # SparseNavEnv
│   │   └── dense_nav.py          # DenseNavEnv
│   └── utils/env_utils.py
├── scripts/
│   └── go_to_pose_camera         # SB3 training entry point
└── test/
    └── test_gymnasium_api.py
```

## Citation

```bibtex
@misc{ros_gymnasium2025,
  author = {Miguel Quiñones},
  title = {ROS-Gymnasium},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/MiguelQr/ROS-Gymnasium}}
}

@inproceedings{martini2023pic4rl,
  title={Pic4rl-gym: a ros2 modular framework for robots autonomous navigation with deep reinforcement learning},
  author={Martini, Mauro and Eirale, Andrea and Cerrato, Simone and Chiaberge, Marcello},
  booktitle={2023 3rd International Conference on Computer, Control and Robotics (ICCCR)},
  pages={198--202},
  year={2023},
  organization={IEEE}
}
```
