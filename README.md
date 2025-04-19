# ROS-Gym

<img alt="ROS2" src="https://img.shields.io/badge/ROS2-Foxy-blue">
<img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-Latest-green">
<img alt="Gazebo" src="https://img.shields.io/badge/Gazebo-11-orange">

A framework for reinforcement learning in robotics that connects Gymnasium environments to Gazebo simulations through ROS 2.

## Overview

ROS-Gym provides an interface for connecting the Gymnasium (formerly OpenAI Gym) reinforcement learning library, with the Gazebo physics simulator. This bridge allows reinforcement learning agents to be trained in simulated robotic environments.

## Features

- **Gymnasium Integration**: Compatibility with the Gymnasium API for reinforcement learning.
- **ROS 2 Based**: Built on ROS 2 for  robotics development.
- **Sensor Support**: Handles camera, laser and odom sensors.
- **Configurable Environments**: Allows to define new environments with different robots, sensors, and tasks.

## Installation

### Prerequisites

- ROS 2 Foxy
- Gazebo 11
- Python 3.8+

### Setup

1. **Create a ROS 2 workspace**:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/rosgym.git
   ```

3. **Install dependencies**:
   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

4. **Build the workspace**:
   ```bash
   colcon build
   ```

## Usage

(WIP) Run the following command to launch a navigation task:
```bash
ros2 launch rosgym starter.launch.py task:=goToPose sensor:=camera mode:=training
```

## Creating Environments

Environments can be createdby extending the `ROSGymEnv` base class:

```python
from rosgym.ros_gym_env import ROSGymEnv

class CustomEnv(ROSGymEnv):
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

## Project Structure

```
rosgym/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── main_params.yaml
│   └── training_params.yaml
├── goals_and_poses/
│   ├── training/
│   └── testing/
├── launch/
│   └── starter.launch.py
├── rosgym/
│   ├── __init__.py
│   ├── register_env.py
│   ├── sensors.py
│   ├── ros_gym_env.py
│   ├── tasks/
│   │   └── goToPose/
│   │       └── nav_depth.py
│   └── utils/
│       ├── env_utils.py
│       └── launch_utils.py
└── scripts/
    └── go_to_pose_camera
```

## Citation

```bibtex
@misc{rosgym2025,
  author = {Miguel Quiñones},
  title = {ROS-Gym},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/MiguelQr/ROSRL}}
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