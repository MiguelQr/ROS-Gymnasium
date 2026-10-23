"""
Gymnasium API conformance tests for the ROS-Gymnasium environments.

Run: colcon test --packages-select ros_gymnasium
"""

import os

import numpy as np
import pytest
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gymnasium.utils.env_checker import check_env

from ros_gymnasium.ros_gymnasium_env import ROSGymnasiumEnv
from ros_gymnasium.sensors import Sensors
from ros_gymnasium.tasks.go_to_pose.dense_nav import DenseNavEnv
from ros_gymnasium.tasks.go_to_pose.sparse_nav import SparseNavEnv

ENVS = {
    "SparseNavEnvDiscrete": (SparseNavEnv, {"action_type": "discrete"}),
    "SparseNavEnv": (SparseNavEnv, {"action_type": "continuous"}),
    "DenseNavEnv": (DenseNavEnv, {}),
}

CAMERA_DEPTH = 1.5  # [m], fake camera depth


@pytest.fixture(scope="module")
def ros_context(tmp_path_factory):
    """Initialise rclpy with the package's default parameters."""
    share = get_package_share_directory("ros_gymnasium")
    main_params = os.path.join(share, "config", "main_params.yaml")
    training_params = os.path.join(share, "config", "training_params.yaml")

    with open(main_params, "r") as file:
        params = yaml.safe_load(file)["main_node"]["ros__parameters"]
    params.update(
        package_name="ros_gymnasium",
        main_params_path=main_params,
        training_params_path=training_params,
    )

    params_file = tmp_path_factory.mktemp("ros") / "params.yaml"
    params_file.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}))

    rclpy.init(args=["test_gymnasium_api", "--ros-args", "--params-file", str(params_file)])
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def teleports():
    """Teleported entities as (name, x, y, yaw)."""
    return []


@pytest.fixture(params=list(ENVS), ids=list(ENVS))
def env(request, ros_context, monkeypatch, tmp_path, teleports):
    """Environment instance with the simulator replaced by fakes."""
    env_cls, env_kwargs = ENVS[request.param]

    def set_entity_state(self, name, x, y, z=0.05, yaw=0.0):
        teleports.append((name, x, y, yaw))
        return True

    def camera_depth(self):
        p = self.depth_process
        return np.full((p.height, p.width, 1), CAMERA_DEPTH, dtype=np.float32)

    def laser(self, min_obstacle_distance=False):
        p = self.laser_process
        return np.full(p.num_points, p.max_dist), False

    monkeypatch.setattr(ROSGymnasiumEnv, "_spin_sensors_callbacks", lambda self, max_attempts=100: None)
    monkeypatch.setattr(ROSGymnasiumEnv, "_set_entity_state", set_entity_state)
    monkeypatch.setattr(ROSGymnasiumEnv, "_create_logdir", lambda self: str(tmp_path))
    monkeypatch.setattr(env_cls, "_reset_simulation", lambda self: None)
    monkeypatch.setattr(Sensors, "get_depth", camera_depth)
    monkeypatch.setattr(Sensors, "get_laser", laser)
    monkeypatch.setattr(Sensors, "get_odom", lambda self, vel=False: [0.0, 0.0, 0.0])
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    environment = env_cls(**env_kwargs)
    yield environment
    environment.close()


def test_check_env(env):
    """Runs Gymnasium's environment checker."""
    env.render_mode = "rgb_array"
    check_env(env)


def test_timeout(env):
    """Timeout sets truncated, not terminated."""
    env.timeout_steps = 3
    env.reset(seed=0)

    for step in range(1, env.timeout_steps + 1):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert terminated is False
        assert truncated is (step == env.timeout_steps)

    assert info["event"] == "timeout"


def test_seeding(env, teleports):
    """Same seed gives the same goal and spawn poses."""
    def episode_start(seed):
        teleports.clear()
        env.reset(seed=seed)
        return list(env.goal_pose), list(teleports)

    assert episode_start(42) == episode_start(42)
    assert len({str(episode_start(seed)) for seed in range(10)}) > 1


def test_reset_clear(env):
    """Reset clears previous frames from the observation stack."""
    env.reset(seed=0)
    for _ in range(env.stacked_images):
        env.step(env.action_space.sample())

    obs, _ = env.reset(seed=0)

    assert np.all(obs[0] > 0.0)
    assert np.all(obs[1:] == 0.0)


def test_render(env):
    """rgb_array render returns the latest depth frame as an RGB image."""
    env.render_mode = "rgb_array"
    env.reset(seed=0)

    frame = env.render()

    assert frame.dtype == np.uint8
    assert frame.shape == (env.image_height, env.image_width, 3)
    assert np.all(frame == int(CAMERA_DEPTH / env.max_depth * 255))
