"""
Gymnasium API conformance tests for the ROS-Gym environments.

Run: colcon test --packages-select rosgym
"""

import os

import numpy as np
import pytest
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gymnasium.utils.env_checker import check_env

from rosgym.ros_gym_env import ROSGymEnv
from rosgym.sensors import Sensors
from rosgym.tasks.goToPose import nav_depth, sparse_nav, sparse_nav_discrete

ENV_CLASSES = {
    "SparseNavEnvDiscrete": sparse_nav_discrete.DepthNavEnv,
    "SparseNavEnv": sparse_nav.DepthNavEnv,
    "DepthNavEnv": nav_depth.DepthNavEnv,
}

CAMERA_DEPTH = 1.5  # [m], fake camera depth


@pytest.fixture(scope="module")
def ros_context(tmp_path_factory):
    """Initialise rclpy with the package's default parameters."""
    share = get_package_share_directory("rosgym")
    main_params = os.path.join(share, "config", "main_params.yaml")
    training_params = os.path.join(share, "config", "training_params.yaml")

    with open(main_params, "r") as file:
        params = yaml.safe_load(file)["main_node"]["ros__parameters"]
    params.update(
        package_name="rosgym",
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


@pytest.fixture(params=list(ENV_CLASSES), ids=list(ENV_CLASSES))
def env(request, ros_context, monkeypatch, tmp_path, teleports):
    """Environment instance with the simulator replaced by fakes."""
    env_cls = ENV_CLASSES[request.param]

    def set_entity_state(self, name, x, y, z=0.05, yaw=0.0):
        teleports.append((name, x, y, yaw))
        return True

    def camera_depth(self):
        p = self.depth_process
        return np.full((p.height, p.width, 1), CAMERA_DEPTH, dtype=np.float32)

    def laser(self, min_obstacle_distance=False):
        p = self.laser_process
        return np.full(p.num_points, p.max_dist), False

    monkeypatch.setattr(ROSGymEnv, "_spin_sensors_callbacks", lambda self, max_attempts=100: None)
    monkeypatch.setattr(ROSGymEnv, "_set_entity_state", set_entity_state)
    monkeypatch.setattr(ROSGymEnv, "_create_logdir", lambda self: str(tmp_path))
    monkeypatch.setattr(env_cls, "_reset_simulation", lambda self: None)
    if env_cls is nav_depth.DepthNavEnv:
        monkeypatch.setattr(env_cls, "_respawn_robot", lambda self: None)
    monkeypatch.setattr(Sensors, "get_depth", camera_depth)
    monkeypatch.setattr(Sensors, "get_laser", laser)
    monkeypatch.setattr(Sensors, "get_odom", lambda self, vel=False: [0.0, 0.0, 0.0])
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    environment = env_cls()
    yield environment
    environment.destroy_node()


def test_check_env(env):
    """Runs Gymnasium's environment checker."""
    check_env(env, skip_render_check=True)


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
