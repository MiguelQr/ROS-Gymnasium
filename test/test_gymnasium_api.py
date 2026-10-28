"""Gymnasium API conformance tests for the ROS-Gymnasium environments."""

import math
import os

import numpy as np
import pytest
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from ros_gymnasium.ros_gymnasium_env import ROSGymnasiumEnv
from ros_gymnasium.sensors import DepthCamera, Sensors
from ros_gymnasium.tasks.go_to_pose.dense_nav import DenseNavEnv
from ros_gymnasium.tasks.go_to_pose.sparse_nav import SparseNavEnv
from ros_gymnasium.wrappers import SelectObservation

ENVS = {
    "SparseNavEnvDiscrete": (SparseNavEnv, {"action_type": "discrete"}),
    "SparseNavEnv": (SparseNavEnv, {"action_type": "continuous"}),
    "DenseNavEnv": (DenseNavEnv, {}),
}

OBSERVATION_SENSORS = ["depth", "scan", "odom"]
CAMERA_DEPTH = 1.5


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
        observation_sensors=OBSERVATION_SENSORS,
        goal_observation=True,
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
        return np.full((p.height, p.width), CAMERA_DEPTH, dtype=np.float32)

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


def test_observation_space(env):
    """Dict observation with stacked readings of each observed sensor and the goal."""
    sensor_spaces = env.sensors.observation_spaces()

    assert isinstance(env.observation_space, spaces.Dict)
    assert set(env.observation_space.spaces) == {*OBSERVATION_SENSORS, "goal"}
    assert env.observation_space["goal"].shape == (2,)
    for name in OBSERVATION_SENSORS:
        assert env.observation_space[name].shape == (env.stacked_frames, *sensor_spaces[name].shape)


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


def test_goal_box_follows_goal(env, teleports):
    """Reset moves the goal entity to the new goal pose."""
    env.reset(seed=0)

    goal_teleports = [(x, y) for name, x, y, _ in teleports if name == "goal"]
    assert goal_teleports[-1] == (env.goal_pose[0], env.goal_pose[1])


def test_reset_clear(env):
    """Reset clears previous frames from the observation stack."""
    env.reset(seed=0)
    for _ in range(env.stacked_frames):
        env.step(env.action_space.sample())

    obs, _ = env.reset(seed=0)

    assert np.all(obs["depth"][0] > 0.0)
    assert np.all(obs["depth"][1:] == 0.0)


def test_depth_too_close_is_obstacle():
    """Too close (-inf) maps to 0 m; beyond range (+inf) and no return (NaN) map to the cutoff."""
    camera = DepthCamera(width=4, height=1, cutoff_dist=3.0)
    frame = np.array([[-np.inf, np.inf, np.nan, 1.5]], dtype=np.float32)

    np.testing.assert_allclose(camera.process_data(frame), [[0.0, 3.0, 3.0, 1.5]])


def test_render(env):
    """rgb_array render returns the latest depth frame as an RGB image."""
    env.render_mode = "rgb_array"
    env.reset(seed=0)
    camera = env.sensors.depth_process

    frame = env.render()

    assert frame.dtype == np.uint8
    assert frame.shape == (camera.height, camera.width, 3)
    assert np.all(frame == int(CAMERA_DEPTH / camera.cutoff_dist * 255))


def test_goal_observation(env):
    """Goal entry is [distance, angle] from the robot (at the origin, yaw 0) to the goal."""
    obs, _ = env.reset(seed=0)
    goal_x, goal_y = env.goal_pose[0], env.goal_pose[1]

    np.testing.assert_allclose(obs["goal"], [math.hypot(goal_x, goal_y), math.atan2(goal_y, goal_x)], rtol=1e-6)


def test_select_observation(env):
    """SelectObservation turns the Dict observation into one of its Box entries."""
    wrapped = SelectObservation(env, "depth")

    obs, _ = wrapped.reset(seed=0)

    assert wrapped.observation_space == env.observation_space["depth"]
    assert isinstance(obs, np.ndarray) and obs in wrapped.observation_space
    check_env(wrapped)
