import gymnasium as gym


def register_envs():
    """Register the ROS-Gymnasium environments with Gymnasium."""
    gym.envs.registration.register(
        id="DenseNavEnv-v0",
        entry_point="ros_gymnasium.tasks.go_to_pose.dense_nav:DenseNavEnv",
        nondeterministic=True,
    )
    print("Registered DenseNavEnv-v0 environment.")
    gym.envs.registration.register(
        id="SparseNavEnvDiscrete-v0",
        entry_point="ros_gymnasium.tasks.go_to_pose.sparse_nav:SparseNavEnv",
        kwargs={"action_type": "discrete", "timeout_steps": 500},
        nondeterministic=True,
    )
    print("Registered SparseNavEnvDiscrete-v0 environment.")
    gym.envs.registration.register(
        id="SparseNavEnv-v0",
        entry_point="ros_gymnasium.tasks.go_to_pose.sparse_nav:SparseNavEnv",
        kwargs={"action_type": "continuous", "timeout_steps": 300},
        nondeterministic=True,
    )
    print("Registered SparseNavEnv-v0 environment.")
