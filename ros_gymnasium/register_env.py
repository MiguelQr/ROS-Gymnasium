import gymnasium as gym

# Episode length is enforced inside each env (``timeout_steps``) so that a
# timeout is reported as ``truncated``; no extra TimeLimit wrapper is added.
# Gazebo physics is not bit-reproducible, hence ``nondeterministic=True``.

def register_envs():
    gym.envs.registration.register(
        id="DenseNavEnv-v0",  # Unique ID for the environment
        entry_point="ros_gymnasium.tasks.go_to_pose.dense_nav:DenseNavEnv",  # Path to the environment class
        nondeterministic=True,
    )
    print("Registered DenseNavEnv-v0 environment.")
    gym.envs.registration.register(
        id="SparseNavEnvDiscrete-v0",  # Unique ID for the environment
        entry_point="ros_gymnasium.tasks.go_to_pose.sparse_nav:SparseNavEnv",  # Path to the environment class
        kwargs={"action_type": "discrete", "timeout_steps": 500},
        nondeterministic=True,
    )
    print("Registered SparseNavEnvDiscrete-v0 environment.")
    gym.envs.registration.register(
        id="SparseNavEnv-v0",  # Unique ID for the environment
        entry_point="ros_gymnasium.tasks.go_to_pose.sparse_nav:SparseNavEnv",  # Path to the environment class
        kwargs={"action_type": "continuous", "timeout_steps": 300},
        nondeterministic=True,
    )
    print("Registered SparseNavEnv-v0 environment.")
    # MultiRobotNavEnv-v0 is work in progress and not registered yet (see README).
