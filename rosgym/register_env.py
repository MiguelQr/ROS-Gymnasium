import gymnasium as gym

# Episode length is enforced inside each env (``timeout_steps``) so that a
# timeout is reported as ``truncated``; no extra TimeLimit wrapper is added.
# Gazebo physics is not bit-reproducible, hence ``nondeterministic=True``.

def register_envs():
    gym.envs.registration.register(
        id="DepthNavEnv-v0",  # Unique ID for the environment
        entry_point="rosgym.tasks.goToPose.nav_depth:DepthNavEnv",  # Path to the environment class
        nondeterministic=True,
    )
    print("Registered DepthNavEnv-v0 environment.")
    gym.envs.registration.register(
        id="SparseNavEnvDiscrete-v0",  # Unique ID for the environment
        entry_point="rosgym.tasks.goToPose.sparse_nav_discrete:DepthNavEnv",  # Path to the environment class
        nondeterministic=True,
    )
    print("Registered SparseNavEnvDiscrete-v0 environment.")
    gym.envs.registration.register(
        id="SparseNavEnv-v0",  # Unique ID for the environment
        entry_point="rosgym.tasks.goToPose.sparse_nav:DepthNavEnv",  # Path to the environment class
        nondeterministic=True,
    )
    print("Registered SparseNavEnv-v0 environment.")
    # MultiRobotNavEnv-v0 is work in progress and not registered yet (see README).
