import gymnasium as gym
from pic4rl.tasks.goToPose.nav_depth import DepthNavEnv

def register_depth_nav_env():
    gym.envs.registration.register(
        id="DepthNavEnv-v0",  # Unique ID for the environment
        entry_point="pic4rl.tasks.goToPose.nav_depth:DepthNavEnv",  # Path to the environment class
        max_episode_steps=500,  # Maximum steps per episode
    )
    print("Registered DepthNavEnv-v0 environment.")