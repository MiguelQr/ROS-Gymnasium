"""Gymnasium wrappers for ROS-Gymnasium environments."""

import gymnasium as gym


class SelectObservation(gym.ObservationWrapper):
    """Replace the Dict observation with one of its entries, e.g. SelectObservation(env, "depth")."""

    def __init__(self, env, key):
        super().__init__(env)
        self.key = key
        self.observation_space = env.observation_space[key]

    def observation(self, observation):
        return observation[self.key]
