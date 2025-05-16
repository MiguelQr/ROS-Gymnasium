# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union
from gymnasium.vector import VectorEnv

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

ObsShape = Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]


def process_observation_space(observation_space: gym.Space) -> ObsShape:
    """Process the observation space.

    Args:
        observation_space (gym.Space): Observation space.

    Returns:
        Information of the observation space.
    """
    if isinstance(observation_space, spaces.Box):
        # Observation is a vector
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: process_observation_space(subspace)  # type: ignore[misc]
            for (key, subspace) in observation_space.spaces.items()
        }
    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported")


def process_action_space(action_space: gym.Space) -> Tuple[Tuple[int, ...], int, int, str]:
    """Get the dimension of the action space.

    Args:
        action_space (gym.Space): Action space.

    Returns:
        Information of the action space.
    """
    # TODO: revise the action_range
    assert action_space.shape is not None, "The action data shape cannot be `None`!"
    action_shape = action_space.shape
    if isinstance(action_space, spaces.Discrete):
        policy_action_dim = int(action_space.n)
        action_dim = 1
        action_type = "Discrete"
    elif isinstance(action_space, spaces.Box):
        policy_action_dim = int(np.prod(action_space.shape))
        action_dim = policy_action_dim
        action_type = "Box"
    elif isinstance(action_space, spaces.MultiDiscrete):
        policy_action_dim = sum(list(action_space.nvec))
        action_dim = int(len(action_space.nvec))
        action_type = "MultiDiscrete"
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(
            action_space.n, int
        ), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        policy_action_dim = int(action_space.n)
        action_dim = policy_action_dim
        action_type = "MultiBinary"
    else:
        raise NotImplementedError(
            f"{action_space} action space is not supported")

    return action_shape, action_dim, policy_action_dim, action_type


class RewardForwardFilter:
    """Reward forward filter."""

    def __init__(self, gamma: float = 0.99) -> None:
        self.rewems = None
        self.gamma = gamma

    def update(self, rews: th.Tensor) -> th.Tensor:
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class TorchRunningMeanStd:
    """Running mean and std for torch tensor."""

    def __init__(self, epsilon=1e-4, shape=(), device=None) -> None:
        self.mean = th.zeros(shape, device=device)
        self.var = th.ones(shape, device=device)
        self.count = epsilon

    def update(self, x) -> None:
        """Update mean and std with batch data."""
        with th.no_grad():
            batch_mean = th.mean(x, dim=0)
            batch_var = th.var(x, dim=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        """Update mean and std with batch moments."""
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self) -> th.Tensor:
        return th.sqrt(self.var)

    def update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + th.pow(delta, 2) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class BaseReward(ABC):
    """Base class of reward module.

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

    Returns:
        Instance of the base reward module.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "rms",
    ) -> None:
        # get environment information
        if isinstance(envs, VectorEnv):
            self.observation_space = envs.single_observation_space
            self.action_space = envs.single_action_space
        else:
            self.observation_space = envs.observation_space
            self.action_space = envs.action_space
        self.n_envs = envs.unwrapped.num_envs
        # process the observation and action space
        self.obs_shape: Tuple = process_observation_space(
            self.observation_space)  # type: ignore
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type = (
            process_action_space(self.action_space)
        )
        # set device and parameters
        self.device = th.device(device)
        self.beta = beta
        self.kappa = kappa
        self.rwd_norm_type = rwd_norm_type
        self.obs_norm_type = obs_norm_type
        # build the running mean and std for normalization
        self.rwd_norm = TorchRunningMeanStd() if self.rwd_norm_type == "rms" else None
        self.obs_norm = (
            TorchRunningMeanStd(shape=self.obs_shape)
            if self.obs_norm_type == "rms"
            else None
        )
        # initialize the normalization parameters if necessary
        if self.obs_norm_type == "rms":
            self.envs = envs
            self.init_normalization()
        # build the reward forward filter
        self.rff = RewardForwardFilter(gamma) if gamma is not None else None
        # training tracker
        self.global_step = 0
        self.metrics = {"loss": [], "intrinsic_rewards": []}

    @property
    def weight(self) -> float:
        """Get the weighting coefficient of the intrinsic rewards."""
        return self.beta * np.power(1.0 - self.kappa, self.global_step)

    def scale(self, rewards: th.Tensor) -> th.Tensor:
        """Scale the intrinsic rewards.

        Args:
            rewards (th.Tensor): The intrinsic rewards with shape (n_steps, n_envs).

        Returns:
            The scaled intrinsic rewards.
        """
        # update reward forward filter if necessary
        if self.rff is not None:
            for step in range(rewards.size(0)):
                rewards[step] = self.rff.update(rewards[step])

        # scale the intrinsic rewards
        if self.rwd_norm_type == "rms":
            self.rwd_norm.update(rewards.ravel())
            return (rewards / self.rwd_norm.std) * self.weight
        elif self.rwd_norm_type == "minmax":
            return (
                (rewards - rewards.min())
                / (rewards.max() - rewards.min())
                * self.weight
            )
        else:
            return rewards * self.weight

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize the observations data, especially useful for images-based observations."""
        if self.obs_norm:
            mean = self.obs_norm.mean.to(self.device)
            std = th.sqrt(self.obs_norm.var.to(self.device))
            x = ((x - mean) / std).clamp(-5, 5)
        else:
            x = x / 255.0 if len(self.obs_shape) > 2 else x
        return x

    def init_normalization(self) -> None:
        """Initialize the normalization parameters for observations if the RMS is used."""
        if self.obs_norm_type != "rms":
            return
        
        num_steps, num_iters = 10, 2  # Reduced for faster initialization
        all_next_obs = []
        
        try:
            # Try modern Gymnasium API first
            self.envs.reset()
            
            for step in range(num_steps * num_iters):
                # Sample random actions
                actions = np.stack([self.action_space.sample() for _ in range(self.n_envs)])
                
                try:
                    # Handle both old and new step return formats
                    step_result = self.envs.step(actions)
                    
                    # Check if the result has 5 elements (obs, reward, terminated, truncated, info)
                    if len(step_result) == 5:
                        next_obs = step_result[0]
                    else:
                        # Old gym format (obs, reward, done, info)
                        next_obs = step_result[0]
                    
                    if isinstance(next_obs, np.ndarray):
                        tensor_obs = th.as_tensor(next_obs).float()
                        if len(self.obs_shape) == 3:
                            # For image observations, ensure proper shape
                            tensor_obs = tensor_obs.reshape(-1, *self.obs_shape)
                        else:
                            # For vector observations
                            tensor_obs = tensor_obs.reshape(-1, *self.obs_shape)
                        
                        all_next_obs.append(tensor_obs)
                    
                    # Update less frequently to avoid overhead
                    if len(all_next_obs) >= num_steps:
                        if all_next_obs:
                            all_obs_tensor = th.cat(all_next_obs, dim=0)
                            self.obs_norm.update(all_obs_tensor)
                            all_next_obs = []
                
                except Exception as e:
                    print(f"Error during normalization step: {e}")
                    break
        
        except Exception as e:
            print(f"Skipping full normalization due to: {e}")
            # Do minimal initialization with zeros instead
            if self.obs_norm is not None:
                dummy_obs = th.zeros((1, *self.obs_shape), dtype=th.float32)
                self.obs_norm.update(dummy_obs)

    @abstractmethod
    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors,
                whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
                For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.

        Returns:
            The intrinsic rewards.
        """
        for key in [
            "observations",
            "actions",
            "rewards",
            "terminateds",
            "truncateds",
            "next_observations",
        ]:
            assert key in samples.keys(), f"Key {key} is not in samples."

        # update the obs RMS if necessary
        if self.obs_norm_type == "rms" and sync:
            self.obs_norm.update(
                samples["observations"].reshape(-1, *self.obs_shape).cpu()
            )
        # update the global step
        self.global_step += 1

    @abstractmethod
    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
