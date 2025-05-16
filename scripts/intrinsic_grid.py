
import collections
import minigrid
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch as th

import operator
from functools import reduce
from gymnasium.core import ObservationWrapper

from adaptative_rnd_action import AdaptiveRND

# MiniGrid-FourRooms-v0
# MiniGrid-LavaCrossingS9N3-v0
# MiniGrid-MultiRoom-N6-v0
# MiniGrid-KeyCorridorS4R3-v0
# MiniGrid-ObstructedMaze-1Dlhb-v0
env_id = "MiniGrid-ObstructedMaze-1Dlhb-v0"
# "FourRooms-A2C-intrinsic-doublenoise-forward_train_target"
run = "ObstructedMaze-A2C-vanilla"
SEED = 10
INTRINSIC_REWARD = False
USE_SELF_SUPERVISION = False
USE_PREDICTOR_HEAD = False
USE_NOISY = True
USE_NOVELTY_BUFFER = False


class FlatObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(imgSize,), dtype='uint8')

    def observation(self, obs):
        return obs['image'].flatten()


class OnPolicyCallback(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """

    def __init__(self, irs, verbose=0):
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.device = irs.device if hasattr(irs, 'device') else 'cpu'

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer
        self.int_rewards_buffer = collections.deque(maxlen=100)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.int_rewards_buffer:
            mean_int_rewards = sum(self.int_rewards_buffer) / \
                len(self.int_rewards_buffer)
            self.logger.record('rollout/mean_int_rewards', mean_int_rewards)

        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations, device=self.device)
        actions = th.as_tensor(self.buffer.actions, device=self.device)
        rewards = th.as_tensor(self.buffer.rewards, device=self.device)
        dones = th.as_tensor(self.buffer.episode_starts,
                             device=self.device)
        # get the new observations
        new_obs = th.roll(obs, -1, dims=0)
        new_obs[-1] = th.as_tensor(self.locals["new_obs"],
                                   device=self.device)
        # print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # compute the intrinsic rewards
        intrinsic_rewards = irs.compute(
            samples=dict(observations=obs, actions=actions,
                         rewards=rewards, terminateds=dones,
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        rewards_np = intrinsic_rewards.detach().cpu().numpy()
        self.buffer.advantages += rewards_np
        self.buffer.returns += rewards_np
        # ===================== compute the intrinsic rewards ===================== #

        self.int_rewards_buffer.append(intrinsic_rewards.mean().item())


def make_env(seed=None):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)  # Proper seeding
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env = Monitor(env)
        env = FlatObsWrapper(env)
        return env
    return _init


if __name__ == '__main__':

    device = 'cuda'
    num_envs = 16
    # obs_dim = 2835 147
    # action_dim = 7
    th.manual_seed(SEED)
    th.cuda.manual_seed_all(SEED)

    vec_env = SubprocVecEnv([make_env(seed=SEED + i) for i in range(num_envs)])

    config = {
        "policy_type": "MLP",
        "total_timesteps": 1e7,
        "env_id": env_id,
        "num_envs": num_envs,
    }

    model = A2C("MlpPolicy", vec_env, seed=1,
                verbose=1, tensorboard_log=f"./results/{run}", device=device)

    eval_callback = EvalCallback(vec_env, best_model_save_path=f"./models/{run}/best_model",
                                 log_path=f"./results/{run}/eval", eval_freq=5000, deterministic=True)

    if not INTRINSIC_REWARD:
        model.learn(total_timesteps=int(
            config["total_timesteps"]), callback=CallbackList([eval_callback]), progress_bar=True)

    else:
        irs = AdaptiveRND(vec_env, device=device, latent_dim=64, use_noisy=USE_NOISY, beta=0.1,
                          use_self_supervision=USE_SELF_SUPERVISION, use_novelty_buffer=USE_NOVELTY_BUFFER, use_predictor_head=USE_PREDICTOR_HEAD)
        model.learn(total_timesteps=int(config["total_timesteps"]),
                    callback=CallbackList(
            [OnPolicyCallback(irs), eval_callback]),
            progress_bar=True)

    model.save(f"./results/{run}/ppo_minigrid_final")
