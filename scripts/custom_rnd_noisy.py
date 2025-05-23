from collections import deque
from gymnasium.vector import VectorEnv
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional

from base_reward import BaseReward
from model import ObservationEncoder, PredictorMLP


class AdaptiveRND(BaseReward):
    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "none",
        obs_norm_type: str = "rms",
        latent_dim: int = 64,
        lr: float = 0.001,
        batch_size: int = 256,
        weight_init: str = "orthogonal",
        alpha: float = 0.5,
        buffer_size: int = 500,
        use_self_supervision: bool = True,
        use_noisy: bool = True
    ) -> None:
        super().__init__(envs, device, beta, kappa, gamma, rwd_norm_type, obs_norm_type)

        self.use_self_supervision = use_self_supervision
        self.alpha = alpha
        self.use_novelid = False
        self.use_id = True

        self.predictor = ObservationEncoder(
            obs_shape=self.obs_shape,
            latent_dim=latent_dim,
            weight_init=weight_init,
            use_noisy=use_noisy
        ).to(self.device)
        self.target = ObservationEncoder(
            obs_shape=self.obs_shape,
            latent_dim=latent_dim,
            weight_init=weight_init,
            use_noisy=use_noisy and not use_self_supervision
        ).to(self.device)

        self.opt = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.batch_size = batch_size

        self.id_head = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.envs.action_space.n)
        ).to(self.device)
        self.id_opt = th.optim.Adam(
            list(self.target.parameters()) + list(self.id_head.parameters()),
            lr=lr * 0.1  # Lower LR to avoid destabilizing RND
        )

    def compute_prediction_error(self, obs_tensor: th.Tensor) -> th.Tensor:
        with th.no_grad():
            tgt = self.target(obs_tensor)
            pred = self.predictor(obs_tensor)
        error = F.mse_loss(pred, tgt, reduction="none").mean(dim=1)  # [T*E]
        return error

    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        super().compute(samples)

        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        next_obs_tensor = self.normalize(
            samples.get("next_observations").to(self.device)).view(-1, *self.obs_shape)
        obs_tensor = self.normalize(
            samples.get("observations").to(self.device)).view(-1, *self.obs_shape)
        actions_tensor = samples["actions"].to(self.device).view(-1)

        novelty_tp1 = self.compute_prediction_error(next_obs_tensor)

        if self.use_self_supervision:
            z_t = self.target(obs_tensor)
            z_t_plus_1 = self.target(next_obs_tensor).detach()

            action_pred = self.id_head(th.cat([z_t, z_t_plus_1], dim=1))
            id_loss = F.cross_entropy(action_pred, actions_tensor.long())

            self.id_opt.zero_grad()
            id_loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.target.parameters(), 1.0)  # Stability
            self.id_opt.step()

        if self.use_novelid:
            novelty_t = self.compute_prediction_error(obs_tensor)
            intrinsic_rewards = th.clamp(
                novelty_tp1 - self.alpha * novelty_t, min=0.0)
            intrinsic_rewards = th.nan_to_num(
                intrinsic_rewards, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            intrinsic_rewards = novelty_tp1.view(n_steps, n_envs)

        if sync:
            self.update(obs_tensor)

        return self.scale(intrinsic_rewards)

    def update(self, obs_tensor: th.Tensor) -> None:
        loader = DataLoader(TensorDataset(obs_tensor),
                            batch_size=self.batch_size, shuffle=True)

        total_loss = 0.0

        for batch_count, (obs_actions,) in enumerate(loader, 1):

            self.opt.zero_grad(set_to_none=True)

            src_feats = self.predictor(obs_actions)
            with th.no_grad():
                tgt_feats = self.target(obs_actions)

            loss = F.mse_loss(src_feats, tgt_feats,
                              reduction="none").mean(dim=-1)
            final_loss = loss.mean()

            final_loss.backward()
            self.opt.step()

            total_loss += final_loss.item()

        if batch_count:
            avg_loss = total_loss / batch_count
            try:
                self.metrics["loss"].append([self.global_step, avg_loss])
            except:
                pass
