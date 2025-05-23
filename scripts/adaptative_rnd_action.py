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
        use_noisy: bool = True,
        use_novelty_buffer: bool = False,
        use_predictor_head: bool = True
    ) -> None:
        super().__init__(envs, device, beta, kappa, gamma, rwd_norm_type, obs_norm_type)

        self.use_novelty_buffer = use_novelty_buffer
        self.use_self_supervision = use_self_supervision
        self.alpha = alpha
        self.use_novelid = False

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

        self.max_buffer_size = buffer_size
        self.revisit_scale = 0.1

        if self.use_novelty_buffer:
            self.goal_update_interval = 100
            self.goal_update_counter = 0
            self.goal_bonus_scale = 0.1
            self.top_k_novel = 16

            self.top_novelty_states = th.zeros(
                self.top_k_novel, latent_dim, device=self.device
            )
            self.top_novelty_scores = th.full(
                (self.top_k_novel,), -float("inf"), device=self.device
            )

            self.goal_z = None  # currently selected goal

    def compute_prediction_error(self, obs_tensor: th.Tensor) -> th.Tensor:
        """
        Compute prediction error between predictor and target network.
        Returns tensor of shape [T, E].
        """
        with th.no_grad():
            tgt = self.target(obs_tensor)
            pred = self.predictor(obs_tensor)

        # pred = pred.view(-1, self.n_envs, tgt.shape[-1])
        # tgt = tgt.view(-1, self.n_envs, tgt.shape[-1])

        error = F.mse_loss(pred, tgt, reduction="none").mean(dim=1)  # [T, E]
        return error

    def sample_new_goal(self):
        if self.top_novelty_scores.max() > -float("inf"):
            top_idx = th.randint(0, self.top_k_novel, (1,))
            self.goal_z = self.top_novelty_states[top_idx].clone()

    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        super().compute(samples)

        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        next_obs_tensor = self.normalize(
            samples.get("next_observations").to(self.device)).view(-1, *self.obs_shape)  # torch.Size([5, 16, 1])
        obs_tensor = self.normalize(samples.get("observations").to(
            self.device)).view(-1, *self.obs_shape)

        # --- Use prediction error as novelty ---
        novelty_tp1 = self.compute_prediction_error(next_obs_tensor)

        if self.use_novelid:
            novelty_t = self.compute_prediction_error(obs_tensor)
            intrinsic_rewards = th.clamp(
                novelty_tp1 - self.alpha * novelty_t, min=0.0)
            intrinsic_rewards = th.nan_to_num(
                intrinsic_rewards, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            intrinsic_rewards = novelty_tp1.view(
                n_steps, n_envs)

        # === Optional: Buffer-based goal exploration ===
        if self.use_novelty_buffer:

            self.goal_update_counter += 1

            with th.no_grad():
                z_tp1 = self.target(next_obs_tensor).view(n_steps, n_envs, -1)

            if not hasattr(self, 'prev_latents'):
                self.prev_latents = z_tp1.clone()
                latent_movement = th.zeros(n_steps, n_envs, device=self.device)
            else:
                latent_movement = (z_tp1 - self.prev_latents).norm(dim=2)
                self.prev_latents = z_tp1.clone()

            # Mean movement across envs per step
            movement_mask = latent_movement.mean(dim=1) < 0.05

            if movement_mask.any():
                # Compute goal bonus only if low movement is detected
                if self.goal_z is not None:
                    goal_dist = F.mse_loss(z_tp1, self.goal_z.expand_as(
                        z_tp1), reduction="none").mean(dim=2)
                    goal_bonus = 1.0 / (1.0 + goal_dist)
                    goal_bonus = goal_bonus * \
                        movement_mask.unsqueeze(1)  # Apply mask
                    intrinsic_rewards += self.goal_bonus_scale * goal_bonus

            if self.goal_update_counter % self.goal_update_interval == 0:
                novelty_flat = novelty_tp1.view(-1)
                z_flat = z_tp1.view(-1, z_tp1.shape[-1])

                for i in range(z_flat.size(0)):
                    min_score, min_idx = self.top_novelty_scores.min(dim=0)
                    if novelty_flat[i] > min_score:
                        self.top_novelty_scores[min_idx] = novelty_flat[i]
                        self.top_novelty_states[min_idx] = z_flat[i]

        if sync:
            self.update(obs_tensor)

        return self.scale(intrinsic_rewards)

    def update(self, obs_tensor: th.Tensor) -> None:

        dataset = TensorDataset(obs_tensor)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        total_loss = 0.0
        batch_count = 0

        for batch_data in loader:
            batch_count += 1
            obs_actions = batch_data[0]
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

        if batch_data:
            avg_loss = total_loss / batch_count
            try:
                self.metrics["loss"].append([self.global_step, avg_loss])
            except:
                pass
