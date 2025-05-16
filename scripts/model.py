from typing import Tuple
from torch import nn
from torch.nn import functional as F

import numpy as np
import torch as th
import math


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.1, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

        self.weight_sigma = nn.Parameter(
            th.empty(out_features, in_features))
        self.register_buffer(
            'weight_epsilon', th.empty(out_features, in_features))

        self.bias_sigma = nn.Parameter(th.empty(out_features))
        self.register_buffer('bias_epsilon', th.empty(out_features))

        self.sigma_init = sigma_init
        self.init_parameters()
        self.reset_noise()

    def init_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)

        self.bias.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight + self.weight_sigma * self.weight_epsilon
            bias = self.bias + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight
            bias = self.bias
        return nn.functional.linear(x, weight, bias)

    def _scale_noise(self, size):
        x = th.randn(size)
        return x.sign().mul(x.abs().sqrt())


def default_layer_init(layer):
    stdv = 1. / math.sqrt(layer.weight.size(1))
    layer.weight.data.uniform_(-stdv, stdv)
    if layer.bias is not None:
        layer.bias.data.uniform_(-stdv, stdv)
    return layer


def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    # torch.nn.init.zeros_(m.bias)
    return layer

def _get_conv_output_size(self, shape):
        """Calculate output size of conv layers for a given input shape."""
        # Create a dummy tensor to forward through the convolutional layers
        batch_size = 1
        input_tensor = th.zeros(batch_size, *shape)
        output_tensor = self.conv_layers(input_tensor)
        return int(np.prod(output_tensor.shape[1:]))

class ObservationEncoder(nn.Module):

    def __init__(self, obs_shape: Tuple, latent_dim: int, weight_init: str = "default", use_noisy: bool = False) -> None:
        super().__init__()

        if weight_init == "orthogonal":
            init_ = orthogonal_layer_init
        elif weight_init == "default":
            init_ = default_layer_init
        else:
            raise ValueError("Invalid weight_init")

        Layer = NoisyLinear if use_noisy else nn.Linear
        
        if len(obs_shape) == 3:

            self.is_image = True
            stacked_frames, height, width = obs_shape
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(stacked_frames, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            conv_output_size = self._get_conv_output_size((stacked_frames, height, width))
            
            self.trunk = nn.Sequential(
                Layer(conv_output_size, 256),
                nn.ReLU(),
                Layer(256, latent_dim)
            )
            
        else:
            self.is_image = False
            self.trunk = nn.Sequential(
                Layer(obs_shape[0], 256),
                nn.ReLU(),
                Layer(256, latent_dim)
            )
        
        if not use_noisy:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    init_(module)
                elif isinstance(module, nn.Conv2d):
                    th.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    if module.bias is not None:
                        module.bias.data.zero_()
    
    def _get_conv_output_size(self, shape):
        """Calculate output size of conv layers for a given input shape."""
        # Create a dummy tensor to forward through the convolutional layers
        batch_size = 1
        input_tensor = th.zeros(batch_size, *shape)
        output_tensor = self.conv_layers(input_tensor)
        return int(np.prod(output_tensor.shape[1:]))
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        if self.is_image:
            # For image observations: use convolutional layers first
            # Ensure proper dimensions: [batch_size, channels, height, width]
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Add batch dimension if missing
                
            features = self.conv_layers(obs)
            features = features.view(features.size(0), -1)  # Flatten
            return self.trunk(features)
        else:
            # For vector observations: directly use the MLP trunk
            return self.trunk(obs)


class PredictorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class ObservationActionEncoder(nn.Module):

    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int, weight_init: str = "default", use_noisy: bool = False) -> None:
        super().__init__()

        if weight_init == "orthogonal":
            init_ = orthogonal_layer_init
        elif weight_init == "default":
            init_ = default_layer_init
        else:
            raise ValueError("Invalid weight_init")

        input_dim = obs_shape[0] + action_dim
        Layer = NoisyLinear if use_noisy else nn.Linear

        self.trunk = nn.Sequential(
            Layer(input_dim, 256),
            nn.ReLU(),
            Layer(256, latent_dim)
        )

        if not use_noisy:
            for layer in self.trunk:
                if isinstance(layer, nn.Linear):
                    init_(layer)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        combined_input = th.cat([obs, actions], dim=-1)
        return self.trunk(combined_input)


class ForwardDynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, use_noisy: str = False):
        super().__init__()

        Layer = NoisyLinear if use_noisy else nn.Linear

        self.model = nn.Sequential(
            Layer(latent_dim + action_dim, 128),
            nn.ReLU(),
            Layer(128, latent_dim)
        )

    def forward(self, z, a):
        return self.model(th.cat([z, a], dim=-1))


class BayesianObservationEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple, latent_dim: int, weight_init="default", dropout_rate=0.1):
        super().__init__()

        if weight_init == "orthogonal":
            init_ = orthogonal_layer_init
        elif weight_init == "default":
            init_ = default_layer_init
        else:
            raise ValueError("Invalid weight_init")

        self.trunk = nn.Sequential(
            init_(nn.Linear(obs_shape[0], 256)),
            nn.ReLU(),
            # Dropout for Monte Carlo uncertainty estimation
            nn.Dropout(dropout_rate),
            # Output both mean and variance
            init_(nn.Linear(256, latent_dim * 2)),
        )

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        output = self.trunk(obs)
        # Split into mean and log-variance
        mean, log_var = output.chunk(2, dim=-1)
        var = F.softplus(log_var) + 1e-6  # Ensure positive variance
        return mean, var


class VAE(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super(VAE, self).__init__()

        init_ = orthogonal_layer_init

        self.encoder = nn.Sequential(
            init_(nn.Linear(obs_shape[0], 256)),
            nn.ReLU(),
            # Output both mean and log-variance
            init_(nn.Linear(256, latent_dim * 2))
        )
        self.decoder = nn.Sequential(
            init_(nn.Linear(latent_dim, 256)),
            nn.ReLU(),
            init_(nn.Linear(256, obs_shape[0])),
            nn.Sigmoid()  # Assuming input is normalized between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = th.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class SNDVEncoder(nn.Module):
    def __init__(self, obs_shape, latent_dim, encoder_model: str = "mnih", weight_init="default"):
        super(SNDVEncoder, self).__init__()

        init_ = orthogonal_layer_init

        self.trunk = nn.Sequential(
            init_(nn.Linear(obs_shape[0], 256)),
            nn.ReLU()
        )
        self.trunk.append(init_(nn.Linear(256, latent_dim)))

    def forward(self, state):
        return self.trunk(state)

    def loss_function(self, states_a, states_b, target=1):
        xa = states_a.clone()
        xb = states_b.clone()

        # normalise states
        # if normalise is not None:
        #     xa = normalise(xa)
        #     xb = normalise(xb)

        # obtain features from model
        za = self(xa)
        zb = self(xb)

        # predict close distance for similar, far distance for different states
        predicted = ((za - zb) ** 2).mean(dim=1)

        # similarity MSE loss
        loss_sim = ((target - predicted) ** 2).mean()

        # L2 magnitude regularisation
        magnitude = (za ** 2).mean() + (zb ** 2).mean()

        # care only when magnitude above 200
        loss_magnitude = th.relu(magnitude - 200.0)

        loss = loss_sim + loss_magnitude

        return loss


class InverseDynamicsEncoder(nn.Module):
    """Encoder with inverse dynamics prediction.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int, encoder_model: str = "mnih", weight_init="default") -> None:
        super().__init__()

        self.encoder = ObservationEncoder(
            obs_shape, latent_dim, encoder_model=encoder_model, weight_init=weight_init)
        self.policy = InverseDynamicsModel(
            latent_dim, action_dim, encoder_model=encoder_model, weight_init=weight_init)

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs (th.Tensor): Current observations.
            next_obs (th.Tensor): Next observations.

        Returns:
            Predicted actions.
        """
        h = self.encoder(obs)
        next_h = self.encoder(next_obs)

        actions = self.policy(h, next_h)
        return actions

    def encode(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.encoder(obs)


class InverseDynamicsModel(nn.Module):
    """Inverse model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """

    def __init__(self, latent_dim, action_dim, encoder_model="mnih", weight_init="default") -> None:
        super().__init__()

        self.trunk = ObservationEncoder(obs_shape=(
            latent_dim * 2,), latent_dim=action_dim, encoder_model=encoder_model, weight_init=weight_init)

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs (th.Tensor): Current observations.
            next_obs (th.Tensor): Next observations.

        Returns:
            Predicted actions.
        """
        return self.trunk(th.cat([obs, next_obs], dim=1))


class ForwardDynamicsModel(nn.Module):
    """Forward model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """

    def __init__(self, latent_dim, action_dim, encoder_model="mnih", weight_init="default") -> None:
        super().__init__()

        self.trunk = ObservationEncoder(obs_shape=(
            latent_dim + action_dim,), latent_dim=latent_dim, encoder_model=encoder_model, weight_init=weight_init)

    def forward(self, obs: th.Tensor, pred_actions: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted next-obs.

        Args:
            obs (th.Tensor): Current observations.
            pred_actions (th.Tensor): Predicted observations.

        Returns:
            Predicted next-obs.
        """
        return self.trunk(th.cat([obs, pred_actions], dim=1))


class InverseDynamicsTransformer(nn.Module):
    """Inverse model using a transformer to predict actions."""

    def __init__(self, latent_dim: int, action_dim: int, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim*2, nhead=num_heads, dim_feedforward=128),
            num_layers=num_layers
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Predict actions given current and next latent observations."""
        features = th.cat([obs, next_obs], dim=-1)
        transformed_features = self.transformer(
            features.unsqueeze(0)).squeeze(0)
        return self.action_predictor(transformed_features)


class ForwardDynamicsTransformer(nn.Module):
    """Forward model using a transformer to predict next observations."""

    def __init__(self, latent_dim: int, action_dim: int, num_heads: int = 3, num_layers: int = 2) -> None:
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim+action_dim, nhead=num_heads, dim_feedforward=128),
            num_layers=num_layers
        )
        self.obs_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Predict next latent observations."""
        features = th.cat([obs, actions], dim=-1)
        transformed_features = self.transformer(
            features.unsqueeze(0)).squeeze(0)
        return self.obs_predictor(transformed_features)
