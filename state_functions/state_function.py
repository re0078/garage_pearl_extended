"""This modules creates a continuous Q-function network."""

import torch

from garage.torch.modules import MLPModule


class ContinuousMLPStateFunction(MLPModule):
    """
    Implements a continuous MLP next state network.

    It predicts the next state for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of f(s, a).
    """

    def __init__(self, env_spec, **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim + self._action_dim,
                           output_dim=self._obs_dim,
                           **kwargs)

    def forward(self, obs_actions, latent):
        """Predicts next state"""
        return super().forward(torch.cat(obs_actions, latent))
