"""
Models for the Pendulum test.
"""

import torch
from torch import nn


class PendulumValueNet(nn.Module):
    def __init__(self, observation_size: int = 3, action_size: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat((observations, actions), dim=-1))


class PendulumActorNet(nn.Module):
    def __init__(
        self,
        observation_size: int = 3,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        low = torch.tensor([-2.0]) if action_low is None else action_low.float()
        high = torch.tensor([2.0]) if action_high is None else action_high.float()
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("action_bias", (high + low) / 2)
        self.net = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, low.numel()),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations) * self.action_scale + self.action_bias
