"""Fixed MLP used for every run."""

import torch
from torch import nn


class MLP(nn.Module):
    """in_features -> [Linear -> ReLU] x len(hidden_sizes) -> Linear(1).

    No dropout or normalization layers: explicit regularization is off by design.
    """

    def __init__(self, in_features: int = 8, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))  # linear output for regression
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(seed: int, in_features: int, hidden_sizes: list[int], device: torch.device) -> MLP:
    # Weights are initialized on the CPU from the seed, then moved, so the initial
    # weights are bit-identical on every device and for every configuration.
    torch.manual_seed(seed)
    model = MLP(in_features, tuple(hidden_sizes))
    return model.to(device)
