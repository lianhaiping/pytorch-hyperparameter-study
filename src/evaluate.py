"""Evaluation: metrics and predictions from a model, with no gradient tracking.

Both functions put the model in eval mode (model.eval()) and run under torch.no_grad().
Callers that continue training must switch back with model.train().
"""

import numpy as np
import torch
from torch import nn


def rmse(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """RMSE over the whole set in one pass (not an average of per-batch RMSEs)."""
    model.eval()
    with torch.no_grad():
        pred = model(X)
    assert pred.shape == y.shape, f"shape mismatch {pred.shape} vs {y.shape}"
    return torch.sqrt(torch.mean((pred - y) ** 2)).item()


def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Model outputs for X, moved back to the CPU as a NumPy array of shape (n, 1)."""
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy()
