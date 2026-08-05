"""Training utilities for AlphaNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: dict[str, torch.Tensor],
    device: str = "cpu",
) -> tuple[float, float, float]:
    model.train()
    state = data["state"].to(device)
    legal_mask = data.get("legal_mask", data.get("mask")).to(device).bool()
    policy_target = data["policy"].to(device).float()
    value_target = data["value"].to(device).float().view(-1)

    optimizer.zero_grad()
    pred_log_policy, pred_value = model(state, legal_mask=legal_mask)
    policy_loss = soft_policy_cross_entropy(pred_log_policy, policy_target)
    value_loss = F.mse_loss(pred_value, value_target)
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def soft_policy_cross_entropy(
    pred_log_policy: torch.Tensor,
    policy_target: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy for MCTS visit-count distributions."""

    return -(policy_target * pred_log_policy).sum(dim=1).mean()
