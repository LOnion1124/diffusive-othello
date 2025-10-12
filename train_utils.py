import torch
from model import AlphaNet

def train_step(model, optimizer, data, device="cuda"):
    model.train()
    state = data["state"].to(device)  # (N, 3, size, size)
    mask = data["mask"].to(device)    # (N, size*size)
    policy_target = data["policy"].to(device)  # (N, size*size)
    value_target = data["value"].to(device).float()  # (N,)

    optimizer.zero_grad()
    pred_log_policy, pred_value = model(state, legal_mask=mask)
    # Policy loss: NLLLoss expects class indices, but we have onehot
    policy_target_idx = policy_target.argmax(dim=1)
    policy_loss = torch.nn.functional.nll_loss(pred_log_policy, policy_target_idx)
    # Value loss: MSE
    value_loss = torch.nn.functional.mse_loss(pred_value, value_target)
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()
