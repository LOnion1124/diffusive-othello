import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaNet(nn.Module):
    def __init__(self, board_size=9, in_channels=3, num_filters=64, num_res_blocks=3, action_size=None):
        super().__init__()
        H = board_size
        A = action_size if action_size is not None else H*H
        self.conv0 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(num_filters)
        # residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters)
            ) for _ in range(num_res_blocks)
        ])
        # policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * H * H, A)
        # value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * H * H, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, legal_mask=None):
        # x: (B, C, H, W)
        out = F.relu(self.bn0(self.conv0(x)))
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = F.relu(out + residual)
        # policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)  # raw logits for actions
        # apply legal mask by setting logits to -inf on illegal
        if legal_mask is not None:
            illegal = (legal_mask == 0)
            logits = logits.masked_fill(illegal, -1e9)
        policy = F.log_softmax(logits, dim=1)  # use log probs for NLLLoss
        # value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(1)  # in [-1,1]
        return policy, value  # policy: log-prob, value: scalar
    
if __name__ == "__main__":
    pass