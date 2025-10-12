import torch

data = torch.load("dataset.pt");

print(data["state"].shape)
print(data["mask"].shape)
print(data["policy"].shape)
print(data["value"].shape)