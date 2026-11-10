import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 2], [3, 4]])

c = torch.stack((a, b), dim=-1)

print(c)