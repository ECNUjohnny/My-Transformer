import torch
import numpy as np

a = torch.rand((4, 5, 6))
b = torch.randint(0, 2, (4, 5))

print(a[b])

