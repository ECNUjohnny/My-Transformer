import torch
import numpy as np
from torch import distributed as dist

c = torch.arange(10)

c = c[:, None]

d = np.random.uniform(size=(2, 3, 3, ))



print(c)