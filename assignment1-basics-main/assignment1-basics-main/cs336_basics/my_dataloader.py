import numpy as np
import torch
import os
from typing import BinaryIO, IO

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    # 标签要读到 i + context_length，所以起点上限是 n - context_length - 1（randint 上界开区间）
    starts = np.random.randint(0, n - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)          # 广播出 (batch, context) 索引矩阵
    x = torch.from_numpy(dataset[idx].astype(np.int64)).to(device)
    y = torch.from_numpy(dataset[idx + 1].astype(np.int64)).to(device)
    return x, y 


def save_checkpoint(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        iteration: int, 
        out: str | os.PathLike | BinaryIO | IO[bytes],
):    
    dict_to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': iteration,
    }

    torch.save(dict_to_save, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']
