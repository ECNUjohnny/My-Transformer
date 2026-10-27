import torch
from torch import nn
import einops
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = 2 / (in_features + out_features)
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight @ x


class Embedding(nn.Moudle):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.to(torch.float32)
        rms = torch.sqrt((x**2).mean(dim=2) + self.eps)
        x = x / rms * self.g
        x.to(self.dtype)
        return x

class FFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_model * 8 // 3
        self.weight1 = nn.Parameter(torch.rand(self.d_ff, self.d_model))
        self.weight3 = nn.Parameter(torch.rand(self.d_ff, self.d_model))
        self.weight2 = nn.Parameter(torch.rand(self.d_ff, self.d_ff))

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.weight2 @ (self.SiLU((self.weight1 @ x)) * (self.weight3 @ x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device=None):
        super().__init__()
        freq = theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k) 
        pos = torch.arange(0, max_seq_len-1, device=device, dtype=torch.float32)
        angles = torch.outer(pos, freq)
        self.register_buffer('cos_cached', angles.cos(), persistent=False)
        self.register_buffer('sin_cached', angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1 = x[:, :, 0::2]
        x2 = x[:, :, 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        return torch.stack((y1, y2), dim=-1).flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x -= torch.max(x, dim=dim, keepdim=True)
    xsum = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    return x / xsum


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: list[list[bool]]):
    d_k = query.shape[-1]
    raw_atten = query @ key / math.sqrt(d_k)
    masked_atten = torch.fill(raw_atten, -1e9)
     

