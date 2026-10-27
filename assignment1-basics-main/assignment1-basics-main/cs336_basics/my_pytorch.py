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
        std = math.sqrt(2 / (in_features + out_features))  # 讲义给的是方差，std 参数要开根号
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T  # 权重按 (d_out, d_in) 存，计算时转置


class Embedding(nn.Module):
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
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))  # 可学习权重，初始全 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # .to() 不是 in-place，要重新赋值
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.g
        return x.to(in_dtype)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 64 * math.ceil(d_model * 8 / 3 / 64)  # ≈8/3 且取整到 64 的倍数
        std = math.sqrt(2 / (self.d_ff + self.d_model))
        self.weight1 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_ff, self.d_model), mean=0, std=std, a=-3*std, b=3*std))
        self.weight3 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_ff, self.d_model), mean=0, std=std, a=-3*std, b=3*std))
        self.weight2 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_model, self.d_ff), mean=0, std=std, a=-3*std, b=3*std))

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return (self.SiLU(x @ self.weight1.T) * (x @ self.weight3.T)) @ self.weight2.T


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device=None):
        super().__init__()
        freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)  # 负指数：低维转得快
        pos = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)  # 含最后一个位置
        angles = torch.outer(pos, freq)
        self.register_buffer('cos_cached', angles.cos(), persistent=False)
        self.register_buffer('sin_cached', angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1 = x[..., 0::2]  # 任意 leading 维度，别写死 [:, :, ...]
        x2 = x[..., 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        return torch.stack((y1, y2), dim=-1).flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 先减最大值防溢出（.values 取 max 的结果张量）；不用 in-place，别改动调用方的张量
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = query.shape[-1]
    # einsum 直接标明轴含义，等价于 query @ key.transpose(-2, -1)
    raw_atten = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # 讲义约定：True = 保留，False = 屏蔽。用 -1e9 而不是 -inf，避免整行全被屏蔽时出 NaN
        raw_atten = raw_atten.masked_fill(~mask, -1e9)
    return softmax(raw_atten, dim=-1) @ value

