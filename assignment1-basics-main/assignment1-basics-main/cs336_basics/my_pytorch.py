import torch
from torch import nn
from einops import rearrange, reduce, repeat, einsum
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
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))  # 可学习权重，初始全 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # .to() 不是 in-place，要重新赋值
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weight
        return x.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 64 * math.ceil(d_model * 8 / 3 / 64)  # ≈8/3 且取整到 64 的倍数
        # w1/w3 上投影、w2 下投影，都用 Linear 子模块（与参考 state dict 的 ffn.w*.weight 对齐）
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def SiLU(self, x):
        return silu(x)

    def forward(self, x):
        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))


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

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(*q.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        k = k.view(*k.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        v = v.view(*v.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 因果 mask：下三角 True（第 i 行只许看 j <= i），必须是 bool 类型才能取反
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        out = scaled_dot_product_attention(q, k, v, mask)

        out = out.transpose(-3, -2).reshape(*x.shape)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads:int, d_ff: int, theta: float, max_seq_len:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        self.attn = MultiheadSelfAttention(d_model, num_heads, self.rope, device, dtype)
        self.ffn = FFN(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)   # 两个子层各自的 norm，权重独立
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None):
        # pre-norm + 残差：y = x + attn(ln1(x))，再 y + ffn(ln2(y))（讲义公式 15）
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers,
                 d_model, num_heads, d_ff, theta, device=None, dtype=None):
        super().__init__()
        # ModuleList 的下标命名 layers.0..N-1，与参考实现 state dict 键名一致
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device, dtype)
            for _ in range(num_layers)
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)  # 输出到词表大小，不是 d_model

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        # 返回未归一化的 logits；softmax 留给损失函数（cross-entropy 内部会做 log-softmax）
        return self.lm_head(self.ln_final(x))


def cross_entropy(logits: torch.Tensor, target: torch.Tensor): 
    probs = softmax(logits, dim=-1)
    