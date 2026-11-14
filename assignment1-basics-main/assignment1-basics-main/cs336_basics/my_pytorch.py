import torch
import numpy as np
from torch import nn
from einops import rearrange, reduce, repeat, einsum
import math
import triton
import triton.language as tl

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))  # 讲义给的是方差,std 参数要开根号
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T  # 权重按 (d_out, d_in) 存,计算时转置


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
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))  # 可学习权重,初始全 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # .to() 不是 in-place,要重新赋值
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
        # w1/w3 上投影、w2 下投影,都用 Linear 子模块(与参考 state dict 的 ffn.w*.weight 对齐)
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
        freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)  # 负指数:低维转得快
        pos = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)  # 含最后一个位置
        angles = torch.outer(pos, freq)
        self.register_buffer('cos_cached', angles.cos(), persistent=False)
        self.register_buffer('sin_cached', angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1 = x[..., 0::2]  # 任意 leading 维度,别写死 [:, :, ...]
        x2 = x[..., 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        return torch.stack((y1, y2), dim=-1).flatten(-2)

def softmax(x: torch.Tensor, dim: int, tao: float | None=None) -> torch.Tensor:
    # 先减最大值防溢出(.values 取 max 的结果张量)；不用 in-place,别改动调用方的张量
    x_max = torch.max(x, dim=dim, keepdim=True).values
    if tao is None:
        e = torch.exp(x - x_max)
    else:
        e = torch.exp((x - x_max) / tao)
    return e / e.sum(dim=dim, keepdim=True)
    

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = query.shape[-1]
    # einsum 直接标明轴含义,等价于 query @ key.transpose(-2, -1)
    raw_atten = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # 讲义约定:True = 保留,False = 屏蔽。用 -1e9 而不是 -inf,避免整行全被屏蔽时出 NaN
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

        # 因果 mask:下三角 True(第 i 行只许看 j <= i),必须是 bool 类型才能取反
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
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)   # 两个子层各自的 norm,权重独立
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None):
        # pre-norm + 残差:y = x + attn(ln1(x)),再 y + ffn(ln2(y))(讲义公式 15)
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers,
                 d_model, num_heads, d_ff, theta, device=None, dtype=None):
        super().__init__()
        # ModuleList 的下标命名 layers.0..N-1,与参考实现 state dict 键名一致
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device, dtype)
            for _ in range(num_layers)
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)  # 输出到词表大小,不是 d_model

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        # 返回未归一化的 logits；softmax 留给损失函数(cross-entropy 内部会做 log-softmax)
        return self.lm_head(self.ln_final(x))


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # -log softmax(logits)_t = lse - logit_t:全程在 log 域,正确类概率再小也不会下溢出 inf
    x_max = torch.max(logits, dim=-1, keepdim=True).values          # 防溢出的最大值
    lse = x_max + torch.log(torch.exp(logits - x_max).sum(dim=-1))  # 稳定的 log-sum-exp
    picked = torch.gather(logits, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # 正确类的 logit
    return torch.mean(lse - picked)



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=5e-4, weight_decay=1e-2, betas=(0.9, 0.95), eps=1e-8):
        beta1, beta2 = betas
        defaults = {
            'lr': lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'lambda': weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            lam = group['lambda']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:                      # 懒初始化
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                t = state['t'] + 1                 # t 从 1 开始(讲义规定)
                state['t'] = t
                m, v = state['m'], state['v']
                g = p.grad
                alpha = lr * math.sqrt(1-beta2**t) / (1-beta1**t)

                with torch.no_grad():
                    p *= (1 - lr * lam)
                    m = beta1 * m + (1-beta1) * g
                    v = beta2 * v + (1-beta2) * (g**2)
                    state['m'], state['v'] = m, v
                    p -= alpha * m / (torch.sqrt(v) + eps)

            
def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw: return alpha_max * t / Tw
    elif Tw <= t <= Tc: return alpha_min + (1 + math.cos((t-Tw)/(Tc-Tw)*math.pi)) * (alpha_max - alpha_min) / 2
    else: return alpha_min

@torch.no_grad()
def gradient_clipping(params, gmax: float, eps: float=1e-6):
    norm = 0
    for p in params:
        if p.grad is None:
            continue
        norm += torch.linalg.norm(p.grad)**2
    if math.sqrt(norm) > gmax:
        for p in params:
            if p.grad is None:
                continue
            p.grad *= gmax / (eps+norm**0.5)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out) -> None:
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }, out)


def load_checkpoint(src, model: nn.Module, optimizer: torch.optim.Optimizer | None=None) -> int:
    ckpt = torch.load(src, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None: optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['iteration']


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    # 标签要读到 i + context_length,所以起点上限是 n - context_length - 1(randint 上界开区间)
    starts = np.random.randint(0, n - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)          # 广播出 (batch, context) 索引矩阵
    x = torch.from_numpy(dataset[idx].astype(np.int64)).to(device)
    y = torch.from_numpy(dataset[idx + 1].astype(np.int64)).to(device)
    return x, y

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # 启动网格 = (Tq, batch):每个程序实例只负责一个 batch、一个 query 分块。
    # 注意:Triton 3.9 已移除块指针(make_block_ptr)API,这里用裸指针算术代替,
    # 语义与 handout 的块指针版本完全一致。
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offs_d = tl.arange(0, D)

    q_i = tl.load(Q_ptr + batch_index * stride_qb
                  + offs_q[:, None] * stride_qq + offs_d[None, :] * stride_qd)

    # 片上缓冲区用 fp32 保证数值稳定(handout 准则)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    # 只有这一个循环:迭代 key 分块 j = 1..Tk
    for j in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        k_j = tl.load(K_ptr + batch_index * stride_kb
                      + offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd)
        v_j = tl.load(V_ptr + batch_index * stride_vb
                      + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd)

        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale

        if IS_CAUSAL:
            # 因果掩码:key 全局索引大于 query 全局索引处加 -1e6
            s_ij = s_ij + tl.where(offs_q[:, None] >= offs_k[None, :], 0.0, -1e6)

        m_ij = tl.maximum(m_i, tl.max(s_ij, axis=1))
        p_ij = tl.exp(s_ij - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p_ij, axis=1)
        o_ij = alpha[:, None] * o_i + tl.dot(p_ij.to(v_j.dtype), v_j)
        o_i, l_i, m_i = o_ij, l_ij, m_ij

    o_i = o_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_ptr + batch_index * stride_ob
             + offs_q[:, None] * stride_oq + offs_d[None, :] * stride_od,
             o_i.to(O_ptr.dtype.element_ty))
    tl.store(L_ptr + batch_index * stride_lb + offs_q * stride_lq,
             L_i.to(L_ptr.dtype.element_ty))


def _flash_backward_pytorch(Q, K, V, O, dO, L, scale, is_causal):
    """按 handout 式 (13)-(19) 用重计算实现反向传播;不依赖 softmax。
    L: 前向保存的 logsumexp, 形状 (batch, Nq)。所有计算在 fp32 中进行。"""
    Q, K, V, O, dO, L = (t.to(torch.float32) for t in (Q, K, V, O, dO, L))
    # D = rowsum(O ∘ dO),等于 rowsum(P ∘ dP)
    D_vec = (O * dO).sum(dim=-1, keepdim=True)
    # 重计算注意力分数与概率(式 13-14)
    S = Q @ K.transpose(-1, -2) * scale
    if is_causal:
        Nq, Nk = Q.shape[-2], K.shape[-2]
        mask = torch.arange(Nq, device=Q.device)[:, None] >= torch.arange(Nk, device=Q.device)[None, :]
        S = S.masked_fill(~mask, -1e6)
    P = torch.exp(S - L[..., None])
    # 式 15-16
    dV = P.transpose(-1, -2) @ dO
    dP = dO @ V.transpose(-1, -2)
    # 式 17-19
    dS = P * (dP - D_vec)
    dQ = dS @ K * scale
    dK = dS.transpose(-1, -2) @ Q * scale
    return dQ, dK, dV


_compiled_flash_bwd = torch.compile(_flash_backward_pytorch)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal=False) -> torch.Tensor:
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), \
            "Our pointer arithmetic will assume contiguous tensors"
        batch, Nq, D = q.shape
        Nk = k.shape[1]
        assert D in (16, 32, 64, 128), "head dim must be a power of 2 >= 16"

        scale = 1.0 / math.sqrt(D)
        o = torch.empty_like(q)
        l = torch.empty((batch, Nq), dtype=torch.float32, device=q.device)

        Q_TILE_SIZE, K_TILE_SIZE = 16, 64
        grid = (triton.cdiv(Nq, Q_TILE_SIZE), batch)
        flash_fwd_kernel[grid](
            q, k, v,
            o, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            Nq, Nk,
            scale,
            IS_CAUSAL=is_causal,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )
        # 保存反向传播所需的激活检查点
        ctx.save_for_backward(q, k, v, o, l)
        ctx.scale = scale
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        dq, dk, dv = _compiled_flash_bwd(q, k, v, o, do, l, ctx.scale, ctx.is_causal)
        # 梯度 dtype 与输入对齐
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


flash_attention = FlashAttention.apply

