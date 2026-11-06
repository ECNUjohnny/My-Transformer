import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import timeit
from my_pytorch import scaled_dot_product_attention, MultiheadSelfAttention
import triton
import triton.language as tl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_models = [16, 32, 64, 128]
context_lens = [256, 1024, 4096]

product = [(d_model, context_len) for d_model in d_models for context_len in context_lens]


MIB = 1024**2

def timeit_gpu(fn, number: int, warmup: int=3) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        y = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer = timeit.Timer(fn)
    t = timer.timeit(number) / number
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 时间测量结束时必须同步,否则只测到 kernel 发射耗时
    return y, t

def measure_memory(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return base / MIB, peak / MIB

for d_model, context_len in product:
    x = torch.rand((8, context_len, d_model), dtype=torch.bfloat16, device=device)
    attention = MultiheadSelfAttention(d_model, 1, device=device, dtype=torch.bfloat16)
    attention = torch.compile(attention, dynamic=True)

    # ---- 显存测量:先 warmup 一次(触发 RoPE 缓存等惰性分配),再测前向与 fwd+bwd 峰值 ----
    with torch.no_grad():
        attention(x)
    fwd_base, fwd_peak = measure_memory(lambda: attention(x))
    def fwd_bwd():
        y_mem = attention(x)
        y_mem.mean().backward()
    _, fb_peak = measure_memory(fwd_bwd)

    # ---- 时间测量:编译图是一次性的(donated buffer 优化),不能复用 y 反复 backward ----
    # 每次迭代重新前向再反向,retain_graph=False 正好成立;bwd 单独耗时 ≈ fwd+bwd − fwd
    def atten_fn():
        return attention(x)
    y, time_f = timeit_gpu(atten_fn, 100)

    def atten_fb():
        attention.zero_grad(set_to_none=True)
        y_iter = attention(x)
        l = F.mse_loss(y_iter, x, reduction='none')
        l.mean().backward()
    _, time_fb = timeit_gpu(atten_fb, 100)
    time_b = time_fb - time_f

    print(f'd_model={d_model}, context_len={context_len}: fwd: {time_f*1e3:.3f} ms, fwd+bwd: {time_fb*1e3:.3f} ms, bwd≈{time_b*1e3:.3f} ms, '
          f'mem: base {fwd_base:.0f} MiB, fwd_peak {fwd_peak:.0f} MiB, fwd+bwd_peak {fb_peak:.0f} MiB')

    del y, x, attention
    torch.cuda.empty_cache()
