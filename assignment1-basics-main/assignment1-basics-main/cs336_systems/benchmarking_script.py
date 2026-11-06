"""Assignment 2 benchmarking：测 Transformer 前向 / 反向 / 优化器步的单次耗时。

用法（在 assignment 根目录）：
    .venv/Scripts/python.exe cs336_systems/benchmarking_script.py \
        --d-model 512 --num-layers 4 --num-heads 16 --batch-size 32 --context-length 256 --max-iters 20

计时要点：
  * CUDA 异步执行——每个计时区间前后必须 torch.cuda.synchronize()，否则 timeit 只测到 kernel 启动延迟（微秒级）
  * 先 warmup 几轮，排除 CUDA 上下文初始化、显存分配器扩容等一次性开销
  * timeit.Timer.timeit(number=N) 返回 N 次【总】秒数，要除以 N 才是单次
  * 测 forward 时用 torch.no_grad()，否则隐式建计算图，时间和显存都不真实
"""
import timeit

import torch

try:
    from cs336_basics.my_pytorch import Transformer, AdamW, cross_entropy, gradient_clipping
    from cs336_basics.train_my_transformer import build_parser
except ImportError:  # 直接以脚本方式运行、包未安装时的兜底
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from cs336_basics.my_pytorch import Transformer, AdamW, cross_entropy, gradient_clipping
    from cs336_basics.train_my_transformer import build_parser


def timeit_gpu(fn, number: int, warmup: int = 3) -> float:
    """返回 fn 单次执行的平均秒数（含 CUDA 同步，带预热）。"""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer = timeit.Timer(fn)
    return timer.timeit(number) / number

def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    model = Transformer(args.vocab_size, args.context_length, args.num_layers, args.d_model,
                        args.num_heads, args.d_ff, args.theta, device)
    # 基准用固定随机 batch（计时不需要真实数据，label 也无所谓）
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = x.clone()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                      betas=(args.beta1, args.beta2), eps=args.eps)
    n_params = sum(p.numel() for p in model.parameters())
    number = min(args.max_iters, 30)  # 每个阶段重复 number 次；别让默认的 30000 把基准跑成训练

    def forward_only():
        torch.cuda.nvtx.range_push('forward')
        with torch.no_grad():
            model(x)
        torch.cuda.nvtx.range_pop()

    def forward_backward():
        torch.cuda.nvtx.range_push('forward')
        optimizer.zero_grad()
        loss = cross_entropy(model(x).view(-1, args.vocab_size), y.view(-1))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('backward')
        loss.backward()
        torch.cuda.nvtx.range_pop()

    def optimizer_step():
        torch.cuda.nvtx.range_push('optimizer_step')
        optimizer.step()
        torch.cuda.nvtx.range_pop()

    def full_iteration():
        torch.cuda.nvtx.range_push('forward')
        loss = cross_entropy(model(x).view(-1, args.vocab_size), y.view(-1))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('backward')
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('optimizer_step')
        optimizer.step()
        torch.cuda.nvtx.range_pop()

    # 预热 + 同步
    for _ in range(3):
        full_iteration()

    t_fwd = timeit_gpu(forward_only, number)
    t_fb = timeit_gpu(forward_backward, number)
    t_step = timeit_gpu(optimizer_step, number)
    t_all = timeit_gpu(full_iteration, number)

    # 稳态峰值显存
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        full_iteration()
        peak_gib = torch.cuda.max_memory_allocated() / 2**30

    tok_per_sec = args.batch_size * args.context_length / t_all
    print(f'model: {n_params:,} params | batch {args.batch_size} x context {args.context_length} '
          f'on {device} | number={number}')
    print(f'forward           : {t_fwd * 1000:8.2f} ms/iter')
    print(f'forward+backward  : {t_fb * 1000:8.2f} ms/iter   (backward ≈ {(t_fb - t_fwd) * 1000:.2f} ms)')
    print(f'optimizer step    : {t_step * 1000:8.2f} ms/iter')
    print(f'full iteration    : {t_all * 1000:8.2f} ms/iter   ({tok_per_sec:,.0f} tokens/s)')
    if torch.cuda.is_available():
        print(f'peak VRAM         : {peak_gib:8.2f} GiB')


if __name__ == '__main__':
    main()
