"""训练脚本：把 cs336_basics.my_pytorch 的组件拼成完整训练循环。

用法（在 assignment 根目录）：
    1) 先把语料转成 token id 数组（uint16 .npy），例如用训练好的分词器：
       tokenizer = Tokenizer.from_files('data/TinyStories-train_bpe_10000.pkl')
       ids = []
       with open('data/TinyStories-train.txt') as f:
           ids.extend(tokenizer.encode_iterable(f))
       np.save('data/tinystories_ids.npy', np.array(ids, dtype=np.uint16))

    2) 训练：
       python ./cs336_basics/train_my_transformer.py --out-dir runs/ts 

    3) 断点续训：
       python train_my_transformer.py --dataset ... --out-dir runs/ts --resume-from runs/ts/checkpoint.pt
"""
import argparse
import os
import time
import array
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from my_pytorch import (
    Transformer,
    AdamW,
    cross_entropy,
    gradient_clipping,
    learning_rate_schedule,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)

from my_bpe import Tokenizer

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='训练 CS336 Transformer LM')
    # 数据
    ap.add_argument('--dataset', default='data/TinyStories-train.npy', help='token id 的 .npy 文件（uint16）')
    ap.add_argument('--vocab-size', type=int, default=10000)
    ap.add_argument('--context-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=32)
    # 模型结构
    ap.add_argument('--d-model', type=int, default=512)
    ap.add_argument('--num-layers', type=int, default=16)
    ap.add_argument('--num-heads', type=int, default=16)
    ap.add_argument('--d-ff', type=int, default=None, help='缺省按 8/3*d_model 取整到 64 的倍数')
    ap.add_argument('--theta', type=float, default=10000.0, help='RoPE 的 Θ')
    # 优化器
    ap.add_argument('--lr', type=float, default=1e-3, help='alpha_max，cosine 调度的峰值学习率')
    ap.add_argument('--min-lr', type=float, default=1e-4, help='alpha_min，cosine 末端学习率')
    ap.add_argument('--warmup-iters', type=int, default=100, help='线性预热的迭代数 T_w')
    ap.add_argument('--beta1', type=float, default=0.9)
    ap.add_argument('--beta2', type=float, default=0.95)
    ap.add_argument('--eps', type=float, default=1e-8)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--grad-clip', type=float, default=1.0, help='全局梯度范数上限')
    # 训练过程
    ap.add_argument('--max-iters', type=int, default=30000, help='总训练迭代数，也是 cosine 周期 T_c')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-interval', type=int, default=10)
    ap.add_argument('--ckpt-interval', type=int, default=500, help='每多少迭代存一次 checkpoint')
    ap.add_argument('--milestone-interval', type=int, default=5000, help='每多少迭代额外留一份不可覆盖的里程碑快照')
    ap.add_argument('--out-dir', default='runs/run01', help='checkpoint 输出目录')
    ap.add_argument('--resume-from', default=None, help='从此 checkpoint 恢复训练')
    return ap


def tokenize(args):
    """把 --dataset 对应的语料转成 token id 数组：data/X.npy <- data/X.txt，
    分词器取 data/X_bpe_<vocab_size>.pkl。已存在 .npy 时跳过。"""
    dataset_path = Path(args.dataset)
    corpus = dataset_path.with_suffix('.txt')
    if dataset_path.exists():
        return
    assert corpus.exists(), f'corpus not found: {corpus}'
    pkl = corpus.parent / f'{corpus.stem}_bpe_{args.vocab_size}.pkl'
    tokenizer = Tokenizer.from_files(str(pkl), special_tokens=['<|endoftext|>'])
    print(f'tokenizing {corpus} with {pkl.name} ...')
    token_ids = array.array('H')  # uint16 紧凑存储，list[int] 装 4.5 亿 token 要 16GB 内存
    with open(corpus, encoding='utf-8', errors='ignore') as f:  # 必须文本模式，正则只吃 str
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            token_ids.append(token_id)
            if i % 50_000_000 == 0 and i:
                print(f'  {i:,} tokens so far')
    np.save(dataset_path, np.frombuffer(token_ids, dtype=np.uint16))
    print(f'saved {len(token_ids):,} tokens -> {dataset_path}')


def main():
    args = build_parser().parse_args()

    tokenize(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(args.dataset, mmap_mode='r')
    print(f'dataset: {args.dataset}, {len(dataset):,} tokens, mmap 加载')

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model: {n_params:,} parameters on {args.device}')

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f'resumed from {args.resume_from} at iteration {start_iter}')

    ckpt_path = out_dir / 'checkpoint.pt'
    csv_path = out_dir / 'train_log.csv'
    if start_iter == 0:
        csv_path.write_text('iter,loss,lr,tok_per_sec\n')  # 覆盖旧日志；续训时追加

    t0 = time.perf_counter()
    consecutive_bad = 0
    progress = tqdm(range(start_iter, args.max_iters), initial=start_iter, total=args.max_iters, ncols=110)
    for it in progress:
        # cosine 学习率调度：直接改 param_groups 里的 lr，优化器 step 时现读
        lr = learning_rate_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr

        x, y = get_batch(dataset, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))

        # 崩塌保险：loss 不是有限值就跳过这一步；连续 3 次则回滚到最近 checkpoint
        if not torch.isfinite(loss):
            consecutive_bad += 1
            print(f'\n[warn] iter {it}: non-finite loss ({loss.item()}), skipped')
            if consecutive_bad >= 3 and ckpt_path.exists():
                bad_iter = load_checkpoint(ckpt_path, model, optimizer)
                consecutive_bad = 0
                print(f'[warn] rolled back to checkpoint at iteration {bad_iter}')
            optimizer.zero_grad()
            continue
        consecutive_bad = 0

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0 or it == args.max_iters - 1:
            tok_per_sec = args.batch_size * args.context_length * (it - start_iter + 1) / (time.perf_counter() - t0 + 1e-9)
            progress.set_postfix(loss=f'{loss.item():.3f}', lr=f'{lr:.2e}', tok_s=f'{tok_per_sec:.0f}')
            with open(csv_path, 'a') as f:  # 落盘日志，断连后也能事后画曲线
                f.write(f'{it},{loss.item():.4f},{lr:.6e},{tok_per_sec:.0f}\n')

        if (it + 1) % args.ckpt_interval == 0 or it == args.max_iters - 1:
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
        # 里程碑：每 milestone-interval 步额外留一份不可覆盖的快照，崩塌后有好据点可回
        if (it + 1) % args.milestone_interval == 0:
            save_checkpoint(model, optimizer, it + 1, out_dir / f'checkpoint_{it + 1}.pt')

    print(f'done. final loss {loss.item():.4f}, checkpoint -> {ckpt_path}')


if __name__ == '__main__':
    main()

