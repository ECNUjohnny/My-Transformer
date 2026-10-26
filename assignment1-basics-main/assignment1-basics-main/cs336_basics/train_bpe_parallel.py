"""大语料（TinyStories / OpenWebText）的多进程 BPE 训练脚本。

BPE 训练是纯 CPU 任务，瓶颈在预分词计数——用 multiprocessing 按块并行。
分块沿 <|endoftext|> 边界切（cs336_basics.pretokenization_example.find_chunk_boundaries），
每个 worker 独立统计自己块内的 pretoken 计数，主进程合并 Counter 后跑 merge 循环。

用法（在 assignment 根目录）：
    python train_bpe_parallel.py --input data/owt_train.txt --vocab-size 32000
    python train_bpe_parallel.py --input data/TinyStories-train.txt --vocab-size 10000

产出 data/<语料名>_bpe_<vocab_size>.pkl，含 vocab 与 merges。
"""
import argparse
import os
import sys

HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import pickle
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import regex as re
from tqdm import tqdm

from bpe import PAT, init_vocab, merge_loop, to_byte_tuples
from parallel import find_chunk_boundaries

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def count_chunk(args):
    """worker：读 [start, end) 字节块 -> 按 special token 切 -> 预分词 -> 返回 str pretoken 计数"""
    path, start, end, special_tokens = args
    with open(path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    freq = Counter()
    if special_tokens:
        splitter = re.compile('|'.join(re.escape(t) for t in special_tokens))
        pieces = splitter.split(chunk)
    else:
        pieces = (chunk,)
    for piece in pieces:
        freq.update(re.findall(PAT, piece))
    return freq


def train_bpe_parallel(input_path, vocab_size, special_tokens=None, num_workers=None):
    special_tokens = ['<|endoftext|>'] if special_tokens is None else special_tokens
    num_workers = num_workers or os.cpu_count()

    # 块数取 worker 数的 8 倍做负载均衡（块边界只能落在 special token 上，大小不均）
    with open(input_path, 'rb') as f:
        split_token = special_tokens[0].encode('utf-8')
        boundaries = find_chunk_boundaries(f, num_workers * 8, split_token)
    tasks = [(str(input_path), s, e, tuple(special_tokens))
             for s, e in zip(boundaries[:-1], boundaries[1:]) if e > s]
    print(f'{len(tasks)} chunks, {num_workers} workers')

    t0 = time.perf_counter()
    str_counts = Counter()
    with Pool(num_workers) as pool:
        for chunk_counts in tqdm(pool.imap_unordered(count_chunk, tasks, chunksize=1),
                                 total=len(tasks), desc='pre-tokenize', ncols=100):
            str_counts.update(chunk_counts)
    print(f'[profile] pre-tokenization: {time.perf_counter() - t0:.1f}s, '
          f'unique pretokens: {len(str_counts):,}')

    t1 = time.perf_counter()
    words, counts = to_byte_tuples(str_counts)
    del str_counts
    print(f'[profile] to byte tuples: {time.perf_counter() - t1:.1f}s, unique words: {len(words):,}')

    vocab = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)
    t2 = time.perf_counter()
    merges = merge_loop(words, counts, vocab, num_merges)
    print(f'[profile] merge loop: {time.perf_counter() - t2:.1f}s, merges: {len(merges):,}')
    return vocab, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(DATA / 'owt_train.txt'))
    ap.add_argument('--vocab-size', type=int, default=32000)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    vocab, merges = train_bpe_parallel(args.input, args.vocab_size, num_workers=args.workers)

    out = DATA / f'{Path(args.input).stem}_bpe_{args.vocab_size}.pkl'
    with open(out, 'wb') as f:
        pickle.dump({'vocab': vocab, 'merges': merges}, f)
    print(f'saved -> {out}')

    longest = max(vocab.values(), key=len)
    print(f'longest token ({len(longest)} bytes): {longest!r}')


if __name__ == '__main__':
    main()
