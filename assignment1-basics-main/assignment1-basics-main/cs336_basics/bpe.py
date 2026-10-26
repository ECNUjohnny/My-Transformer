import regex as re
from pathlib import Path
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from tqdm import tqdm
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PAT)


def init_vocab(special_lists: list[str]) -> dict[int, bytes]:
    """special tokens 在前，之后是 256 个单字节 token"""
    vocab = {i: t.encode('utf-8') for i, t in enumerate(special_lists)}
    for b in range(256):
        vocab[len(vocab)] = bytes([b])
    return vocab


def count_words(text: str, special_lists: list[str]) -> Counter:
    """按 special token 切分 -> GPT-2 正则预分词 -> Counter[str pretoken]
    用 findall + Counter.update 让计数留在 C 层，最后只对 unique pretoken 做转码"""
    freq = Counter()
    if special_lists:
        splitter = re.compile("|".join(re.escape(t) for t in special_lists))
        segments = splitter.split(text)
    else:
        segments = (text,)
    for seg in tqdm(segments, ncols=100, leave=False):
        freq.update(re.findall(PAT, seg))
    return freq


def to_byte_tuples(freq: Counter) -> tuple[list[tuple[bytes, ...]], list[int]]:
    """Counter[str] -> (words, counts)，每个 unique pretoken 转成 tuple[bytes, ...]"""
    words, counts = [], []
    for s, c in freq.items():
        words.append(tuple(bytes([b]) for b in s.encode('utf-8')))
        counts.append(c)
    return words, counts


def merge_loop(words: list[tuple[bytes, ...]], counts: list[int],
               vocab: dict[int, bytes], num_merges: int) -> list[tuple[bytes, bytes]]:
    """增量维护 pair 频次与倒排索引(pair -> 含它的词下标)，每轮合并一个最高频 pair。
    并列时取字典序更大的 pair（讲义规定）。vocab 会被就地扩展。"""
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for idx, word in enumerate(words):
        cnt = counts[idx]
        for pair in zip(word, word[1:]):
            pair_counts[pair] += cnt
            pair_to_words[pair].add(idx)

    merges = []
    for _ in tqdm(range(num_merges), ncols=100):
        if not pair_counts:
            break
        # max 按元组比较：先比频次，频次并列时比 pair 字典序，正好是讲义的规定
        _, best = max(zip(pair_counts.values(), pair_counts.keys()))
        merges.append(best)
        new_token = best[0] + best[1]
        vocab[len(vocab)] = new_token

        for idx in pair_to_words.pop(best, ()):
            word, cnt = words[idx], counts[idx]
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= cnt
                if not pair_counts[pair]:
                    del pair_counts[pair]
                pair_to_words[pair].discard(idx)

            # 重叠序列必须从左到右贪心：(a,a,a) 合并 (a,a) 应得 (aa,a)
            merged, j = [], 0
            while j < len(word):
                if j + 1 < len(word) and word[j] == best[0] and word[j + 1] == best[1]:
                    merged.append(new_token)
                    j += 2
                else:
                    merged.append(word[j])
                    j += 1
            word = tuple(merged)
            words[idx] = word
            for pair in zip(word, word[1:]):
                pair_to_words[pair].add(idx)
                pair_counts[pair] += cnt
    return merges


def bpe(input_path: str, vocab_size: int,
        special_lists: list[str] = ['<|endoftext|>']) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """串行版，适合小语料（corpus.en / TinyStories sample）。大语料用 train_bpe_parallel.py"""
    with open(input_path, 'rb') as f:
        text = f.read().decode(encoding='utf-8', errors='ignore')

    vocab = init_vocab(special_lists)
    freq = count_words(text, special_lists)
    words, counts = to_byte_tuples(freq)

    num_merges = vocab_size - 256 - len(special_lists)
    merges = merge_loop(words, counts, vocab, num_merges)
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)
        self.antivocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}  # rank 小 = 学得早 = 先合并
        self.special_tokens = list(special_tokens) if special_tokens else []
        for tok in self.special_tokens:
            bseq = tok.encode('utf-8')
            if bseq in self.antivocab:
                continue
            self.vocab[len(self.vocab)] = bseq
            self.antivocab[bseq] = len(self.vocab) - 1
        # special token 切分正则：长的放前面，重叠时（如 <|endoftext|><|endoftext|>）优先整体匹配
        self._special_pat = None
        if self.special_tokens:
            alts = '|'.join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self._special_pat = re.compile('(' + alts + ')')
        self._cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str = None, special_tokens=None):
        """加载 train_bpe_parallel.py 存的 pickle（单文件含 'vocab' 和 'merges' 两个键）"""
        with open(vocab_filepath, 'rb') as f:
            saved = pickle.load(f)
        return cls(saved['vocab'], saved['merges'], special_tokens)

    def _pretokenize(self, text: str) -> Iterator[tuple[str, bool]]:
        """切成 (piece, is_special)。
        与训练预分词的关键区别：special token 必须保留为独立 piece（训练统计时才丢弃），
        所以 split 要带捕获组，把切分符留在返回的列表里。"""
        if self._special_pat is None:
            yield from ((m.group(0), False) for m in _PAT.finditer(text))
            return
        for piece in self._special_pat.split(text):
            if not piece:
                continue
            if piece in self.special_tokens:
                yield piece, True
            else:
                yield from ((m.group(0), False) for m in _PAT.finditer(piece))

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        """pretoken -> bytes 序列，反复套用当前 rank 最小（最早学到）的 merge 直到无法再合并。
        pretoken 大量重复出现，用 cache 避免重复编码。"""
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached
        parts = [bytes([b]) for b in pretoken.encode('utf-8')]
        while len(parts) >= 2:
            pair = min((p for p in zip(parts, parts[1:]) if p in self.merge_ranks),
                       key=self.merge_ranks.get, default=None)
            if pair is None:
                break
            merged, j = [], 0
            while j < len(parts):
                if j + 1 < len(parts) and (parts[j], parts[j + 1]) == pair:
                    merged.append(pair[0] + pair[1])
                    j += 2
                else:
                    merged.append(parts[j])
                    j += 1
            parts = merged
        ids = tuple(self.antivocab[b] for b in parts)
        self._cache[pretoken] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece, is_special in self._pretokenize(text):
            if is_special:
                ids.append(self.antivocab[piece.encode('utf-8')])
            else:
                ids.extend(self._encode_pretoken(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐块（如逐行）编码。训练大语料转 token ids 时用它，不用整读进内存"""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        # errors='replace'：无效 UTF-8 序列用替换符而非报错（讲义规定）
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')


def main():
    data = Path(__file__).resolve().parents[1] / 'data'
    vocab_path = data / 'owt_train_bpe_32000.pkl'
    owt_path = data / 'owt_valid.txt'
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path)
    with open(owt_path, 'r', encoding='utf-8') as f:
        text = f.read(1000)
        text = text[105:205]

    print(f'{len(tokenizer.encode(text)) / len(text)}')


if __name__ == '__main__':
    main()
