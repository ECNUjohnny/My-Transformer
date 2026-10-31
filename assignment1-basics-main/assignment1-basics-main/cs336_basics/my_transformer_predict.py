import torch
from my_pytorch import (
    Transformer,
    load_checkpoint,
    softmax,
)

from train_my_transformer import (
    build_parser,
)

from my_bpe import Tokenizer

TOKENS = 'Once upon a time, '

class Predict:
    def __init__(self, args):
        self.transformer = Transformer(args.vocab_size, args.context_length, args.num_layers, args.d_model, 
                                       args.num_heads, args.d_ff, args.theta, args.device)
        self.tokenizer = Tokenizer.from_files(args.vocab, special_tokens=['<|endoftext|>'])
        self.args = args
        load_checkpoint(args.ckpt, self.transformer)
    
    @torch.no_grad()
    def predict(self, tokens: str, max_tokens: int, tao: float=1.0, p: float=1.0):
        tokens_id = torch.tensor([self.tokenizer.encode(tokens)], dtype=torch.long).to(self.args.device)
        for _ in range(max_tokens):
            probs = softmax(self.transformer(tokens_id[:, -self.args.context_length:])[-1, -1], dim=-1, tao=tao)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            drop = (cum - sorted_probs) >= p
            sorted_probs[drop] = 0.0
            sorted_probs /= sorted_probs.sum()
            ids = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_ids[ids] 
            if self.tokenizer.vocab[next_id.item()].decode('utf-8', errors='replace') == '<|endoftext|>':
                break
            tokens_id = torch.cat((tokens_id, next_id.view(1, 1)), dim=-1)
        return self.tokenizer.decode(tokens_id.detach().cpu().tolist()[-1])


def main():
    args = build_parser().parse_args()
    predictor = Predict(args)
    print(predictor.predict(TOKENS, args.max_tokens, args.temperature, args.top_p))

if __name__ == '__main__':
    main()