# TODO Support batch sizes

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer

from typing import Optional
import torch
import torch.nn.functional as F
import sys


class LangGen():
    def __init__(self, model, tokenizer=None, initial_txt: str='', max_to_generate: int=128, min_p: float=0.2, temperature: float=0.75):
        """
        We use min_p https://arxiv.org/abs/2407.01082 
        which performs better than top_p and top_k.

        Implementing top_p and top_k should be easy, should you like.
        """
        assert temperature >= 0.0
        assert 0.0 <= min_p <= 1.0

        self.model = model
        if tokenizer is None:
            tokenizer = Ft4Tokenizer
        self.max_to_generate = max_to_generate
        self.tokenizer = tokenizer
        self.min_p = min_p
        self.temperature = temperature

        # TODO make a dict or some holder for special tokens; incorporate this into the string
        _tokens = [self.tokenizer.RES_START] + self.tokenizer.tokenize(initial_txt)
        self.tokens = torch.tensor(_tokens, dtype=torch.int64, device=model.device)
        self.generated_count = 0
        self.hit_max = False

    def __next__(self) -> str:
        if self.hit_max:
            raise StopIteration
        
        # The unsqueeze and [0] are to add and remove a dummy batch
        # TODO Instead, just use a batch size of 1
        logits = self.model(torch.unsqueeze(self.tokens, 0))
        next_token = self.sample(logits)[0]
        self.tokens = torch.cat((self.tokens, next_token))
        
        if next_token == self.tokenizer.RES_STOP:
            raise StopIteration
        self.generated_count += 1
        next_str = self.tokenizer.detokenize([next_token.item()]).decode(errors='surrogateescape') # type:ignore
        
        if self.generated_count >= self.max_to_generate:
            self.hit_max = True
            next_str += '…'
        return next_str
        
    def __iter__(self):
        return self

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample a token based on the given logits.

        logits:  For each pos in every seq in the batch, a distribution over the vocab for the *next* token.
                 logits[b,i,v] is the predicted logit that v is the token at position i+1 of sequence b
                 Shape: (B, L, V) float.
        returns: For every seq in the batch, the index of our selected (i.e. sampled) next token.
                 Shape: (B, 1) int.
                 The second dim is pos. While it's normally L, here it's always 1, since we only provide a 
                 sample for the *last* pos in each seq.
        where:
            B = batch size
            L = tokens per sequence
            V = vocab size
        """
        assert len(logits.shape) == 3
        # This could be made faster if we used the cache hidden state, instead of 
        # restarting from scratch on each call. For our purposes, it's not needed.
        
        # Drop all but the last positions
        last_pos_logits = logits[:,-1,:] # shape: (B, V)

        # Never generate PAD or BOS
        last_pos_logits[..., Ft4Tokenizer.RES_PAD] = -torch.inf
        last_pos_logits[..., Ft4Tokenizer.RES_START] = -torch.inf

        if self.temperature == 0:
            # Greedy
            max_logit = last_pos_logits.max(dim=-1, keepdim=True).values
            last_pos_logits = torch.where(last_pos_logits >= max_logit, last_pos_logits, -torch.inf)
        elif self.temperature != 1.0:
            last_pos_logits = last_pos_logits / self.temperature

        probs = F.softmax(last_pos_logits, dim=-1)
        
        if 0.0 < self.min_p < 1.0:
            max_prob = probs.max(dim=-1, keepdim=False).values # shape: (B,)
            # In min_p, we mask out any prob that is less than the *max* prob times the adjustable min_p param
            cutoff = max_prob * self.min_p
            probs = torch.where(probs >= cutoff, probs, 0)
            # Since we've masked some prob values, they won't sum to unity
            # That's fine, since torch.multimodal normalizes automatically
        
        assert torch.any(probs > 0, dim=-1), "All probs are zero"
        
        # torch.multinomial is random, based on the seed
        next_token = torch.multinomial(probs, num_samples=1)
        assert len(next_token.shape) == 2
        assert next_token.shape[0] == logits.shape[0]
        return next_token
