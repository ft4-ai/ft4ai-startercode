# TODO Support batch sizes

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from ft4.models.uniform_model import UniformModel

from typing import Optional
import torch
import torch.nn.functional as F
import sys

class LangGen():
    def __init__(self, model, tokenizer=None, initial_txt: str='', max_to_generate: int=128, top_k: int=0, top_p: float=0.0, temperature: float=1.0):
        self.model = model # TODO
        if tokenizer is None:
            tokenizer = Ft4Tokenizer
        self.max_to_generate = max_to_generate
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
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
        Sample an index from logits with optional top-k, top-p (nucleus) sampling, and temperature.

        Args:
            logits: 1D tensor of shape (vocab_size,) containing unnormalized log probabilities.
            top_k: keep only top_k highest logits; 0 disables.
            top_p: keep only enough top logits to reach cumulative probability >= top_p; 0.0 disables.
            temperature: scaling factor for logits; 1.0 disables.

        Returns:
            Selected token index as tensor dtype=int64 shape=(1,)
        """
        # TODO Add min-p

        # Never generate PAD or BOS
        logits[..., Ft4Tokenizer.RES_PAD] = float("-inf")
        logits[..., Ft4Tokenizer.RES_START] = float("-inf") # TODO Rename to BOS
        
        if len(logits.shape) == 3:
            # TODO PERFORMANCE This could be made much faster if we used the cache hidden state as opposed to
            # restarting from scratch each time!
            logits = logits[:,-1,:] # HACK TODO test compatibility with all models
        
        if self.temperature != 1.0:
            logits = logits / self.temperature

        if self.top_k > 0:
            top_k_logits, _ = torch.topk(logits, self.top_k)
            threshold = top_k_logits[-1]
            logits = torch.where(
                logits < threshold,
                torch.tensor(float('-inf'), device=logits.device), # Mask out if < threshold
                logits,
            )

        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Mask tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Always keep at least one token
            # we do this by shifting the mask - TODO explain
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Remove logits
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        #if 0.0 < self.min_p < 1.0:
        #    raise NotImplemented # TODO
        probs = F.softmax(logits, dim=-1)
        # TODO support controlling the seed
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    @classmethod
    def main(cls):
        prompt = sys.stdin.read()
        response = cls(model=UniformModel(start=8, stop=Ft4Tokenizer.last_tid+1), initial_txt=prompt)
        sys.stdout.write(cls.bold(prompt.rstrip()))
        sys.stdout.flush()
        for txt in response:
            sys.stdout.write(txt)
            sys.stdout.flush()
        sys.stdout.write('\n\n')


    @classmethod
    def bold(cls, txt: str) -> str:
        if sys.stdout.isatty():
            return "\033[1m" + txt + "\033[0m"
        return txt

if __name__ == '__main__':
    LangGen.main()

