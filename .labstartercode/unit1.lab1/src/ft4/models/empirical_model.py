from collections import Counter, defaultdict

import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L
import msgpack

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer

class EmpiricalModel(L.LightningModule):
    def __init__(
        self,
        n: int = 3,
        vocab_size: int = Ft4Tokenizer.vocab_size(),
        min_distinct_suffixes: int = 4,
        smoothing: float = 0,
        val_smoothing: float = 1e-1
    ):
        """
        EmpiricalModel predicts the next token according to the empirical probability, 
        conditioned on the previous n-1 tokens.

        n:                     The context window (cw) used for prediction is n - 1.
        vocab_size:            Size of vocabulary.
        min_distinct_suffixes: Prune all prefixes that weren't observed with at least min_distinct_suffixes.
                               (Avoids "reciting from memory".)
        smoothing:             Probability that we predict a suffix that we've *never* observed.
        val_smoothing:         Smoothing used for computing cross-entropy during validation.
        """
        super().__init__()
        self.save_hyperparameters()
        assert n > 0
        self.n = int(n)
        self.cw = self.n - 1
        """cw (context window): The number of preceding tokens we condition on to predict the next one"""
        
        self.vocab_size = vocab_size
        assert (self.vocab_size ** self.cw) <= (2**63 - 1), f'{n=} is too big for {vocab_size=} (must encode to < 64 bits)'
        self.min_distinct_suffixes = min_distinct_suffixes
        self.smoothing_alpha = smoothing / vocab_size
        self.val_smoothing_alpha = val_smoothing / vocab_size

        self.ngram_counts: defaultdict[int, Counter] = defaultdict(Counter)
        """
        ngram_counts: A dict of the form {prefix -> {suffix -> count}}
        
        Usage: 
               self.prefix_suffix_freq[prefix][suffix]     
               # How many times does suffix occur after prefix?

        Observed ngrams are split up into prefix (cw tokens) and suffix (the last token). 
        When predicting, we can query by prefix alone or by (prefix, suffix).

        For performance, the entire prefix is encoded as a single int64.

        (Using a dict is atypical for PyTorch, but necessary for empirical counts.)
        """
        
        self._validating = False
        """_validating: Are we currently validating?"""
        self._padding_token = Ft4Tokenizer.RES_PAD
        """_padding_token: The token used for padding"""
        self.automatic_optimization = False

    def forward(self, tokens: Tensor) -> Tensor:
        """
        tokens:  A batch of sequences of token IDs
                 Shape: (B,L) int.
        returns: Prediction logits:
                 For each position in the sequence, a distribution over the vocab for the *next* token.
                 Shape: (B,L,V) float.
        where:
         B = batch size
         L = tokens per sequence
         V = vocab size
        """
        freq = self._suffix_freq(tokens)
        logits = self._freq2logits(freq)

        # If all logits for a particular (b,l) are -inf (i.e. we've never seen this prefix at all),
        # set them all to 0 (uniform).
        mask_all_neginf = torch.isneginf(logits).all(dim=-1, keepdim=True)
        logits[mask_all_neginf.expand_as(logits)] = 0
        
        return logits

    def _freq2logits(self, freq: Tensor) -> Tensor:
        """
        Transforms a tensor of frequencies (counts) to logits.
        """
        # Determine the logits
        # Hint: This can be done very easily (a single line of code).
        logits = None # TODO-LAB unit1.lab1
        raise NotImplementedError("Implement in unit1.lab1")
        return logits

    def _suffix_freq(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens:  A batch of sequences of token IDs
                 Shape: (B,L) int.
        returns: Empirical suffix frequency: 
                 For each pos in the seq, a distribution over the vocab for the *next* token 
                 showing how often that token followed the cw.
                 Shape: (B,L,V) float.
        where:
         B = batch size
         L = tokens per sequence
         V = vocab size
        """
        # Empirical Models use counts and dicts, not gradients, so vectorizing this in PyTorch 
        # is a bit like fitting a square peg into a round hole.
        #
        # You do not need to understand the details of this method; in Lab 2, when we use 
        # neural networks, the code becomes both simpler and more powerful.
        B, L = tokens.shape
        freqs = torch.zeros((B, L, self.vocab_size), dtype=torch.float32, device=self.device)

        # If cw == 0, there's no context; we simply return the unigram token frequency
        if self.cw == 0:
            suffix_histogram = self.ngram_counts.get(0) # No prefix
            if suffix_histogram:
                suffix_ids = torch.tensor(list(suffix_histogram.keys()), dtype=torch.long, device=self.device)
                suffix_counts = torch.tensor(list(suffix_histogram.values()), dtype=torch.float32, device=self.device)
                freqs[:, :, suffix_ids] = suffix_counts
            return freqs # Skip smoothing

        # Otherwise, cw >= 1, so we use a sliding window to collect the most cw recent tokens at every pos
        # To give us a full prefix for the first pos, we pad the seq with cw-1 tokens on the left
        pad_len = self.cw - 1
        left_padding = torch.full((pad_len,), self._padding_token, dtype=tokens.dtype, device=self.device)

        # We use base_multiplier to compactly encode the entire prefix into a single int64
        base_multiplier = self.vocab_size ** pad_len

        for b in range(B):
            seq = tokens[b]
            if pad_len:
                seq = torch.cat((left_padding, seq), dim=0)

            # Initialize prefix with the first (cw-1) items (no-op when cw==1)
            prefix = 0
            for j in range(pad_len):
                prefix = prefix * self.vocab_size + int(seq[j])

            # For each pos, slide the window once and write the row immediately
            for i in range(L):
                prefix = prefix * self.vocab_size + int(seq[i + self.cw - 1])  # push right
                suffix_histogram = self.ngram_counts.get(prefix)
                if suffix_histogram is not None:
                    suffix_ids = torch.tensor(list(suffix_histogram.keys()), dtype=torch.long, device=self.device)
                    suffix_counts = torch.tensor(list(suffix_histogram.values()), dtype=torch.float32, device=self.device)
                    freqs[b, i, suffix_ids] = suffix_counts
                prefix -= int(seq[i]) * base_multiplier                   # pop left

        # alpha, if > 0, is a form of additive smoothing, added to each count
        # See https://en.wikipedia.org/wiki/Additive_smoothing
        alpha = self.val_smoothing_alpha if self._validating else self.smoothing_alpha
        freqs.add_(float(alpha))

        return freqs

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        """
        We "train" by collecting empirical counts from the token stream.
        This is atypical for PyTorch (no optimizer/gradients), but suits this model.

        batch['tokens']: A batch of sequences of token IDs
                         Shape: (B,L) int.
        batch_idx:       Ignored
        where:
        B = batch size
        L = tokens per sequence
        """
        # See comments in _suffix_freq for implementation details
        tokens: Tensor = batch["tokens"]
        B, L = tokens.shape
        V = self.vocab_size
        cw = self.cw
        n = self.n

        # Pad each sequence with cw PADs on the left and 0 padding on the right
        # This allows us to to have a complete ngram at each pos, including the first
        padded = F.pad(tokens, (self.cw, 0), value=self._padding_token) # shape: (B, L+cw)

        # Collect sliding windows of n tokens each
        windows = padded.unfold(1, size=n, step=1).contiguous() # shape: (B, L, n)

        # Compact each prefix to a single int64
        prefix_expanded = windows[:, :, :cw]  # (B, L, cw)
        prefix_encoded = torch.zeros((B, L), dtype=torch.int64, device=self.device)
        for i in range(cw):
            prefix_encoded = prefix_encoded * V + prefix_expanded[:, :, i]
    
        suffix = windows[:, :, cw].to(torch.int64)  # (B, L)

        prefix_suffix_pairs = torch.stack((prefix_encoded.reshape(-1), suffix.reshape(-1)), dim=1)  # (2, B, L)

        # Row-wise unique
        uniq_pairs, cnt = torch.unique(prefix_suffix_pairs, dim=0, return_counts=True)
        # Shape: (K, 2), (K,), where K is num. uniq. prefix-sufix pairs
        # For j < K:
        #   uniq_pairs[0, j] is an encoded prefix
        #   uniq_pairs[1, j] is a suffix
        #   cnt[j]           is its count

        # Merge into ngram_counts
        uniq_pairs = uniq_pairs.cpu().numpy()
        cnt = cnt.cpu().numpy()
        for (pfx, sfx), c in zip(uniq_pairs, cnt):
            self.ngram_counts[int(pfx)][int(sfx)] += int(c)
    
    def validation_step(self, batch):
        assert self._validating
        tokens = batch['tokens'] # shape: (B, L+1)
        x = tokens[..., :-1] # shape: (B, L) x[b][i] is the token id at pos i (in batch b)
        y = tokens[..., 1:] # shape: (B, L)  y[b][i] is the token id at pos i+1 (and thus the target for pos i)
        logits = self(x) # shape: (B, L, V)
        
        logits_ = logits.flatten(start_dim=0, end_dim=1) # shape: (B*L, V)
        y_ = y.flatten() # shape: (B*L,)
        loss = F.cross_entropy(input=logits_, target=y_, ignore_index=self._padding_token)

        self.log("val_ce", loss, prog_bar=True, logger=True)
        
        return loss

    def on_validation_start(self):
        self._validating = True

    def on_validation_end(self):
        self._validating = False

    def _pruned_prefix_suffix_freq(self) -> dict[int, dict[int, int]]:
        """
        Return a pruned copy of the packed-prefix dict:
          { prefix_id:int -> {suffix:int -> count:int} }
        keeping only prefixes with >= min_distinct_suffixes continuations.
        """
        m = int(self.min_distinct_suffixes)
        if m <= 1:
            # Shallow copy to avoid exposing internals directly in checkpoint
            return {pid: dict(inner) for pid, inner in self.ngram_counts.items()}

        pruned: dict[int, dict[int, int]] = {}
        for pid, counter in self.ngram_counts.items():
            # Keep only positive-count suffixes (normally all)
            positive = {s: c for s, c in counter.items() if c > 0}
            if len(positive) >= m:
                pruned[pid] = positive
        self.print(f"prune: total_prefixes={len(self.ngram_counts)} kept={len(pruned)} "
              f"ratio={len(pruned) / max(1, len(self.ngram_counts)):.3f}")
        return pruned

    def on_save_checkpoint(self, checkpoint):
        """
        Save the pruned packed-prefix dict.
        (PyTorch auto-saves tensors, but dicts require manual saving.)
        """
        pruned = msgpack.packb(self._pruned_prefix_suffix_freq(), use_bin_type=True)
        checkpoint[f"prefix_suffix_freq_pk_n{self.n:03}"] = pruned

    def on_load_checkpoint(self, checkpoint):
        key = f"prefix_suffix_freq_pk_n{self.n:03}"
        raw = checkpoint.get(key, None)
        if raw is None:
            return  # nothing serialized yet
        data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        # Re-wrap into the required type
        self.ngram_counts = defaultdict(Counter, {int(pid): Counter(inner) for pid, inner in data.items()})
        
    def configure_optimizers(self):
        # Counting only; no optimizer required.
        return None
    
    def __str__(self):
        return f'EmpiricalModel(n={self.n})'
