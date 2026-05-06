# Remember: This is *your* code base. Experiment, modify, explore!
# It's helpful to modify the code to print the shape
# and some sample data of important tensors.

import torch
from torch import Tensor
import torch.nn as nn
from hardcoregenai.pipeline.bpe_tokenizer import HcgaiTokenizer

class UniformModel(nn.Module):
    """
    The UniformModel is the simplest possible model that fits the pipeline.

    It predicts the same probability for every candidate token (!), 
    and hence has *no* predictive value. But it's a great way to get started, 
    and smoke tests the pipeline end-to-end.
    """
    def __init__(self, vocab_size=HcgaiTokenizer.last_tid+1):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = None

    def forward(self, tokens: Tensor) -> Tensor:
        """
        tokens:  A batch of sequences of token IDs
                 Shape: (B,L) int.
        returns: Prediction logits: For each position in the sequence, 
                 a distribution over the vocab for the *next* token.
                 Shape: (B,L,V) float.
        where:
         B = batch size
         L = tokens per sequence
         V = vocab size
        """
        B, L = tokens.shape
        V = self.vocab_size
        # Determine the logits:
        # The UniformModel trivially predicts an identical logit for *every* candidate in the vocab.
        logits = None # TODO-LAB unit1.lab1
        raise NotImplementedError("Implement in unit1.lab1")
        return logits

if __name__ == "__main__":
    import sys
    from hardcoregenai.app.language_model import LangGen
    
    model = UniformModel()
    PROMPT = 'Now, she wondered if'

    print('# UniformModel')
    print('# Skipping training (nothing to train)')
    print('# Generating...\n')

    response = LangGen(model=model, initial_txt=PROMPT, max_to_generate=64)

    sys.stdout.write(LangGen.bold(PROMPT))
    sys.stdout.flush()
    for txt in response:
        sys.stdout.write(txt)
        sys.stdout.flush()
    sys.stdout.write('\n\n')

