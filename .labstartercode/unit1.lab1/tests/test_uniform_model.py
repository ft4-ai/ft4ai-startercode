import pytest
import torch
from hardcoregenai.models.uniform_model import UniformModel

@pytest.mark.lab("unit1.lab1")
def test_forward_shape():
    model = UniformModel(vocab_size=16)
    seq = [[0, 1, 2, 3, 4, 5, 0],
           [1, 1, 1, 2, 2, 2, 3]]
    tokens = torch.tensor(seq, dtype=torch.int16)
    
    logits = model(tokens)
    assert logits.shape[0] == tokens.shape[0], 'Shape must be (B,L,V)'
    assert logits.shape[1] == tokens.shape[1], 'Shape must be (B,L,V)'
    assert logits.shape[2] == 16, 'Shape must be (B,L,V)'
    assert torch.is_floating_point(logits), 'Logits must be floats'

@pytest.mark.lab("unit1.lab1")
def test_forward_uniformity():
    model = UniformModel(vocab_size=16)
    seq = [[0, 1, 2, 3, 4, 5, 0],
           [1, 1, 1, 2, 2, 2, 3]]
    tokens = torch.tensor(seq, dtype=torch.int16)
    
    logits = model(tokens)
    torch.testing.assert_close(logits, logits[0,0,0].expand_as(logits), msg='UniformModel should predict *uniform* logits')

@pytest.mark.lab("unit1.lab1")
def test_probability_is_one_over_vocab_size():
    vocab_size = 256
    model = UniformModel(vocab_size=vocab_size)
    seq = [[0, 1, 2, 3, 4, 5, 0],
           [1, 1, 1, 2, 2, 2, 3]]
    tokens = torch.tensor(seq, dtype=torch.int16)
    
    logits = model(tokens)
    probs = torch.softmax(logits, dim=-1)
    
    torch.testing.assert_close(probs, torch.full_like(probs, 1/vocab_size), msg=f'UniformModel should predict probability {1/vocab_size=}')





