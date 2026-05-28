import math

import pytest
import torch
import lightning as L
from ft4.models.uniform_model import UniformModel

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


def test_is_lightning_module():
    assert isinstance(UniformModel(vocab_size=16), L.LightningModule)


def test_configure_optimizers_returns_none():
    assert UniformModel(vocab_size=16).configure_optimizers() is None

@pytest.mark.filterwarnings("ignore:You are trying to `self\\.log\\(\\)`:UserWarning")
def test_validation_step_loss_is_log_vocab_size():
    vocab_size = 32
    model = UniformModel(vocab_size=vocab_size)
    # Tokens > 0 to avoid PAD (RES_PAD=0) being ignored, which would skew the loss.
    tokens = torch.randint(low=1, high=vocab_size, size=(2, 8), dtype=torch.int64)
    loss = model.validation_step({'tokens': tokens})
    torch.testing.assert_close(
        loss, torch.tensor(math.log(vocab_size)),
        msg=f'UniformModel val loss should be log({vocab_size})',
    )

