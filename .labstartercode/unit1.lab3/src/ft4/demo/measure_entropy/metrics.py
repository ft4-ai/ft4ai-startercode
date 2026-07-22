# Test via:      `pytest` or `pytest --only-lab unit1.lab3`
# Then run via:  `python -m ft4.demo.measure_entropy`

# Add any imports you need here
import math

def max_corpus_entropy(guesser_accuracy: float, vocab_size: int) -> float:
    """
    Upper bound on the corpus's true per-token entropy, in nats. If a guesser
    reaches this accuracy, the source cannot have more entropy than this.
    """
    assert vocab_size > 1
    # Return the upper bound on the corpus entropy, in nats.
    # Hint: Use an approach similar to what you used in previous parts of the lab.
    raise NotImplementedError('TODO-LAB unit1.lab3')
