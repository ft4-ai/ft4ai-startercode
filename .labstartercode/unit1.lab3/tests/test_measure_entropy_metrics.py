import pytest

from ft4.demo.measure_entropy import metrics

@pytest.mark.lab("unit1.lab3")
def test_max_corpus_entropy():
    guesser_accuracy = 0.4
    vocab_size = 10000
    expected = 6.2 # nats
    assert metrics.max_corpus_entropy(guesser_accuracy, vocab_size) == pytest.approx(expected, abs=0.05)

@pytest.mark.lab("unit1.lab3")
def test_max_corpus_entropy_zero_accuracy():
    guesser_accuracy = 0
    vocab_size = 10000
    expected = 9.2 # nats
    assert metrics.max_corpus_entropy(guesser_accuracy, vocab_size) == pytest.approx(expected, abs=0.05)
