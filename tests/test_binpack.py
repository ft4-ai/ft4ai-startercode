import pytest

from hardcoregenai.pipeline.binpack import binpack

def test_best_fit_first_and_stable_sort():
    batch = [[1], [2,3], [4,5,6], [7,8], [9], [10], [11, 12], [13]]
    result = binpack(batch, 3, drop_last=True)
    assert result == [[4,5,6], [2,3,1], [7,8,9], [11,12,10]]

def test_fragment_placement():
    batch = [[1], [2,3], [4,5,6], [7,8,9,10], [11,12,13,14,15,16,17,18,19,20], [21,22], [23,24,25,26,27]]
    result = binpack(batch, 3, drop_last=True)
    expected = [[4,5,6], [7,8,9], [11,12,13], [14,15,16], [17,18,19], [23,24,25], [2,3,1], [21,22,10], [26,27,20]]
    assert result == expected

def test_alot_of_fragments_smoke():
    batch = [[1,2,3], [4]*200, [5,6], [7,8,9], [10]*200, [11,12,13,14]]
    binpack(batch, 2)

def test_many_fragments_smoke():
    batch = [[1,2,3], [4]*1000, [5,6]*100, [7,8]*10, [11,12,13,14]*10, [15,16]]
    binpack(batch, 7)

def test_perfect_fit():
    batch = [[1, 2], [3, 4]]
    assert binpack(batch, 2) == [[1, 2], [3, 4]]

def test_best_fit_after_fragmentation():
    batch = [[1, 1, 1], [2]]
    result = binpack(batch, 2, drop_last=False)
    assert _all_full_except_last(result, 2)
    assert all(len(bin) == 2 for bin in result)
    assert result == [[1, 1], [1, 2]]

def test_fragmentation_and_fill():
    batch = [[1, 2, 3], [4], [5]]
    result = binpack(batch, 2, drop_last=False)
    assert _all_full_except_last(result, 2)
    assert len(result) == 3
    assert result == [[1, 2], [3, 4], [5]]

def test_bins_full_except_last():
    batch = [[1] * 5]
    result = binpack(batch, 2, drop_last=False)
    assert len(result) == 3
    assert _all_full_except_last(result, 2)
    assert len(result[-1]) == 1

def test_drop_last_true():
    batch = [[1] * 5]
    # last bin of length 1 dropped
    result = binpack(batch, 2, drop_last=True)
    assert len(result) == 2
    assert all(len(bin) == 2 for bin in result)

def test_empty_and_errors():
    with pytest.raises(ValueError):
        binpack([[1]], 0)
    assert binpack([], 5) == []
    assert binpack([], 5, drop_last=False) == []

def test_two_underfilled_bins():
    # Two sequences shorter than chunk_len could produce two underfilled bins
    results = binpack([[1, 2, 3], [4, 5, 6]], chunk_len=4)
    expected = [[1,2,3,4]]
    assert results == expected

def test_fragmentation_leaves_middle_bin_short():
    results = binpack([[10, 20, 30, 40], [50, 60, 70]], chunk_len=3)
    expected = [[10, 20, 30], [50, 60, 70]]
    assert results == expected

def test_multiple_sequences_and_fragmentation_mid_short():
    # First sequence of length 7 will fragment into [5] and [2];
    # second sequence of length 4 will go into a new bin of length 4.
    # After dropping the last bin (len=4<5), the middle fragment bin (len=2) remains.
    seq1 = list(range(7))
    seq2 = list(range(7, 11))
    results = binpack([seq1, seq2], chunk_len=5)
    expected = [[0, 1, 2, 3, 4], [7, 8, 9, 10, 5]]
    assert results == expected

def _all_full_except_last(bins: list[list[int]], chunk_len: int):
    if not bins:
        return True
    return all(len(bin) == chunk_len for bin in bins[:-1])
