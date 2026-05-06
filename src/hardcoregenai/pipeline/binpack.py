import math
import itertools

def binpack(batch: list[list[int]], chunk_len: int, drop_last: bool=True) -> list[list[int]]:
    """
    Pack sequences into fixed-length bins using Best Fit First with fragmentation,
    ensuring that all bins except possibly the last are exactly chunk_len long.

    Args:
        batch: list of sequences to pack
        chunk_len: length of each bin
        drop_last: if True, drop the last bin if it's shorter than chunk_len

    Returns:
        list of bins (each bin is a list of ints). All bins except possibly the last have length == chunk_len.
    """
    MAX_FRAGMENTS = 512
    frag_cnt = 0
    seq_len = None # Used only for error msgs
    orig_batch_len = len(batch)

    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    
    # Compute how many bins are needed to hold all tokens
    total_tokens = sum(len(seq) for seq in batch)
    if drop_last:
        num_bins = math.floor(total_tokens / chunk_len) if total_tokens > 0 else 0
    else:
        num_bins = math.ceil(total_tokens / chunk_len) if total_tokens > 0 else 0
    bins: list[list[int]] = [[] for _ in range(num_bins)]
    
    # Split any seq larger than chunk_len
    split_batch = []
    for seq in batch:
        it = iter(seq)
        while (chunk := list(itertools.islice(it, chunk_len))):
            split_batch.append(chunk)

    full_chunks    = [c for c in split_batch if len(c) == chunk_len]
    partial_chunks = [c for c in split_batch if len(c) != chunk_len]

    # Place the full chunks in order
    for idx, piece in enumerate(full_chunks):
        bins[idx] = piece

    # Now we place the partial chunks, trying to minimize fragmentation
    # Sorting longest-chunk-first helps
    partial_chunks = sorted(partial_chunks, key=len, reverse=True)

    piece_stack = []
    # Place each sequence into a bin
    for seq in partial_chunks:
        frag_cnt = 0
        seq_len = len(seq)
        piece_stack.append(seq)

        while piece_stack:
            piece = piece_stack.pop() 
            # Best Fit placement: use the smallest bin with available >= len(piece)
            best_i = None
            best_space = chunk_len + 1
            all_bins_full = True
            for i, bn in enumerate(bins):
                available = chunk_len - len(bn)
                if available > 0:
                    all_bins_full = False
                if available >= len(piece) and available < best_space:
                    best_space = available
                    best_i = i
            if all_bins_full:
                # When drop_last==True, we can have all_bins_full and still have some leftovers;
                # so just drop them and return.
                continue
            if best_i is not None:
                bins[best_i].extend(piece)
            else:
                # If we get here, we need to fragment
                if frag_cnt > MAX_FRAGMENTS:
                    raise RuntimeError(f'Too many fragments: Seq of len={seq_len} requires more than {MAX_FRAGMENTS} fragments.  Either truncate it or increase MAX_FRAGMENTS.')
                frag_cnt += 1
                # Size the initial fragment to fill the bin with the most available
                max_i = max(range(len(bins)), key=lambda i: chunk_len - len(bins[i]))
                available = chunk_len - len(bins[max_i])
                # Place the fragment in that bin, filling it
                bins[max_i].extend(piece[:available])
                # and recurse on the remainder
                piece_stack.append(piece[available:])

    for i, bn in enumerate(bins):
        if drop_last:
            assert len(bn) == chunk_len, f'Bin {i} has len={len(bn)}, should be {chunk_len}'

    return bins
