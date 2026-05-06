from __future__ import annotations
import csv
import unicodedata

# TODO Simplify naming (`token` instead of `tid`, etc.)
# Handle reserved tokens better (a dict, and detokenize them back to their dict name, so we get <BOS>)
# Use BOS, EOS instead of START, STOP 

class BpeTokenizer:
    """
    A simple Byte-Pair Encoding (BPE) tokenizer.

    Uses bytes as the basic unit, not unicode chars (e.g. (e.g. 0x41 'A', not U+0041).
    Starts with 256 tokens, one for each byte, and learns new tokens via BPE merge.
    
    Usage example:
        tokenizer = BpeTokenizer())
        tokenizer.learn_tokens([str1, str2, str3])
        tokenizer.save_csv('tokens.csv')
        
        tk = BpeTokenizer.from_csv('tokens.csv')
        tokens = tk.tokenize(a_string) # [343, 4033, 5021, ...]
        assert tk.detokenize(tokens) == a_string

    """
    def __init__(self, init_bytes=True, n_reserved_tokens: int=0) -> None:
        """
        If specified, reserves the first `n_reserved_tokens` ids for control tokens
        """
        self.bytes2tid: dict[bytes, int] = {}
        self.tid2bytes: dict[int, bytes] = {}
        self.last_tid = n_reserved_tokens-1
        self.longest_token_len = 0

        if init_bytes:
            # Initialize all 256 byte values as tokens
            for b in range(256):
                self._add_token(bytes([b]))
        
    @classmethod
    def from_csv(cls, csv_file_path: str) -> BpeTokenizer:
        """
        Load tokenizer from CSV at `csv_file_path`
        CSV file example:
            bytes_hex, tid, comment
            0a0d, 256, '\r\n'
            414242, 257, 'ABB'
            6566, 258, 'ef'
        """
        tokenizer = cls(init_bytes=False, n_reserved_tokens=0)
        with open(csv_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for hexbytes, tid_str, _ in reader:
                b = bytes.fromhex(hexbytes)
                tid = int(tid_str)
                tokenizer.bytes2tid[b] = tid
                tokenizer.tid2bytes[tid] = b
                tokenizer.last_tid = max(tokenizer.last_tid, tid)
        tokenizer.longest_token_len = max((len(b) for b in tokenizer.bytes2tid), default=0)
        
        # validate no duplicate tids
        if len(set(tokenizer.bytes2tid.values())) != len(tokenizer.bytes2tid):
            raise ValueError('Duplicate token ID found in CSV')
        
        return tokenizer

    def tokenize(self, txt: str|bytes|bytearray, unknown_tid=None) -> list[int]:
        """
        `txt` - a str (which is normalized and returned as utf-8 bytes)
                or bytes or bytesarray (which should already be utf-8).

        `unknown_tid`, if provided, will be used to tokenize unknown bytes.
        (However, unknown bytes should _never_ occur under normal usage, as _all_ bytes should have a token.)
        """
        txt_bytes = self._encode(txt)
        tokens: list[int] = []

        p = 0
        data_len = len(txt_bytes)
        while p < data_len:
            for token_len in range(self.longest_token_len, 0, -1):
                piece = txt_bytes[p:p+token_len]
                tid = self.bytes2tid.get(piece)
                if tid is not None:
                    tokens.append(tid)
                    p += token_len
                    break
            else:
                # We can't find a token for this byte!
                # This should never happen, because __init__ initializes a token for all bytes by default.
                # We emit the unknown_token_id (if provided) or just skip it.
                if unknown_tid is not None:
                    tokens.append(unknown_tid)
                p += 1
        return tokens

    def detokenize(self, tids: list[int]) -> bytes:
        """
        Reconstruct bytes from tids; reserved or unknown tids yield markers.
        """
        bs = bytearray()
        for tid in tids:
            tid = int(tid)
            b = self.tid2bytes.get(tid)
            if b is None:
                bs.extend(f'<SPECIALTOKEN:{tid:06d}>'.encode('utf-8'))
            else:
                bs.extend(b)
        return bytes(bs)
    
    def vocab_size(self) -> int:
        return self.last_tid + 1

    def save_csv(self, csv_file_path: str) -> None:
        """
        Save token mappings to CSV at `csv_file_path` (which must not already exist).
        """
        with open(csv_file_path, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['bytes_hex', 'tid', 'comment'])
            for b, tid in self.bytes2tid.items():
                writer.writerow([b.hex(), tid, repr(b.decode(errors='surrogateescape'))])

    def fmt_tokenized(self, txt: str) -> str:
        """
        Formats `txt` grouped by tokens, for debugging.
        """
        tids = self.tokenize(txt)
        strings = (self.detokenize([tid]).decode(errors='replace') for tid in tids)
        return ''.join(f'[{t}]' for t in strings)

    def _add_token(self, token: bytes) -> int:
        """
        Add `token` to vocab and return its new tid.
        """
        self.last_tid += 1
        self.bytes2tid[token] = self.last_tid
        self.tid2bytes[self.last_tid] = token
        self.longest_token_len = max(self.longest_token_len, len(token))
        return self.last_tid

    def _encode(self, txt: str|bytes|bytearray) -> bytes:
        if isinstance(txt, str):
            # For simplicity, normalization is kept minimal
            return unicodedata.normalize('NFKC', txt).replace('\r\n', '\n').encode(errors='replace')
        else:
            # We also accept raw bytes for `txt`, in which case we skip normalization and encoding entirely
            return bytes(txt)

    def percent_of_tokens_that_occur_at_least_n_times(self, texts: list[str| bytes|bytearray], n: int = 100) -> float:
        """
        Fraction of current tokens seen >= n times.
        A rule of thumb is this should be >= 0.95 for n=100:
        > Use the highest possible vocabulary size such that 
        > 95% of [tokens] occur at least 100 times in the data.
        > https://arxiv.org/abs/2306.16842
        > See also https://aclanthology.org/2020.findings-emnlp.352.pdf 
        > (but see https://arxiv.org/abs/2404.00397)

        """
        counts = {id: 0 for id in self.tid2bytes}
        for txt in texts:
            for id in self.tokenize(txt):
                counts[id] += 1
        total = len(counts)
        return (sum(1 for c in counts.values() if c >= n) / total)

    def __repr__(self) -> str:
        return f'BpeTokenizer(tokens={len(self.bytes2tid)}, longest_token_len={self.longest_token_len})'

# TODO The reserved token names should be moved into the base Tokenizer class
# If someone choses not to use them (ie not to reserve), that's fine; 
# but we can certainly standardize their names and numbers.
# Also: BOS and EOS are more common than START and STOP.
# So name these: TOKEN_PAD, TOKEN_UNK, TOKEN_BOS, TOKEN_EOS.
# Then, print out the known ones nicely '<BOS>', '<EOS>', etc.
Ft4Tokenizer = BpeTokenizer.from_csv('src/ft4/pipeline/tokens.v01.csv')
Ft4Tokenizer.RES_PAD = 0
Ft4Tokenizer.RES_UNK = 1
Ft4Tokenizer.RES_START = 2
Ft4Tokenizer.RES_STOP = 3

