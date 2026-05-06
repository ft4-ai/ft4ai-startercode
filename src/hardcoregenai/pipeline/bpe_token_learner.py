from __future__ import annotations
from typing import Optional, Sequence
import re
from collections import defaultdict
import gc

import heapdict
import tqdm

from hardcoregenai.pipeline.bpe_tokenizer import BpeTokenizer

class BpeTokenLearner:
    """
    Learns tokens from `texts` via BPE.

    The stateless algorithm is simple but quadratic runtime:    
        for each merge:                                  O(M)
            (a) tally all bigrams                        O(N)
            (b) find the highest freq                    O(B)
            (c) merge every occurrence in each word      O(N)
    A linear outer loop * linear inner loop = O(M*(N+B+N)) = O(MN) quadratic complexity
        
    Instead, we introduce two stateful data structures which can be updated in log time:
      * `bigram_nfreq`      - a priority queue mapping bigrams to their neg. frequency
      * `bigram_word_idx`   - an index of how many times bigram b appears in word w
    
    Now
      * Update `bigram_nfreq` instead of (a) tally from scratch
      * `bigram_nfreq` is a priority queue, so (b) is logarithmic
      * `bigram_word_count` makes (c) logarithmic as well
    giving runtime O(M log N).
    
    Merging a bigram removes up to 2 old bigrams (left and right) and creates up to 2 new
    ones. So on each merge, we need to make up to 4 updates to the data structures.

    The result is quite fast, but still memory hungry: approx. 100 bytes RAM per input
    byte, most of this due to Python object overhead.

    BPE is a standard algorithm, but implementations differ on  bytes vs.
    Unicode characters, normalization, pretokenization (i.e. recognizing word boundaries,
    handling the space before a word), and performance.  This implementation seeks to be
    as simple as possible while still able to process > 100 MB on a laptop.
    """
    def __init__(self, tk: Optional[BpeTokenizer]=None, init_bytes=True, n_reserved_tokens: int=0) -> None:
        self.tk = tk or BpeTokenizer(init_bytes=init_bytes, n_reserved_tokens=n_reserved_tokens)

        # A single list of tokenized words of all texts
        self.words: list[list[int]] = []
        
        # Since heapdict.peekitem() returns the *lowest* priority,
        # we store the *negative* frequency.
        self.bigram_nfreq = heapdict.heapdict()

        # Usage: 
        #   bigram_word_idx[(1,2)][3] == 4
        #   means: bigram (1,2) appears in word 3 a total of 4 times.
        self.bigram_word_count = defaultdict(dict)

    def learn_tokens(self, texts: Sequence[str|bytes|bytearray], max_tid:int=2**14-1, demo:str|None=None) -> None:
        """
        Learns up to `max_tid` of new tokens into `self.tk`
        (Do not set `demo`: use `BpeTokenLearner.demo()` instead.)
        """
        if demo is not None:
            assert texts == [demo]

        self._slurp_texts(texts)
        del texts # Free memory
        gc.collect()

        pbar = tqdm.tqdm(total=max_tid, initial=self.tk.last_tid, desc='Learning tokens', delay=1.0)
        while self.tk.last_tid < max_tid:
            if demo:
                print(self.tk.fmt_tokenized(demo))
            merger = self._merge_bigram()
            if merger is None:
                pbar.close()
                return
            if demo:
                print(f"Merge {repr(self.tk.tid2bytes[merger[0]].decode(errors='surrogateescape'))} + {repr(self.tk.tid2bytes[merger[1]].decode(errors='surrogateescape'))}: ", end='')
            pbar.update(1)
        pbar.close()

    @classmethod
    def demo(cls, txt: str='fuzzy wuzzy wuzz a bear.') -> None:
        """
        Demo of step-by-step BPE learning and segmentation of `txt`.
        At each step, learns the next merge and shows the new tokenization.
        """
        btl = cls(init_bytes=True, n_reserved_tokens=0)
        btl.learn_tokens(texts=[txt], demo=txt)

    def _merge_bigram(self):
        """
        Tries to merges one bigram (i.e. the most frequent)
        Returns `merger` on success
        or None if no bigrams left to merge.
        """
        if not self.bigram_nfreq:
            return None
        # Select the bigram with the largest freq (ties are arbitary)
        merger, _ = self.bigram_nfreq.peekitem()
        
        # and merge it into a new token
        merged = self.tk.tid2bytes[merger[0]] + self.tk.tid2bytes[merger[1]]
        if merged in self.tk.bytes2tid:
            raise Exception('Tried to merge a previously merged bigram!')
        new_token = self.tk._add_token(merged)
        
        self._replace_merger_with_new_token(merger, new_token)
        del self.bigram_nfreq[merger]
        return merger

    def _replace_merger_with_new_token(self, merger: tuple[int,int], new_token: int):
        """
        Replaces `merger` bigram with a single `new_token`, updating all internal data structures and indexes
        """
        for w in list(self.bigram_word_count[merger].keys()):
            word = self.words[w]
            word_len = len(word)
            p = 0
            while p < word_len - 1:
                if word[p] == merger[0] and word[p+1] == merger[1]:
                    # The bigram occurs at (word[i], word[i+1]).
                    # We now must find the old and new left and right bigrams
                    # and update accordingly.  Example:
                    #
                    # Given word='ABCD' and merge B+C->E,
                    # we delete old left (A,B) and old right (C,D)
                    # and add   new left (A,E) and new right (E,D)
                    old_bigrams = []
                    new_bigrams = []
                    if p > 0: # We have a left bigram, since we're not at the word's beginning
                        old_bigrams.append((word[p-1], word[p]))
                        new_bigrams.append((word[p-1], new_token))

                    if p < word_len - 2: # We have a right bigram, since we're not at the word's end
                        old_bigrams.append((word[p+1], word[p+2]))
                        new_bigrams.append((new_token, word[p+2]))
                        
                    for n in new_bigrams:
                        self.bigram_nfreq[n] = self.bigram_nfreq.get(n, 0) - 1
                        self.bigram_word_count[n][w] = self.bigram_word_count[n].get(w, 0) + 1
                    for o in old_bigrams:
                        if self.bigram_nfreq[o] == -1:
                            del self.bigram_nfreq[o]
                        else:
                            self.bigram_nfreq[o] = self.bigram_nfreq[o] + 1
                        if self.bigram_word_count[o][w] == 1:
                            del self.bigram_word_count[o][w]
                        else:
                            self.bigram_word_count[o][w] = self.bigram_word_count[o][w] - 1
                        
                    word[p] = new_token
                    del word[p+1]
                    word_len -= 1
                p += 1

    def _slurp_texts(self, texts: Sequence[str|bytes|bytearray]):
        """
        Slurp *all* the texts into a single list of words

        We *never* want to merge across word bondaries, so we first split `texts` 
        into `words`, spliting on spaces or other non-letters using a simple regex.
        """
        MAX_TOKEN_LEN = 16 # To prevent resource exhaustion
        pretokenize_regex = re.compile(rf"[ ]?[^\W\d_]{{1,{MAX_TOKEN_LEN}}}|.", flags=re.DOTALL)
        
        word_idx = 0
        for txt in tqdm.tqdm(texts, desc='Reading corpus', delay=1.0, leave=True):
            # Decode bytes to strings, so we can use regex for pretokenization
            if not isinstance(txt, str):
                txt = txt.decode(errors='surrogateescape')
            
            # Pretokenization is minimal:
            # Simply split txt into "words" on space, non-letters, or excessive length
            txt_words = re.findall(pretokenize_regex, txt)

            for w in range(len(txt_words)):
                # Now encode each txt_word back into bytes
                bs = self.tk._encode(txt_words[w])
                word = [self.tk.bytes2tid[bytes((b,))] for b in bs]
                self.words.append(word)

                # and index its bigrams
                for p in range(len(word)-1):
                    bigram = (word[p], word[p+1])
                    self.bigram_nfreq[bigram] = self.bigram_nfreq.get(bigram, 0) - 1
                    self.bigram_word_count[bigram][word_idx] = self.bigram_word_count[bigram].get(word_idx, 0) + 1

                word_idx += 1
        
if __name__ == '__main__':
    BpeTokenLearner.demo()

