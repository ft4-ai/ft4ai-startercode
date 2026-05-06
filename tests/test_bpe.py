import pytest
import unicodedata

from hardcoregenai.pipeline.bpe_tokenizer import BpeTokenizer
from hardcoregenai.pipeline.bpe_token_learner import BpeTokenLearner


TEXTS = []
TEXTS.append("How much wood would a woodchuck chuck if a woodchuck could chuck wood?\n"
              "He would chuck, he would, as much as he could, and chuck as much wood\n"
              "As a woodchuck would if a woodchuck could chuck wood!\n")
TEXTS.append("You know New York you need New York you know you need unique New York.")
TEXTS.append("A AA AAA AAAA B BB BBB BBBB ABAAABBBAB AB ABAB ABA ABAA BAAA ABBAB ABABAB")
TEXTS.append("")
TEXTS.append("Ⓗello world Ⓗello .Ⓗello, how are you?") # Wide denormalized chars, will be normalized away
TEXTS.append("Раз, два, три, четыре, пять, Вышел зайчик погулять.\n"
              "Вдруг охотник выбегает, Прямо в зайчика стреляет.\n"
              "Пиф - паф! ой-ой-ой! Умирает зайчик мой. Привезли домой, Оказался он живой!\n")
TEXTS.append("Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy wasn't fuzzy, was he?")
TEXTS.append("\x00")
TEXTS.append("शशाङ्कः शुभ्रः सुन्दरः।\nरात्रौ नभसि भासते।\nबालक हसति तस्य।\nरम्यत्वात् सर्वदा॥")
TEXTS.append("\t")
TEXTS.append(b"Betty Botter bought some butter\r\n"
              b"But she said the butter's bitter\r\n"
              b"If I put it in my batter, it will make my batter bitter\r\r\n"
              b"But a bit of \tbetter butter \rwill make my batter better\n"
              b"So 'twas better Betty Botter bought a bit of better butter\r\n")
TEXTS.append(b'\xff\xc3\x28\x80\xff\xc3\x41\xff\xc3\x42\xff\xc3\x61') # Does not decode to Unicode

# A quick simple test for initial debugging
def test_smoke():
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens(['fuzzy wuzzy wuzz a bear.'])
   assert tk.detokenize(tk.tokenize('wuzzy wuzz')) == b'wuzzy wuzz'

def test_basics():
    tk = BpeTokenizer()
    btl = BpeTokenLearner(tk)
    btl.learn_tokens(TEXTS, max_tid=400)
    for i in range(400):
       assert i in tk.tid2bytes, "Tokens ids must be contiguous"
    for txt in TEXTS:
      if isinstance(txt, str):
         txt = txt.encode(errors='replace')
      assert tk.detokenize(tk.tokenize(txt)) == txt
    assert_compression(tk)

# CSV I/O: write and read back
def test_csv_round_trip(tmp_path):
    tk = BpeTokenizer()
    tk = BpeTokenizer()
    btl = BpeTokenLearner(tk)
    btl.learn_tokens(TEXTS, max_tid=100003)
    path = tmp_path / "tokens.csv"
    tk.save_csv(str(path))

    tk2 = BpeTokenizer.from_csv(str(path))
    assert tk2.bytes2tid == tk.bytes2tid

    for txt in TEXTS:
      if isinstance(txt, str):
         txt = txt.encode(errors='replace')
      assert tk.detokenize(tk.tokenize(txt)) == txt
      if len(txt) > 64:
         # A rough way to test the algorithm's correctness: 
         # It should compress large TEXTS over 2.1
         assert len(tk.tokenize(txt)) < len(txt) / 2.1
      
    # overwrite should fail
    with pytest.raises(FileExistsError):
        tk.save_csv(str(path))

def test_reserved_tokens():
   tk = BpeTokenizer(n_reserved_tokens=8)
   btl = BpeTokenLearner(tk)
   btl.learn_tokens(TEXTS, max_tid=400)
   tokens = [0, 8, 1, 9, 2, 10]
   expected = b''
   expected += b'<SPECIALTOKEN:000000>'
   expected += tk.detokenize([8])
   expected += b'<SPECIALTOKEN:000001>'
   expected += tk.detokenize([9])
   expected += b'<SPECIALTOKEN:000002>'
   expected += tk.detokenize([10])
   assert tk.detokenize(tokens) == expected 

def assert_roundtrip(tk):
   for txt in TEXTS:
      tokens = tk.tokenize(txt)
      roundtrip = tk.detokenize(tokens).decode(errors='surrogateescape')
      if not isinstance(txt, str):
         txt = txt.decode(errors='surrogateescape')
      assert roundtrip == unicodedata.normalize('NFKC', txt)

# This is the best way to catch bugs in the algorithm that don't cause failures - 
# the compression ratio will drop.
def assert_compression(tk, expected_min_compression_ratio=2.1):
   for txt in TEXTS:
      if len(txt) > 80:
          tokens = tk.tokenize(txt)
          if isinstance(txt, str):
             bs = txt.encode(errors='replace')
          else:
            bs = txt
          assert len(tokens) < len(bs) / expected_min_compression_ratio

@pytest.mark.parametrize("max_tid", [0, 1, 2, 3, 4, 255+0, 255+1, 255+2, 255+3, 255+4, 500,501, 4000, 40001, 100003])
def test_max_tid(max_tid):
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens(TEXTS, max_tid=max_tid)
   assert tk.__repr__() != '' # Smoke test
   assert_roundtrip(tk)
   if max_tid > 400:
      assert_compression(tk)

def test_correctness_1():
   txt = 'peter piper picked a pack of pickled peppers'
   expected_merges = 23
   expected_tokens = 8

   tk = BpeTokenizer()
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens([txt])
   assert tk.last_tid == 255 + expected_merges
   assert len(tk.tokenize(txt)) == expected_tokens

def test_correctness_2():
   txt = "fuzzy wuzzy was a bear. fuzzy wuzzy had no hair. fuzzy wuzzy wasn't very fuzzy, was he?"
   expected_merges = 27
   expected_tokens = 23

   tk = BpeTokenizer()
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens([txt])
   assert tk.last_tid == 255 + expected_merges
   assert len(tk.tokenize(txt)) == expected_tokens

def test_correctness_multiple():
   txt1 = 'peter piper picked a pack of pickled peppers'
   txt2 = "fuzzy wuzzy was a bear. fuzzy wuzzy had no hair. fuzzy wuzzy wasn't very fuzzy, was he?"
   txts = [txt1, txt2]
   expected_merges = 48
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens(txts)
   assert tk.last_tid == 255 + expected_merges
   assert len(tk.tokenize(txt1)) == 8
   assert len(tk.tokenize(txt2)) == 23

def test_multiple_texts():
   # Two texts each with the bigram ("b","c"):
   texts = ["bc bc", "bc"]
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens(texts, max_tid=260)

# Some of these bug actually happened

def test_bcbcbc():
   # A single word with the pattern BC BC BC  —  three repeats
   # It must do at least 3 merges (so that it visits the same old‐bigram twice)
   txt = 'bcbcbc'
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens([txt])
   assert tk.detokenize(tk.tokenize(txt)) == txt.encode()

def test_aaaaaaaa():
   txt = 'AAAAAAAA'
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens([txt])
   assert tk.detokenize(tk.tokenize(txt)) == txt.encode()

def test_abcbc():
   txt = 'abcbc'
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens([txt])
   assert tk.detokenize(tk.tokenize(txt)) == txt.encode()

# A few of the tests below may be superficial or even meaningless

# Initialization tests
def test_initial_reserved_tokens_and_base_vocab():
    # 0 reserved => IDs 0-255 for byte tokens
    tk0 = BpeTokenizer(n_reserved_tokens=0)
    assert tk0.last_tid == 255
    assert len(tk0.bytes2tid) == 256

    # 10 reserved => byte tokens start at ID=10
    tk10 = BpeTokenizer(n_reserved_tokens=10)
    assert tk10.last_tid == 10 + 255
    assert len(tk10.bytes2tid) == 256

# Round-trip tokenization/detokenization
@pytest.mark.parametrize("text", ["hello world", "café", "", "@", "\xff\xfe", "hi there!, hi, how are you? you are there!\n", "we need to think about\xfe how we need to think"])
def test_tokenize_detokenize_round_trip(text):
    tk = BpeTokenizer()
    btl = BpeTokenLearner(tk)
    btl.learn_tokens([text], max_tid=255+len(text)//3)
    ids = tk.tokenize(text)
    s = tk.detokenize(ids)
    assert s == text.encode('utf-8', errors='replace')

def test_not_enough_text_for_tokens():
   tk = BpeTokenizer()
   btl = BpeTokenLearner(tk)
   btl.learn_tokens(TEXTS, max_tid=2**16 - 1)
   assert tk.detokenize(tk.tokenize(b"its")) == b"its"
   for txt in TEXTS:
     if isinstance(txt, str):
       txt = txt.encode(errors='replace')
     assert tk.detokenize(tk.tokenize(txt)) == txt
     if len(txt) > 64:
       # A rough way to test the algorithm's correctness: 
       # It should compress large TEXTS by over 2.1
       assert len(tk.tokenize(txt)) < len(txt) / 2.1

def test_tokenize_empty_bytes_and_detokenize_empty():
    tk = BpeTokenizer()
    btl = BpeTokenLearner(tk)
    btl.learn_tokens([b"abc", b"123", b"ABC"], max_tid=256)
    assert tk.tokenize(b"") == []
    assert tk.detokenize([]) == b""
    assert tk.detokenize(tk.tokenize(b"its")) == b"its"

def test_learn_empty_string():
    tk = BpeTokenizer()
    btl = BpeTokenLearner(tk)
    btl.learn_tokens([b""])
    assert tk.tokenize(b"") == []
    assert tk.detokenize([]) == b""
    assert tk.detokenize(tk.tokenize(b"its")) == b"its"

def test_learn_empty_list():
    tk = BpeTokenizer()
    btl = BpeTokenLearner(tk)
    btl.learn_tokens([])
    assert tk.tokenize(b"") == []
    assert tk.detokenize([]) == b""
    assert tk.detokenize(tk.tokenize(b"its")) == b"its"
    
# End-of-input consumption
def test_end_of_string_last_byte():
    tk = BpeTokenizer()
    data = b"Z"
    ids = tk.tokenize(data)
    assert len(ids) == 1
    assert tk.detokenize(ids) == data



