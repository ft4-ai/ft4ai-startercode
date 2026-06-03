"""Tests for the measure_entropy engine (Option C: score the next token).

These drive the *project* tokenizer (Ft4Tokenizer), not a mock: streams are
built with Ft4Tokenizer.tokenize(...), and assertions are derived from the
tokenizer's actual output rather than hardcoded ids — so they hold whatever
merges the loaded tokens.v01.csv happens to have.

Marked @pytest.mark.lab("unit1.lab3"); run with `-m lab`, skip with `-m "not lab"`.
"""
import pytest

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer as TK
from ft4.demo.measure_entropy.engine import (
    LEAD_IN_CONTENT_TOKENS, MAX_ROUNDS_PER_STORY,
    Session, StoryQueue, correct_guess, is_content, starts_word)

pytestmark = pytest.mark.lab("unit1.lab3")

BOS, EOS = TK.RES_START, TK.RES_STOP


def stream(text):
    """A full <BOS> … <EOS> token stream for a story string."""
    return [BOS] + TK.tokenize(text) + [EOS]


def sess(*texts):
    return Session(queue=StoryQueue.from_token_lists(TK, [stream(t) for t in texts]))


def first_token(word):
    """The first token of ' <word>' (its bytes are a prefix of the word)."""
    return TK.tokenize(" " + word)[0]


def multitoken_word():
    """A word the tokenizer splits as [word-initial content, continuation, …].
    Used to exercise mid-word behavior; skips if the vocab has none handy."""
    for w in ("garden", "happily", "beautiful", "wonderful", "mysterious",
              "information", "adventure", "butterfly", "strawberry", "elephant"):
        toks = TK.tokenize(" " + w)
        if (len(toks) >= 2
                and starts_word(TK, toks[0]) and is_content(TK, toks[0])
                and not starts_word(TK, toks[1]) and is_content(TK, toks[1])):
            return w
    pytest.skip("no suitable multi-token word in this tokenizer")


# --------------------------------------------------------------------------- #
# correct_guess(): the whole matcher, one token.
# --------------------------------------------------------------------------- #

def test_typing_the_word_credits_its_first_token():
    assert correct_guess(TK, first_token("garden"), "garden") is True


def test_wrong_word_is_a_miss():
    assert correct_guess(TK, first_token("garden"), "elephant") is False


def test_casefold_and_surrounding_whitespace():
    t = first_token("the")
    assert correct_guess(TK, t, "THE") is True
    assert correct_guess(TK, t, "   the   ") is True


def test_empty_guess_vs_content_is_a_miss():
    assert correct_guess(TK, first_token("the"), "") is False


def test_empty_guess_predicts_eos():
    assert correct_guess(TK, EOS, "") is True
    assert correct_guess(TK, EOS, "more") is False


# --------------------------------------------------------------------------- #
# Session: one story.
# --------------------------------------------------------------------------- #

STORY = " The cat sat on the mat and they played in the garden very happy"


def test_first_target_is_a_fresh_word():
    s = sess(STORY)
    assert s.view().continuing_word is False
    assert starts_word(TK, s.walker.next_token())


def test_hit_and_miss_update_the_per_token_tally():
    s = sess(STORY)
    # type the current word correctly, then wrong, then correctly
    def cur():
        return TK.detokenize([s.walker.next_token()]).decode("utf-8", "surrogateescape").strip()
    s.commit(cur())                      # hit
    assert s.last_correct is True
    s.commit("definitelynotthisword")    # miss
    assert s.last_correct is False
    s.commit(cur())                      # hit
    assert (s.tally.correct, s.tally.trials) == (2, 3)
    assert abs(s.view().accuracy - 2 / 3) < 1e-9


def test_feedback_correct_shows_the_token():
    s = sess(STORY)
    word = TK.detokenize([s.walker.next_token()]).decode("utf-8", "surrogateescape").strip()
    s.commit(word)
    assert s.last_correct is True and s.last_guess == word and s.last_truth == word


def test_feedback_incorrect_shows_first_word_and_truth():
    s = sess(STORY)
    truth = TK.detokenize([s.walker.next_token()]).decode("utf-8", "surrogateescape").strip()
    s.commit("Mia and her friends")      # multi-word wrong guess
    assert s.last_correct is False
    assert s.last_guess == "Mia"          # only what we scored against (#7)
    assert s.last_truth == truth


def test_empty_enter_shows_eos_label():
    s = sess(STORY)
    s.commit("")                          # predicted <EOS>, but a word follows
    assert s.last_correct is False and s.last_guess == "<EOS>"


def test_round_cap_per_story():
    long_story = " The cat sat on the mat and " + " ".join(["dog"] * 20)
    s = sess(long_story)
    for _ in range(MAX_ROUNDS_PER_STORY):
        assert s.state == "prompting"
        s.commit("x")
    assert s.state == "story_done" and s.round_num == MAX_ROUNDS_PER_STORY


# --------------------------------------------------------------------------- #
# Session: across stories and the <EOS> ending.
# --------------------------------------------------------------------------- #

def test_advance_and_finish():
    s = sess(STORY, STORY)
    assert s.stories_seen == 1
    while s.state == "prompting":          # walk story 1 to <EOS>
        s.commit("")
    assert s.state == "story_done"
    s.advance_story()
    assert s.state == "prompting" and s.stories_seen == 2
    while s.state == "prompting":          # walk story 2
        s.commit("")
    s.advance_story()
    assert s.finished


def test_eos_empty_enter_is_correct():
    s = sess(" The cat sat on the mat and dog")
    while not s.walker.at_end() and s.walker.next_token() != EOS:
        s.commit("x")
        if s.state != "prompting":
            break
    # if we stopped exactly on <EOS>, an empty Enter is a hit that ends the story
    if s.state == "prompting" and s.walker.next_token() == EOS:
        before = s.tally.correct
        s.commit("")
        assert s.last_correct is True and s.tally.correct == before + 1
        assert s.state == "story_done"


# --------------------------------------------------------------------------- #
# Mid-word continuation: the peek-ahead flag tracks the tokenizer.
# --------------------------------------------------------------------------- #

def test_continuing_word_tracks_mid_word():
    w = multitoken_word()
    s = sess(" The cat sat on the mat " + w + " ran home and the dog")
    flags = []
    while s.state == "prompting":
        flags.append(s.view().continuing_word)
        cur = TK.detokenize([s.walker.next_token()]).decode("utf-8", "surrogateescape").strip()
        s.commit(cur)                      # always type the true token, so we keep walking
    assert flags[0] is False                # a story never opens mid-word
    assert True in flags                    # the multi-token word produced a continuation


# --------------------------------------------------------------------------- #
# Queue: short stories are skipped.
# --------------------------------------------------------------------------- #

def test_queue_skips_too_short_stories():
    tiny = " cat"                          # fewer than LEAD_IN + 1 content tokens
    assert len(TK.tokenize(tiny)) < LEAD_IN_CONTENT_TOKENS + 1
    s = sess(tiny, STORY)
    assert s.stories_seen == 1             # the tiny one was skipped, STORY loaded
