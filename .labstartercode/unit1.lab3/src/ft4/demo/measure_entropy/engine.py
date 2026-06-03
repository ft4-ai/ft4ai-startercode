# Run the lab:  python -m ft4.demo.measure_entropy
"""
Upper bounds corpus entropy based on human predictions.

The unit is the TOKEN, and we score one token per round. A word the tokenizer 
split into several tokens is guessed a piece at a time — which is precisely what 
a language model does, and the player is standing in for one.

The engine is a compromise between accuracy and usability:

  - We compare bytes, not token boundaries. Typing more than the next token (a
    whole word when the token is just its start) still counts: the token is a
    prefix of what you typed. Fairness, not leniency — you aren't punished for
    not knowing where the tokenizer split.
  - We ignore case. "The" and "the" are the same guess. This one does flatter.
  - Whitespace is structural: it isn't something you predict, so stray or
    doubled spaces never cost a token.
  - Punctuation is free — revealed automatically, never scored — so we're really
    measuring the entropy of the content tokens, even though V stays the full
    vocabulary. A small, deliberately conservative mismatch.
  - One token per round. When a word spans several tokens you guess them in turn,
    with the already-revealed prefix on screen as context.
  - We peek one token ahead only to lay out the line: a leading space is shown
    before you guess (so you can see a new word is starting). That leaks the
    word boundary — a deliberate, minor cheat in the player's favor.
  - Every story gets a running start; we never ask for a story's first word cold.
  - "The story ends here" is a prediction too: empty Enter means "next is <EOS>".
  - We play on whole stories, not the packed, shuffled stream the model trains
    on. Cleaner for a human, one step removed from the model's real diet.

Round flow (one TOKEN per round):

    prepare: reveal any leading punctuation (unscored); the target is the next
             token (a fresh word, or the continuation of one) or <EOS>
                                  |
              show the story, cursor at the prediction point; if the target
              starts a new word, show its leading space; player types, Enter
                                  |
                                  v
              correct_guess(): does the typed text begin with the token's bytes?
                                  |
                    +-------------+-------------+
                    v                           v
              yes -> correct               no -> miss (the true token
              (a hit, one trial)           is revealed in its place)
                    +-------------+-------------+
                                  |
              reveal that one token . tally it . advance one token
                                  |
        round < 8 and story not over -> next token
        8 rounds / <EOS> / end of story -> pause on a recap, then next story
"""
import re
from dataclasses import dataclass
from typing import Callable, Literal

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from ft4.pipeline.stories_data_module import StoriesDataModule

LEAD_IN_CONTENT_TOKENS = 6      # context revealed free at the start of each story
MAX_ROUNDS_PER_STORY = 8        # a ceiling, not a target
NOISY_TRIALS_BELOW = 50         # below this, the summary warns that accuracy is rough
STORY_WINDOW_CHARS = 320        # how much trailing story we keep on screen
PROMPT_TAIL_CHARS = 56          # how much of the tail the continuation line repeats
DATA_SIZE = "small"
EOS_LABEL = "<EOS>"

# How a revealed token is drawn: a correct guess, the true token you missed,
# an unscored token (punctuation), or unscored context (lead-in / older text).
Status = Literal["correct", "miss", "skip", "context"]
State = Literal["prompting", "story_done", "finished"]

_WS_RUN = re.compile(r"\s+")

# --------------------------------------------------------------------------- #
# Token helpers. Each takes the tokenizer `tok` explicitly so the matcher is a
# pure function of (tokenizer, token, typed text) and is trivial to test.
# --------------------------------------------------------------------------- #

def is_special(tok, token_id: int) -> bool:
    """True for the reserved control ids (PAD/UNK/BOS/EOS)."""
    return token_id in (tok.RES_PAD, tok.RES_UNK, tok.RES_START, tok.RES_STOP)


def _piece(tok, token_id: int) -> str:
    return tok.detokenize([token_id]).decode("utf-8", "surrogateescape")


def is_content(tok, token_id: int) -> bool:
    """A token is content if, after stripping one leading space, any letter or
    digit remains. Punctuation/whitespace and control tokens are not content."""
    if is_special(tok, token_id):
        return False
    return any(ch.isalnum() for ch in _piece(tok, token_id))


def starts_word(tok, token_id: int) -> bool:
    """A content token begins a new word if its bytes start with a space."""
    return _piece(tok, token_id).startswith(" ")


def tokens2str(tok, token_ids) -> str:
    """Render tokens as a display string, naming the BOS/EOS markers."""
    names = {tok.RES_START: "<BOS>", tok.RES_STOP: EOS_LABEL}
    return "".join(names.get(t) or tok.detokenize([t]).decode("utf-8", "replace")
                   for t in token_ids)


def _normalize(s: str) -> str:
    """Comparison-only normalization: collapse whitespace runs to one space,
    strip the ends, and casefold. Display and reveal always use the raw bytes."""
    return _WS_RUN.sub(" ", s).strip().casefold()


def correct_guess(tok, token: int, user_input: str) -> bool:
    """Did the player's guess predict this one token? The token is correct iff
    the typed text begins with its bytes (so typing a whole word credits its
    first token); empty Enter predicts <EOS>. The whole matcher."""
    typed = _normalize(user_input)
    if token == tok.RES_STOP:
        return typed == ""
    return bool(typed) and typed.startswith(_normalize(_piece(tok, token)))


@dataclass
class Tally:
    """Tallies correct guesses and accuracy"""
    correct: int = 0
    trials: int = 0

    def record(self, hit: bool) -> None:
        self.trials += 1
        self.correct += int(hit)

    @property
    def accuracy(self) -> float:
        return self.correct / self.trials if self.trials else 0.0


class StoryWalker:
    """Per-story walker: the token stream, a cursor (= prediction point), and a
    display status for every revealed token. The cursor rests on a content token
    or <EOS>; it may sit mid-word (on a continuation token)."""
    def __init__(self, tok, tokens: list[int]):
        self.tok = tok
        self.tokens = tokens
        self.status: list[Status | None] = [None] * len(tokens)
        self.cursor = 0
        self._reveal_lead_in()

    def reveal(self, status: Status) -> None:
        """Mark the token at the cursor with `status` and step past it."""
        self.status[self.cursor] = status
        self.cursor += 1

    def _reveal_lead_in(self) -> None:
        """Reveal <BOS> plus the first LEAD_IN_CONTENT_TOKENS content tokens as
        unscored context, then finish the current word so the FIRST guess of a
        story is always a fresh word, not a mid-word continuation"""
        seen = 0
        while self.cursor < len(self.tokens) and seen < LEAD_IN_CONTENT_TOKENS:
            t = self.tokens[self.cursor]
            if t == self.tok.RES_STOP:
                break
            if is_content(self.tok, t):
                seen += 1
            self.reveal("context")
        while self.cursor < len(self.tokens):
            t = self.tokens[self.cursor]
            if (t == self.tok.RES_STOP or not is_content(self.tok, t)
                    or starts_word(self.tok, t)):
                break
            self.reveal("context")       # a sub-word continuation: finish the word

    def prepare_round(self) -> None:
        """Auto-advance over leading punctuation/control tokens (revealed
        unscored) so the target is always a content token or <EOS>."""
        while self.cursor < len(self.tokens):
            t = self.tokens[self.cursor]
            if t == self.tok.RES_STOP or is_content(self.tok, t):
                break
            self.reveal("context" if is_special(self.tok, t) else "skip")

    def next_token(self) -> int:
        return self.tokens[self.cursor]

    def at_end(self) -> bool:
        return self.cursor >= len(self.tokens)

    def revealed_tokens(self) -> list[tuple[int, Status]]:
        return [(self.tokens[i], self.status[i] or "context")
                for i in range(self.cursor)]


class StoryQueue:
    def __init__(self, tok, sources, tokenize: Callable[[object], list[int]]):
        self.tok = tok
        self._sources = sources
        self._tokenize = tokenize
        self._ptr = 0

    @classmethod
    def from_dataset(cls, seed: int = 0) -> "StoryQueue":
        sdm = StoriesDataModule(seq_len=256, batch_size=1, data_size=DATA_SIZE)
        sdm.prepare_data()
        sdm.setup("fit")
        sources = list(sdm.story_datasets["train"].shuffle(seed=seed)["story"])
        wrap = lambda s: [Ft4Tokenizer.RES_START] + Ft4Tokenizer.tokenize(s) + [Ft4Tokenizer.RES_STOP]
        return cls(Ft4Tokenizer, sources, wrap)

    @classmethod
    def from_token_lists(cls, tok, lists) -> "StoryQueue":
        # Test seam: caller supplies ready-made token streams (each already
        # wrapped with <BOS>…<EOS>); no dataset, no network.
        return cls(tok, list(lists), tokenize=lambda x: x) #type:ignore

    def next_story(self) -> "StoryWalker | None":
        while self._ptr < len(self._sources):
            tokens = self._tokenize(self._sources[self._ptr])
            self._ptr += 1
            if sum(1 for t in tokens if is_content(self.tok, t)) < LEAD_IN_CONTENT_TOKENS + 1:
                continue  # too short for a lead-in plus a target
            walker = StoryWalker(self.tok, tokens)
            walker.prepare_round()
            return walker
        return None


@dataclass
class ViewModel:
    state: State
    # stats
    correct: int
    trials: int
    accuracy: float
    vocab_size: int
    story_index: int                            # 1-based count of stories seen
    round_num: int                              # 1-based round within this story
    # story render
    story_segments: list[tuple[str, Status]]    # full windowed story, grouped by status
    prompt_tail: str                            # the recent tail the prompt line repeats
    tail_truncated: bool                        # is there revealed text before the tail?
    continuing_word: bool                       # is the next token a mid-word continuation?
    # feedback for the round that just committed (None on a story's first round)
    last_guess: str | None                      # the token guessed (or the wrong guess / <EOS>)
    last_truth: str | None                      # the true token
    last_correct: bool | None
    # per-story (for the end-of-story recap)
    story_correct: int
    story_trials: int
    show_onboarding_cue: bool                   # the one-time "piece at a time" hint


@dataclass
class SummaryModel:
    stories_seen: int
    correct: int
    trials: int
    accuracy: float
    vocab_size: int


class Session:
    """Session: ties the queue, walker, and tally together as a small state machine."""
    def __init__(self, seed: int = 0, queue: "StoryQueue | None" = None):
        self.queue = queue if queue is not None else StoryQueue.from_dataset(seed)
        self.tok = self.queue.tok
        self.tally = Tally()
        self.story_tally = Tally()
        self.stories_seen = 0
        self.round_num = 0
        self.last_guess: str | None = None
        self.last_truth: str | None = None
        self.last_correct: bool | None = None
        self.walker: StoryWalker = None
        self.state: State = "prompting"
        self._load_next_story()

    def _load_next_story(self) -> None:
        walker = self.queue.next_story()
        if walker is None:
            self.state = "finished"
            self.walker = None
            return
        self.walker = walker
        self.stories_seen += 1
        self.round_num = 0
        self.story_tally = Tally()
        self.last_guess = self.last_truth = self.last_correct = None
        self.state = "prompting"

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size()

    @property
    def finished(self) -> bool:
        return self.state == "finished"

    def _next_is_continuation(self) -> bool:
        # Peek one token ahead (after prepare_round): a mid-word continuation is a
        # content token with no leading space.
        if self.walker is None or self.walker.at_end():
            return False
        t = self.walker.next_token()
        return t != self.tok.RES_STOP and is_content(self.tok, t) and not starts_word(self.tok, t)

    def commit(self, user_input: str) -> None:
        if self.state != "prompting":
            return
        token = self.walker.next_token()
        hit = correct_guess(self.tok, token, user_input)
        self.walker.reveal("correct" if hit else "miss")
        self.tally.record(hit)
        self.story_tally.record(hit)
        self.round_num += 1
        # Feedback shows exactly the one token we scored (#6/#7): on a hit, the
        # token itself; on a miss, what the player typed plus the true token.
        truth = tokens2str(self.tok, [token]).strip()
        self.last_truth = truth
        self.last_correct = hit
        self.last_guess = (truth if hit else
                           EOS_LABEL if not user_input.strip() else user_input.split()[0])

        if (token == self.tok.RES_STOP or self.walker.at_end()
                or self.round_num >= MAX_ROUNDS_PER_STORY):
            self.state = "story_done"
        else:
            self.walker.prepare_round()
            if self.walker.at_end():            # ran off the end skipping punctuation
                self.state = "story_done"

    def advance_story(self) -> None:
        # Called by the UI after showing the end-of-story recap.
        if self.state == "story_done":
            self._load_next_story()

    def view(self) -> ViewModel:
        segments, tail, truncated = self._render_story()
        return ViewModel(
            state=self.state,
            correct=self.tally.correct,
            trials=self.tally.trials,
            accuracy=self.tally.accuracy,
            vocab_size=self.vocab_size,
            story_index=self.stories_seen,
            round_num=self.round_num + (0 if self.state == "story_done" else 1),
            story_segments=segments,
            prompt_tail=tail,
            tail_truncated=truncated,
            continuing_word=(self.state == "prompting" and self._next_is_continuation()),
            last_guess=self.last_guess,
            last_truth=self.last_truth,
            last_correct=self.last_correct,
            story_correct=self.story_tally.correct,
            story_trials=self.story_tally.trials,
            show_onboarding_cue=(self.tally.trials == 0),
        )

    def _render_story(self) -> tuple[list[tuple[str, Status]], str, bool]:
        """The full windowed story-so-far as colored segments, plus the recent
        tail the continuation line repeats. The story block is rendered whole
        (Rich wraps it); the tail is a separate repeat, so there's no mid-word
        split between them."""
        if self.walker is None:
            return [], "", False
        chars: list[tuple[str, Status]] = []
        for tid, st in self.walker.revealed_tokens():
            for ch in tokens2str(self.tok, [tid]):
                chars.append((ch, st))
        chars = chars[-STORY_WINDOW_CHARS:]
        segments: list[tuple[str, Status]] = []
        for ch, st in chars:
            if segments and segments[-1][1] == st:
                segments[-1] = (segments[-1][0] + ch, st)
            else:
                segments.append((ch, st))
        tail_chars = chars[-PROMPT_TAIL_CHARS:]
        tail = "".join(ch for ch, _ in tail_chars)
        return segments, tail, len(chars) > len(tail_chars)

    def summary(self) -> SummaryModel:
        return SummaryModel(self.stories_seen, self.tally.correct,
                            self.tally.trials, self.tally.accuracy, self.vocab_size)
