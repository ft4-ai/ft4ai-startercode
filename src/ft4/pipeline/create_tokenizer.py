from datasets import load_dataset

from ft4.pipeline.bpe_token_learner import BpeTokenLearner

TOKENS_CSV_FILE = None # Set this to the file of your choice
N_STORIES_TO_TRAIN = 100_000 # BPE is slow and RAM hungry, so be careful how high you set this
N_TOKENS = 12_288


# --- TinyStories ---
print('Fetching TinyStories')
tiny_stories = load_dataset("roneneldan/TinyStories")
print(tiny_stories)  # e.g. DatasetDict({'train': Dataset(...)})
i = 0
for r in tiny_stories["train"].select(range(4,10)): #type:ignore
    i += 1
    print(f"\n# Story {i}\n")
    story = r["text"] #type:ignore
    print(story)


print('Tokenizing...')
tokenizer = BpeTokenLearner(n_reserved_tokens=8)
tokenizer.learn_tokens(list(r["text"] for r in tiny_stories["train"].select(range(N_STORIES_TO_TRAIN))), max_id=N_TOKENS-1) #type:ignore
if TOKENS_CSV_FILE:
    tokenizer.save_csv(TOKENS_CSV_FILE)
