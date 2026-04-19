# Branch: v3-reward-model

Snapshot of the project at the point where a **reward model** first exists as a tool, but is not yet wired into generation. Frozen at commit `57a9891` ("add a reward model").

## What's new here (vs v2-bpe-tokenizer)

- `checker.py` - 4-layer heuristic text quality scorer:
  1. **Word check**: is every word real?
  2. **N-gram check**: is the word order author-like?
  3. **Repetition check**: is the output repeating itself?
  4. **Distinctive vocab**: does it use author-specific words?
- Standalone usage: `python3 checker.py "text to score"`
- Pluggable: swap the training corpus to score against a different author.

## What's still missing (lives on `main`)

- Reward model not yet called during generation (no best-of-N reranking)
- No dialogue fine-tuning
- No `--character` chatbot flag

## Why this branch exists

The reward model was worth its own snapshot because it represents a conceptual leap: **a language model alone cannot tell you whether its own output is good.** You need an external scorer. This branch shows the scorer existing before we wired it into the inference loop.

## Decisions captured here

1. **Heuristic reward model over neural one**: we do not have human preference data, and we have one person (the user) writing code. A 4-layer rule-based scorer gives us a directionally useful signal in one file.

2. **Pluggable design**: `Checker("data.txt")` lets us re-target the scorer to any corpus. Same code could score Shakespeare output by loading Shakespeare text.

3. **Weighted combination**: overall = 0.25*word + 0.35*ngram + 0.25*repetition + 0.15*distinctive. N-grams get the highest weight because they capture word-order quality, which is the most Wodehouse-flavoured signal.

## Next branch

`main` fine-tunes the base model on extracted dialogue and wires the reward model in via best-of-N sampling in `generate.py`. That is where Goodhart's Law bites - the scorer rewards single-word replies with 100% scores because every layer trivially passes on 1 token. See `local-md-files/transformers/28-reward-model-and-best-of-n.md` for how we caught and patched that.
