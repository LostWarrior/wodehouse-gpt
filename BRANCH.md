# Branch: v2-bpe-tokenizer

Snapshot of the **pure BPE base model** stage of the Wodehouse-GPT project. Frozen at commit `82ce12a` ("increase context window to 512").

## What's here

- BPE tokenizer (vocab 4000, 3.7x compression vs raw bytes)
- 21M parameter decoder-only transformer
- `max_seq_len = 512` (tokens, ~2000 characters with BPE)
- Dropout 0.1 (added because v5-no-dropout overfit hard)
- Trained on 30 Wodehouse books as flat prose
- Dialogue extractor with pronoun resolution (for future work, not used in training here)

## What's NOT here (lives on later branches)

- No reward model / quality checker -> `v3-reward-model`
- No dialogue fine-tuning -> `main`
- No `--character` chatbot -> `main`
- No best-of-N sampling -> `main`

## Why this branch exists

Each BPE-era milestone gets preserved so learners can walk the evolution end-to-end. See `v1-character-tokenizer` for the pre-BPE era.

## Decisions captured here

1. **BPE over character-level**: bigger lesson than tripling parameters. Character models (`character-tokenizer` branch) topped out at val 1.25 with output that looked English but meant nothing. BPE gives the model real words to predict, unlocking actual vocabulary.

2. **Dropout matters**: v5-no-dropout climbed to val 5.30 (overfit). Adding dropout 0.1 dropped val to 4.39. A 3-line change beat a 3x parameter increase.

3. **Context window bump to 512**: was 256, doubled for more long-range coherence. Cost: quadratic in attention. Benefit: tangible on dialogue-heavy passages.

## Next branch

`v3-reward-model` adds `checker.py` - a 4-layer text quality scorer - as a standalone tool. Not wired into generation yet.
