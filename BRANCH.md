# Branch: v5-sampling-techniques

Snapshot of the project after the **inference-time sampling** chapter. Frozen at commit `2fb424f` (add separate sampling settings for best-of-N candidate generation).

## What's new here (vs v4-dialogue-chatbot)

Everything v4 had, plus production-grade sampling controls in `generate.py`:

- **Top-k** (`--top-k N`): hard cap on the candidate pool per step
- **Top-p nucleus sampling** (`--top-p 0.9`): adaptive cutoff matching model confidence
- **Repetition penalty** (`--rep-penalty 1.15`): HuggingFace-style multiplicative downweighting
- **Frequency penalty** (`--freq-penalty`): OpenAI-style additive, scales with count
- **Presence penalty** (`--presence-penalty`): OpenAI-style additive, binary hit
- **Separate settings for best-of-N** (`--best-of-temp`, `--best-of-top-p`): loose sampling during candidate generation for variance, strict reranking to return the winner

## Why this branch exists

Captures "everything we could do with inference alone" before going back to retraining (next branch will be combined-corpus BPE). Represents ~30 minutes of code that gave as much visible quality improvement as a 27-minute fine-tune.

## Decisions captured here

1. **Top-p by default, top-k optional**. They're redundant; top-p is adaptive. Top-k is there for belt-and-suspenders cases but defaults to off.

2. **Both rep-penalty styles available**. Multiplicative (HuggingFace) and additive (OpenAI) ship side by side. Users pick one - or combine them knowingly. Educational value of seeing both.

3. **Separate sampling for best-of-N candidates**. This came from a real finding during testing: `--best-of 5 --top-p 0.9` produced worse results than `--best-of 5 --top-p 1.0`. The reason: reranking needs *variance* between candidates. Top-p destroys variance. Solution: two-knob schedule.

4. **No hard validation between rep styles or conflicting flags**. We trust the operator to know what they're doing. Matches how production inference servers behave.

## Key findings (see `local-md-files/transformers/29-sampling-top-k-top-p-repetition.md`)

- **Sampling + best-of-N can hurt each other**: naive stacking reduces variance that reranking needs. Fix is the separate-settings approach.
- **The semantic ceiling sits above sampling**: no combination of filters makes the model relevant to the prompt. 96% checker score can still be off-topic.
- **Goodhart drift is subtle**: checker sometimes prefers bland outputs over distinctive-but-risky ones.
- **Inference-time engineering is higher ROI than retraining**: fast, reversible, immediate.

## What's still missing (next branches)

- BPE was trained on raw prose only. `<jeeves>` gets chopped into 3-4 subtokens. Retraining BPE on prose + dialogue together would make speaker tags single tokens and probably lift val 0.2-0.4 further. That's the next chapter.
- No instruction-following training. No SFT. Outputs continue text but don't respond to user intent.
- No neural reward model. Heuristic `checker.py` is the closest we have.
