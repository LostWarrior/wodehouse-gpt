# Branch: v4-dialogue-chatbot

Snapshot of the project at the end of the **dialogue fine-tune + reward-model-reranking** chapter, before we added production-grade sampling controls. Frozen at commit `bb41c0e` ("reduce min reply tokens").

## What's new here (vs v3-reward-model)

- **Dialogue fine-tuning** (`finetune.py`): base model specialised on `dialogue_wodehouse.txt` (all 385 characters) at LR 3e-5, best-val checkpoint saving, early stop. Output weights in `model_dialogue.pt`. Val dropped from base 4.46 -> 3.79.
- **Character-conditioned chat** (`generate.py --character NAME`): random pick if omitted. Prompt framed as `<bertie>{user}\n<{character}>` so the model draws on its thousands of recorded Bertie conversations.
- **Best-of-N reward reranking** (`--best-of N`): generates N candidates, scores each with `checker.py`, returns the highest.
- **Stop-token logic fixed**: stop only fires after `MIN_REPLY_TOKENS` have been produced, and only scans text generated past that threshold - prevents 1-word cut-offs.

## What's NOT here (lives on `main`)

- No top-k / top-p / repetition-penalty sampling controls
- Naive multinomial sampling still allows long-tail gibberish (e.g. "Bomington")

## Why this branch exists

Captures the end of the "data + training" chapter. Everything up to this point was about teaching the model more (better data, fine-tuning, reward signal). The next chapter (`main`) is about **inference-time engineering** - making the existing model's output cleaner via sampling. Preserving this snapshot lets learners see the before/after of sampling improvements clearly.

## Decisions captured here

1. **Cast the user as a character**, not narration. We first tried `<narration>{user_input}` - the model confused narration format with dialogue. Switching to `<bertie>` (2,398 training lines) gave the model familiar conversation shape. See `local-md-files/transformers/27-fine-tuning-for-dialogue.md` for the journey.

2. **Best-checkpoint saving + early stopping**. v6 base training overfit past step 7K; we saved the final (worst) checkpoint by accident. Fine-tuning fixed this from day one - saves only when val improves, stops after 5 non-improving evals.

3. **Heuristic scorer as reward model**. Used `checker.py` for best-of-N reranking. Directly hit **Goodhart's law**: 1-word "Really," scored 100% because every layer trivially passes on a single word. Fixed with `MIN_REPLY_TOKENS` length guard. See `local-md-files/transformers/28-reward-model-and-best-of-n.md`.

## Honest outcome

Model captures Wodehouse **form** (speaker tags, narration, vocabulary, character tics like Psmith's "Comrade") but not **semantics** (cannot answer questions meaningfully). This is the 21M-params + 11M-training-chars ceiling. The next improvement chapter (`main`) addresses sampling quality, not capability - the ceiling is still there.

## Next branch

`main` adds top-p nucleus sampling, repetition penalty, and optional top-k to `generate.py`. Result: noticeable end-user quality jump with zero retraining, because the long-tail gibberish path is cut at sampling time. See `local-md-files/transformers/29-sampling-top-k-top-p-repetition.md`.
