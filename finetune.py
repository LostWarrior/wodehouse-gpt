"""
Fine-tune the base Wodehouse model on extracted dialogue (all 385 characters).

Takes model.pt (trained on all Wodehouse prose) and specializes it
on dialogue_wodehouse.txt so it learns character tags like <jeeves>,
<bertie>, <caroline>, <narration>. Any character in the training data
can be prompted at generation time. Saves best-val checkpoint to
model_dialogue.pt.

Key differences vs train.py:
- Loads existing model.pt instead of starting fresh
- Reuses existing merges.json (no BPE retraining)
- Lower learning rate (3e-5 vs 3e-4)
- Fewer steps (2000 vs 20000)
- Saves on val-loss improvement, not every eval
- Early-stops if val hasn't improved in N evals
"""

import os
import torch
import torch.nn.functional as F
from model import WodehouseGPT
from bpe_tokenizer import apply_merges, load as bpe_load
from config import vocab_size, embed_dim, num_heads, num_layers, max_seq_len, \
    dropout, batch_size

# === FINE-TUNING CONFIG (different from train.py) ===
BASE_MODEL = 'model.pt'
OUT_MODEL = 'model_dialogue.pt'
DIALOGUE_FILE = 'dialogue_wodehouse.txt'

ft_learning_rate = 3e-5
ft_max_steps = 2000
ft_eval_interval = 100
ft_patience = 5            # stop if val doesn't improve for this many evals

# === DEVICE ===
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

# === LOAD DATA + EXISTING MERGES ===
print("Loading dialogue and merges...")
with open(DIALOGUE_FILE, 'r') as f:
    text = f.read()

merges = bpe_load()
print(f"Loaded {len(merges)} merge rules from merges.json")

data = torch.tensor(apply_merges(text, merges))
print(f"Total tokens: {len(data):,}")

# 95/5 split - dataset is small, give more to training
split = int(0.95 * len(data))
train_data = data[:split]
val_data = data[split:]
print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")


def get_batch(split_name):
    data_source = train_data if split_name == 'train' else val_data
    max_start = len(data_source) - max_seq_len
    random_starts = torch.randint(max_start, (batch_size,))
    input_chunks = [data_source[s : s + max_seq_len] for s in random_starts]
    target_chunks = [data_source[s + 1 : s + 1 + max_seq_len] for s in random_starts]
    inputs = torch.stack(input_chunks)
    targets = torch.stack(target_chunks)
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split_name in ['train', 'val']:
        batch_losses = []
        for _ in range(50):
            x, y = get_batch(split_name)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            batch_losses.append(loss.item())
        losses[split_name] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


# === LOAD BASE MODEL ===
print(f"\nLoading base model from {BASE_MODEL}...")
model = WodehouseGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout)
model.load_state_dict(torch.load(BASE_MODEL, map_location=device))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=ft_learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
print(f"Learning rate: {ft_learning_rate} (10x lower than base)")

# === FINE-TUNING LOOP ===
print(f"\nFine-tuning for up to {ft_max_steps} steps (early stop after {ft_patience} evals without improvement)...\n")

best_val = float('inf')
evals_since_improvement = 0

for step in range(ft_max_steps + 1):

    if step % ft_eval_interval == 0:
        losses = estimate_loss()
        improved = losses['val'] < best_val
        marker = ' <- best' if improved else ''
        print(f"Step {step:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f}{marker}")

        if improved:
            best_val = losses['val']
            evals_since_improvement = 0
            torch.save(model.state_dict(), OUT_MODEL)
        else:
            evals_since_improvement += 1
            if evals_since_improvement >= ft_patience:
                print(f"\nEarly stop: val loss hasn't improved in {ft_patience} evals")
                break

    if step == ft_max_steps:
        break

    x, y = get_batch('train')
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

print(f"\nBest val loss: {best_val:.4f}")
print(f"Saved to {OUT_MODEL}")
