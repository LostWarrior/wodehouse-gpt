"""
Training loop for WodehouseGPT.

1. Forward pass  - make predictions
2. Loss          - measure how wrong
3. Backward pass - figure out which way to nudge
4. Optimizer     - nudge the weights
"""

import torch
import torch.nn.functional as F
from model import WodehouseGPT
from tokenizer import build_vocab, encode

# === CONFIGURATION ===
vocab_size = 76
embed_dim = 64
num_heads = 4
num_layers = 4
max_seq_len = 256
batch_size = 32
learning_rate = 3e-4
max_steps = 5000
eval_interval = 500

# === DEVICE (Apple GPU > NVIDIA GPU > CPU) ===
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

# === LOAD AND ENCODE DATA ===
print("Loading data...")
with open('data.txt', 'r') as f:
    text = f.read()

char_to_idx, idx_to_char = build_vocab(text)
vocab_size = len(char_to_idx)

data = torch.tensor(encode(text, char_to_idx))
print(f"Total tokens: {len(data):,}")

# 90% for training, 10% for validation
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]
print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")


def get_batch(split_name):
    """
    Grab random input/target pairs. Target is input shifted by one character.
        Text:   "Jeeves shimmered"
        Input:  "Jeeves s"       (positions 0-7)
        Target: "eeves sh"       (positions 1-8)
    """
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
    """Average loss over 50 batches for train and val. No gradient tracking."""
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


# === CREATE MODEL AND OPTIMIZER ===
print(f"\nCreating model...")
model = WodehouseGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# === TRAINING LOOP ===
print(f"\nTraining for {max_steps} steps...")
print()

for step in range(max_steps):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

    # Forward pass
    x, y = get_batch('train')
    logits = model(x)

    # Loss
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# Final evaluation
losses = estimate_loss()
print(f"Step {max_steps:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
print("\nTraining complete!")

torch.save(model.state_dict(), 'model.pt')
print("Model saved to model.pt")
