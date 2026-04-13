"""
Generate text from a trained WodehouseGPT model.

Give it a starting prompt, it predicts one character at a time.
"""

import torch
import torch.nn.functional as F
from model import WodehouseGPT
from tokenizer import build_vocab, encode, decode

# === LOAD VOCAB ===
with open('data.txt', 'r') as f:
    text = f.read()
char_to_idx, idx_to_char = build_vocab(text)

# === SAME CONFIG AS TRAINING ===
vocab_size = len(char_to_idx)
embed_dim = 256
num_heads = 8
num_layers = 8
max_seq_len = 256

# === LOAD TRAINED MODEL ===
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = WodehouseGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
model.load_state_dict(torch.load('model.pt', map_location=device))
model = model.to(device)
model.eval()


def generate(prompt, num_chars=500, temperature=0.8):
    """
    Generate text starting from a prompt.

    temperature controls randomness:
        0.1 = very predictable, repeats common patterns
        0.8 = balanced (default)
        1.5 = creative but messy
    """
    token_ids = encode(prompt, char_to_idx)
    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(num_chars):
            # Only feed the last max_seq_len characters (model's window)
            input_tokens = tokens[:, -max_seq_len:]

            # Get predictions
            logits = model(input_tokens)

            # Take the last position's predictions (what comes next?)
            next_logits = logits[0, -1, :] / temperature

            # Convert to probabilities and sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    # Decode back to text
    generated_ids = tokens[0].tolist()
    return decode(generated_ids, idx_to_char)


# === GENERATE ===
print("=" * 60)
print("WodehouseGPT - Trained on P.G. Wodehouse")
print("=" * 60)
print()

prompts = [
    "Jeeves",
    "It was a",
    "The morning",
]

for prompt in prompts:
    print(f"--- Prompt: '{prompt}' ---")
    print(generate(prompt, num_chars=300))
    print()
