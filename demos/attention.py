"""
Self-Attention - the core of the transformer.

Each character looks at every other character and creates
a new vector that's a weighted mix of everyone's information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        # Three separate linear layers to create Q, K, V
        # Same input goes through three different transformations
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # e.g.,    (32, 256, 64) = 32 sequences, 256 chars, 64 features each

        # Step 0: Create Q, K, V
        q = self.query(x)    # (32, 256, 64) -> (32, 256, 64)
        k = self.key(x)      # (32, 256, 64) -> (32, 256, 64)
        v = self.value(x)    # (32, 256, 64) -> (32, 256, 64)

        # Step 1: Score = Q @ K transposed
        # k.transpose(-2, -1) flips the last two dimensions: (32, 256, 64) -> (32, 64, 256)
        # Result: (32, 256, 64) @ (32, 64, 256) = (32, 256, 256)
        # Each 256x256 grid = every character's score against every other character
        scores = q @ k.transpose(-2, -1)

        # Step 2: Scale - divide by sqrt(embed_dim) to prevent extreme softmax
        scores = scores / self.scale

        # Step 3: Mask - decoder only sees PAST characters, not future ones
        # Create a triangle of True/False:
        #   [True,  False, False]
        #   [True,  True,  False]
        #   [True,  True,  True ]
        # Set future positions (False) to -infinity so softmax gives them 0%
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        # Step 4: Softmax - turn scores into percentages (0-1, rows sum to 1)
        weights = F.softmax(scores, dim=-1)

        # Step 5: Weighted sum - mix Values according to percentages
        # (32, 256, 256) @ (32, 256, 64) = (32, 256, 64)
        output = weights @ v

        return output


if __name__ == '__main__':
    from tokenizer import build_vocab, encode

    # Load text and build vocab
    with open('data.txt', 'r') as f:
        text = f.read()
    char_to_idx, idx_to_char = build_vocab(text)

    vocab_size = len(char_to_idx)
    embed_dim = 64
    max_seq_len = 256

    # Set up embeddings (same as before)
    char_embedding = nn.Embedding(vocab_size, embed_dim)
    pos_embedding = nn.Embedding(max_seq_len, embed_dim)

    # Create attention layer
    attention = SelfAttention(embed_dim)

    # Encode a sentence
    sentence = "Jeeves shimmered in."
    token_ids = torch.tensor(encode(sentence, char_to_idx)).unsqueeze(0)  # add batch dim: (1, 20)
    positions = torch.arange(len(sentence)).unsqueeze(0)                   # (1, 20)

    # Embed
    x = char_embedding(token_ids) + pos_embedding(positions)  # (1, 20, 64)

    print(f"=== Input: '{sentence}' ===")
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"After embedding: {x.shape}")
    print()

    # Run attention
    output = attention(x)
    print(f"=== After Attention ===")
    print(f"Output shape: {output.shape}")
    print()

    # Show that the values changed
    print(f"=== 'J' vector (first 5 values) ===")
    print(f"  Before attention: [{x[0][0][0]:.3f}, {x[0][0][1]:.3f}, {x[0][0][2]:.3f}, {x[0][0][3]:.3f}, {x[0][0][4]:.3f}]")
    print(f"  After attention:  [{output[0][0][0]:.3f}, {output[0][0][1]:.3f}, {output[0][0][2]:.3f}, {output[0][0][3]:.3f}, {output[0][0][4]:.3f}]")
    print()

    # Compare the two 'e's in "Jeeves" (positions 1 and 4)
    print(f"=== Two 'e's in 'Jeeves' ===")
    print(f"  Before attention (pos 1): [{x[0][1][0]:.3f}, {x[0][1][1]:.3f}, {x[0][1][2]:.3f}]")
    print(f"  Before attention (pos 4): [{x[0][4][0]:.3f}, {x[0][4][1]:.3f}, {x[0][4][2]:.3f}]")
    print(f"  After attention  (pos 1): [{output[0][1][0]:.3f}, {output[0][1][1]:.3f}, {output[0][1][2]:.3f}]")
    print(f"  After attention  (pos 4): [{output[0][4][0]:.3f}, {output[0][4][1]:.3f}, {output[0][4][2]:.3f}]")
