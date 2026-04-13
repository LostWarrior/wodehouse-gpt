"""
Multi-Head Attention.

Same attention mechanism, but split into parallel heads.
Each head looks for different patterns in the data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads   # 64 // 4 = 16 features per head

        # Same three linear layers as single attention
        # They still transform the full 64 features
        # We split into heads AFTER the linear transform
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to mix the heads' outputs together
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)   # sqrt(16) = 4, not sqrt(64)

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape
        # e.g., batch=1, seq_len=20, embed_dim=64

        # Step 0: Create Q, K, V (same as single attention)
        q = self.query(x)   # (1, 20, 64)
        k = self.key(x)     # (1, 20, 64)
        v = self.value(x)   # (1, 20, 64)

        # Step 0.5: Split into heads
        # Reshape from (1, 20, 64) to (1, 20, 4, 16)
        # Then swap dims to (1, 4, 20, 16) so each head is a separate "batch"
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (1, 4, 20, 16) = 1 batch, 4 heads, 20 chars, 16 features per head

        # Steps 1-5: Same attention formula, runs on all heads at once
        # Step 1: Score
        scores = q @ k.transpose(-2, -1)    # (1, 4, 20, 16) @ (1, 4, 16, 20) = (1, 4, 20, 20)

        # Step 2: Scale (by sqrt of head_dim=16, not embed_dim=64)
        scores = scores / self.scale

        # Step 3: Mask future positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        # Step 4: Softmax
        weights = F.softmax(scores, dim=-1)

        # Step 5: Weighted sum
        out = weights @ v    # (1, 4, 20, 20) @ (1, 4, 20, 16) = (1, 4, 20, 16)

        # Step 6: Combine heads back together
        # Swap back: (1, 4, 20, 16) -> (1, 20, 4, 16)
        # Then flatten: (1, 20, 4, 16) -> (1, 20, 64)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)

        # Step 7: Final linear to mix heads' findings
        out = self.out(out)   # (1, 20, 64) -> (1, 20, 64)

        return out


if __name__ == '__main__':
    from tokenizer import build_vocab, encode

    with open('data.txt', 'r') as f:
        text = f.read()
    char_to_idx, idx_to_char = build_vocab(text)

    vocab_size = len(char_to_idx)
    embed_dim = 64
    num_heads = 4
    max_seq_len = 256

    char_embedding = nn.Embedding(vocab_size, embed_dim)
    pos_embedding = nn.Embedding(max_seq_len, embed_dim)

    # Multi-head instead of single attention
    mha = MultiHeadAttention(embed_dim, num_heads)

    sentence = "Jeeves shimmered in."
    token_ids = torch.tensor(encode(sentence, char_to_idx)).unsqueeze(0)
    positions = torch.arange(len(sentence)).unsqueeze(0)

    x = char_embedding(token_ids) + pos_embedding(positions)

    print(f"=== Multi-Head Attention ===")
    print(f"Heads: {num_heads}")
    print(f"Features per head: {embed_dim // num_heads}")
    print(f"Input shape:  {x.shape}")

    output = mha(x)
    print(f"Output shape: {output.shape}")
    print()

    print(f"=== 'J' vector (first 5 values) ===")
    print(f"  Before: [{x[0][0][0]:.3f}, {x[0][0][1]:.3f}, {x[0][0][2]:.3f}, {x[0][0][3]:.3f}, {x[0][0][4]:.3f}]")
    print(f"  After:  [{output[0][0][0]:.3f}, {output[0][0][1]:.3f}, {output[0][0][2]:.3f}, {output[0][0][3]:.3f}, {output[0][0][4]:.3f}]")
    print()

    # Show total parameters
    total = sum(p.numel() for p in mha.parameters())
    print(f"Total learnable parameters in MultiHeadAttention: {total:,}")
