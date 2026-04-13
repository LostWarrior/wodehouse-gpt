"""
Feed-Forward Network.

Two linear layers with ReLU in between.
Expand to 4x size (think), then compress back (summarize).
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        # Expand: 64 -> 256 (4x bigger workspace)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)

        # Compress: 256 -> 64 (back to original size)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

        # ReLU: negative numbers become 0 (adds non-linearity)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, seq_len, 64)

        x = self.linear1(x)    # (batch, seq_len, 64)  -> (batch, seq_len, 256)
        x = self.relu(x)       # (batch, seq_len, 256) -> (batch, seq_len, 256)  values unchanged, negatives become 0
        x = self.linear2(x)    # (batch, seq_len, 256) -> (batch, seq_len, 64)

        return x


if __name__ == '__main__':
    embed_dim = 64
    ff = FeedForward(embed_dim)

    # Simulate input: 1 sequence, 20 characters, 64 features
    x = torch.randn(1, 20, 64)

    print(f"=== Feed-Forward Network ===")
    print(f"Input shape:  {x.shape}")

    output = ff(x)
    print(f"Output shape: {output.shape}")
    print()

    # Show the expand/compress
    intermediate = ff.linear1(x)
    print(f"After expand (linear1):  {intermediate.shape}  <- 4x bigger workspace")
    after_relu = ff.relu(intermediate)
    print(f"After ReLU:              {after_relu.shape}  <- same shape, negatives zeroed")
    final = ff.linear2(after_relu)
    print(f"After compress (linear2): {final.shape}  <- back to original")
    print()

    # Count how many values ReLU zeroed out
    num_zeros = (after_relu == 0).sum().item()
    total = after_relu.numel()
    print(f"ReLU zeroed out {num_zeros}/{total} values ({100*num_zeros/total:.1f}%)")
    print()

    # Parameters
    total_params = sum(p.numel() for p in ff.parameters())
    print(f"Total learnable parameters: {total_params:,}")
    print(f"  linear1: {64*256 + 256:,}  (64->256 weights + 256 biases)")
    print(f"  linear2: {256*64 + 64:,}  (256->64 weights + 64 biases)")
