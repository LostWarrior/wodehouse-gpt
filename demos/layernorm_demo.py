"""
Layer Norm + Residual Connection demo.

Layer Norm: stabilizes values to mean=0, std=1
Residual: adds original input back to output (preserves information)
"""

import torch
import torch.nn as nn


# === LAYER NORM ===
print("=== Layer Norm ===")
print()

# Simulate a vector with wild values (what happens after many multiplications)
x = torch.tensor([[150.0, -200.0, 50.0, 300.0]])
print(f"Before norm: {x.tolist()}")
print(f"  Mean: {x.mean():.1f}, Std: {x.std():.1f}")

# Layer norm stabilizes it
norm = nn.LayerNorm(4)  # 4 = size of the last dimension
x_normed = norm(x)
print(f"After norm:  [{x_normed[0][0]:.3f}, {x_normed[0][1]:.3f}, {x_normed[0][2]:.3f}, {x_normed[0][3]:.3f}]")
print(f"  Mean: {x_normed.mean():.4f}, Std: {x_normed.std():.3f}")
print()

# Key insight: relative order is preserved
print("Relative order preserved:")
print(f"  Before: 300 > 150 > 50 > -200")
print(f"  After:  {x_normed[0][3]:.3f} > {x_normed[0][0]:.3f} > {x_normed[0][2]:.3f} > {x_normed[0][1]:.3f}")
print()

# === RESIDUAL CONNECTION ===
print("=== Residual Connection ===")
print()

# Simple example
original = torch.tensor([1.0, 2.0, 3.0])
layer_output = torch.tensor([0.1, -0.5, 0.3])

# Without residual: original info gone
print(f"Original:     {original.tolist()}")
print(f"Layer output: {layer_output.tolist()}")
print(f"Without residual: {layer_output.tolist()}  <- original lost")
print(f"With residual:    {(original + layer_output).tolist()}  <- original preserved")
print()

# === HOW THEY WORK TOGETHER ===
print("=== Combined: Norm -> Layer -> Residual ===")
print()

embed_dim = 64
norm1 = nn.LayerNorm(embed_dim)
fake_layer = nn.Linear(embed_dim, embed_dim)   # stand-in for attention or feed-forward

x = torch.randn(1, 5, embed_dim)  # 1 batch, 5 chars, 64 features

# The pattern used in our transformer:
x_normed = norm1(x)                # normalize first
layer_out = fake_layer(x_normed)   # pass through layer
result = layer_out + x             # add original back (residual)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {result.shape}")
print()
print("The pattern in our transformer block:")
print("  x -> LayerNorm -> Attention -> + x")
print("  x -> LayerNorm -> FeedForward -> + x")
