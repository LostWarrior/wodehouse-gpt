"""
Positional encoding demo.

Shows how adding position embeddings makes identical characters
at different positions become different vectors.
"""

import torch
import torch.nn as nn
from tokenizer import build_vocab, encode

# Load text and build vocab
with open('data.txt', 'r') as f:
    text = f.read()
char_to_idx, idx_to_char = build_vocab(text)

vocab_size = len(char_to_idx)   # 76
embed_dim = 64                  # features per character
max_seq_len = 256               # max positions our model will handle

# Two embedding tables:
# 1. Character embedding: "what character is this?"
char_embedding = nn.Embedding(vocab_size, embed_dim)

# 2. Position embedding: "where in the sequence is it?"
pos_embedding = nn.Embedding(max_seq_len, embed_dim)

# Encode "Jeeves"
word = "Jeeves"
token_ids = torch.tensor(encode(word, char_to_idx))

# Create position indices: [0, 1, 2, 3, 4, 5]
positions = torch.arange(len(word))

print(f"=== Input: '{word}' ===")
print(f"Token IDs:  {token_ids.tolist()}")
print(f"Positions:  {positions.tolist()}")
print()

# Look up both embeddings
char_vectors = char_embedding(token_ids)   # shape: (6, 64)
pos_vectors = pos_embedding(positions)     # shape: (6, 64)

# Add them together - this is the final input to the transformer
final_vectors = char_vectors + pos_vectors  # shape: (6, 64)

print(f"=== Shapes ===")
print(f"Character embeddings: {char_vectors.shape}")
print(f"Position embeddings:  {pos_vectors.shape}")
print(f"Combined (char + pos): {final_vectors.shape}")
print()

# The key insight: compare the two 'e's in "Jeeves"
# 'e' is at positions 1 and 4
print(f"=== Are the two 'e's different now? ===")
print()

# Without position encoding
print(f"Character embedding only:")
print(f"  'e' at position 1: [{char_vectors[1][0]:.3f}, {char_vectors[1][1]:.3f}, {char_vectors[1][2]:.3f}, ...]")
print(f"  'e' at position 4: [{char_vectors[4][0]:.3f}, {char_vectors[4][1]:.3f}, {char_vectors[4][2]:.3f}, ...]")
print(f"  Identical: {torch.equal(char_vectors[1], char_vectors[4])}")
print()

# With position encoding
print(f"After adding position:")
print(f"  'e' at position 1: [{final_vectors[1][0]:.3f}, {final_vectors[1][1]:.3f}, {final_vectors[1][2]:.3f}, ...]")
print(f"  'e' at position 4: [{final_vectors[4][0]:.3f}, {final_vectors[4][1]:.3f}, {final_vectors[4][2]:.3f}, ...]")
print(f"  Identical: {torch.equal(final_vectors[1], final_vectors[4])}")
print()

# What the position embeddings look like on their own
print(f"=== Position embeddings (first 4 values) ===")
for i in range(len(word)):
    pv = pos_vectors[i]
    print(f"  Position {i}: [{pv[0]:.3f}, {pv[1]:.3f}, {pv[2]:.3f}, {pv[3]:.3f}, ...]")
