"""
Embedding demo - turning token IDs into vectors.

An embedding is a learnable lookup table:
  token ID (integer) -> vector (list of floats)
"""

import torch
import torch.nn as nn
from tokenizer import build_vocab, encode

# Load text and build vocab
with open('data.txt', 'r') as f:
    text = f.read()
char_to_idx, idx_to_char = build_vocab(text)

vocab_size = len(char_to_idx)   # 76 unique characters
embed_dim = 64                  # each character becomes 64 numbers

# Create the embedding table
# This creates a table of shape (76, 64) filled with random numbers
# 76 rows (one per character), 64 columns (features per character)
embedding = nn.Embedding(vocab_size, embed_dim)

# Let's see what's inside
print("=== The Embedding Table ===")
print(f"Shape: {embedding.weight.shape}")
print(f"  {vocab_size} characters x {embed_dim} features each")
print(f"  Total learnable numbers: {vocab_size * embed_dim:,}")
print()

# Encode a word into token IDs
word = "Jeeves"
token_ids = encode(word, char_to_idx)
print(f"=== Encoding '{word}' ===")
print(f"Token IDs: {token_ids}")
print()

# Convert to a PyTorch tensor (embedding expects tensors, not plain lists)
token_tensor = torch.tensor(token_ids)
print(f"As tensor: {token_tensor}")
print(f"Shape: {token_tensor.shape}  (6 tokens)")
print()

# Look up the embeddings
vectors = embedding(token_tensor)
print(f"=== After Embedding ===")
print(f"Shape: {vectors.shape}  (6 tokens, each now 64 features)")
print()

# Show what happened to each character
for i, ch in enumerate(word):
    vec = vectors[i]
    print(f"  '{ch}' (id={token_ids[i]:2d}) -> [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}, ... {vec[-1]:.3f}]")

print()

# Key insight: same character = same vector (before training adds context)
print("=== Same Character, Same Vector ===")
word2 = "ee"
ids2 = torch.tensor(encode(word2, char_to_idx))
vecs2 = embedding(ids2)
print(f"First 'e':  [{vecs2[0][0]:.3f}, {vecs2[0][1]:.3f}, {vecs2[0][2]:.3f}, ...]")
print(f"Second 'e': [{vecs2[1][0]:.3f}, {vecs2[1][1]:.3f}, {vecs2[1][2]:.3f}, ...]")
print(f"Identical: {torch.equal(vecs2[0], vecs2[1])}")
print()
print("(Attention will later make them different based on position and context)")
