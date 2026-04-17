"""
The complete transformer model.

All pieces wired together:
- Character + Position embeddings
- N transformer blocks (each: attention + feed-forward + norm + residual)
- Final linear layer to predict next character
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# === MULTI-HEAD ATTENTION (from step 7) ===

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / self.scale

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        out = self.out(out)
        out = self.out_dropout(out)
        return out


# === FEED-FORWARD (from step 8) ===

class FeedForward(nn.Module):

    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# === TRANSFORMER BLOCK (from step 9 - norm + residual) ===

class TransformerBlock(nn.Module):
    """One block = attention + feed-forward, each with norm and residual."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feedforward = FeedForward(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention with norm and residual
        x = x + self.attention(self.norm1(x))

        # Feed-forward with norm and residual
        x = x + self.feedforward(self.norm2(x))

        return x


# === THE FULL MODEL ===

class WodehouseGPT(nn.Module):
    """The complete transformer: embeddings -> N blocks -> predict next char."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.0):
        super().__init__()

        # Embeddings (from steps 4-5)
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final norm and prediction layer
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)   # 64 -> 76

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape

        # Step 1: Embed characters + positions
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.char_embedding(token_ids) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        # Step 2: Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 3: Final norm
        x = self.final_norm(x)

        # Step 4: Predict next character (64 features -> 76 vocab scores)
        logits = self.output_layer(x)

        return logits


if __name__ == '__main__':
    from tokenizer import build_vocab, encode

    with open('data.txt', 'r') as f:
        text = f.read()
    char_to_idx, idx_to_char = build_vocab(text)

    # Model configuration
    vocab_size = len(char_to_idx)    # 76
    embed_dim = 64                    # features per character
    num_heads = 4                     # attention heads
    num_layers = 4                    # transformer blocks
    max_seq_len = 256                 # max characters the model sees

    # Create the model
    model = WodehouseGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"=== WodehouseGPT ===")
    print(f"Vocab size:    {vocab_size}")
    print(f"Embed dim:     {embed_dim}")
    print(f"Heads:         {num_heads}")
    print(f"Layers:        {num_layers}")
    print(f"Max seq len:   {max_seq_len}")
    print(f"Total parameters: {total_params:,}")
    print()

    # Run a forward pass
    sentence = "Jeeves shimmered in."
    token_ids = torch.tensor(encode(sentence, char_to_idx)).unsqueeze(0)  # (1, 20)

    print(f"Input: '{sentence}'")
    print(f"Token IDs shape: {token_ids.shape}")

    logits = model(token_ids)   # (1, 20, 76)
    print(f"Output shape: {logits.shape}  (1 batch, 20 chars, 76 vocab scores)")
    print()

    # What does the model predict after the full sentence?
    last_logits = logits[0, -1, :]    # scores for position 19 (after ".")
    probs = F.softmax(last_logits, dim=-1)
    top5 = torch.topk(probs, 5)

    print(f"Top 5 predictions for what comes after '{sentence}':")
    for prob, idx in zip(top5.values, top5.indices):
        char = idx_to_char[idx.item()]
        display = repr(char) if char in '\n ' else char
        print(f"  {display}: {100*prob.item():.1f}%")
    print()
    print("(These are random guesses - the model hasn't been trained yet!)")
