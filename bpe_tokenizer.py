"""
Byte Pair Encoding tokenizer.

Starts with 256 byte-level tokens (handles any input),
then learns merge rules from training data to build larger tokens.

Replaces tokenizer.py - same job, smarter splitting.
"""

import json


def _count_pairs(tokens):
    """Count every adjacent pair. Returns {(id_a, id_b): count}."""
    counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(tokens, pair, new_id):
    """Replace every occurrence of pair in tokens with new_id."""
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def _build_vocab(merges):
    """Build token_id -> bytes mapping from merge rules."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def train(text, vocab_size):
    """
    Learn BPE merge rules from text.

    Returns:
        merges: dict {(id_a, id_b): new_id} in learning order
        tokens: the training text fully encoded with all merges applied
    """
    tokens = list(text.encode('utf-8'))
    num_merges = vocab_size - 256
    merges = {}

    initial_len = len(tokens)
    print(f"Training BPE: {initial_len:,} bytes, learning {num_merges} merges...")

    for i in range(num_merges):
        counts = _count_pairs(tokens)
        if not counts:
            print(f"  no more pairs after {i} merges")
            break

        best = max(counts, key=counts.get)
        new_id = 256 + i
        tokens = _merge(tokens, best, new_id)
        merges[best] = new_id

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  merge {i + 1}/{num_merges} | vocab: {256 + i + 1} | tokens: {len(tokens):,}")

    ratio = initial_len / len(tokens)
    print(f"Done. Vocab: {256 + len(merges)} | {initial_len:,} -> {len(tokens):,} tokens ({ratio:.1f}x compression)")

    return merges, tokens


def encode(text, merges):
    """
    Encode a string to token IDs.

    Applies merges sequentially - fine for short text like prompts.
    For the full training corpus, use the tokens returned by train().
    """
    tokens = list(text.encode('utf-8'))
    for pair, new_id in merges.items():
        tokens = _merge(tokens, pair, new_id)
    return tokens


def decode(token_ids, merges):
    """Decode token IDs back to a string."""
    vocab = _build_vocab(merges)
    raw = b''.join(vocab[t] for t in token_ids)
    return raw.decode('utf-8', errors='replace')


def save(merges, path='merges.json'):
    """Save merge rules to JSON (preserves learning order)."""
    data = [{"pair": [a, b], "id": new_id} for (a, b), new_id in merges.items()]
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Saved {len(merges)} merge rules to {path}")


def load(path='merges.json'):
    """Load merge rules from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    merges = {(entry['pair'][0], entry['pair'][1]): entry['id'] for entry in data}
    print(f"Loaded {len(merges)} merge rules from {path}")
    return merges


if __name__ == '__main__':
    with open('data.txt', 'r') as f:
        text = f.read()

    # Demo on a small sample (fast). Full training uses all 11M chars.
    sample = text[:50_000]
    print(f"=== BPE Demo (50K chars, full text is {len(text):,} chars) ===\n")

    merges, tokens = train(sample, vocab_size=512)
    vocab = _build_vocab(merges)

    # Show what merges learned
    print("\nFirst 20 merges (most common byte pairs):")
    for i, ((a, b), new_id) in enumerate(merges.items()):
        if i >= 20:
            break
        piece = vocab[new_id].decode('utf-8', errors='replace')
        print(f"  merge {i + 1}: token {new_id} = '{piece}'")

    merge_list = list(merges.items())
    print("\nLast 10 merges (longer pieces):")
    for (a, b), new_id in merge_list[-10:]:
        piece = vocab[new_id].decode('utf-8', errors='replace')
        print(f"  token {new_id} = '{piece}'")

    # Encode / decode roundtrip
    test = "Jeeves shimmered into the room."
    encoded = encode(test, merges)
    decoded = decode(encoded, merges)

    print(f"\n=== Encode / Decode ===")
    print(f"Text:      '{test}'")
    print(f"Tokens:    {len(test)} chars -> {len(encoded)} tokens ({len(test) / len(encoded):.1f}x)")
    print(f"Decoded:   '{decoded}'")
    print(f"Roundtrip: {'OK' if test == decoded else 'FAIL'}")

    # Show tokenization visually
    print(f"\nTokenized: ", end='')
    for t in encoded:
        piece = vocab[t].decode('utf-8', errors='replace')
        print(f"[{piece}]", end='')
    print()

    # Compare with character tokenizer
    print(f"\n=== Comparison ===")
    print(f"  Character tokenizer: {len(test)} tokens (one per char)")
    print(f"  BPE (vocab 512):     {len(encoded)} tokens")
    effective_context = 256 * len(test) // max(len(encoded), 1)
    print(f"  256-token window sees ~{effective_context} characters instead of 256")
