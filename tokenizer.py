"""
Character-level tokenizer for our transformer.

Job: convert text to numbers and back.
Each unique character gets a unique integer ID.
"""


def build_vocab(text):
    """
    Scan the text and assign a number to each unique character.

    Returns:
        char_to_idx: dict mapping character -> number  {'a': 0, 'b': 1, ...}
        idx_to_char: dict mapping number -> character  {0: 'a', 1: 'b', ...}
    """
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def encode(text, char_to_idx):
    """
    Convert a string into a list of integers.

    "hello" -> [45, 32, 53, 53, 56]  (numbers depend on the vocab)
    """
    return [char_to_idx[ch] for ch in text]


def decode(indices, idx_to_char):
    """
    Convert a list of integers back into a string.

    [45, 32, 53, 53, 56] -> "hello"
    """
    return ''.join(idx_to_char[i] for i in indices)


if __name__ == '__main__':
    # Load our Wodehouse text
    with open('data.txt', 'r') as f:
        text = f.read()

    # Build vocabulary
    char_to_idx, idx_to_char = build_vocab(text)

    # Print stats
    print(f"Text length: {len(text):,} characters")
    print(f"Vocabulary size: {len(char_to_idx)} unique characters")
    print(f"Characters: {''.join(char_to_idx.keys())}")
    print()

    # Demo encode/decode
    sample = "Jeeves shimmered in."
    encoded = encode(sample, char_to_idx)
    decoded = decode(encoded, idx_to_char)
    print(f"Original:  {sample}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded}")
    print(f"Round-trip works: {sample == decoded}")
