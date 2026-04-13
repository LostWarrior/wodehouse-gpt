"""
Generate text from a trained WodehouseGPT model.

Usage:
    ./jeeves "Jeeves"                     Single prompt
    ./jeeves                              Interactive mode
    ./jeeves "Jeeves" --chars 500 --temp 0.8
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from model import WodehouseGPT
from tokenizer import encode, decode
from config import embed_dim, num_heads, num_layers, max_seq_len


def load_vocab():
    """Load vocab from data.txt if available, otherwise fall back to vocab.json."""
    if os.path.exists('data.txt'):
        from tokenizer import build_vocab
        with open('data.txt', 'r') as f:
            return build_vocab(f.read())

    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    char_to_idx = vocab['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
    return char_to_idx, idx_to_char


def load_model():
    """Load trained model and vocab."""
    char_to_idx, idx_to_char = load_vocab()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = WodehouseGPT(len(char_to_idx), embed_dim, num_heads, num_layers, max_seq_len)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.to(device)
    model.eval()

    return model, char_to_idx, idx_to_char, device


def generate(model, char_to_idx, idx_to_char, device,
             prompt, num_chars=500, temperature=0.8):
    """Generate text one character at a time starting from a prompt."""
    tokens = torch.tensor(encode(prompt, char_to_idx), device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(num_chars):
            logits = model(tokens[:, -max_seq_len:])
            probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    return decode(tokens[0].tolist(), idx_to_char)


def interactive(model, char_to_idx, idx_to_char, device, num_chars, temperature):
    """Interactive prompt loop."""
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if prompt.lower() in ('quit', 'exit', 'q'):
            print("Bye!")
            break

        print()
        print(generate(model, char_to_idx, idx_to_char, device,
                       prompt or "\n", num_chars, temperature))
        print()


def main():
    parser = argparse.ArgumentParser(description="WodehouseGPT text generator")
    parser.add_argument('prompt', nargs='?', default=None, help="Starting text")
    parser.add_argument('--chars', type=int, default=500, help="Characters to generate")
    parser.add_argument('--temp', type=float, default=0.8, help="Temperature (0.1-1.5)")
    args = parser.parse_args()

    print("Loading model...")
    model, char_to_idx, idx_to_char, device = load_model()

    print("=" * 60)
    print("Jeeves - Trained on P.G. Wodehouse")
    print(f"Temperature: {args.temp} | Characters: {args.chars}")
    print("=" * 60)
    print()

    if args.prompt:
        print(generate(model, char_to_idx, idx_to_char, device,
                       args.prompt, args.chars, args.temp))
    else:
        interactive(model, char_to_idx, idx_to_char, device,
                    args.chars, args.temp)


if __name__ == '__main__':
    main()
