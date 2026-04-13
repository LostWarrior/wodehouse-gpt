"""
Generate text from a trained WodehouseGPT model.

Usage:
    ./jeeves "Jeeves"                     Single prompt
    ./jeeves                              Interactive mode
    ./jeeves "Jeeves" --chars 500 --temp 0.8
"""

import json
import argparse
import torch
import torch.nn.functional as F
from model import WodehouseGPT
from tokenizer import encode, decode
from config import embed_dim, num_heads, num_layers, max_seq_len


def load_model():
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)

    char_to_idx = vocab['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
    vocab_size = len(char_to_idx)

    model_config = dict(vocab_size=vocab_size, embed_dim=embed_dim,
                        num_heads=num_heads, num_layers=num_layers,
                        max_seq_len=max_seq_len)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = WodehouseGPT(**model_config)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.to(device)
    model.eval()

    return model, char_to_idx, idx_to_char, model_config, device


def generate(model, char_to_idx, idx_to_char, config, device,
             prompt, num_chars=500, temperature=0.8):
    tokens = torch.tensor(
        encode(prompt, char_to_idx), device=device
    ).unsqueeze(0)

    with torch.no_grad():
        for _ in range(num_chars):
            logits = model(tokens[:, -config['max_seq_len']:])
            probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    return decode(tokens[0].tolist(), idx_to_char)


def interactive(model, char_to_idx, idx_to_char, config, device,
                num_chars, temperature):
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
        print(generate(model, char_to_idx, idx_to_char, config, device,
                       prompt or "\n", num_chars, temperature))
        print()


def main():
    parser = argparse.ArgumentParser(description="WodehouseGPT text generator")
    parser.add_argument('prompt', nargs='?', default=None, help="Starting text")
    parser.add_argument('--chars', type=int, default=500, help="Characters to generate")
    parser.add_argument('--temp', type=float, default=0.8, help="Temperature (0.1-1.5)")
    args = parser.parse_args()

    print("Loading model...")
    model, char_to_idx, idx_to_char, config, device = load_model()

    print("=" * 60)
    print("Jeeves - Trained on P.G. Wodehouse")
    print(f"Temperature: {args.temp} | Characters: {args.chars}")
    print("=" * 60)
    print()

    if args.prompt:
        print(generate(model, char_to_idx, idx_to_char, config, device,
                       args.prompt, args.chars, args.temp))
    else:
        interactive(model, char_to_idx, idx_to_char, config, device,
                    args.chars, args.temp)


if __name__ == '__main__':
    main()
