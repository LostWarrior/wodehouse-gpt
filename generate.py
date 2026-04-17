"""
Generate text from a trained WodehouseGPT model.

Usage:
    ./jeeves "Jeeves"                     Single prompt
    ./jeeves                              Interactive mode
    ./jeeves "Jeeves" --chars 500 --temp 0.8
"""

import argparse
import torch
import torch.nn.functional as F
from model import WodehouseGPT
from bpe_tokenizer import encode, decode, load as bpe_load
from config import vocab_size, embed_dim, num_heads, num_layers, max_seq_len


def load_model():
    """Load trained model and merges."""
    merges = bpe_load()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = WodehouseGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.to(device)
    model.eval()

    return model, merges, device


def generate(model, merges, device,
             prompt, num_tokens=200, temperature=0.8):
    """Generate text one token at a time starting from a prompt."""
    tokens = torch.tensor(encode(prompt, merges), device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(tokens[:, -max_seq_len:])
            probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    return decode(tokens[0].tolist(), merges)


def interactive(model, merges, device, num_tokens, temperature):
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
        print(generate(model, merges, device,
                       prompt or "\n", num_tokens, temperature))
        print()


def main():
    parser = argparse.ArgumentParser(description="WodehouseGPT text generator")
    parser.add_argument('prompt', nargs='?', default=None, help="Starting text")
    parser.add_argument('--tokens', type=int, default=200, help="Tokens to generate")
    parser.add_argument('--temp', type=float, default=0.8, help="Temperature (0.1-1.5)")
    args = parser.parse_args()

    print("Loading model...")
    model, merges, device = load_model()

    print("=" * 60)
    print("Jeeves - Trained on P.G. Wodehouse")
    print(f"Temperature: {args.temp} | Tokens: {args.tokens}")
    print("=" * 60)
    print()

    if args.prompt:
        print(generate(model, merges, device,
                       args.prompt, args.tokens, args.temp))
    else:
        interactive(model, merges, device,
                    args.tokens, args.temp)


if __name__ == '__main__':
    main()
