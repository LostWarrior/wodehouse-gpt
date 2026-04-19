"""
Generate text from a trained WodehouseGPT model.

Defaults to the dialogue fine-tuned model (model_dialogue.pt) and picks
a random Wodehouse character to respond. Use --base to generate from
the raw pre-dialogue model instead.

Usage:
    ./jeeves "What shall I wear?"                     # random character replies
    ./jeeves --character jeeves "What shall I wear?"  # specific character replies
    ./jeeves                                          # interactive mode
    ./jeeves --base "Jeeves"                          # base model, no character
    ./jeeves "hi" --tokens 300 --temp 0.9
"""

import argparse
import random
import re
from collections import Counter
import torch
import torch.nn.functional as F
from model import WodehouseGPT
from bpe_tokenizer import encode, decode, load as bpe_load
from config import vocab_size, embed_dim, num_heads, num_layers, max_seq_len

BASE_MODEL_PATH = 'model.pt'
DIALOGUE_MODEL_PATH = 'model_dialogue.pt'
DIALOGUE_FILE = 'dialogue_wodehouse.txt'
MIN_LINES_FOR_RANDOM = 20  # exclude rarely-spoken characters from random pick


def load_model(path):
    """Load trained model weights and BPE merges."""
    merges = bpe_load()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = WodehouseGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, merges, device


def load_characters():
    """
    Parse character tags from dialogue file.
    Returns (all_chars, frequent_chars) where frequent_chars is the
    subset safe for random selection (appears MIN_LINES_FOR_RANDOM+ times).
    """
    tag_re = re.compile(r'^<([a-z_]+)(?::[a-z_]+)?>', re.MULTILINE)
    with open(DIALOGUE_FILE, 'r') as f:
        text = f.read()
    counts = Counter(tag_re.findall(text))
    counts.pop('narration', None)
    all_chars = set(counts.keys())
    frequent = [c for c, n in counts.items() if n >= MIN_LINES_FOR_RANDOM]
    return all_chars, frequent


def generate(model, merges, device, prompt, num_tokens=200, temperature=0.8,
             stop_at=None):
    """
    Generate text one token at a time starting from a prompt.
    If stop_at is given (e.g. '\\n<'), stop as soon as that string appears
    in the decoded output past the prompt length.
    """
    tokens = torch.tensor(encode(prompt, merges), device=device).unsqueeze(0)
    prompt_len = len(prompt)

    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(tokens[:, -max_seq_len:])
            probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            if stop_at:
                decoded = decode(tokens[0].tolist(), merges)
                if stop_at in decoded[prompt_len:]:
                    idx = decoded.index(stop_at, prompt_len)
                    return decoded[:idx]

    return decode(tokens[0].tolist(), merges)


def character_reply(model, merges, device, user_input, character,
                    num_tokens, temperature):
    """
    Build a prompt so the chosen character responds to the user.
    Uses <narration> for the user's side (model already knows this tag),
    then opens <character> and lets the model continue.
    """
    prompt = f"<narration>{user_input}\n<{character}>"
    output = generate(model, merges, device, prompt,
                      num_tokens, temperature, stop_at='\n<')
    reply = output[len(prompt):].strip()
    return reply


def interactive_dialogue(model, merges, device, frequent_chars,
                         fixed_character, num_tokens, temperature):
    """Interactive loop. Either a fixed character replies each turn, or random each turn."""
    print("Type a message and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Bye!")
            break
        character = fixed_character or random.choice(frequent_chars)
        reply = character_reply(model, merges, device, user_input,
                                character, num_tokens, temperature)
        print(f"\n<{character}> {reply}\n")


def main():
    parser = argparse.ArgumentParser(description="WodehouseGPT text generator")
    parser.add_argument('prompt', nargs='?', default=None,
                        help="Starting text (if --base) or user message")
    parser.add_argument('--character', type=str, default=None,
                        help="Specific character name (default: random)")
    parser.add_argument('--base', action='store_true',
                        help="Use raw base model, no character wrapping")
    parser.add_argument('--tokens', type=int, default=200, help="Tokens to generate")
    parser.add_argument('--temp', type=float, default=0.8, help="Temperature")
    args = parser.parse_args()

    print("Loading model...")

    if args.base:
        model, merges, device = load_model(BASE_MODEL_PATH)
        print("=" * 60)
        print("Wodehouse-GPT (base model) | Temp:", args.temp, "| Tokens:", args.tokens)
        print("=" * 60)
        print()
        if args.prompt:
            print(generate(model, merges, device, args.prompt, args.tokens, args.temp))
        else:
            while True:
                try:
                    p = input("> ")
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    break
                if p.lower() in ('quit', 'exit', 'q'):
                    break
                print()
                print(generate(model, merges, device, p or "\n", args.tokens, args.temp))
                print()
        return

    model, merges, device = load_model(DIALOGUE_MODEL_PATH)
    all_chars, frequent = load_characters()

    if args.character and args.character not in all_chars:
        print(f"Warning: '{args.character}' not in training data.")
        print(f"Known characters (top 10 by line count): see dialogue_wodehouse.txt")
        print(f"Proceeding anyway - model will try its best.\n")

    print("=" * 60)
    print(f"Wodehouse-GPT (dialogue) | Temp: {args.temp} | Tokens: {args.tokens}")
    print("=" * 60)
    print()

    if args.prompt:
        character = args.character or random.choice(frequent)
        reply = character_reply(model, merges, device, args.prompt,
                                character, args.tokens, args.temp)
        print(f"<{character}> {reply}")
    else:
        interactive_dialogue(model, merges, device, frequent,
                             args.character, args.tokens, args.temp)


if __name__ == '__main__':
    main()
