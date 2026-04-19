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
from checker import Checker
from config import vocab_size, embed_dim, num_heads, num_layers, max_seq_len

BASE_MODEL_PATH = 'model.pt'
DIALOGUE_MODEL_PATH = 'model_dialogue.pt'
DIALOGUE_FILE = 'dialogue_wodehouse.txt'
MIN_LINES_FOR_RANDOM = 50    # exclude rare or junk tags from random pick
MIN_REPLY_TOKENS = 20        # chat-sized reply: 1-2 sentences from one character
USER_TAG_DEFAULT = 'bertie'  # stand-in for the user - most prolific speaker, richest conversation data
USER_TAG_ALT = 'psmith'      # if the character being asked IS bertie


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
             stop_at=None, min_new_tokens=0):
    """
    Generate up to num_tokens new tokens one at a time from a prompt.
    If stop_at is given (e.g. '\\n<'), stop at the first occurrence that
    appears AFTER min_new_tokens tokens have been produced (so short
    authentic Wodehouse one-liners extend into scenes before stopping).
    """
    tokens = torch.tensor(encode(prompt, merges), device=device).unsqueeze(0)
    min_char_pos = None  # char position after min_new_tokens tokens generated

    with torch.no_grad():
        for i in range(num_tokens):
            logits = model(tokens[:, -max_seq_len:])
            probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            if stop_at and i + 1 == min_new_tokens:
                min_char_pos = len(decode(tokens[0].tolist(), merges))

            if stop_at and min_char_pos is not None and i + 1 > min_new_tokens:
                decoded = decode(tokens[0].tolist(), merges)
                if stop_at in decoded[min_char_pos:]:
                    idx = decoded.index(stop_at, min_char_pos)
                    return decoded[:idx]
    return decode(tokens[0].tolist(), merges)


def character_reply(model, merges, device, user_input, character,
                    num_tokens, temperature, checker=None, best_of=1):
    """
    Build a prompt so the chosen character responds to the user, generate
    best_of candidate scenes, and (if checker is given) return the best-
    scoring one. Early stop kicks in once MIN_REPLY_TOKENS have been
    produced, so replies end on a speaker change but aren't cut too short.
    """
    user_tag = USER_TAG_ALT if character == USER_TAG_DEFAULT else USER_TAG_DEFAULT
    prompt = f"<{user_tag}>{user_input}\n<{character}>"
    prompt_visible_start = len(prompt) - len(f"<{character}>")

    best_scene = None
    best_score = -1.0
    best_detail = None

    for _ in range(best_of):
        output = generate(model, merges, device, prompt,
                          num_tokens, temperature,
                          stop_at='\n<', min_new_tokens=MIN_REPLY_TOKENS)
        scene_raw = output[prompt_visible_start:]
        scene = _format_scene(scene_raw)

        if checker is None:
            return scene, None

        result = checker.score(_strip_tags(scene))
        if result['overall'] > best_score:
            best_score = result['overall']
            best_scene = scene
            best_detail = result

    return best_scene, best_detail


def _strip_tags(text):
    """Remove <character> tags so the checker scores just the prose."""
    return re.sub(r'<[a-z_]+(?::[a-z_]+)?>', ' ', text)


def _format_scene(text):
    """Turn '<bertie>...\\n<jeeves>...' runs into readable lines.
    Skips empty turns (e.g. model stutters like <jeeves><jeeves>) and
    drops trailing partial tags the token budget cut off.
    """
    import re
    tag_re = re.compile(r'<([a-z_]+)(?::[a-z_]+)?>')
    pieces = []
    last_end = 0
    last_tag = None
    for m in tag_re.finditer(text):
        if last_tag is not None:
            content = text[last_end:m.start()].strip()
            if content:
                pieces.append(f"<{last_tag}> {content}")
        last_tag = m.group(1)
        last_end = m.end()
    if last_tag is not None:
        content = text[last_end:].strip()
        if content and '<' not in content[-10:]:  # drop if cut off mid-tag
            pieces.append(f"<{last_tag}> {content}")
    return "\n\n".join(pieces)


def interactive_dialogue(model, merges, device, frequent_chars,
                         fixed_character, num_tokens, temperature,
                         checker, best_of):
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
        scene, detail = character_reply(model, merges, device, user_input,
                                        character, num_tokens, temperature,
                                        checker, best_of)
        print(f"\n{scene}\n")
        if detail:
            print(f"[score {detail['overall']:.0%} - "
                  f"words {detail['word']:.0%} / ngram {detail['ngram']:.0%} / "
                  f"rep {detail['repetition']:.0%} / distinct {detail['distinctive']:.0%}]\n")


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
    parser.add_argument('--best-of', type=int, default=1,
                        help="Sample N candidates and return highest checker score")
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

    checker = None
    if args.best_of > 1:
        checker = Checker('data.txt')

    if args.character and args.character not in all_chars:
        print(f"Warning: '{args.character}' not in training data.")
        print(f"Known characters (top 10 by line count): see dialogue_wodehouse.txt")
        print(f"Proceeding anyway - model will try its best.\n")

    print("=" * 60)
    print(f"Wodehouse-GPT (dialogue) | Temp: {args.temp} | "
          f"Tokens: {args.tokens} | Best-of: {args.best_of}")
    print("=" * 60)
    print()

    if args.prompt:
        character = args.character or random.choice(frequent)
        scene, detail = character_reply(model, merges, device, args.prompt,
                                        character, args.tokens, args.temp,
                                        checker, args.best_of)
        print(scene)
        if detail:
            print(f"\n[score {detail['overall']:.0%} - "
                  f"words {detail['word']:.0%} / ngram {detail['ngram']:.0%} / "
                  f"rep {detail['repetition']:.0%} / distinct {detail['distinctive']:.0%}]")
    else:
        interactive_dialogue(model, merges, device, frequent,
                             args.character, args.tokens, args.temp,
                             checker, args.best_of)


if __name__ == '__main__':
    main()
