"""
Reusable dialogue extractor for literature.

Extracts conversations between characters from novels and formats
them as training pairs. Works with any book - just change the config.

Features:
    - Attribution via character names and verbs ("said Jeeves")
    - Pronoun resolution ("he said" -> resolves to last male character mentioned)
    - Tone/emotion tags from attribution verbs ("whispered" -> quiet)
    - Narration context between dialogue lines
    - Alternation for unattributed lines

Usage:
    python3 extract_dialogue.py                  # Jeeves/Bertie from Wodehouse
    python3 extract_dialogue.py --config holmes   # Holmes/Watson (bring your own data)
"""

import re
import argparse


# ============================================================
# CONFIGS - add new ones here for different books/characters
# ============================================================

CONFIGS = {
    "jeeves": {
        "description": "Jeeves and Bertie Wooster from P.G. Wodehouse",
        "characters": {
            "jeeves": {
                "names": ["Jeeves"],
                "pronouns": ["he", "him", "his"],
                "output_tag": "<jeeves>",
            },
            "bertie": {
                "names": ["I", "Bertie", "Wooster"],
                "pronouns": ["I", "me", "my"],
                "output_tag": "<human>",
            },
        },
        "narrator": "bertie",
        "data_file": "data.txt",
    },
    "holmes": {
        "description": "Sherlock Holmes and Watson from Arthur Conan Doyle",
        "characters": {
            "holmes": {
                "names": ["Holmes", "Sherlock"],
                "pronouns": ["he", "him", "his"],
                "output_tag": "<holmes>",
            },
            "watson": {
                "names": ["I", "Watson"],
                "pronouns": ["I", "me", "my"],
                "output_tag": "<human>",
            },
        },
        "narrator": "watson",
        "data_file": "holmes_data.txt",
    },
}


# ============================================================
# VERB TONES - map attribution verbs to emotional tone
# ============================================================
# None means neutral (no tone tag added). Only distinctive tones get tagged.

VERB_TONES = {
    # neutral - no tag
    "said": None, "asked": None, "replied": None, "answered": None,
    "added": None, "continued": None, "repeated": None, "agreed": None,
    "inquired": None, "announced": None, "admitted": None, "explained": None,
    "declared": None, "responded": None, "began": None, "went on": None,
    # quiet
    "whispered": "quiet", "murmured": "quiet",
    # loud
    "shouted": "loud", "cried": "loud", "exclaimed": "loud", "called": "loud",
    # emotional
    "protested": "upset", "pleaded": "desperate",
    # thoughtful
    "suggested": "thoughtful", "observed": "thoughtful", "remarked": "thoughtful",
    # forceful
    "urged": "forceful", "insisted": "forceful",
}

ATTRIBUTION_VERBS = list(VERB_TONES.keys())


# ============================================================
# EXTRACTION
# ============================================================

def find_quotes(text):
    """
    Find all quoted text with their positions.

    Handles both 'single' and "double" quotes.
    Returns list of (start_pos, end_pos, quoted_text).
    """
    quotes = []

    # Strategy: scan for quote marks and pair them up.
    i = 0
    while i < len(text):
        # Check for opening quote (double or single)
        if text[i] in ('"', '\u201c'):
            close_char = '"' if text[i] == '"' else '\u201d'
            start = i + 1
            end = text.find(close_char, start)
            if end == -1:
                end = text.find(text[i], start)
            if end != -1 and end - start < 1000:
                quotes.append((i, end + 1, text[start:end]))
                i = end + 1
                continue

        elif text[i] in ("'", '\u2018'):
            # Single quotes are tricky - also used for apostrophes.
            # Only treat as dialogue if after whitespace/newline
            if i == 0 or text[i - 1] in (' ', '\n', '\t'):
                close_char = "'" if text[i] == "'" else '\u2019'
                start = i + 1
                end = text.find(close_char, start)
                # Skip apostrophes: if close is within same word, skip
                while end != -1 and end < len(text) - 1 and text[end + 1].isalpha():
                    end = text.find(close_char, end + 1)
                if end != -1 and end - start > 1 and end - start < 1000:
                    content = text[start:end]
                    if ' ' in content or (content and content[0].isupper()):
                        quotes.append((i, end + 1, content))
                        i = end + 1
                        continue

        i += 1

    return quotes


def find_speaker(text, quote_start, quote_end, config, recent_by_pronoun):
    """
    Try to find who said a quote by looking for attribution nearby.

    Checks in order:
        1. Named attribution: "text," said Jeeves
        2. Pronoun attribution: "text," he said (resolved via recent mentions)

    Returns (character_key, verb) or (None, None).
    """
    # Build lookups
    name_to_key = {}
    for char_key, char_info in config["characters"].items():
        for name in char_info["names"]:
            name_to_key[name] = char_key

    pronoun_to_keys = {}
    for char_key, char_info in config["characters"].items():
        for pronoun in char_info.get("pronouns", []):
            p = pronoun.lower()
            if p not in pronoun_to_keys:
                pronoun_to_keys[p] = []
            pronoun_to_keys[p].append(char_key)

    after = text[quote_end:quote_end + 80]
    before = text[max(0, quote_start - 80):quote_start]

    # --- Pass 1: named attribution ---

    for verb in ATTRIBUTION_VERBS:
        for name, char_key in name_to_key.items():
            # After: "..." said Jeeves / "..." Jeeves said
            if re.search(rf'^\s*{re.escape(verb)}\s+{re.escape(name)}\b', after, re.IGNORECASE):
                return char_key, verb
            if re.search(rf'^\s*{re.escape(name)}\s+{re.escape(verb)}\b', after, re.IGNORECASE):
                return char_key, verb

            # Before: Jeeves said, "..."
            if re.search(rf'{re.escape(name)}\s+{re.escape(verb)}\s*,?\s*$', before, re.IGNORECASE):
                return char_key, verb
            if re.search(rf'{re.escape(verb)}\s+{re.escape(name)}\s*,?\s*$', before, re.IGNORECASE):
                return char_key, verb

    # --- Pass 2: pronoun attribution ---
    # "he said", "she whispered" - resolve using recently mentioned characters

    for verb in ATTRIBUTION_VERBS:
        for pronoun, possible_keys in pronoun_to_keys.items():
            if re.search(rf'^\s*{re.escape(verb)}\s+{re.escape(pronoun)}\b', after, re.IGNORECASE):
                resolved = recent_by_pronoun.get(pronoun)
                if resolved and resolved in possible_keys:
                    return resolved, verb

            if re.search(rf'^\s*{re.escape(pronoun)}\s+{re.escape(verb)}\b', after, re.IGNORECASE):
                resolved = recent_by_pronoun.get(pronoun)
                if resolved and resolved in possible_keys:
                    return resolved, verb

    return None, None


def update_recent_pronouns(text_chunk, config, recent_by_pronoun):
    """
    Scan a chunk of narration and update which character was most recently
    mentioned, so pronoun resolution knows who "he"/"she" refers to.

    If we see "Jeeves" in the text, then "he" now means jeeves.
    """
    # Find the last occurrence of each character name in the chunk
    last_pos = {}
    for char_key, char_info in config["characters"].items():
        for name in char_info["names"]:
            if name == "I":
                continue  # "I" appears everywhere, skip
            pos = text_chunk.rfind(name)
            if pos != -1:
                existing = last_pos.get(char_key, -1)
                if pos > existing:
                    last_pos[char_key] = pos

    # The character mentioned latest in the text "claims" their pronouns
    if last_pos:
        latest_key = max(last_pos, key=last_pos.get)
        char_info = config["characters"][latest_key]
        for pronoun in char_info.get("pronouns", []):
            recent_by_pronoun[pronoun.lower()] = latest_key


def get_narration(text, prev_end, curr_start, max_len=200):
    """
    Extract narration text between two quotes.

    Returns cleaned narration string, or None if it's just whitespace
    or too long to be useful context.
    """
    between = text[prev_end:curr_start].strip()

    # Clean up: collapse whitespace, remove quote attribution we already captured
    between = re.sub(r'\s+', ' ', between)

    if len(between) < 3:
        return None

    if len(between) > max_len:
        return None

    return between


def extract_dialogues(text, config):
    """
    Extract all dialogue exchanges between the configured characters.

    Returns list of exchanges, where each exchange is a list of dicts:
        {"speaker": str, "text": str, "tone": str|None,
         "narration": str|None, "pos": int}
    """
    quotes = find_quotes(text)
    character_keys = set(config["characters"].keys())

    # Tracks which character "he"/"she" currently refers to
    recent_by_pronoun = {}

    # Step 1: attribute each quote to a speaker
    attributed = []
    for idx, (quote_start, quote_end, content) in enumerate(quotes):
        content = content.strip()
        if len(content) < 2:
            continue

        # Update pronoun tracking from narration before this quote
        if idx > 0:
            prev_end = quotes[idx - 1][1]
            narration_text = text[prev_end:quote_start]
            update_recent_pronouns(narration_text, config, recent_by_pronoun)
        else:
            narration_text = text[max(0, quote_start - 200):quote_start]
            update_recent_pronouns(narration_text, config, recent_by_pronoun)

        speaker, verb = find_speaker(text, quote_start, quote_end, config, recent_by_pronoun)
        tone = VERB_TONES.get(verb) if verb else None

        # Capture narration between this quote and the previous one
        narration = None
        if idx > 0:
            narration = get_narration(text, quotes[idx - 1][1], quote_start)

        # If speaker was found by name, update pronoun tracking
        if speaker:
            char_info = config["characters"].get(speaker, {})
            for pronoun in char_info.get("pronouns", []):
                recent_by_pronoun[pronoun.lower()] = speaker

        attributed.append({
            "speaker": speaker,
            "named": speaker is not None,  # was this found by name/pronoun, not alternation?
            "text": content,
            "tone": tone,
            "narration": narration,
            "pos": quote_start,
            "end_pos": quote_end,
        })

    # Step 2: fill in missing attributions using alternation
    # Only alternate after we've seen a named attribution to one of our
    # target characters. This prevents random dialogue in non-Jeeves books
    # from being assigned to Jeeves/Bertie.
    last_speaker = None
    seen_named = False

    for entry in attributed:
        if entry["speaker"] is not None:
            last_speaker = entry["speaker"]
            seen_named = True
        elif last_speaker is not None and seen_named:
            others = [k for k in character_keys if k != last_speaker]
            if others:
                entry["speaker"] = others[0]
                last_speaker = entry["speaker"]

    # Step 3: group into exchanges (consecutive quotes from known speakers)
    exchanges = []
    current = []

    for entry in attributed:
        if entry["speaker"] is None:
            if len(current) >= 2:
                exchanges.append(current)
            current = []
            continue

        # Gap detection: large gaps mean different scenes
        if current and entry["pos"] - current[-1]["pos"] > 500:
            if len(current) >= 2:
                exchanges.append(current)
            current = []

        current.append(entry)

    if len(current) >= 2:
        exchanges.append(current)

    # Step 4: filter for exchanges that involve both characters
    # AND have at least one turn attributed by name (not just alternation)
    filtered = []
    for exchange in exchanges:
        speakers_present = set(e["speaker"] for e in exchange)
        has_named = any(e.get("named") for e in exchange)
        if len(speakers_present & character_keys) >= 2 and has_named:
            filtered.append(exchange)

    return filtered


def format_for_training(exchanges, config, include_narration=True, include_tone=True):
    """Format exchanges as tagged training text."""
    tag_map = {}
    for char_key, char_info in config["characters"].items():
        tag_map[char_key] = char_info["output_tag"]

    formatted = []
    for exchange in exchanges:
        lines = []
        for entry in exchange:
            # Narration before this line
            if include_narration and entry.get("narration"):
                lines.append(f"<narration>{entry['narration']}")

            # Speaker tag with optional tone
            base_tag = tag_map.get(entry["speaker"], f"<{entry['speaker']}>")
            if include_tone and entry.get("tone"):
                # <jeeves> -> <jeeves:quiet>
                tag = base_tag[:-1] + f":{entry['tone']}>"
            else:
                tag = base_tag

            lines.append(f"{tag}{entry['text']}")

        formatted.append("\n".join(lines))

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Extract dialogue from literature")
    parser.add_argument('--config', default='jeeves',
                        choices=list(CONFIGS.keys()),
                        help="Which character config to use")
    parser.add_argument('--data', default=None,
                        help="Override data file path")
    parser.add_argument('--output', default=None,
                        help="Output file (default: dialogue_<config>.txt)")
    parser.add_argument('--min-turns', type=int, default=2,
                        help="Minimum turns per exchange")
    parser.add_argument('--preview', type=int, default=5,
                        help="Number of exchanges to preview (0 to skip)")
    parser.add_argument('--no-narration', action='store_true',
                        help="Exclude narration context")
    parser.add_argument('--no-tone', action='store_true',
                        help="Exclude tone/emotion tags")
    args = parser.parse_args()

    config = CONFIGS[args.config]
    data_file = args.data or config["data_file"]
    output_file = args.output or f"dialogue_{args.config}.txt"

    print(f"Config: {config['description']}")
    print(f"Data:   {data_file}")
    print()

    with open(data_file, 'r') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")

    # Extract
    exchanges = extract_dialogues(text, config)

    # Filter by minimum turns
    exchanges = [e for e in exchanges if len(e) >= args.min_turns]

    # Stats
    total_turns = sum(len(e) for e in exchanges)
    char_counts = {}
    tone_counts = {}
    narration_count = 0
    for exchange in exchanges:
        for entry in exchange:
            char_counts[entry["speaker"]] = char_counts.get(entry["speaker"], 0) + 1
            if entry.get("tone"):
                tone_counts[entry["tone"]] = tone_counts.get(entry["tone"], 0) + 1
            if entry.get("narration"):
                narration_count += 1

    print(f"\nFound {len(exchanges)} exchanges ({total_turns} total turns)")
    print("Turns per character:")
    for char_key, count in sorted(char_counts.items(), key=lambda x: -x[1]):
        tag = config["characters"].get(char_key, {}).get("output_tag", char_key)
        print(f"  {tag}: {count}")

    if tone_counts:
        print("Tones detected:")
        for tone, count in sorted(tone_counts.items(), key=lambda x: -x[1]):
            print(f"  {tone}: {count}")

    print(f"Narration lines: {narration_count}")

    # Preview
    include_narration = not args.no_narration
    include_tone = not args.no_tone

    if args.preview > 0:
        formatted = format_for_training(exchanges, config, include_narration, include_tone)
        print(f"\n{'='*60}")
        print(f"Preview (first {args.preview} exchanges):")
        print(f"{'='*60}")
        for i, exchange in enumerate(formatted[:args.preview]):
            print(f"\n--- Exchange {i + 1} ---")
            print(exchange)

    # Save
    formatted = format_for_training(exchanges, config, include_narration, include_tone)
    with open(output_file, 'w') as f:
        f.write("\n\n".join(formatted))
    print(f"\nSaved {len(formatted)} exchanges to {output_file}")


if __name__ == '__main__':
    main()
