"""
Reusable dialogue extractor for literature.

Extracts conversations from novels, tags each line with the speaker's name.
Discovers characters automatically from attribution patterns ("said Jeeves").

Features:
    - Automatic character discovery (no need to list every character)
    - Alias merging (Wooster -> Bertie, Sherlock -> Holmes)
    - Narrator resolution ("I said" -> Bertie in Jeeves stories)
    - Tone/emotion tags from attribution verbs
    - Narration context between dialogue lines
    - Pronoun resolution via recent character mentions

Usage:
    python3 extract_dialogue.py                        # Wodehouse, all characters
    python3 extract_dialogue.py --config holmes         # Holmes stories
    python3 extract_dialogue.py --preview 10            # show more exchanges
    python3 extract_dialogue.py --no-narration --no-tone  # plain dialogue only
"""

import re
import argparse


# ============================================================
# CONFIGS
# ============================================================
# Characters are discovered automatically from attribution patterns.
# "aliases" merges different names for the same character.
# "narrator" resolves "I said" to a character name.

CONFIGS = {
    "wodehouse": {
        "description": "All characters from P.G. Wodehouse novels",
        "aliases": {
            "Wooster": "Bertie",
            "Fink-Nottle": "Gussie",
        },
        "narrator": "Bertie",
        "data_file": "data.txt",
    },
    "holmes": {
        "description": "All characters from Sherlock Holmes",
        "aliases": {
            "Sherlock": "Holmes",
        },
        "narrator": "Watson",
        "data_file": "holmes_data.txt",
    },
}


# ============================================================
# VERB TONES
# ============================================================

VERB_TONES = {
    "said": None, "asked": None, "replied": None, "answered": None,
    "added": None, "continued": None, "repeated": None, "agreed": None,
    "inquired": None, "announced": None, "admitted": None, "explained": None,
    "declared": None, "responded": None, "began": None, "went on": None,
    "whispered": "quiet", "murmured": "quiet",
    "shouted": "loud", "cried": "loud", "exclaimed": "loud", "called": "loud",
    "protested": "upset", "pleaded": "desperate",
    "suggested": "thoughtful", "observed": "thoughtful", "remarked": "thoughtful",
    "urged": "forceful", "insisted": "forceful",
}

ATTRIBUTION_VERBS = list(VERB_TONES.keys())


# ============================================================
# EXTRACTION
# ============================================================

def find_quotes(text):
    """Find all quoted text. Returns list of (start, end, content)."""
    quotes = []

    for m in re.finditer(r'["\u201c](.{1,1000}?)["\u201d]', text, re.DOTALL):
        quotes.append((m.start(), m.end(), m.group(1)))

    for m in re.finditer(r"(?<=[\s\n])['\u2018](.{2,1000}?)['\u2019]", text, re.DOTALL):
        content = m.group(1)
        end_pos = m.end()
        if end_pos < len(text) and text[end_pos].isalpha():
            continue
        if ' ' in content or (content and content[0].isupper()):
            quotes.append((m.start(), m.end(), content))

    quotes.sort(key=lambda q: q[0])
    return quotes


# Words that look like names but aren't - skip these
NOT_NAMES = {
    # pronouns
    'he', 'she', 'it', 'they', 'we', 'you',
    'him', 'her', 'his', 'its', 'them', 'their', 'my', 'me',
    # articles / determiners
    'the', 'this', 'that', 'then', 'there', 'here', 'an', 'a',
    # titles (handled separately as prefixes)
    'mr', 'mrs', 'miss', 'sir', 'lord', 'lady', 'aunt', 'uncle',
    'doctor', 'captain', 'major', 'colonel', 'professor',
    # conjunctions / prepositions / common verbs
    'not', 'but', 'and', 'for', 'nor', 'yet', 'still', 'to', 'of',
    'is', 'was', 'be', 'been', 'are', 'were', 'had', 'has', 'have',
    'do', 'did', 'does', 'will', 'would', 'could', 'should', 'may',
    'can', 'shall', 'might', 'must', 'if', 'or', 'so', 'no', 'yes',
    'at', 'in', 'on', 'up', 'out', 'off', 'by', 'as', 'with',
    # common words
    'one', 'all', 'some', 'very', 'much', 'just', 'now', 'well',
    'our', 'old', 'new', 'own', 'other', 'another', 'any', 'every',
    'last', 'next', 'first', 'only', 'same', 'such', 'both',
    'thing', 'things', 'man', 'woman', 'girl', 'boy', 'fellow',
    'voice', 'face', 'hand', 'head', 'eye', 'eyes',
    'what', 'who', 'how', 'why', 'when', 'where', 'which',
    'nothing', 'something', 'everything', 'anything',
    # adverbs (often appear after closing quote)
    'suddenly', 'quietly', 'slowly', 'quickly', 'softly',
    'eagerly', 'hastily', 'breathlessly', 'sharply', 'firmly',
    'coldly', 'hotly', 'fiercely', 'grimly', 'briskly',
    'bitterly', 'cheerfully', 'sadly', 'wearily', 'anxiously',
    'calmly', 'stiffly', 'huskily', 'thoughtfully', 'dreamily',
    'carelessly', 'helplessly', 'hopefully', 'nervously',
    'politely', 'abruptly', 'doubtfully', 'mildly', 'severely',
    'kindly', 'gently', 'gravely', 'shortly', 'smoothly',
}

# Titles that prefix a character name - capture both words
TITLE_WORDS = {'Mr', 'Mrs', 'Miss', 'Sir', 'Lord', 'Lady', 'Aunt', 'Uncle',
               'Doctor', 'Captain', 'Major', 'Colonel', 'Professor'}


def _build_verb_pattern():
    """Build regexes that match 'VERB Name' or 'Name VERB' patterns."""
    verbs = '|'.join(re.escape(v) for v in ATTRIBUTION_VERBS)
    titles = '|'.join(re.escape(t) for t in TITLE_WORDS)

    # Name = "I" OR "Title Surname" OR plain "Surname"
    # Title names: "Lord Marshmoreton", "Aunt Agatha"
    # Plain names: "Jeeves", "Psmith"
    name_pat = rf'(?:(?:{titles})\s+)?(?P<{{group}}>I|[A-Z][a-z]+(?:\-[A-Z][a-z]+)?)'

    def make(template, group_name, verb_group):
        """Build a compiled regex from a template with name and verb groups."""
        name = name_pat.replace('{group}', group_name)
        verb = rf'(?P<{verb_group}>{verbs})'
        return re.compile(template.format(name=name, verb=verb), re.IGNORECASE)

    after_verb_name = make(r'^\s*{verb}\s+{name}\b', 'name1', 'verb1')
    after_name_verb = make(r'^\s*{name}\s+{verb}\b', 'name2', 'verb2')
    before_name_verb = make(r'{name}\s+{verb}\s*,?\s*$', 'name3', 'verb3')
    before_verb_name = make(r'{verb}\s+{name}\s*,?\s*$', 'name4', 'verb4')

    return after_verb_name, after_name_verb, before_name_verb, before_verb_name


def _resolve_name(match, name_group, verb_group, aliases, narrator):
    """Extract and resolve a name from a regex match. Returns (name, verb) or (None, None)."""
    name = match.group(name_group)
    verb = match.group(verb_group).lower()

    if name == 'I':
        return narrator, verb

    if name.lower() in NOT_NAMES or len(name) < 3:
        return None, None

    # Check if there's a title prefix in the full match
    full = match.group(0).strip()
    for title in TITLE_WORDS:
        if full.startswith(title + ' ') or full.endswith(' ' + title):
            # "said Lord Marshmoreton" -> "Lord Marshmoreton"
            # Find the title + name in the match
            title_pattern = re.search(rf'({re.escape(title)})\s+({re.escape(name)})', full)
            if title_pattern:
                name = f"{title} {name}"
                break

    canonical = aliases.get(name, name)
    return canonical, verb


def find_speaker(after, before, verb_patterns, aliases, narrator):
    """
    Find who said a quote by looking for 'VERB Name' patterns nearby.

    Returns (canonical_name, verb) or (None, None).
    """
    avn, anv, bnv, bvn = verb_patterns

    for pattern, name_group, verb_group in [
        (avn, 'name1', 'verb1'),
        (anv, 'name2', 'verb2'),
    ]:
        m = pattern.search(after)
        if m:
            name, verb = _resolve_name(m, name_group, verb_group, aliases, narrator)
            if name:
                return name, verb

    for pattern, name_group, verb_group in [
        (bnv, 'name3', 'verb3'),
        (bvn, 'name4', 'verb4'),
    ]:
        m = pattern.search(before)
        if m:
            name, verb = _resolve_name(m, name_group, verb_group, aliases, narrator)
            if name:
                return name, verb

    return None, None


def find_pronoun_speaker(after, recent_male, recent_female):
    """
    Resolve 'he said' / 'she said' using recently mentioned characters.

    Returns (name, verb) or (None, None).
    """
    verbs = '|'.join(re.escape(v) for v in ATTRIBUTION_VERBS)

    # "he said" / "said he" / "she said" / "said she"
    m = re.search(rf'^\s*(?P<verb>{verbs})\s+(?P<pronoun>he|she)\b', after, re.IGNORECASE)
    if not m:
        m = re.search(rf'^\s*(?P<pronoun>he|she)\s+(?P<verb>{verbs})\b', after, re.IGNORECASE)

    if m:
        pronoun = m.group('pronoun').lower()
        verb = m.group('verb').lower()
        if pronoun == 'he' and recent_male:
            return recent_male, verb
        if pronoun == 'she' and recent_female:
            return recent_female, verb

    return None, None


def get_narration(text, prev_end, curr_start, max_len=200):
    """Extract short narration between two quotes."""
    between = text[prev_end:curr_start].strip()
    between = re.sub(r'\s+', ' ', between)
    if len(between) < 3 or len(between) > max_len:
        return None
    return between


def update_pronoun_context(text_chunk, known_characters, aliases):
    """
    Scan narration for character names. Returns (last_male, last_female).
    Simple heuristic: all characters assumed male unless in FEMALE_NAMES.
    """
    female_names = {
        'Agatha', 'Florence', 'Honoria', 'Madeline', 'Dahlia', 'Stiffy',
        'Bobbie', 'Pauline', 'Angela', 'Maud', 'Phyllis', 'Alice',
        'Cynthia', 'Mary', 'Eve', 'Joan', 'Jill', 'Sally', 'Ann',
        'Irene', 'Adler', 'Hudson', 'Mrs',
    }

    last_male = None
    last_female = None

    for name in known_characters:
        pos = text_chunk.rfind(name)
        if pos != -1:
            canonical = aliases.get(name, name)
            if name in female_names:
                if last_female is None or pos > text_chunk.rfind(last_female or ''):
                    last_female = canonical
            else:
                if last_male is None or pos > text_chunk.rfind(last_male or ''):
                    last_male = canonical

    return last_male, last_female


def extract_dialogues(text, config):
    """
    Extract all dialogue with speaker attribution.

    Returns list of exchanges (list of dicts with speaker, text, tone, narration).
    """
    quotes = find_quotes(text)
    aliases = config.get("aliases", {})
    narrator = config.get("narrator", "Narrator")
    verb_patterns = _build_verb_pattern()

    # Pass 1: attribute speakers and discover character names
    recent_male = None
    recent_female = None
    known_names = set()
    attributed = []

    for idx, (quote_start, quote_end, content) in enumerate(quotes):
        content = content.strip()
        if len(content) < 2:
            continue

        # Update pronoun context from narration before this quote
        if idx > 0:
            narration_text = text[quotes[idx - 1][1]:quote_start]
        else:
            narration_text = text[max(0, quote_start - 200):quote_start]

        if known_names:
            m, f = update_pronoun_context(narration_text, known_names, aliases)
            if m:
                recent_male = m
            if f:
                recent_female = f

        after = text[quote_end:quote_end + 80]
        before = text[max(0, quote_start - 80):quote_start]

        # Try named attribution first
        speaker, verb = find_speaker(after, before, verb_patterns, aliases, narrator)

        # Fall back to pronoun resolution
        if speaker is None:
            speaker, verb = find_pronoun_speaker(after, recent_male, recent_female)

        tone = VERB_TONES.get(verb) if verb else None

        # Capture narration
        narration = None
        if idx > 0:
            narration = get_narration(text, quotes[idx - 1][1], quote_start)

        # Track discovered names
        if speaker and speaker != narrator:
            known_names.add(speaker)
            # Update pronoun context
            female_names = {
                'Agatha', 'Florence', 'Honoria', 'Madeline', 'Dahlia', 'Stiffy',
                'Bobbie', 'Pauline', 'Angela', 'Maud', 'Phyllis', 'Alice',
                'Cynthia', 'Mary', 'Eve', 'Joan', 'Jill', 'Sally', 'Ann',
            }
            if speaker in female_names:
                recent_female = speaker
            else:
                recent_male = speaker

        attributed.append({
            "speaker": speaker,
            "named": speaker is not None,
            "text": content,
            "tone": tone,
            "narration": narration,
            "pos": quote_start,
        })

    # Pass 2: limited alternation
    # For unattributed lines, alternate between characters already named
    # in the current run of dialogue. Only works when exactly 2 characters
    # have been identified (otherwise we don't know who to alternate to).
    last_speaker = None
    named_in_run = set()

    for entry in attributed:
        if entry["speaker"] is not None:
            last_speaker = entry["speaker"]
            named_in_run.add(entry["speaker"])
        elif last_speaker is not None and len(named_in_run) == 2:
            # Alternate to the other named character
            other = [s for s in named_in_run if s != last_speaker]
            if other:
                entry["speaker"] = other[0]
                last_speaker = entry["speaker"]
        else:
            # Can't resolve - reset the run
            last_speaker = None
            named_in_run = set()

        # Reset run tracking on large gaps
        if entry["speaker"] is None:
            last_speaker = None
            named_in_run = set()

    # Pass 3: group into exchanges
    exchanges = []
    current = []

    for entry in attributed:
        if entry["speaker"] is None:
            if len(current) >= 2:
                exchanges.append(current)
            current = []
            continue

        if current and entry["pos"] - current[-1]["pos"] > 500:
            if len(current) >= 2:
                exchanges.append(current)
            current = []

        current.append(entry)

    if len(current) >= 2:
        exchanges.append(current)

    # Pass 3: filter - require at least 2 different speakers
    filtered = []
    for exchange in exchanges:
        speakers = set(e["speaker"] for e in exchange)
        if len(speakers) >= 2:
            filtered.append(exchange)

    return filtered


def format_for_training(exchanges, config, include_narration=True, include_tone=True):
    """Format exchanges with <character_name> tags."""
    formatted = []
    for exchange in exchanges:
        lines = []
        for entry in exchange:
            if include_narration and entry.get("narration"):
                lines.append(f"<narration>{entry['narration']}")

            name = entry["speaker"].lower().replace(' ', '_')
            if include_tone and entry.get("tone"):
                tag = f"<{name}:{entry['tone']}>"
            else:
                tag = f"<{name}>"

            lines.append(f"{tag}{entry['text']}")

        formatted.append("\n".join(lines))

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Extract dialogue from literature")
    parser.add_argument('--config', default='wodehouse',
                        choices=list(CONFIGS.keys()),
                        help="Which config to use")
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

    exchanges = extract_dialogues(text, config)
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
    print(f"\nCharacters found ({len(char_counts)}):")
    for name, count in sorted(char_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  <{name.lower()}>: {count} turns")
    if len(char_counts) > 20:
        print(f"  ... and {len(char_counts) - 20} more")

    if tone_counts:
        print(f"\nTones:")
        for tone, count in sorted(tone_counts.items(), key=lambda x: -x[1]):
            print(f"  {tone}: {count}")

    print(f"Narration lines: {narration_count}")

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

    formatted = format_for_training(exchanges, config, include_narration, include_tone)
    with open(output_file, 'w') as f:
        f.write("\n\n".join(formatted))
    print(f"\nSaved {len(formatted)} exchanges to {output_file}")


if __name__ == '__main__':
    main()
