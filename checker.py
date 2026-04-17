"""
Text quality checker for generated output.

Scores generated text against the training corpus using layered checks:
  Layer 1: Word check       - is every word real?
  Layer 2: N-gram check     - is the word order author-like?
  Layer 3: Repetition check - is it repeating itself?
  Layer 4: Distinctive vocab - does it use author-specific words?

Pluggable: swap training text for a different author/style.

Usage:
    # Standalone
    python3 checker.py "Jeeves entered the room and shimmered"

    # From code
    from checker import Checker
    c = Checker("data.txt")
    result = c.score("Jeeves entered the room")
    print(result["overall"], result["failures"])
"""

import re
import sys
from collections import Counter


class Checker:

    def __init__(self, training_text_path, ngram_n=3):
        with open(training_text_path, 'r') as f:
            text = f.read()

        words = self._tokenize(text)
        self.ngram_n = ngram_n

        # Layer 1: all known words
        self.known_words = set(w.lower() for w in words)

        # Layer 2: all n-grams from training text
        lowered = [w.lower() for w in words]
        self.known_ngrams = set()
        for i in range(len(lowered) - ngram_n + 1):
            self.known_ngrams.add(tuple(lowered[i:i + ngram_n]))

        # Layer 4: distinctive vocabulary
        # Words that appear frequently in training text but are unusual in
        # general English. We approximate "unusual" by word length and
        # frequency - short common words like "the" aren't distinctive.
        word_counts = Counter(w.lower() for w in words)
        total = len(words)
        self.distinctive_words = set()
        for word, count in word_counts.items():
            freq = count / total
            # Distinctive = appears often enough to matter (>0.001%)
            # but not so common it's just English (< 0.5%)
            # and is at least 4 chars (filters "the", "and", etc.)
            if freq > 0.00001 and freq < 0.005 and len(word) >= 4:
                self.distinctive_words.add(word)

        print(f"Checker loaded: {len(self.known_words):,} words, "
              f"{len(self.known_ngrams):,} {ngram_n}-grams, "
              f"{len(self.distinctive_words):,} distinctive words")

    def _tokenize(self, text):
        """Split text into words. Simple whitespace + punctuation split."""
        return re.findall(r"[a-zA-Z']+", text)

    def score(self, text):
        """
        Score generated text. Returns dict with:
            overall:    float 0-1 (weighted average of all layers)
            word:       float 0-1 (fraction of known words)
            ngram:      float 0-1 (fraction of known n-grams)
            repetition: float 0-1 (1.0 = no repetition, 0.0 = all repeated)
            distinctive: float 0-1 (fraction of distinctive vocab used)
            failures:   list of specific problems found
        """
        words = self._tokenize(text)
        if not words:
            return {"overall": 0, "word": 0, "ngram": 0,
                    "repetition": 0, "distinctive": 0, "failures": ["empty text"]}

        failures = []

        # Layer 1: Word check
        lowered = [w.lower() for w in words]
        unknown = [w for w in lowered if w not in self.known_words]
        word_score = 1.0 - (len(unknown) / len(lowered))
        if unknown:
            unique_unknown = list(set(unknown))[:5]
            failures.append(f"unknown_words: {', '.join(unique_unknown)}")

        # Layer 2: N-gram check
        if len(lowered) >= self.ngram_n:
            ngrams = [tuple(lowered[i:i + self.ngram_n])
                      for i in range(len(lowered) - self.ngram_n + 1)]
            matched = sum(1 for ng in ngrams if ng in self.known_ngrams)
            ngram_score = matched / len(ngrams)
            unmatched = [ng for ng in ngrams if ng not in self.known_ngrams][:3]
            if ngram_score < 0.5:
                failures.append(
                    f"low_ngram_match ({ngram_score:.0%}): "
                    f"{[' '.join(ng) for ng in unmatched]}"
                )
        else:
            ngram_score = 1.0

        # Layer 3: Repetition check
        rep_score, rep_failures = self._check_repetition(words)
        failures.extend(rep_failures)

        # Layer 4: Distinctive vocabulary
        text_distinctive = set(lowered) & self.distinctive_words
        # What fraction of the text's unique words are distinctive?
        unique_words = set(lowered)
        non_trivial = {w for w in unique_words if len(w) >= 4}
        if non_trivial:
            distinctive_score = len(text_distinctive) / len(non_trivial)
        else:
            distinctive_score = 0.0

        # Overall: weighted average
        overall = (
            word_score * 0.25 +
            ngram_score * 0.35 +
            rep_score * 0.25 +
            distinctive_score * 0.15
        )

        return {
            "overall": round(overall, 3),
            "word": round(word_score, 3),
            "ngram": round(ngram_score, 3),
            "repetition": round(rep_score, 3),
            "distinctive": round(distinctive_score, 3),
            "failures": failures,
        }

    def _check_repetition(self, words):
        """Check for repeated words and phrases."""
        failures = []
        score = 1.0

        # Consecutive word repetition: "the the the"
        for i in range(len(words) - 2):
            if words[i].lower() == words[i + 1].lower() == words[i + 2].lower():
                failures.append(f"triple_repeat: '{words[i]}' x3")
                score -= 0.2
                break

        # Phrase repetition within a window
        # Check if any 4-word phrase appears 3+ times
        if len(words) >= 4:
            phrase_counts = Counter()
            for i in range(len(words) - 3):
                phrase = ' '.join(w.lower() for w in words[i:i + 4])
                phrase_counts[phrase] += 1

            for phrase, count in phrase_counts.most_common(3):
                if count >= 3:
                    failures.append(f"repeated_phrase: '{phrase}' x{count}")
                    score -= 0.15 * (count - 2)

        return max(0, score), failures

    def score_verbose(self, text):
        """Score and print a human-readable report."""
        result = self.score(text)
        print(f"\n{'='*60}")
        print(f"Overall: {result['overall']:.0%}")
        print(f"  Words known:    {result['word']:.0%}")
        print(f"  N-gram match:   {result['ngram']:.0%}")
        print(f"  No repetition:  {result['repetition']:.0%}")
        print(f"  Distinctive:    {result['distinctive']:.0%}")
        if result['failures']:
            print(f"\nIssues:")
            for f in result['failures']:
                print(f"  - {f}")
        else:
            print(f"\nNo issues found.")
        print(f"{'='*60}")
        return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 checker.py \"text to check\"")
        print("       python3 checker.py --file output.txt")
        sys.exit(1)

    checker = Checker('data.txt')

    if sys.argv[1] == '--file':
        with open(sys.argv[2]) as f:
            text = f.read()
    else:
        text = ' '.join(sys.argv[1:])

    print(f"\nChecking: {text[:100]}{'...' if len(text) > 100 else ''}")
    checker.score_verbose(text)
