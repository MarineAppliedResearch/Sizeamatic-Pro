#!/usr/bin/env python3
"""
Find the most common contiguous word sequences (n-grams) of length >= 3
in a text file, ranked from most common to least common.

Usage:
  python ngram_common_phrases.py
  (then pick a file in the file dialog)

Notes:
- Tokenization: words are sequences of letters/digits/apostrophes.
- Case-insensitive by default.
- Default: report the Top 200 phrases across all n >= 3, up to a max length.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from tkinter import Tk
from tkinter.filedialog import askopenfilename


@dataclass
class Settings:
    min_n: int = 3              # minimum phrase length (words)
    max_n: int = 12             # maximum phrase length to consider
    top_k: int = 200            # how many phrases to print
    min_count: int = 2          # only show phrases that appear at least this many times
    case_insensitive: bool = True


def pick_text_file() -> str:
    # Create a minimal Tk root solely for the file dialog.
    root = Tk()
    root.withdraw()
    root.update()
    path = askopenfilename(
        title="Select a text file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def read_text(path: str) -> str:
    # Read as UTF-8, replacing invalid bytes so the tool doesn’t crash on odd encodings.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def tokenize(text: str, case_insensitive: bool) -> list[str]:
    # Keep words and contractions; drop punctuation.
    if case_insensitive:
        text = text.lower()

    # Example matches: "don't", "rovs", "2026", "cec-epc" becomes "cec" "epc"
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def count_ngrams(tokens: list[str], min_n: int, max_n: int) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()

    # Count all contiguous n-grams for n in [min_n, max_n]
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            break

        # Sliding window
        for i in range(0, len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            counts[gram] += 1

    return counts


def choose_best_unique_phrases(
    counts: Counter[tuple[str, ...]],
    min_count: int,
    top_k: int
) -> list[tuple[int, str]]:
    """
    Returns phrases sorted by count desc, then length desc, then lexicographically.
    Also suppresses phrases that are fully contained inside a higher-ranked phrase
    with the same count (to reduce redundant outputs like:
      "in the" vs "in the future" (same count) etc.)
    """
    items = [
        (c, gram)
        for gram, c in counts.items()
        if c >= min_count
    ]

    items.sort(key=lambda x: (-x[0], -len(x[1]), x[1]))

    selected: list[tuple[int, tuple[str, ...]]] = []
    selected_set = set()

    # Group by count so we only do "contained suppression" within same-frequency phrases.
    by_count: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for c, gram in items:
        by_count[c].append(gram)

    # Iterate counts high-to-low in the same order as items.
    for c in sorted(by_count.keys(), reverse=True):
        grams = sorted(by_count[c], key=lambda g: (-len(g), g))

        # For this count bucket, keep longer phrases first.
        kept: list[tuple[str, ...]] = []
        for g in grams:
            if g in selected_set:
                continue

            # If g is contained in any already-kept phrase of same count, skip it.
            contained = False
            for k in kept:
                if is_subsequence_contiguous(g, k):
                    contained = True
                    break

            if not contained:
                kept.append(g)
                selected.append((c, g))
                selected_set.add(g)

            if len(selected) >= top_k:
                break

        if len(selected) >= top_k:
            break

    return [(c, " ".join(g)) for c, g in selected]


def is_subsequence_contiguous(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    """True if `shorter` appears as a contiguous slice inside `longer`."""
    if len(shorter) > len(longer):
        return False
    if len(shorter) == len(longer):
        return shorter == longer

    n = len(shorter)
    for i in range(0, len(longer) - n + 1):
        if longer[i:i + n] == shorter:
            return True
    return False


def main() -> None:
    settings = Settings()

    path = pick_text_file()
    if not path:
        print("No file selected. Exiting.")
        return

    text = read_text(path)
    tokens = tokenize(text, settings.case_insensitive)

    if len(tokens) < settings.min_n:
        print(f"Not enough tokens to form sequences of length {settings.min_n}.")
        return

    counts = count_ngrams(tokens, settings.min_n, settings.max_n)
    ranked = choose_best_unique_phrases(counts, settings.min_count, settings.top_k)

    print(f"\nFile: {path}")
    print(f"Tokens: {len(tokens)}")
    print(f"Counting n-grams for n = {settings.min_n}..{settings.max_n}")
    print(f"Showing up to {settings.top_k} phrases with count >= {settings.min_count}\n")

    for c, phrase in ranked:
        print(f"{c:6d}  {phrase}")


if __name__ == "__main__":
    main()