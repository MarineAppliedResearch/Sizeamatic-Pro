#!/usr/bin/env python3
"""
Word cloud generator for a plain text report, excluding common English words.

Outputs to: ./report_wordcloud_output/
  - wordcloud.png
  - top_words.png

Install:
  pip install wordcloud matplotlib

Run:
  python report_wordcloud.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from collections import Counter
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
from wordcloud import WordCloud


@dataclass
class Settings:
    output_dir: str = "report_wordcloud_output"
    max_words: int = 250
    min_word_len: int = 3
    drop_numbers: bool = True

    # Add domain specific stopwords here (things that will dominate but aren’t informative)
    extra_stopwords: tuple[str, ...] = (
        "figure", "table", "section", "chapter",
        "shall", "must", "should",
    )


# A solid default English stopword set (kept local so you don’t need nltk).
ENGLISH_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each",
    "few","for","from","further",
    "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's",
    "me","more","most","mustn't","my","myself",
    "no","nor","not",
    "of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these",
    "they","they'd","they'll","they're","they've","this","those","through","to","too",
    "under","until","up",
    "very",
    "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's",
    "where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves",
}


def pick_text_file() -> str:
    root = Tk()
    root.withdraw()
    root.update()
    path = askopenfilename(
        title="Select your report plain-text file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    # Keep words and contractions; drop punctuation.
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    settings = Settings()

    path = pick_text_file()
    if not path:
        print("No file selected. Exiting.")
        return

    text = read_text(path)
    tokens = tokenize(text)

    stopwords = set(ENGLISH_STOPWORDS)
    stopwords.update(w.lower() for w in settings.extra_stopwords)

    filtered: list[str] = []
    for w in tokens:
        if len(w) < settings.min_word_len:
            continue
        if w in stopwords:
            continue
        if settings.drop_numbers and w.isdigit():
            continue
        filtered.append(w)

    if not filtered:
        print("No words left after filtering. Loosen filters or check input.")
        return

    counts = Counter(filtered)

    ensure_dir(settings.output_dir)

    # Word cloud
    wc = WordCloud(
        width=2000,
        height=1200,
        background_color="white",
        max_words=settings.max_words,
        collocations=False,  # avoids common bigrams overwhelming results
    ).generate_from_frequencies(counts)

    wordcloud_path = os.path.join(settings.output_dir, "wordcloud.png")
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(wordcloud_path, dpi=200)
    plt.close()

    # Top words bar chart (top 30)
    top_n = 30
    top_items = counts.most_common(top_n)
    labels = [w for (w, _) in top_items][::-1]
    values = [c for (_, c) in top_items][::-1]

    topwords_path = os.path.join(settings.output_dir, "top_words.png")
    plt.figure()
    plt.barh(labels, values)
    plt.title(f"Top {top_n} words (stopwords removed)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(topwords_path, dpi=200)
    plt.close()

    print("Done.")
    print(f"Input: {path}")
    print(f"Tokens: {len(tokens)}")
    print(f"Filtered tokens: {len(filtered)}")
    print(f"Output folder: {os.path.abspath(settings.output_dir)}")
    print(f"Generated: {wordcloud_path}")
    print(f"Generated: {topwords_path}")
    print("")
    print("Tip: add domain-specific junk words to Settings.extra_stopwords to clean it up.")


if __name__ == "__main__":
    main()