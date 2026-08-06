#!/usr/bin/env python3
"""
Report Dashboard (plain text)

Reads a plain-text version of your report and generates graphs without needing
chapter/section parsing. It uses sliding word-windows across the document.

Outputs (PNG) to: ./report_dashboard_output/
  1) requirements_density.png       (shall/must/required/etc per window)
  2) term_trends.png                (chosen term frequencies per window)
  3) acronym_first_appearance.png   (first appearance position + total count)

How it works:
- Tokenizes words (case-insensitive).
- Builds overlapping windows of N words (default 800) stepping by M (default 200).
- Computes metrics per window.
- Plots and saves charts.

Dependencies:
  pip install matplotlib

Run:
  python report_dashboard.py
  (select your .txt file in the dialog)
"""

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename


@dataclass
class Settings:
    # Sliding window configuration (word-based)
    window_words: int = 800
    step_words: int = 200

    # Output configuration
    output_dir: str = "report_dashboard_output"

    # Requirements / modal language
    requirement_patterns: Tuple[str, ...] = (
        r"\bshall\b",
        r"\bmust\b",
        r"\brequired\b",
        r"\bshall\s+not\b",
        r"\bmust\s+not\b",
        r"\bwill\b",
        r"\bshould\b",
    )

    # Term trends (edit this list to match your report vocabulary)
    # Keep terms lower-case; matching is case-insensitive
    terms_to_track: Tuple[str, ...] = (
        "rov",
        "sensor",
        "camera",
        "stereo",
        "lighting",
        "sonar",
        "navigation",
        "ethernet",
        "udp",
        "tcp",
        "pressure",
        "vessel",
        "connector",
        "power",
        "cec",
        "calibration",
    )

    # Acronym detection heuristic:
    # sequences like ROV, FOSW, CEC, IP, TCP, UDP, etc.
    acronym_regex: str = r"\b[A-Z][A-Z0-9]{1,9}\b"  # 2-10 chars, starts with A-Z

    # Ignore these as "acronyms" (too generic / noisy)
    acronym_stoplist: Tuple[str, ...] = (
        "A", "AN", "AND", "AS", "AT", "BE", "BY", "DO", "FOR", "FROM", "IF", "IN",
        "IS", "IT", "NO", "NOT", "OF", "ON", "OR", "THE", "TO", "US", "WE", "WITH",
    )

    # Limit acronyms displayed to top N by total count
    top_acronyms: int = 40


def pick_text_file() -> str:
    root = Tk()
    root.withdraw()
    root.update()
    path = askopenfilename(
        title="Select the report plain-text file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def tokenize_words(text: str) -> List[str]:
    # Words are letters/digits and optional internal apostrophes (e.g., don't)
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def build_windows(tokens: List[str], window_words: int, step_words: int) -> List[Tuple[int, int]]:
    if window_words <= 0 or step_words <= 0:
        raise ValueError("window_words and step_words must be positive integers.")
    if len(tokens) < window_words:
        return [(0, len(tokens))]

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + window_words)
        windows.append((start, end))
        if end == len(tokens):
            break
        start += step_words
    return windows


def count_requirements_in_window(window_text: str, patterns: Tuple[str, ...]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pat in patterns:
        # compile each time is fine at this scale; if huge, precompile
        counts[pat] = len(re.findall(pat, window_text, flags=re.IGNORECASE))
    return counts


def compute_window_metrics(
    tokens: List[str],
    windows: List[Tuple[int, int]],
    settings: Settings,
) -> Dict[str, List[float]]:
    # Pre-join each window to a string once for regex counts
    requirement_total_per_window: List[float] = []
    requirement_per_1k_words: List[float] = []

    term_series: Dict[str, List[float]] = {t: [] for t in settings.terms_to_track}

    for (start, end) in windows:
        w_tokens = tokens[start:end]
        w_len = max(1, len(w_tokens))
        w_text = " ".join(w_tokens)

        req_counts = count_requirements_in_window(w_text, settings.requirement_patterns)
        total_req = float(sum(req_counts.values()))
        requirement_total_per_window.append(total_req)
        requirement_per_1k_words.append(total_req * 1000.0 / w_len)

        w_counter = Counter(w_tokens)
        for term in settings.terms_to_track:
            # normalized per 1k words so window length differences don’t skew
            term_series[term].append(w_counter[term] * 1000.0 / w_len)

    metrics: Dict[str, List[float]] = {
        "requirement_total": requirement_total_per_window,
        "requirement_per_1k": requirement_per_1k_words,
    }
    for term, series in term_series.items():
        metrics[f"term::{term}"] = series

    return metrics


def extract_acronyms(text: str, settings: Settings) -> Tuple[Counter[str], Dict[str, int]]:
    # Count acronyms and find their first occurrence position (character index)
    acr_counts: Counter[str] = Counter()
    first_pos: Dict[str, int] = {}

    for m in re.finditer(settings.acronym_regex, text):
        acr = m.group(0)
        if acr in settings.acronym_stoplist:
            continue

        acr_counts[acr] += 1
        if acr not in first_pos:
            first_pos[acr] = m.start()

    return acr_counts, first_pos


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_requirements(windows: List[Tuple[int, int]], metrics: Dict[str, List[float]], out_path: str) -> None:
    x = list(range(len(windows)))

    plt.figure()
    plt.plot(x, metrics["requirement_per_1k"])
    plt.title("Requirement language density (per 1000 words)")
    plt.xlabel("Window index (start to end of document)")
    plt.ylabel("Count per 1000 words")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_term_trends(windows: List[Tuple[int, int]], metrics: Dict[str, List[float]], terms: Tuple[str, ...], out_path: str) -> None:
    x = list(range(len(windows)))

    plt.figure()
    for term in terms:
        plt.plot(x, metrics[f"term::{term}"], label=term)

    plt.title("Term trends (per 1000 words)")
    plt.xlabel("Window index (start to end of document)")
    plt.ylabel("Occurrences per 1000 words")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_acronym_timeline(
    acr_counts: Counter[str],
    first_pos: Dict[str, int],
    text_len: int,
    top_n: int,
    out_path: str
) -> None:
    # Select top acronyms by count
    items = [(acr, acr_counts[acr], first_pos.get(acr, None)) for acr in acr_counts]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:top_n]

    # Map to x = first occurrence as percent of document, y = index
    y_labels = [acr for (acr, _, _) in items]
    y = list(range(len(items)))

    x = []
    sizes = []
    for (acr, count, pos) in items:
        if pos is None:
            pos = 0
        pct = 100.0 * (pos / max(1, text_len))
        x.append(pct)
        # marker size proportional to sqrt(count) to compress range
        sizes.append(20.0 + 30.0 * math.sqrt(count))

    plt.figure()
    plt.scatter(x, y, s=sizes)
    plt.title("Acronym first appearance timeline (size ~ total count)")
    plt.xlabel("First appearance (% of document)")
    plt.yticks(y, y_labels)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    settings = Settings()

    path = pick_text_file()
    if not path:
        print("No file selected. Exiting.")
        return

    raw_text = read_text(path)
    tokens = tokenize_words(raw_text)

    if len(tokens) < 50:
        print("File is very short after tokenization. Are you sure this is the report text?")
        return

    windows = build_windows(tokens, settings.window_words, settings.step_words)
    metrics = compute_window_metrics(tokens, windows, settings)

    acr_counts, first_pos = extract_acronyms(raw_text, settings)

    ensure_output_dir(settings.output_dir)

    req_out = os.path.join(settings.output_dir, "requirements_density.png")
    terms_out = os.path.join(settings.output_dir, "term_trends.png")
    acr_out = os.path.join(settings.output_dir, "acronym_first_appearance.png")

    plot_requirements(windows, metrics, req_out)
    plot_term_trends(windows, metrics, settings.terms_to_track, terms_out)
    plot_acronym_timeline(acr_counts, first_pos, len(raw_text), settings.top_acronyms, acr_out)

    print("Done.")
    print(f"Input: {path}")
    print(f"Tokens: {len(tokens)}")
    print(f"Windows: {len(windows)}  (window={settings.window_words} words, step={settings.step_words} words)")
    print(f"Output folder: {os.path.abspath(settings.output_dir)}")
    print("Generated:")
    print(f"  {req_out}")
    print(f"  {terms_out}")
    print(f"  {acr_out}")
    print("")
    print("Next tweaks you likely want:")
    print("  - Edit Settings.terms_to_track to match your actual subsystem vocabulary.")
    print("  - Increase window_words for smoother curves; decrease for more localized detail.")
    print("  - If acronym list is noisy, add items to acronym_stoplist.")


if __name__ == "__main__":
    main()