"""Evasion attacks on message filters: the tricks real scammers use to slip past keyword and ML filters.

Each attack keeps the message readable for a human but changes the characters a filter sees,
the same way attackers fragment or encode packets to slip past an intrusion detection system.
Attacks are seeded, so every run produces the same evasive messages.
"""
import random
import re

# Latin letters and their Cyrillic / Greek look-alikes (they render the same in most fonts)
HOMOGLYPHS = {"a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "x": "х", "y": "у",
              "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х"}
LEET = {"o": "0", "O": "0", "a": "@", "A": "4", "i": "1", "I": "1", "e": "3", "E": "3", "s": "$", "S": "5", "t": "7"}
ZWSP = "​"  # zero-width space: invisible, but splits the word for a filter
WORD = re.compile(r"[A-Za-z]{3,}")


def _per_word(text, fn, rate, seed):
    rng = random.Random(seed)
    return WORD.sub(lambda m: fn(m.group(0), rng) if rng.random() < rate else m.group(0), text)


def leetspeak(text, rate=0.6, seed=0):
    """'Share your OTP' -> 'Sh@r3 y0ur 0TP'"""
    return _per_word(text, lambda w, r: "".join(LEET.get(ch, ch) if r.random() < 0.7 else ch for ch in w), rate, seed)


def spacing(text, rate=0.5, seed=0):
    """'pay now' -> 'p a y now'"""
    return _per_word(text, lambda w, r: " ".join(w), rate, seed)


def punctuation(text, rate=0.5, seed=0):
    """'verify' -> 'v.e.r.i.f.y'"""
    return _per_word(text, lambda w, r: r.choice([".", "-", "_", "*"]).join(w), rate, seed)


def homoglyphs(text, rate=0.8, seed=0):
    """Swap Latin letters for identical-looking Cyrillic ones. Looks unchanged to a person."""
    return _per_word(text, lambda w, r: "".join(HOMOGLYPHS.get(ch, ch) if r.random() < 0.7 else ch for ch in w), rate, seed)


def zero_width(text, rate=0.8, seed=0):
    """Hide invisible characters inside words. Looks unchanged to a person."""
    return _per_word(text, lambda w, r: ZWSP.join(w), rate, seed)


def defang_links(text, rate=1.0, seed=0):
    """'bit.ly/x' -> 'bit[.]ly/x', 'https' -> 'hxxps', 'site.com' -> 'site dot com'"""
    rng = random.Random(seed)
    text = re.sub(r"\bhttp", "hxxp", text, flags=re.I)
    return re.sub(r"\b([\w-]+)\.(com|net|org|in|io|ly|xyz|info|link|top|co|me|ru)\b",
                  lambda m: f"{m.group(1)}{rng.choice(['[.]', '(.)', ' dot '])}{m.group(2)}", text)


def combined(text, seed=0):
    """What a determined attacker does: several tricks at once."""
    return zero_width(homoglyphs(leetspeak(defang_links(text, seed=seed), rate=0.4, seed=seed), rate=0.5, seed=seed + 1),
                      rate=0.3, seed=seed + 2)


ATTACKS = {
    "Leetspeak (0TP, p@y)": leetspeak,
    "Spaced letters (p a y)": spacing,
    "Punctuation (v.e.r.i.f.y)": punctuation,
    "Look-alike letters (Cyrillic)": homoglyphs,
    "Invisible characters": zero_width,
    "Disguised links (bit[.]ly)": defang_links,
    "All tricks combined": combined,
}
