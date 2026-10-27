"""Defence against evasion: normalise a message before any rule or model looks at it.

The same idea as traffic normalisation in an intrusion detection system (for example Snort's
preprocessors): undo the encodings an attacker uses, so signatures see the real content.
The ORIGINAL text is always kept for display; only the detector reads the normalised copy.

normalise(text) returns (clean_text, tricks_found). tricks_found lists the evasion tricks seen,
because hiding words from a filter is itself a warning sign.
"""
import re
import unicodedata

# Cyrillic and Greek look-alikes back to Latin
CONFUSABLE = {
    "а": "a", "с": "c", "е": "e", "і": "i", "ј": "j", "о": "o", "р": "p", "ѕ": "s", "х": "x", "у": "y", "һ": "h", "ԁ": "d",
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J", "К": "K", "М": "M", "О": "O", "Р": "P",
    "Ѕ": "S", "Т": "T", "Х": "X", "У": "Y",
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "τ": "t", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
}
CONF_RE = re.compile("[" + "".join(CONFUSABLE) + "]")
DELEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
# A run of 3+ single letters split by one kind of separator: "p a y", "v.e.r.i.f.y", "O-T-P"
SPLIT_RE = re.compile(r"(?<![\w])((?:[A-Za-z][ ._*\-]){2,}[A-Za-z])(?![\w])")
# A token mixing letters with leet symbols: "0TP", "p@y", "g00gle", "sh@r3" (not "2000" or "XX4521")
LEET_TOKEN = re.compile(r"(?<![\w@$])[\w@$]*[A-Za-z][\w@$]*(?![\w@$])")


# Tokens that are normal text, not leetspeak: amounts and units ($25K, 5pm, 2nd, 10x),
# names with a trailing number (Web3, Covid19, mp3, XX4521) and codes like 9a8b7c.
KEEP_TOKEN = re.compile(r"^(\$?\d[\d,.]*(am|pm|st|nd|rd|th|k|m|gb|mb|kg|km|x|rs|hrs?|min|s)?|[a-z]+\d+|\d+[a-z]\d+\w*)$", re.I)


def _deleet(tok):
    letters = sum(ch.isalpha() for ch in tok)
    symbols = sum(ch in DELEET for ch in tok)
    if symbols == 0 or letters == 0 or KEEP_TOKEN.match(tok):   # 5pm, 2nd, 10k, XX4521, UPI refs stay intact
        return tok
    if letters < symbols and len(tok) > 4:                       # mostly a number: an amount or account number
        return tok
    return "".join(DELEET.get(ch, ch) for ch in tok)


def normalise(text: str):
    tricks = []
    s = unicodedata.normalize("NFKC", str(text))   # full-width and styled letters to plain ones

    no_invisible = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    if no_invisible != s:
        tricks.append("invisible characters")
    s = no_invisible

    if CONF_RE.search(s) and re.search(r"[A-Za-z]", s):   # look-alikes mixed with normal Latin text
        tricks.append("look-alike letters")
        s = CONF_RE.sub(lambda m: CONFUSABLE[m.group(0)], s)

    def join_split(m):
        seps = set(m.group(1)[1::2])
        return m.group(1)[::2] if len(seps) == 1 else m.group(1)
    joined = SPLIT_RE.sub(join_split, s)
    if joined != s:
        tricks.append("split-up letters")
    s = joined

    def deleet_in_place(m):
        tok, before, after = m.group(0), s[m.start() - 1:m.start()], s[m.end():m.end() + 1]
        if "@" in tok and (before == "." or after == "."):
            return tok   # part of an email address or UPI ID (rahul.k@okicici, @www.swiggy.com), not leetspeak
        return _deleet(tok)
    deleeted = LEET_TOKEN.sub(deleet_in_place, s)
    if deleeted != s:
        tricks.append("numbers or symbols in place of letters")
    s = deleeted

    refanged = re.sub(r"\bhxxp", "http", s, flags=re.I)
    refanged = re.sub(r"\s*(\[\.\]|\(\.\)|\[dot\]|\(dot\))\s*", ".", refanged, flags=re.I)
    refanged = re.sub(r"\b([\w-]+) dot (com|net|org|in|io|ly|xyz|info|link|top|co|me|ru)\b", r"\1.\2", refanged, flags=re.I)
    if refanged != s:
        tricks.append("disguised link")
    s = refanged
    return s, tricks
