""" Telugu text front-end with number normalisation and a rule-based G2P.

Telugu orthography is close to phonemic, so a deterministic Unicode parser is
enough to obtain a phone sequence.

  * every consonant carries the inherent vowel /a/ unless followed by a
    dependent vowel sign (matra) or the virama (halant),
  * Telugu words do not delete the inherent vowel word-finally (unlike Hindi),
  * the anusvara (U+0C02) is realised as the nasal homorganic with the next
    consonant (word-finally / before sonorants it is /m/),
  * the visarga is realised as /h/.

Phone labels are lowercase ASCII (retroflexes carry an 'x', aspirates an 'h')
so they never collide with the upper-case ARPAbet symbols used for English.
"""

import re

# Phone set
VOWEL_PHONES = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
CONSONANT_PHONES = [
    "k", "kh", "g", "gh", "ng",
    "c", "ch", "j", "jh", "ny",
    "tx", "txh", "dx", "dxh", "nx",
    "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "l", "lx", "v", "sh", "sx", "s", "h",
]
TELUGU_PHONES = VOWEL_PHONES + CONSONANT_PHONES

INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
MATRAS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"],
    "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tx", "ఠ": "txh", "డ": "dx", "ఢ": "dxh", "ణ": "nx",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "lx", "ఴ": "lx", "వ": "v",
    "శ": "sh", "ష": "sx", "స": "s", "హ": "h", "ౘ": "c", "ౙ": "j",
}
VIRAMA, ANUSVARA, VISARGA, CANDRABINDU = "్", "ం", "ః", "ఁ"
IGNORED = {"‌", "‍", "﻿", "ఀ", "఼"}  # ZWNJ, ZWJ, BOM, nukta...

_HOMORGANIC = {}
for _nasal, _group in [
    ("ng", ["k", "kh", "g", "gh", "ng"]),
    ("ny", ["c", "ch", "j", "jh", "ny"]),
    ("nx", ["tx", "txh", "dx", "dxh", "nx"]),
    ("n", ["t", "th", "d", "dh", "n"]),
    ("m", ["p", "ph", "b", "bh", "m"]),
]:
    for _c in _group:
        _HOMORGANIC[_c] = _nasal


def word_to_phones(word):
    """Convert a single Telugu word (no spaces/punctuation) to a phone list."""
    chars = [c for c in word if c not in IGNORED]
    phones = []
    pending_anusvara = []  # indices in `phones` whose nasal depends on what follows
    i = 0
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else None
        if c in CONSONANTS:
            cons = CONSONANTS[c]
            for k in pending_anusvara:
                phones[k] = _HOMORGANIC.get(cons, "m")
            pending_anusvara = []
            phones.append(cons)
            if nxt == VIRAMA:
                i += 2
                continue
            if nxt in MATRAS:
                phones.extend(MATRAS[nxt])
                i += 2
                continue
            phones.append("a")  # inherent vowel
        elif c in INDEPENDENT_VOWELS:
            pending_anusvara = []
            phones.extend(INDEPENDENT_VOWELS[c])
        elif c == ANUSVARA:
            pending_anusvara.append(len(phones))
            phones.append("m")
        elif c == VISARGA:
            phones.append("h")
        elif c == CANDRABINDU:
            pass  # nasalisation of the preceding vowel, not modelled
        # anything else (a stray matra, digit or Latin letter) is dropped. Digits
        # have already been expanded by normalize_numbers().
        i += 1
    return phones


# Number expansion
_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు",
         "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు",
         "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై",
         8: "ఎనభై", 9: "తొంభై"}


def _below_100(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("" if o == 0 else " " + _ONES[o])


def _below_1000(n):
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    if h == 1:
        head = "వంద" if r == 0 else "నూట"
    else:
        head = _ONES[h] + (" వందలు" if r == 0 else " వందల")
    return head if r == 0 else head + " " + _below_100(r)


def number_to_words(n):
    """Cardinal number to Telugu words (0 <= n < 10,000,000)."""
    n = int(n)
    if n < 1000:
        return _below_1000(n)
    if n < 100000:
        th, r = divmod(n, 1000)
        if th == 1:
            head = "వెయ్యి"
        else:
            head = _below_100(th) + (" వేలు" if r == 0 else " వేల")
        return head if r == 0 else head + " " + _below_1000(r)
    lakh, r = divmod(n, 100000)
    head = ("ఒక లక్ష" if lakh == 1 else _below_100(lakh) + " లక్షల")
    return head if r == 0 else head + " " + number_to_words(r)


def _ordinal(words):
    """Turn cardinal words into the ordinal stem, e.g. పన్నెండు becomes పన్నెండవ."""
    if words[-1] in ("ు", "ి"):
        return words[:-1] + "వ"
    return words + "వ"


def normalize_numbers(text):
    # times, e.g. "10:45" becomes "పది గంటల నలభై ఐదు నిమిషాల" plus the case suffix of the next word
    def _time(m):
        hh, mm, suffix = int(m.group(1)), int(m.group(2)), m.group(3) or ""
        if suffix.startswith("గంటల"):  # in "10:45 గంటలకు" the case ending moves to the minutes
            case = suffix[len("గంటల"):]
            out = f"{number_to_words(hh)} గంటల"
            return out + (f" {number_to_words(mm)} నిమిషాల{case}" if mm else case)
        out = f"{number_to_words(hh)} గంటల"
        return out + (f" {number_to_words(mm)} నిమిషాలు" if mm else "") + (" " + suffix if suffix else "")

    text = re.sub(r"(\d{1,2}):(\d{2})\s*(గంటల\S*)?", _time, text)
    # ordinals, e.g. "12వ" becomes "పన్నెండవ"
    text = re.sub(r"(\d+)వ", lambda m: _ordinal(number_to_words(m.group(1))), text)
    # remaining cardinals
    text = re.sub(r"\d+", lambda m: number_to_words(m.group(0)), text)
    return text


_PUNCT = r"[,;:.!?\-\"'“”‘’()\[\]]"


def clean_telugu(text):
    text = "".join(c for c in text if c not in IGNORED)
    text = normalize_numbers(text)
    return re.sub(r"\s+", " ", text).strip()


def words_for_alignment(text):
    """Normalised word list (punctuation stripped) used for MFA .lab files."""
    text = re.sub(_PUNCT, " ", clean_telugu(text))
    return text.split()


def text_to_phone_groups(text):
    """Return a list of phone lists, with ['sp'] inserted at punctuation (inference)."""
    text = clean_telugu(text)
    groups = []
    for tok in re.split(r"(" + _PUNCT + r"|\s+)", text):
        if not tok or tok.isspace():
            continue
        if re.fullmatch(_PUNCT, tok):
            if groups and groups[-1] != ["sp"]:
                groups.append(["sp"])
            continue
        ph = word_to_phones(tok)
        if ph:
            groups.append(ph)
    while groups and groups[-1] == ["sp"]:
        groups.pop()
    return groups
