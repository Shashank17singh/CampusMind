"""Converter for legacy Hindi typewriter fonts (Kruti Dev 010 / Chanakya) to Unicode Devanagari."""

import re

# Characteristic KrutiDev frequent words:
# gS (है), esa (में), dk (का), dh (की), ds (के), dks (को), vkSj (और),
# gks (हो), Fkk (था), Fks (थे), Fkh (थी), ugha (नहीं), ;g (यह), og (वह)
_KRUTIDEV_MARKERS = re.compile(
    r"\b(gS|esa|dk|dh|ds|dks|vkSj|gks|Fkk|Fks|Fkh|ugha|;g|og|fd|ls|ij)\b"
)

# Multi-character mappings (ordered by length descending to match longest first)
_MULTI_CHAR_MAP = [
    ("vkS", "औ"),
    ("vks", "ओ"),
    ("vk", "आ"),
    ("bZ", "ई"),
    ("=k", "त्र"),
    ("?k", "घ"),
    (".k", "ण"),
    ("Fk", "थ"),
    ("/k", "ध"),
    ("Hk", "भ"),
    ("'k", "श"),
    ('"k', "ष"),
    ("[k", "ख"),
    ("ks", "ो"),
    ("kS", "ौ"),
    ("ZZ", "र्"),
]

_SINGLE_CHAR_MAP = {
    "k": "ा",
    "h": "ी",
    "q": "ु",
    "w": "ू",
    "`": "ृ",
    "s": "े",
    "S": "ै",
    "a": "ं",
    "A": "ँ",
    "%": ":",
    "d": "क",
    "x": "ग",
    "p": "च",
    "N": "छ",
    "t": "ज",
    "T": "झ",
    "V": "ट",
    "B": "ठ",
    "M": "ड",
    "<": "ढ",
    "r": "त",
    "n": "द",
    "u": "न",
    "i": "प",
    "Q": "फ",
    "c": "ब",
    "e": "म",
    ";": "य",
    "j": "र",
    "y": "ल",
    "o": "व",
    "l": "स",
    "g": "ह",
    "K": "ज्ञ",
    "v": "अ",
    "b": "इ",
    "m": "उ",
    "Å": "ऊ",
    ",": "ए",
    "D": "क्",
    "X": "ग्",
    "P": "च्",
    "R": "त्",
    "U": "न्",
    "I": "प्",
    "C": "ब्",
    "E": "म्",
    "Y": "ल्",
    "O": "व्",
    "L": "स्",
    "~": "्",
    "+": "़",
    "0": "०",
    "1": "१",
    "2": "२",
    "3": "३",
    "4": "४",
    "5": "५",
    "6": "६",
    "7": "७",
    "8": "८",
    "9": "९",
}


def is_krutidev(text: str) -> bool:
    """Return True if text appears to be encoded in legacy KrutiDev font."""
    if not text:
        return False
    matches = _KRUTIDEV_MARKERS.findall(text)
    # If 3 or more distinctive KrutiDev words match in the block
    return len(matches) >= 3


def krutidev_to_unicode(text: str) -> str:
    """Convert KrutiDev encoded string to Unicode Devanagari."""
    result = text

    # Handle 'f' matra (chhoti 'i' matra which appears BEFORE consonant in KrutiDev)
    # e.g. 'fd' -> 'क' + 'ि' = 'कि', 'f'k' -> 'श' + 'ि' = 'शि'
    result = re.sub(r"f(['\"\[]?[a-zA-Z])", r"\1ि", result)

    for src, dst in _MULTI_CHAR_MAP:
        result = result.replace(src, dst)

    out = []
    for char in result:
        out.append(_SINGLE_CHAR_MAP.get(char, char))
    converted = "".join(out)

    # Reph 'Z' (half 'r' on top of following letter)
    converted = re.sub(r"([क-ह])Z", r"र्\1", converted)
    converted = converted.replace("Z", "र्")

    return converted
