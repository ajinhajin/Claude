"""Character sets and the cells a user has to hand-write.

Hangul is not written syllable by syllable (there are 11,172 of them).
Instead each jamo is written once per *position variant* and the syllables
are composed automatically:

* 초성 (initial) – shape depends on the vowel type and whether there is a 받침
    V = vertical vowels (ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅣ)      e.g. 가 / 간
    H = horizontal vowels (ㅗ ㅛ ㅜ ㅠ ㅡ)              e.g. 고 / 곤
    C = compound vowels (ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ)           e.g. 과 / 관
* 중성 (vowel)   – with / without 받침                   e.g. 아 / 안
* 종성 (final)   – one set                               e.g. 각

Every Hangul cell on the template represents a full syllable box, so the
written jamo is used exactly where it was written – no scaling needed.
"""

from dataclasses import dataclass, field

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")  # index + 1 = T

V_VOWELS = {0, 1, 2, 3, 4, 5, 6, 7, 20}
H_VOWELS = {8, 12, 13, 17, 18}
C_VOWELS = {9, 10, 11, 14, 15, 16, 19}


def vowel_class(v: int) -> str:
    if v in V_VOWELS:
        return "V"
    if v in H_VOWELS:
        return "H"
    return "C"


def compose(l: int, v: int, t: int = 0) -> str:
    return chr(0xAC00 + (l * 21 + v) * 28 + t)


# Representative vowel used for the gray guide syllable of each initial set.
_CLASS_VOWEL = {"V": JUNG.index("ㅏ"), "H": JUNG.index("ㅗ"), "C": JUNG.index("ㅘ")}
_N = JONG.index("ㄴ") + 1
_IEUNG = CHO.index("ㅇ")
_GIYEOK = CHO.index("ㄱ")


@dataclass
class Cell:
    name: str            # glyph name in the font
    label: str           # character printed above the cell
    kind: str            # "latin" | "hangul"
    guide: str = ""      # light-gray guide text drawn inside the cell
    codepoint: int = 0   # for directly mapped glyphs (latin, punctuation)


@dataclass
class CellSet:
    key: str
    title: str
    cells: list = field(default_factory=list)


def cho_name(l: int, variant: str) -> str:
    return f"cho{l:02d}.{variant}"


def jung_name(v: int, variant: str) -> str:
    return f"jung{v:02d}.{variant}"


def jong_name(t: int) -> str:
    return f"jong{t:02d}"


def _cho_set(variant: str) -> CellSet:
    cls, fin = variant[0], variant[1] == "1"
    v = _CLASS_VOWEL[cls]
    names = {"V": "ㅏ형", "H": "ㅗ형", "C": "ㅘ형"}
    title = f"초성 · {names[cls]} · 받침{'있음' if fin else '없음'}"
    s = CellSet(f"cho.{variant}", title)
    for l, ch in enumerate(CHO):
        s.cells.append(Cell(cho_name(l, variant), ch, "hangul",
                            guide=compose(l, v, _N if fin else 0)))
    return s


def _jung_set(variant: str) -> CellSet:
    fin = variant == "1"
    s = CellSet(f"jung.{variant}", f"중성 · 받침{'있음' if fin else '없음'}")
    for v, ch in enumerate(JUNG):
        s.cells.append(Cell(jung_name(v, variant), ch, "hangul",
                            guide=compose(_IEUNG, v, _N if fin else 0)))
    return s


def _jong_set() -> CellSet:
    s = CellSet("jong", "종성(받침)")
    a = JUNG.index("ㅏ")
    for i, ch in enumerate(JONG):
        t = i + 1
        s.cells.append(Cell(jong_name(t), ch, "hangul", guide=compose(_GIYEOK, a, t)))
    return s


# Small hint printed next to characters that are easy to misread.
HINTS = {
    "\"": "큰따옴표", "'": "작은따옴표", "`": "억음부호", ",": "쉼표", ".": "마침표",
    "\\": "역슬래시", "|": "세로선", "_": "밑줄", "-": "하이픈", "~": "물결",
    "^": "캐럿", ":": "쌍점", ";": "쌍반점", "l": "엘", "I": "아이", "O": "오", "0": "영",
    "\u201c": "여는 큰따옴표", "\u201d": "닫는 큰따옴표",
    "\u2018": "여는 작은따옴표", "\u2019": "닫는 작은따옴표",
    "\u2026": "말줄임표", "\u00b7": "가운뎃점", "\u20a9": "원화",
}


def _latin_set(key: str, title: str, chars: str) -> CellSet:
    s = CellSet(key, title)
    for ch in chars:
        s.cells.append(Cell(f"uni{ord(ch):04X}", ch, "latin",
                            guide=HINTS.get(ch, ""), codepoint=ord(ch)))
    return s


UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
EXTRA = "“”‘’…·₩"  # “ ” ‘ ’ … · ₩


def latin_sets():
    return [
        _latin_set("upper", "영문 대문자", UPPER),
        _latin_set("lower", "영문 소문자", LOWER),
        _latin_set("digit", "숫자", DIGITS),
        _latin_set("punct", "문장부호", PUNCT),
        _latin_set("extra", "추가 기호", EXTRA),
    ]


# Template presets.  "quick" asks only for the no-받침 variants; the
# with-받침 variants are then derived automatically by squashing.
PRESETS = {
    "full": ["cho.V0", "cho.V1", "cho.H0", "cho.H1", "cho.C0", "cho.C1",
             "jung.0", "jung.1", "jong"],
    "quick": ["cho.V0", "cho.H0", "cho.C0", "jung.0", "jong"],
}
PRESET_IDS = {"full": 0, "quick": 1}


def hangul_set(key: str) -> CellSet:
    if key.startswith("cho."):
        return _cho_set(key[4:])
    if key.startswith("jung."):
        return _jung_set(key[5:])
    if key == "jong":
        return _jong_set()
    raise KeyError(key)


def preset_sets(preset: str):
    return latin_sets() + [hangul_set(k) for k in PRESETS[preset]]
