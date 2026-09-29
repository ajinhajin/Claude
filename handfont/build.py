"""Assemble traced outlines into a TrueType font."""

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.areaPen import AreaPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

from . import layout as L
from .charset import (CHO, JONG, JUNG, cho_name, jong_name, jung_name,
                      vowel_class)

SQUASH = 0.72  # vertical scale used to derive the with-받침 variants

# Hangul compatibility jamo (ㄱ … ㅎ, ㅏ … ㅣ) shown on their own.
COMPAT_CONSONANTS = "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"

# If a variant is missing entirely, borrow the closest one.
CHO_FALLBACK = {"V0": ["V1"], "V1": ["V0"], "H0": ["C0", "H1"], "H1": ["C1", "H0"],
                "C0": ["H0", "C1"], "C1": ["H1", "C0"]}
JUNG_FALLBACK = {"0": ["1"], "1": ["0"]}


def _bounds(rec):
    bp = BoundsPen(None)
    rec.replay(bp)
    return bp.bounds


def _transformed(rec, t):
    out = RecordingPen()
    rec.replay(TransformPen(out, t))
    return out


def _tt_glyph(rec):
    ap = AreaPen()
    rec.replay(ap)
    tt = TTGlyphPen(None)
    # TrueType wants clockwise outer contours (negative area)
    rec.replay(Cu2QuPen(tt, max_err=1.0, reverse_direction=ap.value > 0))
    return tt.glyph()


def _empty_glyph():
    return TTGlyphPen(None).glyph()


def _notdef():
    pen = TTGlyphPen(None)
    for pts in ([(50, -200), (50, 800), (550, 800), (550, -200)],
                [(100, -150), (500, -150), (500, 750), (100, 750)]):
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    return pen.glyph()


def derive_missing(outlines, log=print):
    """Create with-받침 variants from the plain ones when they were not written."""
    tops = [_bounds(r)[3] for n, r in outlines.items()
            if n.startswith(("cho", "jung")) and _bounds(r)]
    if not tops:
        return
    top = sorted(tops)[int(len(tops) * 0.9)]
    t = (1, 0, 0, SQUASH, 0, top * (1 - SQUASH))   # y' = top - (top - y) * SQUASH
    derived = 0
    for l in range(len(CHO)):
        for cls in "VHC":
            src, dst = cho_name(l, cls + "0"), cho_name(l, cls + "1")
            if dst not in outlines and src in outlines:
                outlines[dst] = _transformed(outlines[src], t)
                derived += 1
    for v in range(len(JUNG)):
        src, dst = jung_name(v, "0"), jung_name(v, "1")
        if dst not in outlines and src in outlines:
            outlines[dst] = _transformed(outlines[src], t)
            derived += 1
    if derived:
        log(f"  받침 있는 모양 {derived}개를 자동으로 만들었습니다")


def _pick(outlines, base, variant, fallback):
    for v in [variant] + fallback[variant]:
        name = base(v)
        if name in outlines:
            return name
    return None


def build_font(outlines, out_path, family="MyHandwriting", family_ko=None,
               hangul_advance=1000, space_width=280, sidebearing=40,
               version="1.000", log=print):
    outlines = dict(outlines)
    derive_missing(outlines, log)

    glyphs = {".notdef": _notdef(), "space": _empty_glyph()}
    metrics = {".notdef": (600, 50), "space": (space_width, 0)}
    cmap = {0x20: "space", 0xA0: "space"}
    bounds = {}

    # --- Latin, digits, punctuation --------------------------------------
    latin = 0
    for name, rec in outlines.items():
        if not name.startswith("uni"):
            continue
        b = _bounds(rec)
        if not b:
            continue
        dx = sidebearing - b[0]
        rec = _transformed(rec, (1, 0, 0, 1, dx, 0))
        glyphs[name] = _tt_glyph(rec)
        metrics[name] = (int(round(b[2] - b[0] + 2 * sidebearing)), 0)
        cmap[int(name[3:], 16)] = name
        latin += 1

    # --- Hangul jamo components -------------------------------------------
    for name, rec in outlines.items():
        if name.startswith(("cho", "jung", "jong")):
            b = _bounds(rec)
            if b:
                glyphs[name] = _tt_glyph(rec)
                metrics[name] = (hangul_advance, 0)
                bounds[name] = b

    # --- Standalone jamo (ㄱ, ㅏ …) ----------------------------------------
    ref = bounds.get(jung_name(0, "0")) or bounds.get(jung_name(20, "0"))
    ref_cy = (ref[1] + ref[3]) / 2 if ref else 300
    ref_h = (ref[3] - ref[1]) if ref else 800
    jamo = 0
    for i, ch in enumerate(COMPAT_CONSONANTS):
        src = None
        if ch in CHO:
            l = CHO.index(ch)
            src = _pick(bounds, lambda v: cho_name(l, v), "V0", CHO_FALLBACK)
        if src is None and ch in JONG:
            src = jong_name(JONG.index(ch) + 1)
            src = src if src in bounds else None
        if src is None:
            continue
        b = bounds[src]
        w, h = b[2] - b[0], b[3] - b[1]
        s = min(0.62 * ref_h / max(h, 1), 0.8 * hangul_advance / max(w, 1), 2.5)
        t = (s, 0, 0, s, hangul_advance / 2 - s * (b[0] + b[2]) / 2, ref_cy - s * (b[1] + b[3]) / 2)
        gname = f"uni{0x3131 + i:04X}"
        glyphs[gname] = _tt_glyph(_transformed(outlines[src], t))
        metrics[gname] = (hangul_advance, 0)
        cmap[0x3131 + i] = gname
        jamo += 1
    for v, ch in enumerate(JUNG):
        src = _pick(bounds, lambda x: jung_name(v, x), "0", JUNG_FALLBACK)
        if src is None:
            continue
        b = bounds[src]
        dx = hangul_advance / 2 - (b[0] + b[2]) / 2
        gname = f"uni{0x314F + v:04X}"
        glyphs[gname] = _tt_glyph(_transformed(outlines[src], (1, 0, 0, 1, dx, 0)))
        metrics[gname] = (hangul_advance, 0)
        cmap[0x314F + v] = gname
        jamo += 1

    # --- Hangul syllables (composites) ------------------------------------
    syllables = 0
    missing_parts = set()
    for l in range(19):
        for v in range(21):
            cls = vowel_class(v)
            for t in range(28):
                fin = "1" if t else "0"
                parts = [
                    _pick(bounds, lambda x: cho_name(l, x), cls + fin, CHO_FALLBACK),
                    _pick(bounds, lambda x: jung_name(v, x), fin, JUNG_FALLBACK),
                ]
                if t:
                    parts.append(jong_name(t) if jong_name(t) in bounds else None)
                if None in parts:
                    if parts[0] is None:
                        missing_parts.add(f"초성 {CHO[l]}")
                    if parts[1] is None:
                        missing_parts.add(f"중성 {JUNG[v]}")
                    if t and parts[-1] is None:
                        missing_parts.add(f"종성 {JONG[t - 1]}")
                    continue
                x0 = min(bounds[p][0] for p in parts)
                x1 = max(bounds[p][2] for p in parts)
                dx = int(round(hangul_advance / 2 - (x0 + x1) / 2))
                pen = TTGlyphPen(glyphs)
                for p in parts:
                    pen.addComponent(p, (1, 0, 0, 1, dx, 0))
                code = 0xAC00 + (l * 21 + v) * 28 + t
                gname = f"uni{code:04X}"
                glyphs[gname] = pen.glyph()
                metrics[gname] = (hangul_advance, 0)
                cmap[code] = gname
                syllables += 1

    log(f"  영문·숫자·기호 {latin}자, 한글 자모 {jamo}자, 한글 음절 {syllables}/11172자")
    if missing_parts:
        log("  ! 빠진 자모 때문에 만들지 못한 음절이 있습니다: " + ", ".join(sorted(missing_parts)))

    # --- Assemble ------------------------------------------------------------
    order = [".notdef", "space"] + sorted(n for n in glyphs if n not in (".notdef", "space"))
    fb = FontBuilder(L.UPM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)

    glyf = fb.font["glyf"]
    ymax, ymin = 800, -200
    hmtx = {}
    for n in order:
        g = glyf[n]
        g.recalcBounds(glyf)
        adv = metrics[n][0]
        lsb = getattr(g, "xMin", 0) if g.numberOfContours else 0
        hmtx[n] = (adv, lsb)
        if g.numberOfContours:
            ymax, ymin = max(ymax, g.yMax), min(ymin, g.yMin)
    fb.setupHorizontalMetrics(hmtx)

    asc, desc = 900, -300
    fb.setupHorizontalHeader(ascent=asc, descent=desc, lineGap=0)
    ps = "".join(ch for ch in family if ch.isalnum()) or "Handwriting"
    names = {
        "familyName": {"en": family, **({"ko": family_ko} if family_ko else {})},
        "styleName": "Regular",
        "uniqueFontIdentifier": f"{ps}-Regular;{version}",
        "fullName": {"en": family, **({"ko": family_ko} if family_ko else {})},
        "psName": f"{ps}-Regular",
        "version": f"Version {version}",
    }
    fb.setupNameTable(names)
    fb.setupOS2(sTypoAscender=asc, sTypoDescender=desc, sTypoLineGap=0,
                usWinAscent=max(asc, int(ymax)), usWinDescent=max(-desc, int(-ymin)),
                sxHeight=L.X_H, sCapHeight=L.CAP_H, usWeightClass=400,
                fsType=0, fsSelection=0x40 | 0x80, achVendID="HNDF", version=4)
    fb.setupPost(keepGlyphNames=False)
    fb.setupHead(unitsPerEm=L.UPM, fontRevision=float(version))

    os2 = fb.font["OS/2"]
    os2.recalcUnicodeRanges(fb.font)
    if hasattr(os2, "recalcCodePageRanges"):
        os2.recalcCodePageRanges(fb.font)
    fb.save(out_path)

    woff2 = None
    try:
        import brotli  # noqa: F401
        from fontTools.ttLib import TTFont
        f = TTFont(out_path)
        f.flavor = "woff2"
        woff2 = out_path.rsplit(".", 1)[0] + ".woff2"
        f.save(woff2)
    except ImportError:
        pass
    return out_path, woff2
