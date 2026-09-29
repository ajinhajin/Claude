"""Find the template in a scan/photo, straighten it and cut out the cells."""

import itertools
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import layout as L

UPSCALE = 2          # cells are upsampled before tracing for smoother outlines
BORDER_INSET = 10    # px (page scale) ignored along the cell border


def load_images(path):
    """Return a list of grayscale uint8 arrays (PDF files may hold several pages)."""
    if path.lower().endswith(".pdf"):
        try:
            import pymupdf as fitz
        except ImportError:
            raise SystemExit("PDF 스캔을 읽으려면 `pip install pymupdf` 가 필요합니다. "
                             "(또는 PNG/JPG로 저장해서 넣어주세요)")
        doc = fitz.open(path)
        out = []
        for page in doc:
            pix = page.get_pixmap(dpi=L.DPI, colorspace=fitz.csGRAY)
            out.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy())
        return out
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return [np.array(im)]


def normalize(gray):
    """Divide out uneven lighting so paper ≈ 1.0 and ink ≈ 0."""
    h, w = gray.shape
    f = 600.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    k = max(9, int(max(small.shape) / 12) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(gray.astype(np.float32) / np.maximum(bg, 1), 0, 1)


def find_marks(norm):
    """Locate the four solid corner squares; returns centers ordered tl, tr, bl, br."""
    bw = (norm < 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (max(norm.shape) / 250.0) ** 2
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        if rw == 0 or rh == 0 or not 0.6 < rw / rh < 1.67:
            continue
        if area / (rw * rh) < 0.8:
            continue
        m = cv2.moments(c)
        cands.append((area, m["m10"] / m["m00"], m["m01"] / m["m00"]))
    cands.sort(reverse=True)
    cands = cands[:12]

    best, best_score = None, 0
    for combo in itertools.combinations(cands, 4):
        areas = [a for a, _, _ in combo]
        if max(areas) / min(areas) > 3:
            continue
        pts = np.array([[x, y] for _, x, y in combo], np.float32)
        hull = cv2.convexHull(pts)
        if len(hull) != 4:
            continue
        score = cv2.contourArea(hull)
        if score > best_score:
            best, best_score = pts, score
    if best is None:
        return None
    pts = best[np.argsort(best[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bot[0], bot[1]], np.float32)


def _read_bits(page):
    b = L.BIT_SIZE // 2 - 8
    bits = []
    for cx, cy in L.bit_centers():
        bits.append(1 if page[cy - b:cy + b, cx - b:cx + b].mean() < 0.5 else 0)
    return L.decode_bits(bits)


def straighten(norm):
    """Warp the page to canonical 300 dpi A4. Returns (page, preset, page_index)."""
    marks = find_marks(norm)
    if marks is None:
        raise ValueError("모서리의 검은 사각형 4개를 찾지 못했습니다")
    dst = np.array(L.FID_CENTERS, np.float32)
    # try upright first, then upside-down
    for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
        M = cv2.getPerspectiveTransform(marks[order], dst)
        page = cv2.warpPerspective(norm, M, (L.PAGE_W, L.PAGE_H), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        decoded = _read_bits(page)
        if decoded:
            pid, index = decoded
            return page, L.preset_by_id(pid), index
    raise ValueError("페이지 번호 표시(상단 작은 사각형)를 읽지 못했습니다")


def extract_cell(page, pc, threshold=0.55, weight=0, min_blob=40):
    """Return a boolean ink mask (upscaled) for one cell, or None if empty."""
    crop = page[pc.y:pc.y + L.CELL, pc.x:pc.x + L.CELL]
    up = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    ink = (up < threshold).astype(np.uint8)
    i = BORDER_INSET * UPSCALE
    ink[:i, :] = 0
    ink[-i:, :] = 0
    ink[:, :i] = 0
    ink[:, -i:] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob * UPSCALE * UPSCALE
    ink = keep[labels].astype(np.uint8)

    if weight:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(weight) + 1,) * 2)
        ink = cv2.dilate(ink, k) if weight > 0 else cv2.erode(ink, k)
    if ink.sum() < 150 * UPSCALE * UPSCALE:
        return None
    return ink.astype(bool)


def _target(pc):
    from .segment import CHO, JONG, JUNG
    return {"cho": CHO, "jung": JUNG, "jong": JONG}[pc.set_key.split(".")[0]]


def split_hangul(raw, mode="auto", debug_dir=None, log=print):
    """Reduce Hangul cells to the one jamo each is for.

    ``raw`` maps glyph name -> (ink, placed cell).  People fill the whole
    template the same way, so in "auto" mode every cell votes on whether
    whole syllables were written and the majority decides for all of them.
    """
    from .charset import syllable_name
    from .segment import colorize, normalize_syllable, split_jamo, syllable_score
    out = {}
    jamo_cells = {n: v for n, v in raw.items() if v[1].set_key != "syl"}

    # whole syllables from the "common" template are used as written
    for name, (ink, pc) in raw.items():
        if pc.set_key == "syl":
            out[name] = normalize_syllable(ink, pc.cell.guide)

    if jamo_cells and mode == "auto":
        votes = [syllable_score(ink, pc.cell.guide, _target(pc)) > 0
                 for ink, pc in jamo_cells.values()]
        mode = "syllable" if sum(votes) * 2 > len(votes) else "jamo"
        if mode == "syllable":
            log(f"  한글 칸에 글자 전체가 쓰여 있어 자모만 분리합니다 ({sum(votes)}/{len(votes)}칸 판정)")
    if mode == "jamo":
        out.update({name: ink for name, (ink, _) in jamo_cells.items()})
        return out

    for name, (ink, pc) in jamo_cells.items():
        part, _ = split_jamo(ink, pc.cell.guide, _target(pc))
        if debug_dir:
            d = os.path.join(debug_dir, "_split")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{name}.png"), colorize(ink, pc.cell.guide))
        if np.count_nonzero(part) < 60 * UPSCALE * UPSCALE:
            log(f"  ! {pc.cell.label} ({pc.cell.guide}) 칸에서 자모를 분리하지 못했습니다: {name}")
        else:
            out[name] = part
        # the syllable itself was written too – keep it as a finished glyph
        whole = syllable_name(pc.cell.guide)
        if whole not in out:
            out[whole] = normalize_syllable(ink, pc.cell.guide)
    return out


def scan_files(paths, threshold=0.55, weight=0, debug_dir=None, hangul_mode="auto", log=print):
    """Scan all files. Returns {glyph_name: ink_mask}, {glyph_name: set_key}, presets."""
    cells, sets, presets = {}, {}, set()
    hangul = {}
    pages_cache = {}
    for path in paths:
        for n, gray in enumerate(load_images(path)):
            src = f"{os.path.basename(path)}" + (f"#{n + 1}" if n else "")
            try:
                page, preset, index = straighten(normalize(gray))
            except ValueError as e:
                log(f"  ✗ {src}: {e}")
                continue
            if preset not in pages_cache:
                pages_cache[preset] = L.build_pages(preset)
            layout_pages = pages_cache[preset]
            if index >= len(layout_pages):
                log(f"  ✗ {src}: 알 수 없는 페이지 번호 {index + 1}")
                continue
            presets.add(preset)
            found = 0
            for pc in layout_pages[index].cells:
                ink = extract_cell(page, pc, threshold, weight)
                if ink is None:
                    continue
                if pc.cell.kind == "hangul":
                    hangul[pc.cell.name] = (ink, pc)
                else:
                    cells[pc.cell.name] = ink
                sets[pc.cell.name] = pc.set_key
                found += 1
            log(f"  ✓ {src}: {preset} {index + 1}/{len(layout_pages)} 페이지, "
                f"{found}/{len(layout_pages[index].cells)} 칸 인식")
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"_page_{preset}_{index + 1}.jpg"),
                            (page * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 70])

    cells.update(split_hangul(hangul, hangul_mode, debug_dir, log))
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name, ink in cells.items():
            cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), np.where(ink, 0, 255).astype(np.uint8))
    return cells, sets, presets


def load_overrides(folder, threshold=0.55):
    """Per-glyph replacement images: <folder>/<glyph_name>.png sized like a cell."""
    out = {}
    if not folder or not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        name, ext = os.path.splitext(fn)
        if ext.lower() not in (".png", ".jpg", ".jpeg") or name.startswith("_"):
            continue
        gray = np.array(Image.open(os.path.join(folder, fn)).convert("L"), np.float32) / 255
        size = L.CELL * UPSCALE
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        ink = gray < threshold
        if ink.any():
            out[name] = ink
    return out
