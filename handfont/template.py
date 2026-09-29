"""Render printable handwriting templates (PDF + PNG)."""

import os
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

from . import layout as L
from .charset import PRESET_IDS

GUIDE_GRAY = 222     # light enough to be removed by the scanner
LINE_GRAY = 200
BORDER_GRAY = 150

# Bundled copy of the guide font (SIL OFL 1.1). The scanner re-renders the
# guide syllables with it, so it must match what the templates were drawn with.
GUIDE_FONT = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")
GUIDE_SIZE = int(L.CELL * 0.86)

_FONT_CANDIDATES = [
    GUIDE_FONT,
    # Linux
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    # Windows
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
]


def find_korean_font(explicit=None):
    if explicit:
        return explicit
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    if shutil.which("fc-match"):
        out = subprocess.run(["fc-match", "-f", "%{file}", ":lang=ko"],
                             capture_output=True, text=True).stdout.strip()
        if out and os.path.exists(out):
            return out
    raise SystemExit("한글 폰트를 찾을 수 없습니다. --font 로 .ttf/.otf 경로를 지정하세요.")


def _em_to_px(y_units, cell_y):
    return cell_y + (L.EM_TOP - y_units) * L.CELL / L.UPM


def render_page(page, font_path):
    img = Image.new("L", (L.PAGE_W, L.PAGE_H), 255)
    d = ImageDraw.Draw(img)

    # Registration marks
    h = L.FID_SIZE // 2
    for cx, cy in L.FID_CENTERS:
        d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1], fill=0)

    # Page id bits
    bits = L.encode_bits(PRESET_IDS[page.preset], page.index)
    b = L.BIT_SIZE // 2
    for (cx, cy), bit in zip(L.bit_centers(), bits):
        box = [cx - b, cy - b, cx + b - 1, cy + b - 1]
        d.rectangle(box, fill=0 if bit else 255, outline=0, width=3)

    f_head = ImageFont.truetype(font_path, 40)
    f_small = ImageFont.truetype(font_path, 28)
    f_label = ImageFont.truetype(font_path, 36)
    f_guide = ImageFont.truetype(GUIDE_FONT, GUIDE_SIZE)

    head = f"손글씨 폰트 템플릿 ({page.preset})  ·  {page.index + 1} / {page.total} 페이지"
    d.text((L.GRID_X0, 250), head, font=f_head, fill=0)
    d.text((L.GRID_X0, 310), " / ".join(page.titles), font=f_small, fill=90)
    d.text((L.PAGE_W - L.GRID_X0, 310),
           "검은 펜으로, 칸 안에만 쓰세요. 회색 글자는 위치 참고용입니다.",
           font=f_small, fill=90, anchor="ra")

    for pc in page.cells:
        x, y, c = pc.x, pc.y, pc.cell
        # label
        # U+005C is drawn as ₩ by most Korean fonts, so it gets its name only.
        if c.label != "\\":
            d.text((x + 6, y - 8), c.label, font=f_label, fill=0, anchor="ls")
        if c.guide and c.guide != c.label:
            d.text((x + L.CELL - 6, y - 10), c.guide, font=f_small, fill=120, anchor="rs")

        # guides
        if c.kind == "hangul":
            d.text((x + L.CELL / 2, y + L.CELL / 2), c.guide, font=f_guide,
                   fill=GUIDE_GRAY, anchor="mm")
            mid = x + L.CELL // 2
            for yy in range(y + 8, y + L.CELL - 8, 16):
                d.line([mid, yy, mid, yy + 6], fill=235)
            mid = y + L.CELL // 2
            for xx in range(x + 8, x + L.CELL - 8, 16):
                d.line([xx, mid, xx + 6, mid], fill=235)
        else:
            for units, gray, w in ((L.CAP_H, LINE_GRAY, 2), (L.X_H, LINE_GRAY, 2),
                                   (0, 170, 3), (L.DESC, LINE_GRAY, 2)):
                yy = _em_to_px(units, y)
                if units == L.X_H:
                    for xx in range(x + 4, x + L.CELL - 4, 18):
                        d.line([xx, yy, xx + 9, yy], fill=gray, width=w)
                else:
                    d.line([x + 4, yy, x + L.CELL - 4, yy], fill=gray, width=w)

        d.rectangle([x, y, x + L.CELL - 1, y + L.CELL - 1], outline=BORDER_GRAY, width=2)

    return img


def make_templates(preset, out_dir, font_path=None, png=True):
    font_path = find_korean_font(font_path)
    os.makedirs(out_dir, exist_ok=True)
    pages = L.build_pages(preset)
    images = [render_page(p, font_path) for p in pages]
    pdf_path = os.path.join(out_dir, f"template_{preset}.pdf")
    images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=L.DPI)
    paths = [pdf_path]
    if png:
        for p, im in zip(pages, images):
            path = os.path.join(out_dir, f"template_{preset}_p{p.index + 1}.png")
            im.save(path, optimize=True)
            paths.append(path)
    return paths
