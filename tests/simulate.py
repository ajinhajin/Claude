#!/usr/bin/env python3
"""End-to-end test: fill the template with a pen-style font, fake a phone
photo (perspective, shading, noise) and save it as JPEG.

    python tests/simulate.py --preset quick -o /tmp/sim
    python build_font.py /tmp/sim/*.jpg -o /tmp/sim/out
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handfont import layout as L  # noqa: E402
from handfont.charset import JUNG, vowel_class  # noqa: E402
from handfont.template import find_korean_font, render_page  # noqa: E402

PEN = "/usr/share/fonts/truetype/nanum/NanumPen.ttf"

# Rough regions (fractions of the cell: x0, y0, x1, y1) where a person
# would write each jamo variant.
CHO_REGION = {"V0": (.10, .14, .56, .84), "V1": (.10, .08, .56, .52),
              "H0": (.20, .10, .80, .50), "H1": (.24, .06, .76, .36),
              "C0": (.10, .08, .52, .48), "C1": (.12, .06, .50, .34)}
JUNG_REGION = {("V", "0"): (.60, .06, .88, .94), ("V", "1"): (.60, .04, .88, .60),
               ("H", "0"): (.10, .54, .90, .90), ("H", "1"): (.10, .38, .90, .60),
               ("C", "0"): (.10, .06, .90, .94), ("C", "1"): (.10, .04, .90, .60)}
JONG_REGION = (.20, .66, .80, .94)


def _ink(ch, font):
    im = Image.new("L", (600, 600), 255)
    ImageDraw.Draw(im).text((300, 300), ch, font=font, fill=0, anchor="mm")
    a = np.array(im)
    ys, xs = np.where(a < 128)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def fill_page(img, page, pen_path):
    big = ImageFont.truetype(pen_path, 400)
    arr = np.array(img)
    for pc in page.cells:
        c, S = pc.cell, L.CELL
        if c.kind == "latin":
            f = ImageFont.truetype(pen_path, int(S * 1.05))
            tmp = Image.new("L", (S, S), 255)
            base = (L.EM_TOP) * S / L.UPM
            ImageDraw.Draw(tmp).text((S / 2, base), c.label, font=f, fill=0, anchor="ms")
            patch = np.array(tmp)
        else:
            key = pc.set_key
            if key == "syl":
                region = (.08, .08, .92, .92)
            elif key.startswith("cho."):
                region = CHO_REGION[key[4:]]
            elif key.startswith("jung."):
                v = JUNG.index(c.label)
                region = JUNG_REGION[(vowel_class(v), key[5:])]
            else:
                region = JONG_REGION
            glyph = _ink(c.guide if key == "syl" else c.label, big)
            x0, y0, x1, y1 = (int(r * S) for r in region)
            glyph = cv2.resize(glyph, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            patch = np.full((S, S), 255, np.uint8)
            patch[y0:y1, x0:x1] = glyph
        # pen ink is dark blue-black, not pure black
        patch = np.where(patch < 128, 40, 255).astype(np.uint8)
        sub = arr[pc.y:pc.y + S, pc.x:pc.x + S]
        arr[pc.y:pc.y + S, pc.x:pc.x + S] = np.minimum(sub, patch)
    return Image.fromarray(arr)


def fake_photo(img, seed):
    rnd = random.Random(seed)
    a = np.array(img).astype(np.float32)
    h, w = a.shape
    W, H = 3000, 4000
    canvas = np.full((H, W), 60, np.float32)  # dark table
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    j = lambda: rnd.uniform(-120, 120)
    dst = np.float32([[250 + j(), 200 + j()], [2750 + j(), 230 + j()],
                      [200 + j(), 3800 + j()], [2800 + j(), 3750 + j()]])
    M = cv2.getPerspectiveTransform(src, dst)
    page = cv2.warpPerspective(a, M, (W, H), borderValue=-1)
    canvas = np.where(page >= 0, page, canvas)
    # uneven lighting
    yy, xx = np.mgrid[0:H, 0:W]
    light = 0.65 + 0.35 * (xx / W) * (1 - 0.4 * yy / H)
    canvas = canvas * light
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    canvas += np.random.default_rng(seed).normal(0, 6, canvas.shape)
    if seed % 2:
        canvas = canvas[::-1, ::-1]  # upside-down photo
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="quick")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    font = find_korean_font()
    for page in L.build_pages(args.preset):
        img = fill_page(render_page(page, font), page, PEN)
        photo = fake_photo(img, page.index)
        path = os.path.join(args.out, f"photo_{args.preset}_{page.index + 1}.jpg")
        cv2.imwrite(path, photo, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(path)


if __name__ == "__main__":
    main()
