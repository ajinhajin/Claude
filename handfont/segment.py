"""Pull a single jamo out of a cell where the whole syllable was written.

Many people trace the gray guide syllable (e.g. write all of 간 in the
"ㄱ · 간" box) instead of writing just the one jamo.  To handle that, the
guide syllable is re-rendered with the same font, each of its pixels is
labelled initial / vowel / final, the labels are fitted onto the handwriting
and every ink pixel is given the label of the closest guide stroke.

Labelling the guide works by comparing it with syllables that differ in one
jamo: strokes that stay put when the initial changes are vowel/final, strokes
that stay put when the final changes are initial/vowel.
"""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import layout as L
from .scan import UPSCALE
from .template import GUIDE_FONT, GUIDE_SIZE

NONE, CHO, JUNG, JONG = 0, 1, 2, 3
SAMPLE_CHO = [0, 2, 9, 11, 12]     # ㄱ ㄴ ㅅ ㅇ ㅈ
SAMPLE_JONG = [1, 4, 21, 19, 17]   # ㄱ ㄴ ㅇ ㅅ ㅂ


def decompose(ch):
    code = ord(ch) - 0xAC00
    return code // 588, (code % 588) // 28, code % 28


@lru_cache(maxsize=None)
def _font():
    return ImageFont.truetype(GUIDE_FONT, GUIDE_SIZE)


@lru_cache(maxsize=4096)
def render(l, v, t):
    """Guide syllable as drawn on the template, at scan resolution."""
    ch = chr(0xAC00 + (l * 21 + v) * 28 + t)
    im = Image.new("L", (L.CELL, L.CELL), 0)
    ImageDraw.Draw(im).text((L.CELL / 2, L.CELL / 2), ch, font=_font(), fill=255, anchor="mm")
    size = L.CELL * UPSCALE
    return cv2.resize(np.array(im), (size, size), interpolation=cv2.INTER_LINEAR) > 127


def _stable(ref, masks):
    """Pixels of ``ref`` covered (within a small tolerance) in every render.

    Long horizontal strokes (ㅡ, the bars of ㅗ ㅜ …) get extra vertical
    tolerance: the font moves them up or down depending on the other jamo.
    """
    iso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 * UPSCALE + 1,) * 2)
    tall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 * UPSCALE + 1, 30 * UPSCALE + 1))
    bar = np.ones((1, 55 * UPSCALE), np.uint8)
    horizontal = cv2.morphologyEx(ref.astype(np.uint8), cv2.MORPH_OPEN, bar) > 0
    out = np.ones_like(ref)
    for m in masks:
        m = m.astype(np.uint8)
        out &= np.where(horizontal, cv2.dilate(m, tall) > 0, cv2.dilate(m, iso) > 0)
    return out


def _smooth(mask, labels, sigma):
    """Relabel each pixel of ``mask`` by the dominant label around it."""
    scores = [cv2.GaussianBlur((labels == k).astype(np.float32), (0, 0), sigma)
              for k in (CHO, JUNG, JONG)]
    out = (np.argmax(np.stack(scores), axis=0) + 1).astype(np.uint8)
    out[~mask] = NONE
    return out


def _absorb_islands(labels, max_frac=0.05):
    """Merge small label fragments into the label that surrounds them."""
    total = np.count_nonzero(labels)
    ring_k = np.ones((3 * UPSCALE + 1,) * 2, np.uint8)
    out = labels.copy()
    for k in (CHO, JUNG, JONG):
        n, comp, stats, _ = cv2.connectedComponentsWithStats((out == k).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= max_frac * total:
                continue
            sel = comp == i
            ring = (cv2.dilate(sel.astype(np.uint8), ring_k) > 0) & ~sel
            around = np.bincount(out[ring], minlength=4)
            around[NONE] = around[k] = 0
            if around.any():
                out[sel] = around.argmax()
    return out


def _majority(mask, labels, ratio):
    """Give each connected component a single label when one clearly dominates."""
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = labels.copy()
    for i in range(1, n):
        sel = comp == i
        counts = np.bincount(labels[sel], minlength=4)
        counts[NONE] = 0
        if counts.sum() and counts.max() >= ratio * counts.sum():
            out[sel] = counts.argmax()
    return out


@lru_cache(maxsize=512)
def guide_labels(l, v, t):
    """Per-pixel jamo label map of the guide syllable."""
    ref = render(l, v, t)
    lab = np.zeros(ref.shape, np.uint8)
    cho = ref & ~_stable(ref, [render(x, v, t) for x in SAMPLE_CHO if x != l][:4])
    jong = np.zeros_like(ref)
    if t:
        jong = ref & ~_stable(ref, [render(l, v, x) for x in SAMPLE_JONG if x != t][:4]) & ~cho
    lab[ref] = JUNG
    lab[cho] = CHO
    lab[jong] = JONG
    # Stroke edges are noisy (the font nudges shapes between syllables), so
    # decide on the stroke cores and spread those labels outwards.
    core = cv2.erode(ref.astype(np.uint8), np.ones((3 * UPSCALE,) * 2, np.uint8)) > 0
    lab = _smooth(ref, np.where(core, lab, NONE), 5 * UPSCALE)
    return _absorb_islands(_majority(ref, lab, 0.7))


def _assign(ink, lab):
    """Label every ink pixel with the nearest guide label."""
    dist = []
    for k in (CHO, JUNG, JONG):
        src = (lab != k).astype(np.uint8)
        if src.all():
            dist.append(np.full(ink.shape, np.inf, np.float32))
        else:
            dist.append(cv2.distanceTransform(src, cv2.DIST_L2, 5))
    nearest = (np.argmin(np.stack(dist), axis=0) + 1).astype(np.uint8)
    nearest[~ink] = NONE
    # strokes that clearly belong to one jamo stay whole; strokes that run
    # from one jamo into the next (ㄴ flowing into ㅗ …) are cut per pixel
    return _majority(ink, nearest, 0.85)


def _points(mask, n=600):
    ys, xs = np.nonzero(mask)
    step = max(1, len(xs) // n)
    return np.stack([xs[::step], ys[::step]], 1).astype(np.float32)


def _lookup(dt, pts):
    h, w = dt.shape
    x = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
    return np.minimum(dt[y, x], 40 * UPSCALE)


def _dt(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def _fit(guide, ink):
    """Find the scale/shift that lays ``guide`` over ``ink`` most closely.

    Starts by matching bounding boxes, then refines by minimising the
    symmetric (truncated) chamfer distance.  Returns ((sx, sy, tx, ty), cost)
    mapping guide coordinates to ink coordinates.
    """
    ys, xs = np.nonzero(guide)
    iy, ix = np.nonzero(ink)
    sx = (ix.max() - ix.min() + 1) / (xs.max() - xs.min() + 1)
    sy = (iy.max() - iy.min() + 1) / (ys.max() - ys.min() + 1)
    p = [sx, sy, ix.min() - xs.min() * sx, iy.min() - ys.min() * sy]

    g_pts, i_pts = _points(guide), _points(ink)
    dt_ink, dt_guide = _dt(ink), _dt(guide)

    def cost(q):
        a, b, tx, ty = q
        fwd = _lookup(dt_ink, g_pts * (a, b) + (tx, ty)).mean()
        back = _lookup(dt_guide, (i_pts - (tx, ty)) / (a, b)).mean()
        return fwd + back

    best = cost(p)
    for s_step, t_step in ((0.08, 16 * UPSCALE), (0.04, 8 * UPSCALE),
                           (0.02, 4 * UPSCALE), (0.01, 2 * UPSCALE)):
        improved = True
        while improved:
            improved = False
            for i, step in ((0, s_step), (1, s_step), (2, t_step), (3, t_step)):
                for d in (-step, step):
                    q = list(p)
                    q[i] += d
                    if q[0] <= 0.2 or q[1] <= 0.2:
                        continue
                    c = cost(q)
                    if c < best - 1e-6:
                        p, best, improved = q, c, True
    return p, best


def _warp(lab, p):
    sx, sy, tx, ty = p
    M = np.float32([[sx, 0, tx], [0, sy, ty]])
    return cv2.warpAffine(lab, M, (lab.shape[1], lab.shape[0]), flags=cv2.INTER_NEAREST)


def _nudge(part, dt_ink, reach):
    """Best translation (within ``reach`` px) that lays ``part`` onto the ink."""
    pts = _points(part, 400)
    best, best_t = _lookup(dt_ink, pts).mean(), (0, 0)
    for step in (8 * UPSCALE, 4 * UPSCALE, 2 * UPSCALE):
        improved = True
        while improved:
            improved = False
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                t = (best_t[0] + dx, best_t[1] + dy)
                if max(abs(t[0]), abs(t[1])) > reach:
                    continue
                c = _lookup(dt_ink, pts + t).mean()
                if c < best - 1e-6:
                    best, best_t, improved = c, t, True
    return best_t


def _labelled(ink, syllable):
    """Label the ink strokes of a whole handwritten syllable.

    Returns (labels, fit) where ``fit`` maps guide → ink coordinates.  After
    the whole guide is laid over the handwriting, each guide jamo is nudged
    on its own towards the nearest strokes – people rarely keep the exact
    proportions of the guide – and the strokes are assigned again.
    """
    lab = guide_labels(*decompose(syllable))
    p, _ = _fit(lab > 0, ink)
    placed = _warp(lab, p)
    dt_ink = _dt(ink)
    for _ in range(2):
        moved = np.zeros_like(placed)
        for k in (CHO, JUNG, JONG):
            part = placed == k
            if not part.any():
                continue
            tx, ty = _nudge(part, dt_ink, 20 * UPSCALE)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            shifted = cv2.warpAffine(part.astype(np.uint8), M, part.shape[::-1],
                                     flags=cv2.INTER_NEAREST) > 0
            moved[shifted & (moved == NONE)] = k
        placed = moved
    return _assign(ink, placed), (p, placed)


def to_guide_frame(mask, p):
    """Undo the guide→ink fit so the handwriting sits exactly on the guide."""
    sx, sy, tx, ty = p
    inv = np.float32([[1 / sx, 0, -tx / sx], [0, 1 / sy, -ty / sy]])
    out = cv2.warpAffine(mask.astype(np.float32), inv, mask.shape[::-1], flags=cv2.INTER_LINEAR)
    return out > 0.5


def syllable_score(ink, syllable, target):
    """> 0 when the cell looks like a whole syllable rather than a lone jamo."""
    lab = guide_labels(*decompose(syllable))
    _, as_jamo = _fit(lab == target, ink)
    _, as_syllable = _fit(lab > 0, ink)
    return as_jamo - as_syllable


def split_jamo(ink, syllable, target, normalize=True):
    """Keep only the strokes of ``target`` (CHO / JUNG / JONG) from a whole syllable.

    Returns (mask, quality).  ``quality`` is the mean distance (px) from the
    kept strokes to where the guide expects that jamo plus the mean distance
    from the guide jamo to the kept strokes – large when strokes of another
    jamo leaked in or part of this one was cut off.

    With ``normalize`` the result is moved/scaled so the whole handwritten
    syllable would sit exactly on the guide syllable.  Jamo taken from
    different cells then share one frame and line up when composed, even if
    each cell was written a little smaller, larger or off-centre.
    """
    labels, (p, placed) = _labelled(ink, syllable)
    part = (labels == target).astype(np.uint8)
    # drop crumbs cut off from neighbouring jamo
    n, comp, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= max(0.08 * areas.max(), 30 * UPSCALE * UPSCALE)
        part = keep[comp].astype(np.uint8)
    part = part > 0

    expected = placed == target
    if part.any() and expected.any():
        quality = (_lookup(_dt(expected), _points(part)).mean()
                   + _lookup(_dt(part), _points(expected)).mean()) / UPSCALE
    else:
        quality = np.inf
    if normalize:
        part = to_guide_frame(part, p)
    return part, quality


def normalize_syllable(ink, syllable):
    """A whole handwritten syllable, scaled/shifted onto its guide syllable."""
    lab = guide_labels(*decompose(syllable))
    p, _ = _fit(lab > 0, ink)
    return to_guide_frame(ink, p)


def colorize(ink, syllable):
    """Debug view: initial red, vowel blue, final green."""
    lab, _ = _labelled(ink, syllable)
    img = np.full(ink.shape + (3,), 255, np.uint8)
    img[lab == CHO] = (40, 40, 220)
    img[lab == JUNG] = (220, 90, 30)
    img[lab == JONG] = (40, 160, 40)
    return img
