"""Page geometry shared by the template renderer and the scanner.

All coordinates are pixels on an A4 page rendered at 300 dpi.
"""

from dataclasses import dataclass

from .charset import PRESET_IDS, preset_sets

DPI = 300
PAGE_W, PAGE_H = 2480, 3508

# Registration marks (solid squares) near the four page corners.
FID_SIZE = 90
FID_CENTERS = [(165, 165), (PAGE_W - 165, 165),
               (165, PAGE_H - 165), (PAGE_W - 165, PAGE_H - 165)]  # tl, tr, bl, br

# Page id: a row of small squares between the top marks.
#   bit0 = 1 (guard), bit1..3 = preset id (low bit first), bit4..6 = page
#   index, bit7 = 0 (guard).  The first templates used bit2..6 for the page;
#   they never had more than 8 pages, so they decode the same way.
BIT_SIZE = 36
BIT_COUNT = 8
BIT_Y = 165


def bit_centers():
    return [(PAGE_W // 2 + int((i - (BIT_COUNT - 1) / 2) * 60), BIT_Y) for i in range(BIT_COUNT)]


def encode_bits(preset_id: int, page: int):
    assert 0 <= preset_id < 8 and 0 <= page < 8
    return ([1] + [(preset_id >> i) & 1 for i in range(3)]
            + [(page >> (2 - i)) & 1 for i in range(3)] + [0])


def decode_bits(bits):
    if bits[0] != 1 or bits[-1] != 0:
        return None
    preset_id = bits[1] | bits[2] << 1 | bits[3] << 2
    page = bits[4] << 2 | bits[5] << 1 | bits[6]
    return preset_id, page


# Grid
CELL = 270          # square writing box
LABEL_H = 48        # label strip above each box
COLS = 7
ROWS = 8
GRID_X0 = 150
GRID_Y0 = 400
COL_GAP = (PAGE_W - 2 * GRID_X0 - COLS * CELL) // (COLS - 1)
ROW_GAP = 40
ROW_H = LABEL_H + CELL + ROW_GAP

# Font units: a cell is the 1000-unit em box, y from EM_TOP (top) to EM_TOP-1000.
UPM = 1000
EM_TOP = 800

# Latin guide lines (font units)
CAP_H = 680
X_H = 450
DESC = -180


@dataclass
class PlacedCell:
    cell: object
    set_key: str
    set_title: str
    x: int   # top-left of the writing box
    y: int


@dataclass
class Page:
    preset: str
    index: int
    total: int
    cells: list
    titles: list


def build_pages(preset: str):
    """Lay out every cell set; each set starts on a new row."""
    rows = []
    for s in preset_sets(preset):
        for i in range(0, len(s.cells), COLS):
            rows.append((s, s.cells[i:i + COLS]))

    pages = []
    for p in range(0, len(rows), ROWS):
        placed, titles = [], []
        for r, (s, cells) in enumerate(rows[p:p + ROWS]):
            if s.title not in titles:
                titles.append(s.title)
            for c, cell in enumerate(cells):
                x = GRID_X0 + c * (CELL + COL_GAP)
                y = GRID_Y0 + r * ROW_H + LABEL_H
                placed.append(PlacedCell(cell, s.key, s.title, x, y))
        pages.append(Page(preset, len(pages), 0, placed, titles))
    for pg in pages:
        pg.total = len(pages)
    return pages


def preset_by_id(pid: int) -> str:
    for k, v in PRESET_IDS.items():
        if v == pid:
            return k
    raise KeyError(pid)
