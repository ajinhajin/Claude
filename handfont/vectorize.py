"""Bitmap → outline (font units) using potrace."""

import potrace
from fontTools.pens.recordingPen import RecordingPen

from . import layout as L


def trace(ink, turdsize=8, alphamax=1.0, opttolerance=0.4):
    """Trace a boolean mask (True = ink) covering one cell.

    Returns a RecordingPen with cubic outlines in font units, where the cell
    maps to x ∈ [0, UPM], y ∈ [EM_TOP - UPM, EM_TOP].
    """
    size = ink.shape[1]
    s = L.UPM / size

    def pt(p):
        return (round(p.x * s, 1), round(L.EM_TOP - p.y * s, 1))

    # potracer treats True as background
    path = potrace.Bitmap(~ink).trace(turdsize=turdsize, alphamax=alphamax,
                                      opticurve=True, opttolerance=opttolerance)
    pen = RecordingPen()
    for curve in path.curves:
        pen.moveTo(pt(curve.start_point))
        for seg in curve.segments:
            if seg.is_corner:
                pen.lineTo(pt(seg.c))
                pen.lineTo(pt(seg.end_point))
            else:
                pen.curveTo(pt(seg.c1), pt(seg.c2), pt(seg.end_point))
        pen.closePath()
    return pen
