"""«Маска» preview: what Mimiq sees and what it replaces, drawn as points instead of a flat colour.

* the camera picture, dimmed and desaturated;
* the replaced area as a halftone grid that is glued to the face (dots sit in face-aligned space, so they
  move, turn and scale with the head), dot size = mask strength — soft edges get smaller dots;
* parts of the face that are kept from the camera because something covers them (glasses, a hand, a mask,
  a microphone, hair) as small amber dots;
* a thin outline of the mask and the 68 tracked points with their contour lines; points hidden by an
  occluder are drawn as amber rings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import geometry as geo
from . import landmarks as L

ACCENT = (253, 139, 61)        # #3D8BFD (BGR)
ACCENT_HI = (255, 214, 168)
AMBER = (64, 176, 255)         # #FFB040
POINT = (250, 244, 236)
LINE = (236, 196, 150)
GRID = 34                      # dots across the face crop


@dataclass
class FaceMarks:
    tid: int
    alpha: float
    lm68: np.ndarray                      # frame coordinates
    hidden: Optional[np.ndarray] = None   # 68 flags
    approx: bool = False                  # 68 points estimated from 5 (68-point network off)
    matrix: Optional[np.ndarray] = None   # frame → mask space
    mask: Optional[np.ndarray] = None     # mask space, already multiplied by the face's alpha


def fit_matrix(w: int, h: int, width: int, height: int, mode: str) -> np.ndarray:
    """The affine that `sources.fit_frame` applies (frame → output)."""
    if (w, h) == (width, height):
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    if mode == "fit":
        s = min(width / w, height / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        return np.array([[nw / w, 0, (width - nw) // 2], [0, nh / h, (height - nh) // 2]], np.float64)
    s = max(width / w, height / h)
    nw, nh = max(width, int(round(w * s))), max(height, int(round(h * s)))
    return np.array([[nw / w, 0, -((nw - width) // 2)], [0, nh / h, -((nh - height) // 2)]], np.float64)


def _base(view: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    base = cv2.addWeighted(view, 0.25, g3, 0.75, 0.0)
    return cv2.convertScaleAbs(base, alpha=0.36, beta=6)


def _face_hull(lm68: np.ndarray) -> np.ndarray:
    """Jaw line + forehead (brows lifted towards the hairline) — the area where an occluder matters."""
    jaw = lm68[list(L.JAW)]
    brows = lm68[list(L.BROW_R)[::-1] + list(L.BROW_L)[::-1]]
    up = lm68[27] - lm68[8]
    lift = up / max(np.linalg.norm(up), 1e-6) * np.linalg.norm(up) * 0.22
    return np.vstack([jaw, brows + lift])


def _fx(p: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(p) * 16).astype(np.int32)   # 4 bits of sub-pixel precision for cv2 drawing


def _roi(m: np.ndarray, size: int, w: int, h: int):
    corners = geo.transform_points(np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float64), m)
    x1, y1 = np.clip(np.floor(corners.min(0)).astype(int) - 2, 0, [w, h])
    x2, y2 = np.clip(np.ceil(corners.max(0)).astype(int) + 2, 0, [w, h])
    return int(x1), int(y1), int(x2), int(y2)


def _draw_grid(dots: np.ndarray, glow: np.ndarray, frame: np.ndarray, mk: FaceMarks, to_view: np.ndarray,
               box: np.ndarray, hull_view: Optional[np.ndarray]) -> None:
    """Halftone dots in face space: size = mask strength × the brightness of the face under them."""
    vh, vw = dots.shape[:2]
    m_view = geo.compose(to_view, geo.invert(mk.matrix))            # mask space → view
    footprint = mk.mask.shape[0] * geo.matrix_scale(m_view)
    if footprint < GRID * 2:
        return
    n = int(np.clip(round(footprint), 96, 720))                      # ≈ 1 face-space px per screen px
    to_fs = np.array([[n / mk.mask.shape[0], 0, 0], [0, n / mk.mask.shape[0], 0]], np.float64)
    m_fs = geo.compose(to_fs, mk.matrix)                             # frame → face space (n × n)
    step = n / GRID
    cell_mask = cv2.resize(mk.mask, (GRID, GRID), interpolation=cv2.INTER_AREA)
    cell_box = cv2.resize(box, (GRID, GRID), interpolation=cv2.INTER_AREA) * mk.alpha
    face = geo.warp_face(frame, m_fs, n)
    lum = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (GRID, GRID), interpolation=cv2.INTER_AREA).astype(np.float32)
    sel = cell_mask > 0.3
    if sel.sum() > 8:
        lo, hi = np.percentile(lum[sel], 4), np.percentile(lum[sel], 96)
        lum = np.clip((lum - lo) / max(hi - lo, 8.0), 0.0, 1.0)
    else:
        lum = np.full_like(lum, 0.6)
    radius = step * 0.5 * (0.16 + 0.62 * np.sqrt(np.clip(cell_mask, 0, 1)) * (0.42 + 0.58 * lum))
    radius[cell_mask <= 0.04] = 0.0
    hidden = np.zeros((GRID, GRID), bool)
    if hull_view is not None:
        hull_fs = geo.transform_points(hull_view, geo.compose(geo.compose(to_fs, mk.matrix), geo.invert(to_view)))
        hull = np.zeros((GRID, GRID), np.uint8)
        cv2.fillPoly(hull, [np.round(hull_fs / step * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
        hidden = (hull > 0) & (cell_box - cell_mask > 0.35) & (cell_mask < 0.3)
    r_hidden = step * 0.5 * 0.34
    # rasterise every dot at once: distance of each face-space pixel to its cell centre
    c = (np.arange(n, dtype=np.float32) + 0.5)
    idx = np.minimum((c / step).astype(np.int32), GRID - 1)
    off = c - (idx + 0.5) * step
    d = np.sqrt(off[None, :] ** 2 + off[:, None] ** 2)
    r = radius[idx[:, None], idx[None, :]]
    hid = hidden[idx[:, None], idx[None, :]]
    a_face = np.clip(r - d + 0.5, 0.0, 1.0)
    a_hid = np.clip(r_hidden - d + 0.5, 0.0, 1.0) * hid
    t = cell_mask[idx[:, None], idx[None, :]]
    layer = np.empty((n, n, 3), np.float32)
    for ch in range(3):
        col = ACCENT[ch] + (ACCENT_HI[ch] - ACCENT[ch]) * 0.5 * t
        layer[..., ch] = a_face * col + a_hid * AMBER[ch]
    m_draw = geo.compose(m_view, geo.invert(to_fs))                  # face space (n) → view
    x1, y1, x2, y2 = _roi(m_draw, n, vw, vh)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    p = m_draw.copy()
    p[0, 2] -= x1
    p[1, 2] -= y1
    warped = cv2.warpAffine(layer, p, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR, borderValue=0)
    roi = dots[y1:y2, x1:x2]
    np.maximum(roi, np.clip(warped, 0, 255).astype(np.uint8), out=roi)
    g = cv2.warpAffine((a_face * (t > 0.5)).astype(np.float32), p, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR,
                       borderValue=0)
    groi = glow[y1:y2, x1:x2]
    for ch in range(3):
        np.maximum(groi[..., ch], (g * ACCENT[ch]).astype(np.uint8), out=groi[..., ch])


def _draw_outline(dots: np.ndarray, mk: FaceMarks, to_view: np.ndarray) -> None:
    m = geo.compose(to_view, geo.invert(mk.matrix))
    h, w = dots.shape[:2]
    x1, y1, x2, y2 = _roi(m, mk.mask.shape[0], w, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    p = m.copy()
    p[0, 2] -= x1
    p[1, 2] -= y1
    wm = cv2.warpAffine(mk.mask.astype(np.float32), p, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR, borderValue=0)
    binary = (wm > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    roi = dots[y1:y2, x1:x2]
    for cnt in contours:
        if len(cnt) < 12:
            continue
        cnt = cv2.approxPolyDP(cnt, 0.8, True)
        cv2.polylines(roi, [cnt], True, ACCENT_HI, 1, cv2.LINE_AA)


def _draw_points(dots: np.ndarray, glow: np.ndarray, mk: FaceMarks, to_view: np.ndarray) -> None:
    pts = geo.transform_points(mk.lm68, to_view)
    face = float(np.linalg.norm(pts[8] - pts[27])) * 1.6
    if face < 24:
        return
    a = float(np.clip(mk.alpha, 0, 1))
    hidden = mk.hidden if mk.hidden is not None else np.zeros(68, bool)
    line = tuple(int(c * (0.42 if not mk.approx else 0.28) * a) for c in LINE)
    for path in L.OPEN_PATHS + L.CLOSED_PATHS:
        poly = _fx(pts[list(path)])
        cv2.polylines(dots, [poly], path in L.CLOSED_PATHS, line, 1, cv2.LINE_AA, 4)
    r = float(np.clip(face / 150.0, 1.3, 4.2))
    core = tuple(int(c * a) for c in POINT)
    for k, (x, y) in enumerate(pts):
        c = _fx((x, y))
        if hidden[k]:
            cv2.circle(dots, tuple(c), int(r * 1.25 * 16), AMBER, 1, cv2.LINE_AA, 4)
        else:
            cv2.circle(glow, tuple(c), int(r * 2.2 * 16), ACCENT_HI, -1, cv2.LINE_AA, 4)
            cv2.circle(dots, tuple(c), int(r * 16), core, -1, cv2.LINE_AA, 4)


def render(frame: np.ndarray, marks: Sequence[FaceMarks], out_size: Tuple[int, int], fit_mode: str,
           view_size: Optional[Tuple[int, int]], mask_blur: float = 0.3,
           padding: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
    """Draw the mask view at preview resolution (`view_size` = max (w, h) of the preview, None = output)."""
    from .analysis import box_mask
    h, w = frame.shape[:2]
    W, H = out_size
    to_out = fit_matrix(w, h, W, H, fit_mode)
    k = 1.0
    if view_size:
        k = min(1.0, view_size[0] / W, view_size[1] / H)
        if k >= 0.95:
            k = 1.0
    vw, vh = max(2, int(W * k)), max(2, int(H * k))
    to_view = to_out.copy()
    to_view[:, :] *= k
    view = cv2.warpAffine(frame, to_view, (vw, vh), flags=cv2.INTER_AREA if k * to_out[0, 0] < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    base = _base(view)
    dots = np.zeros_like(base)
    glow = np.zeros_like(base)
    for mk in sorted(marks, key=lambda m: m.alpha):
        if mk.alpha <= 0.02:
            continue
        hull = geo.transform_points(_face_hull(mk.lm68), to_view) if mk.lm68 is not None else None
        if mk.mask is not None and mk.matrix is not None:
            box = box_mask(mk.mask.shape[0], mask_blur, padding)
            _draw_grid(dots, glow, frame, mk, to_view, box, hull)
            _draw_outline(dots, mk, to_view)
        if mk.lm68 is not None:
            _draw_points(dots, glow, mk, to_view)
    small = cv2.resize(glow, (max(1, vw // 2), max(1, vh // 2)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(2.0, vw / 400.0))
    glow = cv2.resize(small, (vw, vh), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(base, cv2.convertScaleAbs(glow, alpha=0.55))
    return cv2.max(out, dots)
