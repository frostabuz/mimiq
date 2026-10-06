"""Face alignment templates, warping and anti-aliased paste-back."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

TEMPLATES = {
    "arcface_112_v1": np.array([[0.35473214, 0.45658929], [0.64526786, 0.45658929], [0.50000000, 0.61154464],
                                [0.37913393, 0.77687500], [0.62086607, 0.77687500]], np.float64),
    "arcface_112_v2": np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
                                [0.37097589, 0.82469196], [0.63151696, 0.82325089]], np.float64),
    "arcface_128": np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                             [0.38710391, 0.72160547], [0.61507734, 0.72034453]], np.float64),
    "ffhq_512": np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                          [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float64),
}


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation) src → dst."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n, dim = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    a = dst_d.T @ src_d / n
    d = np.ones(dim)
    if np.linalg.det(a) < 0:
        d[-1] = -1
    u, s, vt = np.linalg.svd(a)
    r = u @ np.diag(d) @ vt
    var = src_d.var(axis=0).sum()
    scale = (s @ d) / var if var > 1e-12 else 1.0
    m = np.zeros((2, 3), np.float64)
    m[:, :2] = scale * r
    m[:, 2] = dst_mean - scale * (r @ src_mean)
    return m


def face_matrix(lm5: np.ndarray, template: str, size: int) -> np.ndarray:
    return umeyama(lm5, TEMPLATES[template] * size)


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    return pts @ m[:, :2].T + m[:, 2]


def invert(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(m.astype(np.float64))


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the affine that applies `b` first and then `a`."""
    a3 = np.vstack([a, [0, 0, 1]])
    b3 = np.vstack([b, [0, 0, 1]])
    return (a3 @ b3)[:2]


def matrix_scale(m: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(m[:, :2]))))


def roll_degrees(lm5: np.ndarray) -> float:
    (lx, ly), (rx, ry) = lm5[0], lm5[1]
    return float(np.degrees(np.arctan2(ry - ly, rx - lx)))


def warp_face(frame: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    """Warp a face crop out of `frame`, pre-shrinking big faces to avoid aliasing."""
    s = matrix_scale(m)
    if s < 0.6:
        h, w = frame.shape[:2]
        inv = invert(m)
        corners = transform_points(np.array([[0, 0], [size, 0], [size, size], [0, size]]), inv)
        x1, y1 = np.floor(corners.min(0)).astype(int) - 2
        x2, y2 = np.ceil(corners.max(0)).astype(int) + 2
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 > 4 and y2 - y1 > 4:
            f = min(1.0, s * 1.5)
            roi = frame[y1:y2, x1:x2]
            small = cv2.resize(roi, (max(1, int(round((x2 - x1) * f))), max(1, int(round((y2 - y1) * f)))),
                               interpolation=cv2.INTER_AREA)
            fx = small.shape[1] / (x2 - x1)
            fy = small.shape[0] / (y2 - y1)
            a = m[:, :2]
            m2 = np.zeros((2, 3), np.float64)
            m2[:, 0] = a[:, 0] / fx
            m2[:, 1] = a[:, 1] / fy
            m2[:, 2] = a @ np.array([x1, y1], np.float64) + m[:, 2]
            return cv2.warpAffine(small, m2, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def paste_region(frame_shape, m: np.ndarray, crop_size: Tuple[int, int]):
    h, w = frame_shape[:2]
    cw, ch = crop_size
    inv = invert(m)
    corners = transform_points(np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]]), inv)
    x1, y1 = np.floor(corners.min(0)).astype(int)
    x2, y2 = np.ceil(corners.max(0)).astype(int)
    return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h), inv


def paste(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Blend `crop` (in face space defined by `m`) back into `frame` in place."""
    ch, cw = crop.shape[:2]
    x1, y1, x2, y2, inv = paste_region(frame.shape, m, (cw, ch))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return frame
    s = matrix_scale(inv)
    if s < 0.7:  # crop is much larger than its footprint → shrink first (prevents shimmering)
        f = min(1.0, s * 1.4)
        nw, nh = max(8, int(round(cw * f))), max(8, int(round(ch * f)))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
        inv = inv.copy()
        inv[:, 0] *= cw / nw
        inv[:, 1] *= ch / nh
    p = inv.copy()
    p[0, 2] -= x1
    p[1, 2] -= y1
    size = (x2 - x1, y2 - y1)
    wm = cv2.warpAffine(mask.astype(np.float32), p, size, flags=cv2.INTER_LINEAR, borderValue=0)
    if float(wm.max()) <= 1e-3:
        return frame
    wc = cv2.warpAffine(crop.astype(np.float32), p, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    wm = np.clip(wm, 0.0, 1.0)[..., None]
    out = roi + (wc - roi) * wm
    frame[y1:y2, x1:x2] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return frame


def similarity_roi(center: Tuple[float, float], side: float, angle: float, size: int) -> np.ndarray:
    """Matrix mapping an (optionally rotated) square ROI of `side` px around `center` to size×size."""
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), float(size) / max(side, 1.0))
    m[0, 2] += size / 2.0 - center[0]
    m[1, 2] += size / 2.0 - center[1]
    return m


def box_from_points(points: np.ndarray) -> np.ndarray:
    x1, y1 = points.min(0)
    x2, y2 = points.max(0)
    return np.array([x1, y1, x2, y2], np.float64)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def paste_mask(canvas: np.ndarray, mask: np.ndarray, m: np.ndarray) -> None:
    """Max-accumulate a face-space mask into a full-frame float mask (debug view)."""
    ch, cw = mask.shape[:2]
    x1, y1, x2, y2, inv = paste_region(canvas.shape, m, (cw, ch))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    p = inv.copy()
    p[0, 2] -= x1
    p[1, 2] -= y1
    wm = cv2.warpAffine(mask.astype(np.float32), p, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR, borderValue=0)
    np.maximum(canvas[y1:y2, x1:x2], wm, out=canvas[y1:y2, x1:x2])
