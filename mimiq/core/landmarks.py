"""Occlusion-robust 68-point face shape.

Glasses, caps, headphones, bandanas, masks and hands hide parts of the face. A landmark network still
returns all 68 points, but the hidden ones are guesses that snap to the edge of the object in front and
jitter from frame to frame. `ShapePrior` keeps a per-face reference shape (in face-aligned coordinates)
and fits it to every new set of points with a robust (iteratively re-weighted) similarity fit: points that
disagree with the rest of the face are recognised as occluded and replaced by where the reference says
they must be. The reference itself slowly adapts to the person (face proportions, expression).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import geometry as geo

# mean 2DFAN4 shape in "arcface_128" unit coordinates (symmetrised)
MEAN_68 = np.array([
    (0.1793, 0.4243), (0.1857, 0.5091), (0.1995, 0.5900), (0.2125, 0.6638),
    (0.2379, 0.7434), (0.2833, 0.8092), (0.3344, 0.8535), (0.3985, 0.8943),
    (0.5000, 0.9197), (0.6015, 0.8943), (0.6656, 0.8535), (0.7167, 0.8092),
    (0.7621, 0.7434), (0.7875, 0.6638), (0.8005, 0.5900), (0.8143, 0.5091),
    (0.8207, 0.4243), (0.2511, 0.3450), (0.2872, 0.3196), (0.3351, 0.3102),
    (0.3793, 0.3145), (0.4186, 0.3252), (0.5814, 0.3252), (0.6207, 0.3145),
    (0.6649, 0.3102), (0.7128, 0.3196), (0.7489, 0.3450), (0.5000, 0.4160),
    (0.5000, 0.4717), (0.5000, 0.5280), (0.5000, 0.5744), (0.4459, 0.6087),
    (0.4688, 0.6151), (0.5000, 0.6215), (0.5312, 0.6151), (0.5541, 0.6087),
    (0.3096, 0.4077), (0.3391, 0.3924), (0.3785, 0.3924), (0.4158, 0.4132),
    (0.3815, 0.4227), (0.3392, 0.4226), (0.5842, 0.4132), (0.6215, 0.3924),
    (0.6609, 0.3924), (0.6904, 0.4077), (0.6608, 0.4226), (0.6185, 0.4227),
    (0.3762, 0.7086), (0.4203, 0.6904), (0.4730, 0.6781), (0.5000, 0.6831),
    (0.5270, 0.6781), (0.5797, 0.6904), (0.6238, 0.7086), (0.5770, 0.7398),
    (0.5395, 0.7565), (0.5000, 0.7597), (0.4605, 0.7565), (0.4230, 0.7398),
    (0.3841, 0.7075), (0.4641, 0.7049), (0.5000, 0.7051), (0.5359, 0.7049),
    (0.6159, 0.7075), (0.5361, 0.7208), (0.5000, 0.7233), (0.4639, 0.7208),
], np.float64)

# index groups (iBUG 68 layout) used for drawing
JAW = tuple(range(0, 17))
BROW_L, BROW_R = tuple(range(17, 22)), tuple(range(22, 27))
NOSE_BRIDGE, NOSE_BASE = tuple(range(27, 31)), tuple(range(31, 36))
EYE_L, EYE_R = tuple(range(36, 42)), tuple(range(42, 48))
LIPS_OUT, LIPS_IN = tuple(range(48, 60)), tuple(range(60, 68))
OPEN_PATHS = (JAW, BROW_L, BROW_R, NOSE_BRIDGE + (33,), NOSE_BASE)
CLOSED_PATHS = (EYE_L, EYE_R, LIPS_OUT, LIPS_IN)


def five_from_68(lm68: np.ndarray) -> np.ndarray:
    return np.array([lm68[36:42].mean(0), lm68[42:48].mean(0), lm68[30], lm68[48], lm68[54]], np.float64)


MEAN_FIVE = five_from_68(MEAN_68)


def weighted_similarity(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares similarity src → dst."""
    w = np.maximum(np.asarray(w, np.float64), 1e-6)
    w = w / w.sum()
    ms, md = (src * w[:, None]).sum(0), (dst * w[:, None]).sum(0)
    s, d = src - ms, dst - md
    a = (d * w[:, None]).T @ s
    u, sv, vt = np.linalg.svd(a)
    e = np.ones(2)
    if np.linalg.det(u @ vt) < 0:
        e[-1] = -1
    r = u @ np.diag(e) @ vt
    var = float((w * (s ** 2).sum(1)).sum())
    scale = float(sv @ e) / var if var > 1e-12 else 1.0
    m = np.zeros((2, 3), np.float64)
    m[:, :2] = scale * r
    m[:, 2] = md - scale * (r @ ms)
    return m


def shape_from_five(lm5: np.ndarray) -> np.ndarray:
    """Approximate 68 points from 5 (used when the 68-point network is off)."""
    return geo.transform_points(MEAN_68, geo.umeyama(MEAN_FIVE, lm5))


class ShapePrior:
    SIGMA = 0.035      # residual scale of a normal point (face-aligned units, face width ≈ 0.62)
    OUTLIER = 0.085    # residual above which a point counts as hidden
    CONF_LOW = 0.30    # heat-map confidence below which a point is not trusted

    def __init__(self):
        self.ref = MEAN_68.copy()
        self.frames = 0

    def reset(self) -> None:
        self.ref = MEAN_68.copy()
        self.frames = 0

    def fuse(self, pts: np.ndarray, conf: Optional[np.ndarray] = None,
             update: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (fused points, hidden flags, quality 0..1) for raw landmarks `pts` (68×2, image space)."""
        pts = np.asarray(pts, np.float64)
        c = np.ones(68) if conf is None else np.asarray(conf, np.float64).reshape(68)
        base = np.clip((c - 0.2) / 0.45, 0.05, 1.0)
        w = base.copy()
        m = weighted_similarity(self.ref, pts, w)
        for _ in range(4):
            scale = max(geo.matrix_scale(m), 1e-6)
            r = np.linalg.norm(geo.transform_points(self.ref, m) - pts, axis=1) / scale
            w = base / (1.0 + (r / self.SIGMA) ** 2)
            m = weighted_similarity(self.ref, pts, w)
        scale = max(geo.matrix_scale(m), 1e-6)
        pred = geo.transform_points(self.ref, m)
        r = np.linalg.norm(pred - pts, axis=1) / scale
        hidden = (r > self.OUTLIER) | (c < self.CONF_LOW)
        # soft hand-over between measurement and prediction
        k = np.clip((r - 0.6 * self.OUTLIER) / (0.6 * self.OUTLIER), 0.0, 1.0)
        k = np.maximum(k, np.clip((self.CONF_LOW + 0.1 - c) / 0.2, 0.0, 1.0))
        fused = pts + (pred - pts) * k[:, None]
        vis = ~hidden
        # quality: share of points that agree with the face, and how sure the network is about them
        quality = float(vis.mean()) * float(np.clip(np.median(c[vis]) / 0.6, 0.0, 1.0)) if vis.any() else 0.0
        if update and quality > 0.55:
            obs = geo.transform_points(pts, geo.invert(m))
            rate = 0.35 if self.frames < 8 else 0.08
            self.ref[vis] += (obs[vis] - self.ref[vis]) * rate
            self.frames += 1
        return fused, hidden, quality
