"""Temporal filters used to stabilise landmarks and masks."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter (Casiez et al.) with a single speed for the whole point set.

    Using one shared speed keeps the landmark constellation rigid, so the face crop
    does not wobble while the head is still and follows quickly when it moves.
    Speeds are measured in face-sizes per second to make the tuning scale-invariant.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 8.0, d_cutoff: float = 1.2):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None
        self.t: Optional[float] = None

    def __call__(self, x: np.ndarray, t: float, scale: float = 1.0) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x is None or self.t is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1.0 - a_d) * self.dx
        speed = float(np.mean(np.abs(self.dx))) / max(scale, 1.0)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(dt, cutoff)
        self.x = a * x + (1.0 - a) * self.x
        self.t = t
        return self.x.copy()

    @property
    def velocity(self) -> Optional[np.ndarray]:
        return None if self.dx is None else self.dx.copy()


def smoothing_to_params(strength: float):
    """Map a 0..1 UI slider to One Euro parameters (0 = raw, 1 = very steady)."""
    s = float(np.clip(strength, 0.0, 1.0))
    min_cutoff = 6.0 * (1.0 - s) ** 2 + 0.35 * s
    beta = 10.0 + 10.0 * s       # follows fast head motion without trailing behind
    return min_cutoff, beta
