"""Background replacement: video matting → clean, stable alpha → blur / picture / colour behind the person.

Quality notes (what makes the edge look "real" instead of a cut-out with a glow):

* Robust Video Matting is recurrent: it keeps hidden state between frames, so hair and shoulders don't flicker
  the way per-frame segmentation does. A motion-adaptive filter calms the remaining sub-pixel noise but never
  lags behind real movement.
* Faint "ghost" alpha in the background (shiny furniture, reflections) is cut off and small detached islands
  are dropped, so nothing of the old room shines through the new background.
* The blur only mixes *background* pixels (normalised convolution): the person's colours never bleed into it,
  so there is no halo around the head.
* Semi-transparent edges (hair) get the matting model's colour-decontaminated foreground instead of the raw
  camera pixels, so the old background's colour doesn't fringe the hair; a light wrap blends the edge with the
  new picture.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("mimiq.background")

MODES = ("off", "blur", "image", "color")
MAX_SIDE = 1280            # matting resolution cap (long side); larger frames are matted at this size
INTERNAL = {"rvm_mobilenetv3": 512, "rvm_resnet50": 640}   # long side seen by the coarse network


@dataclass
class Matte:
    alpha: np.ndarray          # float32 h × w, 0..1 (matting resolution)
    fgr: np.ndarray            # float32 h × w × 3, BGR 0..255 — colour-decontaminated foreground
    src: np.ndarray            # uint8   h × w × 3 — the frame the matte was computed on (same size)
    size: Tuple[int, int]      # (w, h) of the camera frame


def parse_color(text: str) -> Tuple[int, int, int]:
    """'#RRGGBB' → BGR tuple (falls back to graphite)."""
    t = (text or "").strip().lstrip("#")
    try:
        r, g, b = int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
        return b, g, r
    except (ValueError, IndexError):
        return 35, 31, 31


class BackgroundMatter:
    """Robust Video Matting (MobileNetV3 / ResNet-50) with recurrent state and temporal clean-up."""

    def __init__(self, key: str, session):
        self.key = key
        self.session = session
        self.internal = INTERNAL.get(key, 512)
        self._out_names = [o.name for o in session.get_outputs()]
        self.reset()

    def reset(self) -> None:
        z = np.zeros((1, 1, 1, 1), np.float32)
        self._rec = [z, z, z, z]
        self._shape: Optional[Tuple[int, int]] = None
        self._prev: Optional[np.ndarray] = None

    def warmup(self) -> None:
        self.matte(np.full((720, 1280, 3), 127, np.uint8), 0.0)
        self.reset()

    def matte(self, frame: np.ndarray, stability: float = 0.5) -> Matte:
        h, w = frame.shape[:2]
        k = min(1.0, MAX_SIDE / max(h, w))
        img = frame if k >= 1.0 else cv2.resize(frame, (max(2, round(w * k)), max(2, round(h * k))),
                                                interpolation=cv2.INTER_AREA)
        ph, pw = img.shape[:2]
        if self._shape != (ph, pw):
            self.reset()
            self._shape = (ph, pw)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, swapRB=True)          # 1 × 3 × h × w, RGB 0..1
        ratio = np.array([min(1.0, self.internal / max(ph, pw))], np.float32)
        feed = {"src": blob, "r1i": self._rec[0], "r2i": self._rec[1], "r3i": self._rec[2], "r4i": self._rec[3],
                "downsample_ratio": ratio}
        out = dict(zip(self._out_names, self.session.run(None, feed)))
        self._rec = [out["r1o"], out["r2o"], out["r3o"], out["r4o"]]
        a = out["pha"][0, 0]
        f = out["fgr"][0]
        fgr = cv2.merge([f[2], f[1], f[0]])
        fgr *= 255.0
        a = self._clean(a)
        a = self._stabilise(a, stability)
        return Matte(a, fgr, img, (w, h))

    # ------------------------------------------------------------------ clean-up
    @staticmethod
    def _clean(a: np.ndarray) -> np.ndarray:
        # cut faint haze (reflections / glossy furniture) and snap near-solid areas to fully opaque
        a = np.clip((a - 0.035) * (1.0 / 0.94), 0.0, 1.0)
        h, w = a.shape
        sw, sh = max(8, w // 4), max(8, h // 4)
        small = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
        solid = (small > 0.5).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        if n > 2:                         # background + more than one blob → drop small detached islands
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_min = max(0.04 * float(areas.max()), 0.003 * sw * sh)
            keep_ids = np.flatnonzero(areas >= keep_min) + 1
            keep = np.isin(labels, keep_ids).astype(np.uint8)
            keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.5)
            a = a * np.clip(cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR) * 1.5, 0.0, 1.0)
        return a

    def _stabilise(self, a: np.ndarray, stability: float) -> np.ndarray:
        prev = self._prev
        if prev is not None and prev.shape == a.shape and stability > 0:
            d = cv2.absdiff(a, prev)
            kmin = 1.0 - 0.85 * float(np.clip(stability, 0.0, 1.0))
            gain = np.clip(d * 5.0, kmin, 1.0)             # big change = real motion → follow at once
            a = prev + (a - prev) * gain
        self._prev = a
        return a


class BackgroundCompositor:
    """Puts the person (the processed output frame) over the chosen background."""

    def __init__(self):
        self._image_cache: Dict[tuple, np.ndarray] = {}
        self._image_error: Optional[str] = None

    @property
    def image_error(self) -> Optional[str]:
        return self._image_error

    def apply(self, out: np.ndarray, camera: np.ndarray, m: Matte, mode: str, *, blur: float = 0.6,
              image: str = "", image_blur: float = 0.0, color: str = "") -> np.ndarray:
        if mode not in ("blur", "image", "color"):
            return out
        H, W = out.shape[:2]
        mh, mw = m.alpha.shape
        same = (mw, mh) == (W, H)
        a = m.alpha if same else cv2.resize(m.alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        if mode == "image":
            bg = self._image(image, W, H, image_blur)
            if bg is None:
                bg = self._blur(camera, a, blur)
        elif mode == "color":
            bg = np.empty((H, W, 3), np.uint8)
            bg[:] = parse_color(color)
        else:
            bg = self._blur(camera, a, blur)

        # solid person and clean background are plain copies; only the thin edge band needs per-pixel maths
        res = bg.copy()
        cv2.copyTo(out, (a >= 1.0).view(np.uint8), res)
        wrap = glow = None
        if mode != "blur":
            wrap, glow = self._light_wrap_maps(bg, a, 0.32)
            edge = ((a > 0.0) & (a < 1.0)) | ((wrap > 0.003) & (a > 0.0))
        else:
            edge = (a > 0.0) & (a < 1.0)
        ys, xs = np.nonzero(edge)
        if ys.size == 0:
            return res
        av = a[ys, xs][:, None]
        fg = out[ys, xs].astype(np.float32)
        # colour-decontaminated foreground on semi-transparent pixels (hair): no old-background fringe
        band = np.clip((1.0 - av) * 3.0, 0.0, 1.0)
        band[av < 0.004] = 0.0
        if same:
            resid = m.fgr[ys, xs] - m.src[ys, xs].astype(np.float32)
        else:
            sy = np.minimum((ys * (mh / H)).astype(np.int32), mh - 1)
            sx = np.minimum((xs * (mw / W)).astype(np.int32), mw - 1)
            resid = m.fgr[sy, sx] - m.src[sy, sx].astype(np.float32)
        fg += resid * band
        if wrap is not None:
            wv = wrap[ys, xs][:, None]
            fg += (glow[ys, xs].astype(np.float32) - fg) * wv
        bv = bg[ys, xs].astype(np.float32)
        res[ys, xs] = np.clip(bv + (fg - bv) * av, 0, 255).astype(np.uint8)
        return res

    # ------------------------------------------------------------------ backgrounds
    @staticmethod
    def _blur(camera: np.ndarray, a: np.ndarray, strength: float) -> np.ndarray:
        """Blur of the background only: the person is excluded, so its colours can't bleed into a halo."""
        H, W = a.shape
        sigma_full = (6.0 + 54.0 * float(np.clip(strength, 0.0, 1.0))) * (H / 720.0)
        f = max(2.0, sigma_full / 2.5)                       # work where the kernel is ~2.5 px wide
        sw, sh = max(8, round(W / f)), max(8, round(H / f))
        sigma = sigma_full * sw / W
        img = cv2.resize(camera, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        sa = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
        sa = cv2.dilate(sa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))   # keep away from the edge
        wgt = (1.0 - sa) ** 2
        num = cv2.GaussianBlur(img * wgt[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(wgt, (0, 0), sigma)
        # where the person covers the whole kernel, borrow from a much wider neighbourhood
        num2 = cv2.GaussianBlur(num, (0, 0), sigma * 3)
        den2 = cv2.GaussianBlur(den, (0, 0), sigma * 3)
        t = np.clip(den * 8.0, 0.0, 1.0)[..., None]
        bg = num / np.maximum(den, 1e-4)[..., None] * t + num2 / np.maximum(den2, 1e-4)[..., None] * (1.0 - t)
        empty = den2 < 1e-3
        if empty.any():                                    # the person fills the frame: plain blur
            bg[empty] = cv2.GaussianBlur(img, (0, 0), sigma * 3)[empty]
        return cv2.resize(np.clip(bg, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)

    def _image(self, path: str, W: int, H: int, blur: float) -> Optional[np.ndarray]:
        if not path:
            self._image_error = "Изображение для фона не выбрано"
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self._image_error = "Файл фона не найден"
            return None
        key = (path, mtime, W, H, round(float(blur), 2))
        img = self._image_cache.get(key)
        if img is None:
            from ..imaging import read_image
            try:
                src = read_image(path)
            except Exception as exc:
                self._image_error = f"Не удалось открыть фон: {exc}"
                return None
            img = cover(src, W, H)
            if blur > 0.005:
                sw, sh = max(8, W // 4), max(8, H // 4)
                small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), 0.5 + 12.0 * float(blur))
                img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
            if len(self._image_cache) > 6:
                self._image_cache.clear()
            self._image_cache[key] = img
        self._image_error = None
        return img

    @staticmethod
    def _light_wrap_maps(bg: np.ndarray, a: np.ndarray, strength: float):
        """Where (and with what colour) the new background's light spills over the very edge of the person."""
        H, W = a.shape
        sw, sh = max(8, W // 4), max(8, H // 4)
        sa = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
        outside = cv2.GaussianBlur(1.0 - sa, (0, 0), 2.0)
        wrap = np.clip(outside * sa * 2.0, 0.0, 1.0) * strength
        glow = cv2.GaussianBlur(cv2.resize(bg, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 3.0)
        return (cv2.resize(wrap, (W, H), interpolation=cv2.INTER_LINEAR),
                cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR))


def cover(img: np.ndarray, W: int, H: int) -> np.ndarray:
    """Scale and centre-crop `img` to exactly W × H (like CSS background-size: cover)."""
    h, w = img.shape[:2]
    k = max(W / w, H / h)
    nw, nh = max(W, round(w * k)), max(H, round(h * k))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    x, y = (nw - W) // 2, (nh - H) // 2
    return np.ascontiguousarray(img[y:y + H, x:x + W])


def list_images(folder: Path) -> list:
    from ..imaging import IMAGE_EXTS
    try:
        files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    except OSError:
        return []
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
