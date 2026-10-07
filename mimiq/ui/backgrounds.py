"""Background tab helpers: picture gallery, colour swatches and the built-in studio backdrops."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QGridLayout, QMenu, QToolButton, QWidget

from .. import paths
from ..core.background import cover, list_images
from ..imaging import read_image, to_qimage
from . import icons, theme
from .widgets import POINTER

log = logging.getLogger("mimiq.ui.backgrounds")

THUMB = QSize(96, 54)
SWATCHES = [("#202024", "Графит"), ("#F2F2F4", "Светлый"), ("#2B6CD4", "Синий"), ("#00B140", "Хромакей (для OBS)")]
_MARKER = ".defaults-v1"


# ---------------------------------------------------------------------- built-in backdrops
def _backdrop(w: int, h: int, center: Sequence[int], edge: Sequence[int], light=(0.5, 0.32), spread=0.9,
              bokeh: Sequence = (), seed: int = 0) -> np.ndarray:
    """Soft studio backdrop: radial light falloff, optional out-of-focus lights, fine grain against banding."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / w - light[0], (yy / h - light[1]) * (h / w) * 1.6
    r = np.sqrt(nx * nx + ny * ny) / spread
    t = np.clip(r, 0.0, 1.0) ** 1.4
    c, e = np.array(center, np.float32)[::-1], np.array(edge, np.float32)[::-1]   # RGB → BGR
    img = c * (1.0 - t[..., None]) + e * t[..., None]
    for bx, by, br, col, k in bokeh:
        d = np.sqrt((xx - bx * w) ** 2 + (yy - by * h) ** 2) / (br * w)
        m = np.clip(1.0 - d, 0.0, 1.0) ** 0.6 * k
        img = img * (1.0 - m[..., None]) + np.array(col, np.float32)[::-1] * m[..., None]
    img = cv2.GaussianBlur(img, (0, 0), w / 160)
    img += np.random.default_rng(seed).normal(0.0, 1.6, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def ensure_defaults(folder: Optional[Path] = None) -> None:
    """Put a few neutral backdrops into the gallery once (deleted ones don't come back)."""
    folder = folder or paths.backgrounds_dir()
    marker = folder / _MARKER
    if marker.exists():
        return
    w, h = 1920, 1080
    items = {
        "Графит.jpg": _backdrop(w, h, (52, 52, 58), (16, 16, 19), seed=1),
        "Студия.jpg": _backdrop(w, h, (112, 128, 148), (30, 36, 46), light=(0.45, 0.25), seed=2),
        "Тёплая стена.jpg": _backdrop(w, h, (214, 196, 172), (112, 96, 80), light=(0.35, 0.2), spread=1.1, seed=3),
        "Вечер.jpg": _backdrop(w, h, (44, 52, 92), (12, 13, 26), light=(0.6, 0.35), seed=4,
                               bokeh=[(0.12, 0.3, 0.07, (250, 196, 120), 0.35), (0.22, 0.62, 0.05, (255, 170, 90), 0.3),
                                      (0.84, 0.22, 0.06, (140, 170, 255), 0.3), (0.9, 0.7, 0.08, (255, 210, 150), 0.25),
                                      (0.7, 0.12, 0.04, (255, 240, 200), 0.3)]),
    }
    for name, img in items.items():
        p = folder / name
        if not p.exists():
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            if ok:
                buf.tofile(str(p))
    marker.write_text("ok", encoding="utf-8")


def import_image(src: str) -> Path:
    """Copy a picture into the gallery folder, so the background keeps working if the original moves."""
    folder = paths.backgrounds_dir()
    s = Path(src)
    read_image(s)                                    # fail early with a clear message if it's not a picture
    target = folder / s.name
    if target.exists():
        if target.stat().st_size == s.stat().st_size:
            return target
        target = folder / f"{s.stem}-{int(time.time())}{s.suffix}"
    shutil.copy2(s, target)
    return target


# ---------------------------------------------------------------------- gallery
_thumbs: Dict[tuple, QPixmap] = {}


def _rounded(pix: QPixmap, radius: float = 5.0) -> QPixmap:
    out = QPixmap(pix.size())
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, pix.width(), pix.height(), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pix)
    p.end()
    return out


def thumbnail(path: Path, dpr: float = 2.0) -> Optional[QPixmap]:
    try:
        key = (str(path), path.stat().st_mtime, dpr)
    except OSError:
        return None
    pix = _thumbs.get(key)
    if pix is None:
        try:
            img = cover(read_image(path), int(THUMB.width() * dpr), int(THUMB.height() * dpr))
        except Exception as exc:
            log.debug("thumbnail failed for %s: %s", path, exc)
            return None
        pix = QPixmap.fromImage(to_qimage(img))
        pix.setDevicePixelRatio(dpr)
        pix = _rounded(pix, 5.0 * dpr)
        pix.setDevicePixelRatio(dpr)
        _thumbs[key] = pix
    return pix


class BackgroundGallery(QWidget):
    picked = Signal(str)
    addRequested = Signal()
    removeRequested = Signal(str)

    COLS = 3

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self._buttons: List[QToolButton] = []
        self._current = ""

    def _tile(self) -> QToolButton:
        b = QToolButton()
        b.setObjectName("bgTile")
        b.setCursor(POINTER)
        b.setFixedSize(THUMB.width() + 8, THUMB.height() + 8)
        b.setIconSize(THUMB)
        b.setStyleSheet(
            f"QToolButton#bgTile {{ background: {theme.FIELD}; border: 2px solid transparent; border-radius: 8px; "
            f"padding: 0; }} QToolButton#bgTile:hover {{ border-color: {theme.LINE_HI}; }} "
            f"QToolButton#bgTile:checked {{ border-color: {theme.ACCENT}; }}")
        return b

    def set_items(self, files: Sequence[Path], current: str) -> None:
        for b in self._buttons:
            b.deleteLater()
        self._buttons = []
        self._current = current
        dpr = max(1.0, self.devicePixelRatioF())
        add = self._tile()
        add.setIcon(icons.icon("plus", theme.DIM, 20))
        add.setIconSize(QSize(20, 20))
        add.setToolTip("Добавить своё изображение")
        add.clicked.connect(self.addRequested.emit)
        tiles = [add]
        for f in files:
            pix = thumbnail(f, dpr)
            if pix is None:
                continue
            b = self._tile()
            b.setCheckable(True)
            b.setIcon(QIcon(pix))
            b.setToolTip(f.stem + "\nПравая кнопка — удалить")
            b.setChecked(str(f) == current)
            b.clicked.connect(lambda _=False, p=str(f): self._pick(p))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _pos, p=str(f), btn=b: self._menu(p, btn))
            b.setProperty("path", str(f))
            tiles.append(b)
        for i, b in enumerate(tiles):
            self.grid.addWidget(b, i // self.COLS, i % self.COLS, Qt.AlignmentFlag.AlignLeft)
        self._buttons = tiles

    def _pick(self, path: str) -> None:
        self.set_current(path)
        self.picked.emit(path)

    def set_current(self, path: str) -> None:
        self._current = path
        for b in self._buttons:
            p = b.property("path")
            if p:
                b.setChecked(p == path)

    def _menu(self, path: str, btn: QToolButton) -> None:
        m = QMenu(self)
        act = m.addAction(icons.icon("trash", theme.BAD, 16), "Удалить фон")
        if m.exec(btn.mapToGlobal(btn.rect().bottomLeft())) is act:
            self.removeRequested.emit(path)


class ColorSwatches(QWidget):
    picked = Signal(str)
    customRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self._buttons: Dict[str, QToolButton] = {}
        for i, (col, name) in enumerate(SWATCHES):
            b = self._swatch(col)
            b.setToolTip(name)
            b.clicked.connect(lambda _=False, c=col: self.picked.emit(c))
            lay.addWidget(b, 0, i)
            self._buttons[col] = b
        self.custom = self._swatch(None)
        self.custom.setToolTip("Свой цвет…")
        self.custom.clicked.connect(self.customRequested.emit)
        lay.addWidget(self.custom, 0, len(SWATCHES))
        lay.setColumnStretch(len(SWATCHES) + 1, 1)

    @staticmethod
    def _style(fill: str) -> str:
        return (f"QToolButton {{ background: {fill}; border: 2px solid {theme.LINE_HI}; border-radius: 20px; }} "
                f"QToolButton:hover {{ border-color: {theme.DIM}; }} "
                f"QToolButton:checked {{ border: 3px solid {theme.ACCENT}; }}")

    @classmethod
    def _swatch(cls, color: Optional[str]) -> QToolButton:
        b = QToolButton()
        b.setCursor(POINTER)
        b.setCheckable(True)
        b.setFixedSize(40, 40)
        b.setStyleSheet(cls._style(color or theme.FIELD))
        if color is None:
            b.setIcon(icons.icon("palette", theme.DIM, 18))
            b.setIconSize(QSize(18, 18))
        return b

    def set_value(self, color: str) -> None:
        col = (color or "").upper()
        known = False
        for c, b in self._buttons.items():
            on = c.upper() == col
            known |= on
            b.setChecked(on)
        self.custom.setChecked(not known)
        q = QColor(color) if color else QColor()
        self.custom.setStyleSheet(self._style(q.name() if not known and q.isValid() else theme.FIELD))
