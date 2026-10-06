"""Brand rendering: logo pixmaps, the on-video "AI" disclosure label and the virtual-camera placeholder."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .. import paths
from ..imaging import qimage_to_bgra
from . import theme


def _logo_svg() -> bytes:
    return (paths.package_dir() / "assets" / "logo.svg").read_bytes()


@lru_cache(maxsize=16)
def logo_image(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    r = QSvgRenderer(QByteArray(_logo_svg()))
    p = QPainter(img)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
    r.render(p, QRectF(0, 0, size, size))
    p.end()
    return img


@lru_cache(maxsize=16)
def logo_pixmap(size: int) -> QPixmap:
    dpr = 2.0
    pm = QPixmap.fromImage(logo_image(int(size * dpr)))
    pm.setDevicePixelRatio(dpr)
    return pm


def _font(px: float, weight: QFont.Weight) -> QFont:
    f = theme.ui_font(10, weight)
    f.setPixelSize(max(6, int(round(px))))
    return f


def watermark_bgra(out_height: int) -> np.ndarray:
    """Small "AI · Mimiq" label placed in the corner of the outgoing video (disclosure)."""
    h = max(20, int(round(out_height * 0.036)))
    f = _font(h * 0.5, QFont.Weight.DemiBold)
    fm = QFontMetricsF(f)
    text = "AI · Mimiq"
    pad = h * 0.42
    w = int(round(fm.horizontalAdvance(text) + 2 * pad))
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0, 120))
    p.drawRoundedRect(QRectF(0, 0, w, h), h * 0.2, h * 0.2)
    p.setFont(f)
    p.setPen(QColor(255, 255, 255, 220))
    p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, text)
    p.end()
    return qimage_to_bgra(img)


def placeholder_bgr(width: int, height: int, subtitle: str = "Камера на паузе") -> np.ndarray:
    """Frame shown by the virtual camera while no video is being processed."""
    img = QImage(width, height, QImage.Format.Format_RGB32)
    img.fill(QColor(theme.BG))
    p = QPainter(img)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform |
                     QPainter.RenderHint.TextAntialiasing)
    s = int(height * 0.12)
    top = height * 0.36
    p.drawImage(QRectF((width - s) / 2, top, s, s), logo_image(max(64, s * 2)))
    p.setFont(_font(height * 0.05, QFont.Weight.DemiBold))
    p.setPen(QColor(theme.TEXT))
    p.drawText(QRectF(0, top + s + height * 0.03, width, height * 0.07),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "Mimiq")
    p.setFont(_font(height * 0.03, QFont.Weight.Normal))
    p.setPen(QColor(theme.DIM))
    p.drawText(QRectF(0, top + s + height * 0.105, width, height * 0.06),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, subtitle)
    p.end()
    bgra = qimage_to_bgra(img)
    return np.ascontiguousarray(bgra[..., :3])
