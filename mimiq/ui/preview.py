"""Live preview: video canvas with split-compare, a small HUD, empty/waiting states and model progress."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..core.tuner import rate_text
from . import brand, icons, theme
from .widgets import POINTER, Spinner, circle_pixmap, label

RADIUS = 8.0
CANVAS = "#0C0C0E"


def likeness_word(v: float) -> str:
    """ArcFace cosine of the output vs. the photo → plain words (0.7+ is what a good swap reaches)."""
    if v >= 0.68:
        return "высокое"
    if v >= 0.55:
        return "хорошее"
    if v >= 0.4:
        return "среднее"
    return "низкое"


class _EmptyState(QWidget):
    """Centered logo/title/subtitle/buttons shown when there is no live video."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(0)
        lay.addStretch(1)
        self.logo = QLabel()
        self.logo.setPixmap(brand.logo_pixmap(56))
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner = Spinner(36, 3.0)
        spin_row = QHBoxLayout()
        spin_row.addStretch(1)
        spin_row.addWidget(self.spinner)
        spin_row.addStretch(1)
        lay.addWidget(self.logo)
        lay.addLayout(spin_row)
        lay.addSpacing(22)
        self.title = label("", "h1")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-size: 15pt; font-weight: 600;")
        lay.addWidget(self.title)
        lay.addSpacing(8)
        self.subtitle = label("", "dim", wrap=True)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setFixedWidth(450)
        sub_row = QHBoxLayout()
        sub_row.addStretch(1)
        sub_row.addWidget(self.subtitle)
        sub_row.addStretch(1)
        lay.addLayout(sub_row)
        lay.addSpacing(24)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch(1)
        self.primary = QPushButton()
        self.primary.setObjectName("primary")
        self.primary.setCursor(POINTER)
        self.primary.setIconSize(QSize(18, 18))
        self.secondary = QPushButton()
        self.secondary.setCursor(POINTER)
        self.secondary.setIconSize(QSize(18, 18))
        row.addWidget(self.primary)
        row.addWidget(self.secondary)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        self.hint = label("", "faint")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.hint)

    def configure(self, *, busy: bool, title: str, subtitle: str, primary: Optional[tuple], secondary: Optional[tuple],
                  hint: str = "") -> None:
        self.logo.setVisible(not busy)
        self.spinner.setVisible(busy)
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self.subtitle.setVisible(bool(subtitle))
        for btn, spec in ((self.primary, primary), (self.secondary, secondary)):
            btn.setVisible(spec is not None)
            if spec is not None:
                text, ic = spec
                btn.setText("  " + text if ic else text)
                btn.setIcon(icons.icon(ic, "#FFFFFF" if btn is self.primary else theme.TEXT, 18) if ic else QIcon())
        self.hint.setText(hint)
        self._update_hint()

    def set_hint_suppressed(self, on: bool) -> None:
        """Hide the bottom hint line while notifications are shown over it."""
        self._suppress_hint = on
        self._update_hint()

    def set_hint_covered(self, on: bool) -> None:
        """Hide the bottom hint line while the model-loading card is drawn over it."""
        if getattr(self, "_covered", False) != on:
            self._covered = on
            self._update_hint()

    def _update_hint(self) -> None:
        # keep the space reserved, so the centred content does not jump when the hint hides
        self.hint.setVisible(bool(self.hint.text()))
        hidden = getattr(self, "_suppress_hint", False) or getattr(self, "_covered", False)
        self.hint.setStyleSheet("color: transparent;" if hidden else "")


class PreviewWidget(QWidget):
    toggleFullscreen = Signal()
    primaryClicked = Signal(str)     # action id of the empty-state primary button
    secondaryClicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(520, 320)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self._result: Optional[QImage] = None
        self._original: Optional[QImage] = None
        self._mode = "result"
        self._split = 0.5
        self._drag = False
        self._state = "idle"           # idle | waiting | live | error
        self._source_kind = "link"
        self._message = ""
        self._show_hud = True
        self._fps = 0.0
        self._latency = 0.0
        self._source = ""
        self._resolution = ""
        self._faces: List[dict] = []
        self._face_fps = 0.0
        self._fluid = False
        self._likeness: Optional[float] = None
        self._identity_name = ""
        self._identity_avatar: Optional[QPixmap] = None
        self._swap_enabled = True
        self._models_ready = False
        self._loading: Optional[tuple] = None
        self._recording = False
        self._blink = True
        self._actions = ("", "")
        self._f_hud = theme.ui_font(8.8, QFont.Weight.Medium)
        self._f_small = theme.ui_font(8.5, QFont.Weight.Normal)
        self.empty = _EmptyState(self)
        self.empty.primary.clicked.connect(lambda: self.primaryClicked.emit(self._actions[0]))
        self.empty.secondary.clicked.connect(lambda: self.secondaryClicked.emit(self._actions[1]))
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(600)
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._refresh_empty()

    # ------------------------------------------------------------------ public API
    def set_frames(self, result: Optional[QImage], original: Optional[QImage]) -> None:
        self._result = result
        self._original = original
        if self._state != "live":
            self._state = "live"
            self._refresh_empty()
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_state(self, state: str, message: str = "", source_kind: Optional[str] = None) -> None:
        if source_kind is not None:
            self._source_kind = source_kind
        self._state = state
        self._message = message
        if state != "live":
            self._result = self._original = None
        self._refresh_empty()
        self.update()

    def set_source_kind(self, kind: str) -> None:
        self._source_kind = kind
        self._refresh_empty()

    def set_hud(self, fps: float, latency: float, source: str, resolution: str, faces: List[dict]) -> None:
        self._fps, self._latency, self._source, self._resolution, self._faces = fps, latency, source, resolution, faces
        if self._state == "live":
            self.update()

    def set_perf(self, face_fps: float, fluid: bool, likeness: Optional[float]) -> None:
        self._face_fps, self._fluid, self._likeness = face_fps, fluid, likeness

    def set_identity(self, name: str, thumb: Optional[QImage]) -> None:
        self._identity_name = name
        self._identity_avatar = circle_pixmap(thumb, 22) if thumb is not None else None
        self.update()

    def set_swap_enabled(self, on: bool) -> None:
        self._swap_enabled = on
        self.update()

    def set_show_hud(self, on: bool) -> None:
        self._show_hud = on
        self.update()

    def set_models_ready(self, ready: bool) -> None:
        self._models_ready = ready
        self.update()

    def set_loading(self, stage: Optional[str], frac: float = 0.0, text: str = "") -> None:
        self._loading = None if stage in (None, "done") else (stage, frac, text)
        self.empty.set_hint_covered(self._loading is not None)
        self.update()

    def set_recording(self, on: bool) -> None:
        self._recording = on
        if on:
            self._blink_timer.start()
        else:
            self._blink_timer.stop()
        self.update()

    def _toggle_blink(self) -> None:
        self._blink = not self._blink
        self.update()

    # ------------------------------------------------------------------ empty state
    def _refresh_empty(self) -> None:
        live = self._state == "live"
        self.empty.setVisible(not live)
        if live:
            return
        link = self._source_kind == "link"
        if self._state == "waiting":
            if link:
                self._actions = ("qr", "")
                self.empty.configure(busy=True, title="Ждём видео с iPhone…",
                                     subtitle="Отсканируйте QR-код камерой iPhone, откройте ссылку в Safari и "
                                              "нажмите «Включить камеру».",
                                     primary=("Показать QR-код", "qr"), secondary=None,
                                     hint="iPhone и компьютер должны быть в одной сети Wi-Fi")
            else:
                self._actions = ("", "")
                self.empty.configure(busy=True, title="Подключение к камере…",
                                     subtitle=self._message or "Это может занять несколько секунд.",
                                     primary=None, secondary=None)
        elif self._state == "error":
            self._actions = ("retry", "settings")
            self.empty.configure(busy=False, title="Нет сигнала", subtitle=self._message or "Источник недоступен.",
                                 primary=("Повторить", "refresh"), secondary=("Настройки источника", "sliders"))
        else:
            if link:
                self._actions = ("qr", "start")
                self.empty.configure(busy=False, title="Mimiq готов к работе",
                                     subtitle="Подключите iPhone по Wi-Fi через QR-код — видео запустится "
                                              "автоматически. Другую камеру можно выбрать в настройках источника.",
                                     primary=("Подключить iPhone", "qr"), secondary=("Старт", "play"),
                                     hint="Пробел — старт/стоп  ·  F11 — полный экран")
            else:
                self._actions = ("start", "")
                self.empty.configure(busy=False, title="Камера не запущена",
                                     subtitle="Нажмите «Старт», чтобы начать обработку видео.",
                                     primary=("Старт", "play"), secondary=None,
                                     hint="Пробел — старт/стоп  ·  F11 — полный экран")

    def resizeEvent(self, e):  # noqa: N802
        self.empty.setGeometry(self.rect())
        super().resizeEvent(e)

    # ------------------------------------------------------------------ geometry
    def _image_rect(self, img: QImage) -> QRectF:
        r = QRectF(self.rect())
        if img is None or img.isNull() or img.width() == 0:
            return r
        s = min(r.width() / img.width(), r.height() / img.height())
        w, h = img.width() * s, img.height() * s
        return QRectF(r.left() + (r.width() - w) / 2, r.top() + (r.height() - h) / 2, w, h)

    def _divider_x(self) -> float:
        img = self._result or self._original
        ir = self._image_rect(img) if img is not None else QRectF(self.rect())
        return ir.left() + ir.width() * self._split

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform |
                         QPainter.RenderHint.TextAntialiasing)
        full = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(CANVAS))
        p.drawRoundedRect(full, RADIUS, RADIUS)
        if self._state == "live" and (self._result is not None or self._original is not None):
            self._paint_video(p)
            self._mask_corners(p, full)
        border = full.adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(border, RADIUS, RADIUS)
        if self._state == "live" and self._show_hud:
            self._paint_hud(p)
        if self._loading:
            self._paint_loading(p)

    def _mask_corners(self, p: QPainter, r: QRectF) -> None:
        """Round the canvas corners by painting the window colour over them (much cheaper than clipping
        every video frame to an anti-aliased path)."""
        outer = QPainterPath()
        outer.addRect(r)
        inner = QPainterPath()
        inner.addRoundedRect(r, RADIUS, RADIUS)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.BG))
        p.drawPath(outer.subtracted(inner))

    def _paint_video(self, p: QPainter) -> None:
        mode = self._mode
        main = self._original if mode == "original" and self._original is not None else self._result
        if main is None:
            main = self._original
        ir = self._image_rect(main)
        if mode == "split" and self._original is not None and self._result is not None:
            x = ir.left() + ir.width() * self._split
            p.drawImage(ir, self._result)
            p.save()
            p.setClipRect(QRectF(ir.left(), ir.top(), x - ir.left(), ir.height()))
            p.drawImage(ir, self._original)
            p.restore()
            self._paint_divider(p, ir, x)
        else:
            p.drawImage(ir, main)

    def _paint_divider(self, p: QPainter, ir: QRectF, x: float) -> None:
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        p.drawLine(QPointF(x, ir.top()), QPointF(x, ir.bottom()))
        c = QPointF(x, ir.center().y())
        p.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        p.setBrush(QColor(18, 18, 20, 220))
        p.drawEllipse(c, 14, 14)
        p.setPen(QPen(QColor("#FFFFFF"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        for d in (-1, 1):
            tip = c.x() + d * 8
            p.drawPolyline([QPointF(tip - d * 3.5, c.y() - 4.5), QPointF(tip, c.y()), QPointF(tip - d * 3.5, c.y() + 4.5)])
        self._pill(p, QPointF(x - 10, ir.top() + 12), [("Оригинал", None)], anchor="right")
        self._pill(p, QPointF(x + 10, ir.top() + 12), [("Mimiq", None)], anchor="left")

    # ------------------------------------------------------------------ HUD
    def _pill(self, p: QPainter, at: QPointF, parts, anchor: str = "left", dot: Optional[str] = None,
              avatar: Optional[QPixmap] = None, height: float = 24.0) -> QRectF:
        fm = QFontMetricsF(self._f_hud)
        pad = 8.0
        w = pad * 2
        if dot:
            w += 12
        if avatar is not None:
            w += 26
        for text, _c in parts:
            w += fm.horizontalAdvance(text)
        x = at.x() - w if anchor == "right" else at.x()
        rect = QRectF(x, at.y(), w, height)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(rect, 4, 4)
        cx = rect.left() + pad
        if avatar is not None:
            cx -= 3
            p.drawPixmap(QRectF(cx, rect.top() + (height - 20) / 2, 20, 20), avatar, QRectF(avatar.rect()))
            cx += 26
        if dot:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(dot))
            p.drawEllipse(QPointF(cx + 3.5, rect.center().y()), 3.5, 3.5)
            cx += 12
        p.setFont(self._f_hud)
        for text, color in parts:
            p.setPen(QColor(color or theme.TEXT))
            tw = fm.horizontalAdvance(text)
            p.drawText(QRectF(cx, rect.top(), tw + 2, height), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       text)
            cx += tw
        return rect

    def _hud_rect(self) -> QRectF:
        """HUD sits on the picture itself (not on letterbox bars) when there is room."""
        full = QRectF(self.rect())
        img = self._original if self._mode == "original" and self._original is not None else self._result
        if img is None:
            return full
        ir = self._image_rect(img).intersected(full)
        return ir if ir.height() >= 160 and ir.width() >= 360 else full

    def _paint_hud(self, p: QPainter) -> None:
        r = self._hud_rect()
        m = 12.0
        live = self._fps > 0
        parts = [("LIVE", theme.TEXT if live else theme.FAINT)]
        if live:
            slow = self._fps < 15
            parts.append((f"   {self._fps:.0f} к/с" if self._fps >= 2 else f"   {self._fps:.1f} к/с",
                          theme.WARN if slow else theme.DIM))
            if self._fluid and self._faces and 0 < self._face_fps < self._fps * 0.8:
                parts.append((f"  ·  лицо {rate_text(self._face_fps)}", theme.DIM))
            if self._latency > 0 and self._fps >= 2:
                parts.append((f"  ·  {self._latency:.0f} мс", theme.DIM))
        self._pill(p, QPointF(r.left() + m, r.top() + m), parts, dot=theme.LIVE if live else theme.FAINT)
        right_parts = []
        if self._source:
            right_parts.append((self._source, theme.TEXT))
        if self._resolution:
            right_parts.append((("  ·  " if right_parts else "") + self._resolution, theme.DIM))
        y_right = r.top() + m
        if right_parts:
            self._pill(p, QPointF(r.right() - m, y_right), right_parts, anchor="right")
            y_right += 30
        if self._recording:
            self._pill(p, QPointF(r.right() - m, y_right), [("REC", "#FFFFFF")], anchor="right",
                       dot=theme.LIVE if self._blink else "#5A1A16")
        # identity / face status (bottom-left)
        if not self._swap_enabled:
            status, color = "Замена выключена", theme.FAINT
        elif not self._identity_name:
            status, color = "Выберите лицо слева", theme.WARN
        elif not self._models_ready:
            status, color = "Загрузка моделей…", theme.WARN
        elif not self._faces:
            status, color = "Лицо не найдено", theme.WARN
        elif all(f.get("status") == "holding" for f in self._faces):
            status, color = "Удержание…", theme.WARN
        elif len(self._faces) > 1:
            status, color = f"Лиц в кадре: {len(self._faces)}", theme.DIM
        else:
            status, color = "Лицо в кадре", theme.DIM
        parts = []
        active = bool(self._identity_name and self._swap_enabled)
        if active:
            parts.append((self._identity_name, theme.TEXT))
            parts.append(("  ·  ", theme.FAINT))
        parts.append((status, color))
        if active and self._faces and self._likeness is not None and color == theme.DIM:
            parts.append(("  ·  сходство " + likeness_word(self._likeness), theme.DIM))
        h = 28.0
        self._pill(p, QPointF(r.left() + m, r.bottom() - m - h), parts,
                   avatar=self._identity_avatar if active else None, height=h)

    def _paint_loading(self, p: QPainter) -> None:
        stage, frac, text = self._loading
        r = QRectF(self.rect())
        w = min(420.0, r.width() - 40)
        h = 60.0
        box = QRectF(r.center().x() - w / 2, r.bottom() - h - 16, w, h)
        p.setPen(QPen(QColor(theme.LINE_HI), 1))
        p.setBrush(QColor(theme.CARD))
        p.drawRoundedRect(box, 6, 6)
        p.setFont(self._f_hud)
        color = theme.BAD if stage == "error" else theme.TEXT
        p.setPen(QColor(color))
        title = {"download": "Загрузка моделей", "load": "Подготовка нейросетей", "error": "Ошибка моделей"}.get(stage, "")
        p.drawText(QRectF(box.left() + 14, box.top() + 9, w - 28, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        p.setFont(self._f_small)
        p.setPen(QColor(theme.DIM))
        pct = f"{frac * 100:.0f}%" if stage != "error" else ""
        p.drawText(QRectF(box.left() + 14, box.top() + 9, w - 28, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, pct)
        fm = QFontMetricsF(self._f_small)
        p.drawText(QRectF(box.left() + 14, box.top() + 28, w - 28, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(text, Qt.TextElideMode.ElideRight, w - 28))
        bar = QRectF(box.left() + 14, box.bottom() - 12, w - 28, 3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.SEG_ON))
        p.drawRoundedRect(bar, 1.5, 1.5)
        if stage != "error" and frac > 0:
            fill = QRectF(bar.left(), bar.top(), max(3.0, bar.width() * min(1.0, frac)), bar.height())
            p.setBrush(QColor(theme.ACCENT))
            p.drawRoundedRect(fill, 1.5, 1.5)

    # ------------------------------------------------------------------ mouse
    def _near_divider(self, x: float) -> bool:
        return self._mode == "split" and self._state == "live" and abs(x - self._divider_x()) < 22

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self._mode == "split" and self._state == "live":
            self._drag = True
            self._set_split(e.position().x())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):  # noqa: N802
        x = e.position().x()
        if self._drag:
            self._set_split(x)
        self.setCursor(Qt.CursorShape.SplitHCursor if (self._drag or self._near_divider(x)) else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        self._drag = False
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.toggleFullscreen.emit()

    def _set_split(self, x: float) -> None:
        img = self._result or self._original
        ir = self._image_rect(img) if img is not None else QRectF(self.rect())
        if ir.width() > 0:
            self._split = min(0.98, max(0.02, (x - ir.left()) / ir.width()))
            self.update()
