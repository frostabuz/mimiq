"""Reusable Mimiq widgets: toggle switch, segmented control, slider rows, cards, chips, toasts, spinner."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import (Property, QEasingCurve, QEvent, QObject, QPoint, QPropertyAnimation, QRectF, QSize, Qt,
                            QTimer, Signal)
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath,
                           QPen, QPixmap)
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QComboBox, QFrame, QGraphicsOpacityEffect, QHBoxLayout,
                               QLabel, QPushButton, QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget)

from . import icons, theme

POINTER = Qt.CursorShape.PointingHandCursor


# ====================================================================== small helpers
def label(text: str = "", name: str = "", wrap: bool = False, parent: Optional[QWidget] = None) -> QLabel:
    lab = QLabel(text, parent)
    if name:
        lab.setObjectName(name)
    if wrap:
        lab.setWordWrap(True)
    return lab


def hline() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {theme.LINE}; border: none;")
    return line


def icon_label(name: str, color: str = theme.DIM, size: int = 18) -> QLabel:
    lab = QLabel()
    lab.setPixmap(icons.pixmap(name, color, size))
    lab.setFixedSize(size, size)
    return lab


def circle_pixmap(img: QImage, size: int, ring: bool = False, ring_width: float = 2.0) -> QPixmap:
    """Round avatar from a portrait image, optionally with a thin accent ring."""
    dpr = 2.0
    px = int(size * dpr)
    out = QPixmap(px, px)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
    inset = (ring_width + 2.0) * dpr if ring else 0.0
    rect = QRectF(inset, inset, px - 2 * inset, px - 2 * inset)
    path = QPainterPath()
    path.addEllipse(rect)
    p.save()
    p.setClipPath(path)
    if img is not None and not img.isNull():
        side = min(img.width(), img.height())
        src = QRectF((img.width() - side) / 2, (img.height() - side) / 2, side, side)
        p.drawImage(rect, img, src)
    else:
        p.fillRect(rect, QColor(theme.FIELD))
    p.restore()
    if ring:
        pen = QPen(QColor(theme.ACCENT), ring_width * dpr)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        half = ring_width * dpr / 2
        p.drawEllipse(QRectF(half, half, px - 2 * half, px - 2 * half))
    p.end()
    out.setDevicePixelRatio(dpr)
    return out


# ====================================================================== wheel-safe inputs
class NoWheelSlider(QSlider):
    """Slider that does not steal the mouse wheel while the settings panel is scrolled."""

    def wheelEvent(self, e):  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class Combo(QComboBox):
    """Styled combo box that ignores the wheel unless focused and supports (key, text) items."""

    picked = Signal(object)

    def __init__(self, items: Sequence = (), parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCursor(POINTER)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(8)
        self.view().setCursor(POINTER)
        self.set_items(items)
        self.activated.connect(lambda i: self.picked.emit(self.itemData(i)))

    def set_items(self, items: Sequence, current=None) -> None:
        self.blockSignals(True)
        self.clear()
        for it in items:
            key, text = (it, str(it)) if not isinstance(it, (tuple, list)) else (it[0], it[1])
            self.addItem(text, key)
            if isinstance(it, (tuple, list)) and len(it) > 2 and it[2] is False:
                model_item = self.model().item(self.count() - 1)
                if model_item is not None:
                    model_item.setEnabled(False)
        self.blockSignals(False)
        if current is not None:
            self.set_value(current)

    def value(self):
        return self.currentData()

    def set_value(self, key) -> None:
        self.blockSignals(True)
        idx = self.findData(key)
        if idx >= 0:
            self.setCurrentIndex(idx)
        self.blockSignals(False)

    def wheelEvent(self, e):  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


# ====================================================================== toggle switch
class ToggleSwitch(QAbstractButton):
    """Compact animated switch: grey when off, solid accent when on."""

    def __init__(self, checked: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(POINTER)
        self.setFixedSize(38, 22)
        self._knob = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(170)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _get_knob(self) -> float:
        return self._knob

    def _set_knob(self, v: float) -> None:
        self._knob = float(v)
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def set_value(self, on: bool) -> None:
        self.blockSignals(True)
        self.setChecked(bool(on))
        self.blockSignals(False)
        self._anim.stop()
        self._knob = 1.0 if on else 0.0
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(38, 22)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.4)
        r = QRectF(0, 0, self.width(), self.height())
        rad = r.height() / 2
        off, on = QColor(theme.SEG_ON), QColor(theme.ACCENT)
        k = self._knob
        track = QColor(int(off.red() + (on.red() - off.red()) * k), int(off.green() + (on.green() - off.green()) * k),
                       int(off.blue() + (on.blue() - off.blue()) * k))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, rad, rad)
        d = r.height() - 6
        x = r.left() + 3 + (r.width() - 6 - d) * k
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QRectF(x, r.top() + 3, d, d))


# ====================================================================== segmented control
class Segmented(QFrame):
    """Exclusive pill buttons. items: (key, text) or (key, text, icon_name)."""

    changed = Signal(object)

    def __init__(self, items: Sequence, parent: Optional[QWidget] = None, expand: bool = True,
                 tooltips: Optional[Dict[object, str]] = None):
        super().__init__(parent)
        self.setObjectName("segBox")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: Dict[object, QPushButton] = {}
        self._value = None
        for it in items:
            key, text = it[0], it[1]
            ic = it[2] if len(it) > 2 else None
            b = QPushButton(text)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(POINTER)
            if ic:
                b.setIcon(icons.icon(ic, theme.DIM, 16, active="#FFFFFF"))
                b.setIconSize(QSize(16, 16))
            if expand:
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if tooltips and key in tooltips:
                b.setToolTip(tooltips[key])
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self._clicked(k))

    def _clicked(self, key) -> None:
        if key != self._value:
            self._value = key
            self.changed.emit(key)

    def value(self):
        return self._value

    def set_value(self, key) -> None:
        self._value = key
        b = self.buttons.get(key)
        if b is not None:
            b.setChecked(True)
        else:
            self.group.setExclusive(False)
            for bb in self.buttons.values():
                bb.setChecked(False)
            self.group.setExclusive(True)


# ====================================================================== rows & cards
class SliderRow(QWidget):
    """Title + live value on top, slider below, optional hint."""

    valueChanged = Signal(float)

    def __init__(self, title: str, minimum: float, maximum: float, step: float = 1.0,
                 fmt: Optional[Callable[[float], str]] = None, hint: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._min, self._step = float(minimum), float(step)
        self._fmt = fmt or (lambda v: f"{v:g}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.title = label(title)
        self.value_label = label("", "value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.value_label)
        lay.addLayout(top)
        self.slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.slider.setCursor(POINTER)
        self.slider.setRange(0, int(round((float(maximum) - self._min) / self._step)))
        self.slider.setPageStep(max(1, self.slider.maximum() // 10))
        self.slider.valueChanged.connect(self._changed)
        lay.addWidget(self.slider)
        if hint:
            lay.addWidget(label(hint, "faint", wrap=True))
        self._changed(self.slider.value(), emit=False)

    def _to_value(self, pos: int) -> float:
        return round(self._min + pos * self._step, 6)

    def _changed(self, pos: int, emit: bool = True) -> None:
        v = self._to_value(pos)
        self.value_label.setText(self._fmt(v))
        if emit:
            self.valueChanged.emit(v)

    def value(self) -> float:
        return self._to_value(self.slider.value())

    def set_value(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((float(v) - self._min) / self._step)))
        self.slider.blockSignals(False)
        self._changed(self.slider.value(), emit=False)


class SettingRow(QWidget):
    """Title (+hint) on the left, a control on the right."""

    def __init__(self, title: str, control: QWidget, hint: str = "", icon_name: Optional[str] = None,
                 parent: Optional[QWidget] = None, stretch_control: bool = False):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        if icon_name:
            lay.addWidget(icon_label(icon_name, theme.DIM, 18), 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = label(title)
        text.addWidget(self.title)
        self.hint = None
        if hint:
            self.hint = label(hint, "faint", wrap=True)
            text.addWidget(self.hint)
        lay.addLayout(text, 1)
        lay.addWidget(control, 1 if stretch_control else 0, Qt.AlignmentFlag.AlignVCenter)
        self.control = control


class FieldRow(QWidget):
    """Label above a full-width control (combo boxes, line edits)."""

    def __init__(self, title: str, control: QWidget, hint: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.title = label(title, "dim")
        lay.addWidget(self.title)
        lay.addWidget(control)
        self.hint = None
        if hint:
            self.hint = label(hint, "faint", wrap=True)
            lay.addWidget(self.hint)
        self.control = control


class Card(QFrame):
    """Settings group: a title (+ subtitle) over its rows, separated from the previous group by a thin line."""

    def __init__(self, title: str = "", icon_name: Optional[str] = None, subtitle: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(2, 16, 2, 18)
        self.body.setSpacing(12)
        self.header = None
        if title:
            head = QHBoxLayout()
            head.setSpacing(10)
            col = QVBoxLayout()
            col.setSpacing(0)
            self.title = label(title, "h2")
            col.addWidget(self.title)
            if subtitle:
                col.addWidget(label(subtitle, "faint", wrap=True))
            head.addLayout(col, 1)
            self.header = head
            self.body.addLayout(head)

    def add(self, w: QWidget) -> QWidget:
        self.body.addWidget(w)
        return w

    def header_right(self, w: QWidget) -> QWidget:
        if self.header is not None:
            self.header.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
        return w


class IconButton(QToolButton):
    def __init__(self, icon_name: str, tooltip: str = "", size: int = 38, icon_size: int = 18,
                 checkable: bool = False, flat: bool = False, color: str = theme.TEXT,
                 active_color: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("flat" if flat else "iconbtn")
        self.setIcon(icons.icon(icon_name, color, icon_size, active=active_color))
        self.setIconSize(QSize(icon_size, icon_size))
        self.setFixedSize(size, size)
        self.setCheckable(checkable)
        self.setCursor(POINTER)
        if tooltip:
            self.setToolTip(tooltip)

    def set_icon(self, icon_name: str, color: str = theme.TEXT, icon_size: int = 18) -> None:
        self.setIcon(icons.icon(icon_name, color, icon_size))


class Chip(QFrame):
    """Small rounded status chip: dot/icon + text."""

    clicked = Signal()

    def __init__(self, text: str = "", icon_name: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("chip")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 5, 12, 5)
        lay.setSpacing(7)
        self.dot = StatusDot(8)
        self.icon = QLabel()
        self.icon.setFixedSize(16, 16)
        self._icon_name = icon_name
        if icon_name:
            self.icon.setPixmap(icons.pixmap(icon_name, theme.DIM, 16))
            self.dot.hide()
        else:
            self.icon.hide()
        self.text = label(text)
        self.text.setStyleSheet(f"color: {theme.DIM}; font-weight: 500; font-size: 9pt;")
        lay.addWidget(self.dot)
        lay.addWidget(self.icon)
        lay.addWidget(self.text)
        self.setStyleSheet(f"#chip {{ background: transparent; border: 1px solid {theme.LINE}; border-radius: 6px; }}")

    def set(self, text: str, color: Optional[str] = None, tooltip: str = "") -> None:
        self.text.setText(text)
        if color:
            self.dot.set_color(color)
            if self._icon_name:
                self.icon.setPixmap(icons.pixmap(self._icon_name, color, 16))
        self.setToolTip(tooltip)

    def mouseReleaseEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class StatusDot(QWidget):
    """Colored dot with an optional soft pulse (for 'waiting' states)."""

    def __init__(self, size: int = 8, color: str = theme.FAINT, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size = size
        self._phase = 0.0
        self.setFixedSize(size + 8, size + 8)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def set_color(self, color: str, pulse: bool = False) -> None:
        self._color = QColor(color)
        if pulse and not self._timer.isActive():
            self._timer.start()
        elif not pulse and self._timer.isActive():
            self._timer.stop()
            self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.04) % 1.0
        self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QRectF(self.rect()).center()
        if self._timer.isActive():
            halo = QColor(self._color)
            halo.setAlphaF(0.45 * (1.0 - self._phase))
            r = self._size / 2 + 4 * self._phase
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(halo)
            p.drawEllipse(c, r, r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(c, self._size / 2, self._size / 2)


class Spinner(QWidget):
    """Simple arc spinner; animates only while visible."""

    def __init__(self, size: int = 30, width: float = 3.0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._width = width
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self._angle = (self._angle - 7) % 360
        self.update()

    def showEvent(self, e):  # noqa: N802
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):  # noqa: N802
        self._timer.stop()
        super().hideEvent(e)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self._width / 2 + 1
        r = QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)
        p.setPen(QPen(QColor(255, 255, 255, 26), self._width))
        p.drawEllipse(r)
        pen = QPen(QColor(theme.TEXT), self._width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(r, int(self._angle * 16), int(90 * 16))


class Wordmark(QWidget):
    """The "Mimiq" wordmark: plain, slightly tight semibold text."""

    def __init__(self, text: str, point_size: float = 14, weight: QFont.Weight = QFont.Weight.DemiBold,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._text = text
        self._font = theme.ui_font(point_size, weight)
        self._font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 97)
        fm = QFontMetrics(self._font)
        self.setFixedSize(fm.horizontalAdvance(text) + 6, fm.height() + 2)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = QFontMetrics(self._font)
        path = QPainterPath()
        path.addText(2, fm.ascent() + 1, self._font, self._text)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.TEXT))
        p.drawPath(path)


GradientText = Wordmark     # old name


# ====================================================================== toasts
class _Toast(QFrame):
    COLORS = {"ok": theme.OK, "info": theme.ACCENT_HI, "warn": theme.WARN, "error": theme.BAD}
    ICONS = {"ok": "check", "info": "info", "warn": "info", "error": "close"}

    def __init__(self, level: str, text: str, parent: QWidget):
        super().__init__(parent)
        self.text_value = text
        color = self.COLORS.get(level, theme.ACCENT_HI)
        self.setObjectName("toast")
        self.setStyleSheet(f"#toast {{ background: {theme.CARD_HI}; border: 1px solid {theme.LINE_HI}; "
                           f"border-radius: 6px; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 16, 10)
        lay.setSpacing(10)
        badge = QLabel()
        badge.setFixedSize(24, 24)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(icons.pixmap(self.ICONS.get(level, "info"), color, 16, 2.2))
        lay.addWidget(badge)
        msg = label(text, wrap=True)
        msg.setMaximumWidth(520)
        lay.addWidget(msg)
        self.effect = QGraphicsOpacityEffect(self)
        self.effect.setOpacity(0.0)
        self.setGraphicsEffect(self.effect)
        self.anim = QPropertyAnimation(self.effect, b"opacity", self)
        self.anim.setDuration(220)


class ToastManager(QObject):
    """Stacks toasts at the bottom centre of a host widget."""

    def __init__(self, host: QWidget, bottom_margin: int = 64):
        super().__init__(host)
        self.host = host
        self.bottom = bottom_margin
        self.toasts: List[_Toast] = []
        host.installEventFilter(self)

    def eventFilter(self, obj, ev):  # noqa: N802
        if obj is self.host and ev.type() == QEvent.Type.Resize:
            self._layout()
        return False

    def show(self, level: str, text: str, timeout: int = 3800) -> None:
        if any(t.text_value == text for t in self.toasts):
            return  # identical toast already visible
        t = _Toast(level, text, self.host)
        t.adjustSize()
        t.show()
        t.raise_()
        self.toasts.append(t)
        if len(self.toasts) > 3:
            self._close(self.toasts[0])
        self._layout()
        t.anim.setStartValue(0.0)
        t.anim.setEndValue(1.0)
        t.anim.start()
        QTimer.singleShot(timeout, lambda: self._close(t))

    def _close(self, t: _Toast) -> None:
        if t not in self.toasts:
            return
        self.toasts.remove(t)
        t.anim.stop()
        t.anim.setStartValue(t.effect.opacity())
        t.anim.setEndValue(0.0)
        t.anim.finished.connect(t.deleteLater)
        t.anim.start()
        self._layout()

    def _layout(self) -> None:
        y = self.host.height() - self.bottom
        for t in reversed(self.toasts):
            t.adjustSize()
            y -= t.height()
            t.move((self.host.width() - t.width()) // 2, y)
            y -= 8
