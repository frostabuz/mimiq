"""Dialogs: iPhone connection (QR) and the first-run responsible-use agreement."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from . import brand, icons, theme
from .widgets import POINTER, Combo, IconButton, Spinner, StatusDot, label


# ====================================================================== QR code
class QrWidget(QWidget):
    """Branded QR code: rounded modules, gradient finder eyes, logo in the middle (ECC level H)."""

    def __init__(self, size: int = 268, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._matrix: List[List[bool]] = []
        self._text = ""

    def set_text(self, text: str) -> None:
        self._text = text
        self._matrix = []
        if text:
            try:
                import qrcode
                qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, border=0, box_size=1)
                qr.add_data(text)
                qr.make(fit=True)
                self._matrix = qr.get_matrix()
            except Exception:
                self._matrix = []
        self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(r, 8, 8)
        if not self._matrix:
            p.setPen(QColor("#8A8AA0"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Запуск Mimiq Link…" if not self._text else "QR недоступен")
            return
        n = len(self._matrix)
        pad = 20.0
        cell = (r.width() - 2 * pad) / n
        ox = oy = pad
        dark = QColor("#111113")
        logo_cells = int(n * 0.24) | 1
        lo = (n - logo_cells) // 2
        hi = lo + logo_cells

        def in_finder(x: int, y: int) -> bool:
            return (x < 7 and y < 7) or (x >= n - 7 and y < 7) or (x < 7 and y >= n - 7)

        p.setBrush(dark)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)    # crisp square modules, like any QR code
        for y, row in enumerate(self._matrix):
            for x, on in enumerate(row):
                if not on or in_finder(x, y) or (lo - 1 <= x < hi + 1 and lo - 1 <= y < hi + 1):
                    continue
                p.drawRect(QRectF(ox + x * cell, oy + y * cell, cell + 0.5, cell + 0.5))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for fx, fy in ((0, 0), (n - 7, 0), (0, n - 7)):
            outer = QRectF(ox + fx * cell, oy + fy * cell, 7 * cell, 7 * cell)
            path = QPainterPath()
            path.addRect(outer)
            inner = QPainterPath()
            inner.addRect(outer.adjusted(cell, cell, -cell, -cell))
            p.setBrush(dark)
            p.drawPath(path.subtracted(inner))
            p.drawRect(outer.adjusted(2 * cell, 2 * cell, -2 * cell, -2 * cell))
        side = logo_cells * cell
        lr = QRectF(ox + lo * cell, oy + lo * cell, side, side)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(lr.adjusted(-cell * 0.6, -cell * 0.6, cell * 0.6, cell * 0.6))
        p.drawImage(lr.adjusted(cell * 0.3, cell * 0.3, -cell * 0.3, -cell * 0.3), brand.logo_image(256))


# ====================================================================== connect dialog
class _Step(QWidget):
    def __init__(self, number: int, title: str, text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        badge = QLabel(str(number))
        badge.setFixedSize(24, 24)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background: transparent; color: {theme.TEXT}; border: 1px solid {theme.LINE_HI}; "
                            f"border-radius: 12px; font-weight: 600; font-size: 9pt;")
        lay.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        t = label(title)
        t.setStyleSheet("font-weight: 600;")
        col.addWidget(t)
        d = label(text, "dim", wrap=True)
        d.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(d)
        lay.addLayout(col, 1)


class ConnectDialog(QDialog):
    hostChanged = Signal(str)

    def __init__(self, engine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Подключить iPhone")
        self.setModal(False)
        self.setMinimumSize(880, 600)
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 26, 30, 22)
        root.setSpacing(22)
        head = QHBoxLayout()
        head.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label("Подключить iPhone", "h1"))
        col.addWidget(label("Mimiq Link — камера iPhone по Wi-Fi прямо из Safari, без установки приложений", "dim"))
        head.addLayout(col, 1)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(30)
        left = QFrame()
        left.setObjectName("box")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(22, 22, 22, 18)
        ll.setSpacing(14)
        self.qr = QrWidget(268)
        qr_row = QHBoxLayout()
        qr_row.addStretch(1)
        qr_row.addWidget(self.qr)
        qr_row.addStretch(1)
        ll.addLayout(qr_row)
        url_row = QHBoxLayout()
        url_row.setSpacing(8)
        self.url = label("", "faint")
        self.url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.url.setStyleSheet(f"font-family: Consolas, 'Cascadia Mono', monospace; color: {theme.DIM}; font-size: 9pt;")
        url_row.addWidget(self.url, 1)
        self.copy = IconButton("copy", "Скопировать ссылку", size=32, icon_size=16)
        self.copy.clicked.connect(self._copy)
        url_row.addWidget(self.copy)
        ll.addLayout(url_row)
        self.ip_combo = Combo([])
        self.ip_combo.picked.connect(self._pick_ip)
        self.ip_row = QWidget()
        ir = QHBoxLayout(self.ip_row)
        ir.setContentsMargins(0, 0, 0, 0)
        ir.setSpacing(10)
        ir.addWidget(label("Сеть", "dim"))
        ir.addWidget(self.ip_combo, 1)
        ll.addWidget(self.ip_row)
        body.addWidget(left, 0)

        right = QVBoxLayout()
        right.setSpacing(18)
        right.addWidget(_Step(1, "Одна сеть Wi-Fi", "iPhone и компьютер должны быть в одной сети. Лучше всего — "
                                                     "Wi-Fi 5 ГГц и компьютер по кабелю к роутеру."))
        right.addWidget(_Step(2, "Отсканируйте QR-код", "Откройте «Камеру» на iPhone, наведите на код и нажмите на "
                                                         "появившуюся ссылку — она откроется в Safari."))
        right.addWidget(_Step(3, "Подтвердите сертификат", "Safari предупредит о соединении: нажмите <b>Подробнее</b> → "
                                                            "<b>посетить этот веб-сайт</b>. Сертификат создан Mimiq на "
                                                            "вашем ПК — данные не уходят в интернет."))
        right.addWidget(_Step(4, "Включите камеру", "Нажмите «Включить камеру» и разрешите доступ. Видео "
                                                     "запустится в Mimiq автоматически."))
        trouble = label("Не открывается? Сделайте сеть в Windows «Частной», разрешите Python в брандмауэре "
                        "(или запустите tools\\firewall.bat), отключите VPN и не используйте гостевой Wi-Fi — "
                        "он изолирует устройства.", "faint", wrap=True)
        right.addWidget(trouble)
        right.addStretch(1)
        body.addLayout(right, 1)
        root.addLayout(body, 1)

        foot = QFrame()
        foot.setObjectName("box")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(16, 12, 12, 12)
        fl.setSpacing(10)
        self.dot = StatusDot(10, theme.WARN)
        self.dot.set_color(theme.WARN, pulse=True)
        fl.addWidget(self.dot)
        self.status = label("Ожидание iPhone…")
        self.status.setStyleSheet("font-weight: 600;")
        fl.addWidget(self.status, 1)
        done = QPushButton("Готово")
        done.setObjectName("primary")
        done.setCursor(POINTER)
        done.clicked.connect(self.accept)
        fl.addWidget(done)
        root.addWidget(foot)

        self._urls: List[str] = []
        self._closing = False
        engine.linkEvent.connect(self._on_link)
        self.finished.connect(lambda *_: self._disconnect())
        self.refresh()
        if engine.link is not None and engine.link.connected:
            self._set_status("connected", "iPhone подключён")

    def _disconnect(self) -> None:
        try:
            self.engine.linkEvent.disconnect(self._on_link)
        except (RuntimeError, TypeError):
            pass

    def refresh(self) -> None:
        self._urls = self.engine.link_urls()
        if not self._urls:
            self.qr.set_text("")
            self.url.setText("Запуск сервера…")
            self.ip_row.hide()
            if self.engine.link is not None and self.engine.link.error:
                self._set_status("error", self.engine.link.error)
            return
        self._show_url(self._urls[0])
        ips = [u.split("//", 1)[1].split(":", 1)[0] for u in self._urls]
        self.ip_combo.set_items([(ip, ip) for ip in ips], ips[0])
        self.ip_row.setVisible(len(ips) > 1)

    def _show_url(self, url: str) -> None:
        self.qr.set_text(url)
        self.url.setText(url)
        self.url.setToolTip(url)

    def _pick_ip(self, ip: str) -> None:
        url = next((u for u in self._urls if f"//{ip}:" in u), None)
        if url:
            self._show_url(url)
            self.hostChanged.emit(ip)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.url.text())
        self.copy.setToolTip("Скопировано")
        self.copy.set_icon("check", theme.OK, 16)
        QTimer.singleShot(1400, lambda: self.copy.set_icon("copy", theme.TEXT, 16))

    def _set_status(self, state: str, text: str) -> None:
        color = {"connected": theme.OK, "live": theme.OK, "waiting": theme.WARN, "error": theme.BAD}.get(state, theme.WARN)
        self.dot.set_color(color, pulse=state in ("waiting", "connected"))
        self.status.setText(text)
        self.status.setStyleSheet(f"font-weight: 600; color: {theme.BAD if state == 'error' else theme.TEXT};")

    def _on_link(self, kind: str, data: dict) -> None:
        if kind == "listening":
            self.refresh()
        elif kind == "error":
            self._set_status("error", data.get("message", "Ошибка Mimiq Link"))
        elif kind == "connected":
            self._set_status("connected", "iPhone открыл страницу — нажмите «Включить камеру»")
        elif kind == "hello":
            w, h = data.get("width") or 0, data.get("height") or 0
            if w:
                self._set_status("live", f"{data.get('device') or 'iPhone'} подключён · {w}×{h}")
            else:
                self._set_status("connected", f"{data.get('device') or 'iPhone'} на связи — нажмите «Включить камеру»")
        elif kind == "stats":
            if data.get("fps", 0) > 1 and not self._closing:
                self._closing = True
                self._set_status("live", f"Видео идёт · {data.get('fps', 0):.0f} к/с — можно закрыть это окно")
                QTimer.singleShot(1600, self.accept)
        elif kind == "disconnected":
            self._closing = False
            self._set_status("waiting", "iPhone отключился — ожидание…")


# ====================================================================== consent
class ConsentDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Mimiq")
        self.setModal(True)
        self.setFixedWidth(600)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(34, 32, 34, 26)
        lay.setSpacing(18)
        logo = QLabel()
        logo.setPixmap(brand.logo_pixmap(44))
        lay.addWidget(logo)
        t = label("Добро пожаловать в Mimiq", "h1")
        lay.addWidget(t)
        lay.addWidget(label("Mimiq заменяет лицо на видео в реальном времени. Это мощный инструмент — пожалуйста, "
                            "используйте его честно.", "dim", wrap=True))
        card = QFrame()
        card.setObjectName("box")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(18, 16, 18, 16)
        cl.setSpacing(14)
        for ic, title, text in (
                ("shield", "Только с согласия", "Используйте своё лицо или лица людей, которые явно разрешили это."),
                ("info", "Без обмана", "Не выдавайте себя за другого человека, чтобы ввести в заблуждение, получить "
                                       "деньги или доступ к аккаунтам. Это может быть незаконно."),
                ("eye", "Честная метка", "По умолчанию в углу видео есть значок «AI · Mimiq». Его можно отключить "
                                         "в настройках вывода."),
                ("chip", "Всё локально", "Видео и фото обрабатываются только на этом компьютере.")):
            row = QHBoxLayout()
            row.setSpacing(14)
            b = QLabel()
            b.setFixedSize(20, 20)
            b.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b.setPixmap(icons.pixmap(ic, theme.DIM, 18))
            row.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
            c = QVBoxLayout()
            c.setSpacing(2)
            tt = label(title)
            tt.setStyleSheet("font-weight: 600;")
            c.addWidget(tt)
            c.addWidget(label(text, "dim", wrap=True))
            row.addLayout(c, 1)
            cl.addLayout(row)
        lay.addWidget(card)
        self.check = QCheckBox("Я понимаю и буду использовать Mimiq ответственно")
        self.check.setCursor(POINTER)
        lay.addWidget(self.check)
        btns = QHBoxLayout()
        btns.addStretch(1)
        quit_ = QPushButton("Выйти")
        quit_.setObjectName("ghost")
        quit_.setCursor(POINTER)
        quit_.clicked.connect(self.reject)
        self.ok = QPushButton("Продолжить")
        self.ok.setObjectName("primary")
        self.ok.setCursor(POINTER)
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self.accept)
        self.check.toggled.connect(self.ok.setEnabled)
        btns.addWidget(quit_)
        btns.addWidget(self.ok)
        lay.addLayout(btns)
        self.adjustSize()

    def showEvent(self, e):  # noqa: N802
        lay = self.layout()
        if lay is not None and lay.hasHeightForWidth():
            self.setMinimumHeight(lay.totalHeightForWidth(self.width()))
        super().showEvent(e)
