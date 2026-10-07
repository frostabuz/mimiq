"""Right-hand settings column: Source · Swap · Mask · Background · Output tabs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QButtonGroup, QColorDialog, QFileDialog, QFrame, QHBoxLayout, QLineEdit, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from .. import __version__, paths
from ..config import Settings
from ..core import models
from . import backgrounds, icons, theme
from .widgets import (POINTER, Card, Combo, FieldRow, IconButton, Segmented, SettingRow, SliderRow, StatusDot,
                      ToggleSwitch, label)


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def secs(v: float) -> str:
    return "выкл." if v <= 0 else f"{v / 1000:.2f} с".replace(".", ",")


LINK_RES = [("1280x720", "1280 × 720 · HD"), ("1920x1080", "1920 × 1080 · Full HD"), ("960x540", "960 × 540 · экономно")]
CAP_RES = [("1280x720", "1280 × 720"), ("1920x1080", "1920 × 1080"), ("960x540", "960 × 540"), ("640x480", "640 × 480")]
OUT_RES = [("1280x720", "1280 × 720 · HD"), ("1920x1080", "1920 × 1080 · Full HD"), ("960x540", "960 × 540"),
           ("640x480", "640 × 480")]


def model_items(kind: str, none_label: Optional[str] = None) -> List[tuple]:
    items = [("", none_label)] if none_label else []
    for spec in models.by_kind(kind):
        mark = "✓" if models.is_installed(spec.key) else f"↓ {spec.size_mb} МБ"
        items.append((spec.key, f"{spec.title}    {mark}"))
    return items


class TabBar(QWidget):
    changed = Signal(int)

    def __init__(self, items: Sequence[tuple], parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for i, (text, _ic) in enumerate(items):
            b = QToolButton()
            b.setObjectName("tab")
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            b.setCheckable(True)
            b.setCursor(POINTER)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setMinimumHeight(34)
            self.group.addButton(b, i)
            lay.addWidget(b)
        self.group.idClicked.connect(self.changed.emit)
        self.group.button(0).setChecked(True)

    def set_index(self, i: int) -> None:
        b = self.group.button(i)
        if b:
            b.setChecked(True)


class SettingsPanel(QFrame):
    changed = Signal(dict)
    connectRequested = Signal()
    openFolder = Signal(str)
    refreshCameras = Signal()
    vcamSetup = Signal(str)                # install | uninstall (Mimiq Camera)
    installTensorrt = Signal()
    checkUpdates = Signal()                # manual "Проверить сейчас"
    notify = Signal(str, str)              # level, text (toast)

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.s = settings
        self._binds: List[Callable[[Settings], None]] = []
        self._cameras: List = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 6, 10)
        outer.setSpacing(4)
        head = QVBoxLayout()
        head.setContentsMargins(2, 0, 8, 0)
        head.setSpacing(8)
        head.addWidget(label("Настройки", "h2"))
        self.tabs = TabBar([("Источник", "camera"), ("Замена", "face"), ("Маска", "mask"), ("Фон", "image"),
                            ("Вывод", "monitor")])
        tabs_box = QVBoxLayout()
        tabs_box.setSpacing(0)
        tabs_box.addWidget(self.tabs)
        tab_line = QFrame()
        tab_line.setFixedHeight(1)
        tab_line.setStyleSheet(f"background: {theme.LINE}; border: none;")
        tabs_box.addWidget(tab_line)
        head.addLayout(tabs_box)
        outer.addLayout(head)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.tabs.changed.connect(self.stack.setCurrentIndex)
        for build in (self._build_source, self._build_swap, self._build_mask, self._build_background,
                      self._build_output):
            page = QWidget()
            col = QVBoxLayout(page)
            col.setContentsMargins(2, 0, 8, 8)
            col.setSpacing(0)
            build(col)
            first = col.itemAt(0).widget() if col.count() else None
            if isinstance(first, Card):
                first.setObjectName("cardFlat")     # the tab bar line already separates it
            col.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.load(settings)

    # ================================================================== binding helpers
    def _emit(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.s, k, v)
        self.changed.emit(kw)
        self._sync_visibility()

    def _toggle(self, key: str) -> ToggleSwitch:
        t = ToggleSwitch()
        t.toggled.connect(lambda v: self._emit(**{key: bool(v)}))
        self._binds.append(lambda s: t.set_value(getattr(s, key)))
        return t

    def _combo(self, key: str, items: Sequence) -> Combo:
        c = Combo(items)
        c.picked.connect(lambda v: self._emit(**{key: v}))
        self._binds.append(lambda s: c.set_value(getattr(s, key)))
        return c

    def _seg(self, key: str, items: Sequence) -> Segmented:
        g = Segmented(items)
        g.changed.connect(lambda v: self._emit(**{key: v}))
        self._binds.append(lambda s: g.set_value(getattr(s, key)))
        return g

    def _slider(self, key: str, title: str, lo: float, hi: float, step: float, fmt=None, hint: str = "",
                cast=float) -> SliderRow:
        r = SliderRow(title, lo, hi, step, fmt, hint)
        r.valueChanged.connect(lambda v: self._emit(**{key: cast(v)}))
        self._binds.append(lambda s: r.set_value(getattr(s, key)))
        return r

    def _size_combo(self, wkey: str, hkey: str, items: Sequence) -> Combo:
        c = Combo(items)

        def picked(v):
            w, h = (int(x) for x in v.split("x"))
            self._emit(**{wkey: w, hkey: h})
        c.picked.connect(picked)
        self._binds.append(lambda s: c.set_value(f"{getattr(s, wkey)}x{getattr(s, hkey)}"))
        return c

    # ================================================================== Source
    def _build_source(self, col: QVBoxLayout) -> None:
        card = Card("Источник видео", "camera", "Откуда Mimiq берёт картинку")
        self.source_seg = self._seg("source_kind", [("link", "iPhone", "phone"), ("device", "Камера", "camera"),
                                                    ("url", "Поток", "link")])
        card.add(self.source_seg)
        # --- iPhone / Mimiq Link
        self.link_box = QWidget()
        lb = QVBoxLayout(self.link_box)
        lb.setContentsMargins(0, 2, 0, 0)
        lb.setSpacing(14)
        status = QFrame()
        status.setObjectName("linkStatus")
        status.setStyleSheet(f"#linkStatus {{ background: {theme.CARD}; border: 1px solid {theme.LINE}; "
                             f"border-radius: 6px; }}")
        sl = QHBoxLayout(status)
        sl.setContentsMargins(10, 8, 8, 8)
        sl.setSpacing(6)
        self.link_dot = StatusDot(8, theme.WARN)
        self.link_dot.set_color(theme.WARN, pulse=True)
        sl.addWidget(self.link_dot)
        txt = QVBoxLayout()
        txt.setSpacing(0)
        self.link_title = label("Ожидание iPhone", wrap=True)
        self.link_title.setStyleSheet("font-weight: 600;")
        self.link_sub = label("Mimiq Link · Wi-Fi", "faint", wrap=True)
        txt.addWidget(self.link_title)
        txt.addWidget(self.link_sub)
        sl.addLayout(txt, 1)
        qr = QPushButton("  QR-код")
        qr.setIcon(icons.icon("qr", theme.TEXT, 16))
        qr.setStyleSheet("padding: 6px 12px;")
        qr.setCursor(POINTER)
        qr.clicked.connect(self.connectRequested.emit)
        sl.addWidget(qr)
        lb.addWidget(status)
        lb.addWidget(FieldRow("Камера iPhone", self._seg("link_camera", [("user", "Фронтальная"),
                                                                         ("environment", "Основная")])))
        lb.addWidget(FieldRow("Разрешение", self._combo("link_resolution", LINK_RES)))
        lb.addWidget(FieldRow("Кадров в секунду", self._seg("link_fps", [(24, "24"), (30, "30"), (60, "60")])))
        lb.addWidget(self._slider("link_quality", "Качество сжатия", 0.6, 0.95, 0.01, pct,
                                  "Выше — чётче, но больше нагрузка на Wi-Fi."))
        lb.addWidget(label("Работает прямо в Safari, без приложений. Для минимальной задержки — Wi-Fi 5 ГГц.",
                           "faint", wrap=True))
        card.add(self.link_box)
        # --- Webcam / phone apps
        self.device_box = QWidget()
        db = QVBoxLayout(self.device_box)
        db.setContentsMargins(0, 2, 0, 0)
        db.setSpacing(14)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        self.camera_combo = Combo([])
        self.camera_combo.picked.connect(self._camera_picked)
        rl.addWidget(self.camera_combo, 1)
        refresh = IconButton("refresh", "Обновить список камер", size=36, icon_size=16)
        refresh.clicked.connect(self.refreshCameras.emit)
        rl.addWidget(refresh)
        db.addWidget(FieldRow("Камера", row, "Camo, iVCam, Iriun, DroidCam и USB-камеры появляются здесь как "
                                             "обычные веб-камеры."))
        db.addWidget(FieldRow("Разрешение", self._size_combo("capture_width", "capture_height", CAP_RES)))
        db.addWidget(FieldRow("Кадров в секунду", self._seg("capture_fps", [(24, "24"), (30, "30"), (60, "60")])))
        db.addWidget(FieldRow("Драйвер захвата", self._seg("device_backend", [("dshow", "DirectShow"),
                                                                             ("msmf", "Media Foundation")])))
        card.add(self.device_box)
        # --- network stream
        self.url_box = QWidget()
        ub = QVBoxLayout(self.url_box)
        ub.setContentsMargins(0, 2, 0, 0)
        ub.setSpacing(10)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("http://192.168.1.20:4747/video")
        self.url_edit.editingFinished.connect(lambda: self._emit(stream_url=self.url_edit.text().strip())
                                              if self.url_edit.text().strip() != self.s.stream_url else None)
        self._binds.append(lambda s: self.url_edit.setText(s.stream_url))
        ub.addWidget(FieldRow("Адрес потока", self.url_edit, "MJPEG (DroidCam, IP Webcam), RTSP или RTMP. "
                                                             "DroidCam: http://IP-телефона:4747/video"))
        card.add(self.url_box)
        col.addWidget(card)

        card = Card("Ориентация", "refresh")
        card.add(FieldRow("Поворот", self._seg("rotate", [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")])))
        card.add(SettingRow("Зеркально", self._toggle("mirror"), "Отразить картинку по горизонтали"))
        col.addWidget(card)

    def _camera_picked(self, idx) -> None:
        if idx is None:
            return
        name = next((c.name for c in self._cameras if c.index == idx), "")
        self._emit(device_index=int(idx), device_name=name)

    def set_cameras(self, cams: List) -> None:
        self._cameras = cams
        items = [(c.index, ("📱  " if c.is_phone else "") + c.name) for c in cams] or [(None, "Камеры не найдены")]
        self.camera_combo.set_items(items)
        current = next((c.index for c in cams if c.name == self.s.device_name), None)
        if current is None:
            current = self.s.device_index if any(c.index == self.s.device_index for c in cams) else None
        if current is not None:
            self.camera_combo.set_value(current)

    def set_link_status(self, state: str, title: str, subtitle: str = "") -> None:
        color = {"connected": theme.OK, "waiting": theme.WARN, "error": theme.BAD, "off": theme.FAINT}.get(state, theme.FAINT)
        self.link_dot.set_color(color, pulse=state == "waiting")
        self.link_title.setText(title)
        self.link_sub.setText(subtitle or "Mimiq Link · Wi-Fi")

    # ================================================================== Swap
    def _build_swap(self, col: QVBoxLayout) -> None:
        card = Card("Замена лица", "face", "Нейросеть, которая переносит лицо")
        card.add(SettingRow("Включить замену", self._toggle("swap_enabled"), "Ctrl+E — быстро включить/выключить"))
        self.swapper_combo = self._combo("swapper_model", model_items("swapper"))
        self.swapper_row = FieldRow("Модель", self.swapper_combo, " ")
        card.add(self.swapper_row)
        card.add(FieldRow("Детализация лица (Pixel Boost)",
                          self._seg("pixel_boost", [(128, "128"), (256, "256"), (512, "512")]),
                          "Каждая ступень — в 4 раза больше работы нейросети. 128 — плавно, 256 — чётче кожа и "
                          "глаза; на сходство почти не влияет."))
        card.add(SettingRow("Все лица в кадре", self._toggle("multi_face"), "Заменять каждое найденное лицо"))
        col.addWidget(card)

        card = Card("Сходство", "face", "Насколько результат похож на фото")
        card.add(self._slider("identity_strength", "Сила сходства", 0, 100, 5, lambda v: f"{v:.0f}%",
                              "Убирает из результата черты лица перед камерой. 0 — обычная замена, 60 — "
                              "рекомендуется, выше 80 — возможны артефакты.", cast=int))
        card.add(FieldRow("Проходы замены", self._seg("swap_passes", [(1, "1"), (2, "2"), (3, "3")]),
                          "Повторная обработка добавляет сходства (кожа, брови, щетина). Каждый проход — ещё одна "
                          "замена по времени."))
        card.add(label("Больше всего сходства дают 3–10 разных фото одного человека: анфас и лёгкие повороты, "
                       "хороший свет, без очков и фильтров.", "faint", wrap=True))
        col.addWidget(card)

        card = Card("Улучшение лица", "sparkles", "Восстанавливает детали после замены")
        self.enhancer_combo = self._combo("enhancer_model", model_items("enhancer", "Выключено"))
        card.add(FieldRow("Модель", self.enhancer_combo))
        self.blend_row = self._slider("enhancer_blend", "Сила улучшения", 0.0, 1.0, 0.01, pct)
        card.add(self.blend_row)
        self.fidelity_row = self._slider("enhancer_fidelity", "Точность CodeFormer", 0.0, 1.0, 0.01, pct,
                                         "Меньше — красивее, больше — ближе к оригиналу.")
        card.add(self.fidelity_row)
        col.addWidget(card)

        card = Card("Цвет и свет", "palette")
        card.add(self._slider("color_match", "Подгонка тона кожи", 0.0, 1.0, 0.01, pct,
                              "Подстраивает цвет лица под освещение кадра, чтобы не было «маски»."))
        col.addWidget(card)

        card = Card("Отслеживание", "target", "Стабильность и устойчивость к движению")
        card.add(FieldRow("Детектор лиц", self._combo("detector_model", model_items("detector"))))
        card.add(FieldRow("Размер детекции", self._seg("detector_size", [(320, "320"), (480, "480"), (640, "640")])))
        card.add(SettingRow("Уточнение по 68 точкам", self._toggle("landmark_refine"),
                            "Точнее контур и меньше дрожания"))
        card.add(self._slider("smoothing", "Сглаживание движения", 0.0, 1.0, 0.01, pct,
                              "Убирает дрожание, при высоких значениях добавляет инерцию."))
        card.add(self._slider("hold_ms", "Удержание при потере лица", 0, 2000, 50, secs,
                              "Маска не пропадает при поворотах головы и перекрытиях.", cast=int))
        card.add(self._slider("fade_ms", "Плавное появление", 0, 1000, 50, secs, cast=int))
        card.add(self._slider("detector_score", "Порог уверенности", 0.3, 0.9, 0.01, pct))
        col.addWidget(card)

    # ================================================================== Mask
    def _build_mask(self, col: QVBoxLayout) -> None:
        card = Card("Перекрытия", "mask", "Что может оказаться поверх лица")
        card.add(SettingRow("Не перекрывать предметы", self._toggle("mask_occlusion"),
                            "Руки, очки, микрофон и волосы остаются поверх лица"))
        self.occluder_row = FieldRow("Модель окклюзии", self._combo("occluder_model", model_items("occluder")))
        card.add(self.occluder_row)
        col.addWidget(card)

        card = Card("Форма лица", "face")
        card.add(SettingRow("Точная форма (парсинг)", self._toggle("mask_region"),
                            "Маска повторяет кожу, брови, глаза, нос и рот"))
        self.parser_row = FieldRow("Модель парсинга", self._combo("parser_model", model_items("parser")))
        card.add(self.parser_row)
        col.addWidget(card)

        card = Card("Края и стабильность", "layers")
        card.add(self._slider("mask_blur", "Растушёвка краёв", 0.0, 1.0, 0.01, pct))
        card.add(self._slider("mask_temporal", "Стабильность маски", 0.0, 0.9, 0.01, pct,
                              "Сглаживает маску между кадрами — края не мерцают."))
        col.addWidget(card)

        card = Card("Отступы маски", "expand")
        card.add(self._slider("mask_padding_top", "Сверху", 0, 40, 1, lambda v: f"{v:.0f}%", cast=int))
        card.add(self._slider("mask_padding_bottom", "Снизу", 0, 40, 1, lambda v: f"{v:.0f}%", cast=int))
        card.add(self._slider("mask_padding_sides", "По бокам", 0, 40, 1, lambda v: f"{v:.0f}%", cast=int))
        col.addWidget(card)
        col.addWidget(label("Совет: режим «Маска» под превью показывает, какая область заменяется.", "faint",
                            wrap=True))

    # ================================================================== Background
    def _build_background(self, col: QVBoxLayout) -> None:
        card = Card("Фон", "image", "Размытие или замена фона за вами")
        self.bg_seg = self._seg("bg_mode", [("off", "Выкл."), ("blur", "Размытие"), ("image", "Картинка"),
                                            ("color", "Цвет")])
        card.add(self.bg_seg)
        # --- blur
        self.bg_blur_box = QWidget()
        bl = QVBoxLayout(self.bg_blur_box)
        bl.setContentsMargins(0, 2, 0, 0)
        bl.setSpacing(10)
        bl.addWidget(self._slider("bg_blur", "Сила размытия", 0.0, 1.0, 0.01, pct,
                                  "Размывается только комната: контур и волосы остаются чёткими, без ореола."))
        card.add(self.bg_blur_box)
        # --- picture
        self.bg_image_box = QWidget()
        il = QVBoxLayout(self.bg_image_box)
        il.setContentsMargins(0, 2, 0, 0)
        il.setSpacing(12)
        self.bg_gallery = backgrounds.BackgroundGallery()
        self.bg_gallery.picked.connect(lambda p: self._emit(bg_image=p))
        self.bg_gallery.addRequested.connect(self._add_background)
        self.bg_gallery.removeRequested.connect(self._remove_background)
        il.addWidget(self.bg_gallery)
        il.addWidget(self._slider("bg_image_blur", "Размытие картинки", 0.0, 1.0, 0.01, pct,
                                  "Лёгкое размытие (10–25%) — как у настоящей камеры, фон выглядит естественнее."))
        card.add(self.bg_image_box)
        # --- colour
        self.bg_color_box = QWidget()
        cl = QVBoxLayout(self.bg_color_box)
        cl.setContentsMargins(0, 2, 0, 0)
        cl.setSpacing(8)
        self.bg_swatches = backgrounds.ColorSwatches()
        self.bg_swatches.picked.connect(lambda c: self._emit(bg_color=c))
        self.bg_swatches.customRequested.connect(self._pick_bg_color)
        self._binds.append(lambda s: self.bg_swatches.set_value(s.bg_color))
        cl.addWidget(self.bg_swatches)
        cl.addWidget(label("Зелёный — хромакей: в OBS его можно убрать фильтром «Хромакей» и поставить свой "
                           "фон или видео.", "faint", wrap=True))
        card.add(self.bg_color_box)
        col.addWidget(card)

        card = Card("Качество контура", "sparkles", "Нейросеть, которая отделяет вас от комнаты")
        self.bg_model_combo = self._combo("bg_model", model_items("matting"))
        self.bg_model_row = FieldRow("Модель", self.bg_model_combo, " ")
        card.add(self.bg_model_row)
        card.add(self._slider("bg_stability", "Стабильность контура", 0.0, 1.0, 0.01, pct,
                              "Убирает мерцание краёв и волос между кадрами. Резкие движения отслеживаются "
                              "без задержки."))
        col.addWidget(card)
        col.addWidget(label("Лучше всего — ровный свет спереди и стена без зеркал за спиной. Фон работает и без "
                            "замены лица. Быстро включить — Ctrl+B. Модели RVM — лицензия GPL-3.0.", "faint",
                            wrap=True))

    def _refresh_gallery(self) -> None:
        try:
            backgrounds.ensure_defaults()
        except Exception as exc:  # the gallery must never break the panel
            self.notify.emit("warn", f"Не удалось подготовить стандартные фоны: {exc}")
        files = backgrounds.list_images(paths.backgrounds_dir())
        self.bg_gallery.set_items(files, self.s.bg_image)
        self._gallery_ready = True
        if not self.s.bg_image and files and self.s.bg_mode == "image":
            self._emit(bg_image=str(files[0]))
            self.bg_gallery.set_current(str(files[0]))

    def _add_background(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Изображение для фона", "",
                                           "Изображения (*.jpg *.jpeg *.png *.webp *.bmp *.heic)")
        if not f:
            return
        try:
            path = backgrounds.import_image(f)
        except Exception as exc:
            self.notify.emit("error", f"Не удалось добавить фон: {exc}")
            return
        self._emit(bg_image=str(path))
        self._refresh_gallery()

    def _remove_background(self, path: str) -> None:
        try:
            paths_ok = paths.backgrounds_dir()
            p = Path(path)
            if p.parent.resolve() == paths_ok.resolve():
                p.unlink(missing_ok=True)
        except OSError as exc:
            self.notify.emit("error", f"Не удалось удалить: {exc}")
            return
        if self.s.bg_image == path:
            files = backgrounds.list_images(paths.backgrounds_dir())
            self._emit(bg_image=str(files[0]) if files else "")
        self._refresh_gallery()

    def _pick_bg_color(self) -> None:
        c = QColorDialog.getColor(QColor(self.s.bg_color), self, "Цвет фона")
        if c.isValid():
            self._emit(bg_color=c.name())
            self.bg_swatches.set_value(c.name())

    # ================================================================== Output
    def _build_output(self, col: QVBoxLayout) -> None:
        card = Card("Виртуальная камера", "monitor", "Для Zoom, Teams, Discord, Telegram, OBS")
        card.add(SettingRow("Включить", self._toggle("vcam_enabled")))
        st = QHBoxLayout()
        st.setSpacing(4)
        self.vcam_dot = StatusDot(8, theme.FAINT)
        self.vcam_text = label("Запустится вместе с обработкой", "dim", wrap=True)
        st.addWidget(self.vcam_dot, 0, Qt.AlignmentFlag.AlignTop)
        st.addWidget(self.vcam_text, 1)
        holder = QWidget()
        holder.setLayout(st)
        card.add(holder)
        # --- Mimiq Camera (own virtual camera, no OBS needed)
        mc = QFrame()
        mc.setObjectName("mcamBox")
        mc.setStyleSheet(f"#mcamBox {{ background: {theme.CARD}; border: 1px solid {theme.LINE}; "
                         f"border-radius: 6px; }}")
        ml = QHBoxLayout(mc)
        ml.setContentsMargins(10, 8, 8, 8)
        ml.setSpacing(8)
        txt = QVBoxLayout()
        txt.setSpacing(0)
        self.mcam_title = label("Mimiq Camera", wrap=True)
        self.mcam_title.setStyleSheet("font-weight: 600;")
        self.mcam_sub = label("Проверяю…", "faint", wrap=True)
        txt.addWidget(self.mcam_title)
        txt.addWidget(self.mcam_sub)
        ml.addLayout(txt, 1)
        self.mcam_btn = QPushButton("Установить")
        self.mcam_btn.setObjectName("primary")
        self.mcam_btn.setStyleSheet("padding: 6px 12px;")
        self.mcam_btn.setCursor(POINTER)
        self.mcam_btn.clicked.connect(self._on_mcam_button)
        ml.addWidget(self.mcam_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._mcam_installed = False
        card.add(mc)
        card.add(FieldRow("Драйвер", self._combo("vcam_backend", [("auto", "Авто"), ("mimiq", "Mimiq Camera"),
                                                                  ("obs", "OBS Virtual Camera")])))
        card.add(FieldRow("Разрешение", self._size_combo("output_width", "output_height", OUT_RES)))
        card.add(FieldRow("Кадров в секунду", self._seg("output_fps", [(24, "24"), (25, "25"), (30, "30"), (60, "60")])))
        card.add(FieldRow("Заполнение кадра", self._seg("output_fit", [("crop", "Обрезать"), ("fit", "Вписать")])))
        card.add(label("В Zoom, Teams, Discord, Telegram или браузере выберите камеру «Mimiq Camera». "
                       "«Авто» использует её, а если она не установлена — OBS Virtual Camera.", "faint", wrap=True))
        col.addWidget(card)

        card = Card("Метка «AI»", "shield")
        card.add(SettingRow("Метка на видео", self._toggle("watermark"),
                            "Небольшой значок «AI · Mimiq» в углу — честно по отношению к собеседникам"))
        col.addWidget(card)

        card = Card("Производительность", "chip")
        card.add(SettingRow("Авто-плавность", self._toggle("auto_quality"),
                            "Если ПК не успевает, Mimiq временно упрощает самое дорогое и возвращает качество, "
                            "когда появляется запас"))
        self.auto_status = label("", "faint", wrap=True)
        card.add(self.auto_status)
        self.set_auto_status(False, [], False)
        card.add(SettingRow("Плавное видео", self._toggle("fluid_video"),
                            "Видео идёт с частотой камеры, даже если нейросеть медленнее: между заменами лицо "
                            "следует за головой без задержки"))
        self.provider_combo = self._combo("execution_provider", [("auto", "Авто")])
        card.add(FieldRow("Вычисления", self.provider_combo))
        card.add(FieldRow("Видеокарта", self._combo("gpu_device", [(i, f"GPU {i}") for i in range(4)])))
        self.provider_info = label("", "faint", wrap=True)
        card.add(self.provider_info)
        self.trt_btn = QPushButton("Установить TensorRT (≈1,9 ГБ)")
        self.trt_btn.setCursor(POINTER)
        self.trt_btn.setToolTip("Откроется окно установщика. После установки перезапустите Mimiq.")
        self.trt_btn.clicked.connect(self.installTensorrt.emit)
        self.trt_btn.setVisible(False)
        card.add(self.trt_btn)
        col.addWidget(card)

        card = Card("Папки", "folder")
        row = QHBoxLayout()
        row.setSpacing(8)
        for key, text in (("captures", "Снимки"), ("models", "Модели"), ("logs", "Логи")):
            b = QPushButton(text)
            b.setObjectName("ghost")
            b.setCursor(POINTER)
            b.clicked.connect(lambda _=False, k=key: self.openFolder.emit(k))
            row.addWidget(b)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        card.add(holder)
        col.addWidget(card)

        card = Card(f"Mimiq {__version__}", "info")
        card.add(label("Локальная обработка: видео и фото не покидают ваш компьютер. Модели InsightFace / FaceFusion "
                       "распространяются для некоммерческого использования.", "faint", wrap=True))
        card.add(SettingRow("Проверять обновления", self._toggle("check_updates"),
                            "При запуске Mimiq спрашивает у GitHub номер последней версии. Больше ничего "
                            "не отправляется"))
        upd = QHBoxLayout()
        upd.setContentsMargins(0, 0, 0, 0)
        upd.setSpacing(8)
        self.update_btn = QPushButton("Проверить сейчас")
        self.update_btn.setObjectName("ghost")
        self.update_btn.setCursor(POINTER)
        self.update_btn.clicked.connect(self.checkUpdates.emit)
        upd.addWidget(self.update_btn)
        self.update_status = label("", "faint", wrap=True)
        upd.addWidget(self.update_status, 1)
        holder = QWidget()
        holder.setLayout(upd)
        card.add(holder)
        col.addWidget(card)

    def set_update_status(self, text: str, busy: bool = False) -> None:
        self.update_btn.setEnabled(not busy)
        self.update_status.setText(text)

    def set_providers(self, available: Sequence[str], active: str, gpu_name: str = "") -> None:
        items = [("auto", "Авто")]
        for key, text in (("tensorrt", "NVIDIA TensorRT · быстрее"), ("cuda", "NVIDIA CUDA"),
                          ("directml", "DirectML · AMD / Intel"), ("cpu", "CPU")):
            ok = key in available
            if key == "tensorrt" and not ok and "cuda" not in available:
                continue                                   # not an NVIDIA PC — don't tease
            items.append((key, text if ok else f"{text} — не установлено" if key == "tensorrt"
                          else f"{text} — недоступно", ok))
        self.provider_combo.set_items(items, self.s.execution_provider)
        label_ = models.PROVIDER_LABELS.get(active, active)
        info = f"Сейчас: {label_}" + (f" · {gpu_name}" if gpu_name else "")
        if active == "cpu":
            info += ". Для реального времени нужна видеокарта NVIDIA RTX (CUDA) или DirectML."
        elif active == "tensorrt":
            info += ". Первый запуск после смены модели — пара минут на оптимизацию, дальше сразу из кэша."
        want_trt = "cuda" in available and "tensorrt" not in available and sys.platform == "win32"
        if want_trt:
            info += (". С TensorRT лицо обновляется заметно чаще (обычно в 1,5–3 раза) — нужна NVIDIA RTX / "
                     "GTX 16xx.")
        self.trt_btn.setVisible(want_trt)
        self.provider_info.setText(info)

    def set_auto_status(self, running: bool, labels: Sequence[str], exhausted: bool) -> None:
        if not self.s.auto_quality:
            text = ""
        elif not running:
            text = "Сейчас: работает, когда запущена обработка."
        elif labels:
            text = "Сейчас упрощено: " + ", ".join(labels) + (". Это максимум упрощения для этого ПК." if exhausted
                                                              else ".")
        else:
            text = "Сейчас: всё по выбранным настройкам."
        if self.auto_status.text() != text:
            self.auto_status.setText(text)
        self.auto_status.setVisible(bool(text))

    def set_vcam_status(self, active: bool, text: str) -> None:
        if active:
            self.vcam_dot.set_color(theme.OK)
            self.vcam_text.setText(f"{text or 'Виртуальная камера'} · транслируется")
            self.vcam_text.setStyleSheet("")
        elif text:
            self.vcam_dot.set_color(theme.BAD)
            self.vcam_text.setText(text)
            self.vcam_text.setStyleSheet(f"color: {theme.BAD};")
        else:
            self.vcam_dot.set_color(theme.FAINT)
            self.vcam_text.setText("Запустится вместе с обработкой" if self.s.vcam_enabled else "Выключена")
            self.vcam_text.setStyleSheet("")

    def set_mcam_state(self, supported: bool, installed: bool, busy: str = "", other: str = "") -> None:
        """State of the Mimiq Camera box. `busy` = text shown while (un)installing."""
        self._mcam_installed = installed
        self.mcam_btn.setVisible(supported)
        self.mcam_btn.setEnabled(not busy)
        if busy:
            self.mcam_sub.setText(busy)
            return
        if not supported:
            self.mcam_sub.setText("Своя виртуальная камера Mimiq есть только в Windows.")
        elif installed:
            self.mcam_sub.setText("Установлена — выберите «Mimiq Camera» в списке камер приложения. "
                                  "Нет в списке? Полностью перезапустите браузер или приложение для звонков.")
        else:
            note = f" (сейчас вместо неё зарегистрирована «{other}»)" if other else ""
            self.mcam_sub.setText("Своя камера Mimiq, OBS не нужен. Установка — один раз, Windows спросит права "
                                  "администратора" + note + ".")
        self.mcam_btn.setText("Удалить" if installed else "Установить")
        self.mcam_btn.setObjectName("ghost" if installed else "primary")
        self.mcam_btn.style().unpolish(self.mcam_btn)
        self.mcam_btn.style().polish(self.mcam_btn)

    def _on_mcam_button(self) -> None:
        self.vcamSetup.emit("uninstall" if self._mcam_installed else "install")

    def refresh_model_marks(self) -> None:
        self.swapper_combo.set_items(model_items("swapper"), self.s.swapper_model)
        self.bg_model_combo.set_items(model_items("matting"), self.s.bg_model)
        self.enhancer_combo.set_items(model_items("enhancer", "Выключено"), self.s.enhancer_model)

    # ================================================================== load / visibility
    def load(self, s: Settings) -> None:
        self.s = s.copy()
        for bind in self._binds:
            bind(s)
        self._sync_visibility()

    def _sync_visibility(self) -> None:
        s = self.s
        self.link_box.setVisible(s.source_kind == "link")
        self.device_box.setVisible(s.source_kind == "device")
        self.url_box.setVisible(s.source_kind == "url")
        self.fidelity_row.setVisible(s.enhancer_model == "codeformer")
        self.blend_row.setEnabled(bool(s.enhancer_model))
        self.occluder_row.setEnabled(s.mask_occlusion)
        self.parser_row.setEnabled(s.mask_region)
        spec = models.REGISTRY.get(s.swapper_model)
        if spec is not None and self.swapper_row.hint is not None:
            self.swapper_row.hint.setText(spec.note or " ")
        self.bg_blur_box.setVisible(s.bg_mode == "blur")
        self.bg_image_box.setVisible(s.bg_mode == "image")
        self.bg_color_box.setVisible(s.bg_mode == "color")
        if s.bg_mode == "image" and not getattr(self, "_gallery_ready", False):
            self._refresh_gallery()
        spec = models.REGISTRY.get(s.bg_model)
        if spec is not None and self.bg_model_row.hint is not None:
            self.bg_model_row.hint.setText(spec.note or " ")

    def show_tab(self, index: int) -> None:
        self.tabs.set_index(index)
        self.stack.setCurrentIndex(index)
