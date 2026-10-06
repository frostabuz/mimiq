"""Right-hand settings column: Source · Swap · Mask · Output tabs."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from .. import __version__
from ..config import Settings
from ..core import models
from . import icons, theme
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
        self.tabs = TabBar([("Источник", "camera"), ("Замена", "face"), ("Маска", "mask"), ("Вывод", "monitor")])
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
        for build in (self._build_source, self._build_swap, self._build_mask, self._build_output):
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
        card.add(FieldRow("Драйвер", self._combo("vcam_backend", [("auto", "Авто"), ("obs", "OBS Virtual Camera"),
                                                                  ("unitycapture", "Unity Capture")])))
        card.add(FieldRow("Разрешение", self._size_combo("output_width", "output_height", OUT_RES)))
        card.add(FieldRow("Кадров в секунду", self._seg("output_fps", [(24, "24"), (25, "25"), (30, "30"), (60, "60")])))
        card.add(FieldRow("Заполнение кадра", self._seg("output_fit", [("crop", "Обрезать"), ("fit", "Вписать")])))
        card.add(label("В приложении выберите камеру «OBS Virtual Camera». Для работы нужен установленный OBS Studio "
                       "(достаточно один раз запустить и остановить в нём виртуальную камеру).", "faint", wrap=True))
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
        col.addWidget(card)

    def set_providers(self, available: Sequence[str], active: str, gpu_name: str = "") -> None:
        items = [("auto", "Авто")]
        for key, text in (("cuda", "NVIDIA CUDA"), ("tensorrt", "NVIDIA TensorRT"), ("directml", "DirectML · AMD / Intel"),
                          ("cpu", "CPU")):
            ok = key in available
            items.append((key, text if ok else f"{text} — недоступно", ok))
        self.provider_combo.set_items(items, self.s.execution_provider)
        label_ = models.PROVIDER_LABELS.get(active, active)
        info = f"Сейчас: {label_}" + (f" · {gpu_name}" if gpu_name else "")
        if active == "cpu":
            info += ". Для реального времени нужна видеокарта NVIDIA RTX (CUDA) или DirectML."
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

    def refresh_model_marks(self) -> None:
        self.swapper_combo.set_items(model_items("swapper"), self.s.swapper_model)
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

    def show_tab(self, index: int) -> None:
        self.tabs.set_index(index)
        self.stack.setCurrentIndex(index)
