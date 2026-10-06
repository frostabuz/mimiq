"""Mimiq main window: header · faces | preview + controls | settings · status bar."""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from .. import __version__, paths, updates
from ..config import PRESET_LABELS, PRESETS, Settings
from ..core import models
from ..core.identity import Identity, IdentityLibrary
from ..core.tuner import rate_text
from ..engine import Engine
from ..imaging import to_qimage
from ..io import vcam_setup
from . import brand, icons, theme
from .dialogs import ConnectDialog
from .faces import FacesPanel, NewFaceDialog, expand_paths, pick_photos
from .platform import gpu_name
from .preview import PreviewWidget, likeness_word
from .settings_panel import SettingsPanel
from .widgets import (POINTER, Chip, IconButton, Segmented, StatusDot, ToastManager, ToggleSwitch, Wordmark, label)

log = logging.getLogger("mimiq.ui")

PRESET_KEYS = set().union(*[set(v) for v in PRESETS.values()])
PRESET_TIPS = {
    "speed": "Без улучшения лица и лёгкая детекция — для слабых видеокарт и ноутбуков",
    "balanced": "Pixel Boost 128 + GPEN 256 — плавно на большинстве видеокарт",
    "quality": "Pixel Boost 256, GPEN 512 и точная форма лица — для RTX 3060 и выше",
    "ultra": "Как «Качество» + 2 прохода замены (больше сходства) — для RTX 4070 и выше",
}
STAGE_NAMES = (("track", "трекинг"), ("swap", "замена"), ("enhance", "улучшение"), ("occlusion", "перекрытия"),
               ("parser", "форма"))


class _Bridge(QObject):
    cameras = Signal(list)
    gpu = Signal(str)
    mcam = Signal(dict)                    # Mimiq Camera status / (un)install result
    update = Signal(object, str, bool)     # Release | None, error text, manual check


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, engine: Engine, library: IdentityLibrary):
        super().__init__()
        self.settings = settings
        self.engine = engine
        self.library = library
        self._connect_dialog: Optional[ConnectDialog] = None
        self._manual_stop = False
        self._gpu_name = ""
        self._was_maximized = False
        self._link_connected = False
        self._release: Optional[updates.Release] = None
        self._update_busy = False
        self.setWindowTitle("Mimiq")
        self.setWindowIcon(QIcon(str(paths.package_dir() / "assets" / "mimiq.ico")))
        self.setMinimumSize(1280, 740)
        self.setAcceptDrops(True)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(16, 10, 16, 6)
        v.setSpacing(12)
        self.header = self._build_header()
        v.addWidget(self.header)

        body = QHBoxLayout()
        body.setSpacing(12)
        self.faces = FacesPanel(library)
        self.faces.setFixedWidth(288)
        body.addWidget(self.faces)
        center = QVBoxLayout()
        center.setSpacing(12)
        self.preview = PreviewWidget()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center.addWidget(self.preview, 1)
        self.controls = self._build_controls()
        center.addWidget(self.controls)
        body.addLayout(center, 1)
        self.settings_panel = SettingsPanel(settings)
        self.settings_panel.setFixedWidth(368)
        body.addWidget(self.settings_panel)
        v.addLayout(body, 1)
        self.status = self._build_status()
        v.addWidget(self.status)
        self.toasts = ToastManager(root, bottom_margin=118)
        self.toasts.activeChanged.connect(self.preview.empty.set_hint_suppressed)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self._save)

        self._wire()
        self._shortcuts()
        self._apply_initial_state()

    # ================================================================== construction
    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setObjectName("header")
        w.setFixedHeight(52)
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 0, 0, 0)
        h.setSpacing(10)
        mid = Qt.AlignmentFlag.AlignVCenter
        logo = QLabel()
        logo.setPixmap(brand.logo_pixmap(26))
        h.addWidget(logo, 0, mid)
        h.addWidget(Wordmark("Mimiq", 13.5), 0, mid)
        ver = label(__version__, "faint")
        h.addWidget(ver, 0, mid)
        h.addSpacing(22)
        h.addWidget(label("Режим", "dim"), 0, mid)
        self.preset_seg = Segmented([(k, PRESET_LABELS[k]) for k in PRESETS], expand=False, tooltips=PRESET_TIPS)
        self.preset_seg.changed.connect(self.apply_preset)
        h.addWidget(self.preset_seg, 0, mid)
        self.custom_badge = label("Свои настройки", "badge")
        self.custom_badge.setToolTip("Вы изменили параметры вручную. Выберите режим, чтобы вернуть пресет.")
        h.addWidget(self.custom_badge, 0, mid)
        h.addStretch(1)
        self.update_chip = Chip("Доступно обновление")
        self.update_chip.setCursor(POINTER)
        self.update_chip.clicked.connect(self._show_update)
        self.update_chip.hide()
        h.addWidget(self.update_chip, 0, mid)
        self.provider_chip = Chip("Подготовка…", "chip")
        self.provider_chip.setToolTip("Чем считаются нейросети")
        h.addWidget(self.provider_chip, 0, mid)
        self.vcam_chip = Chip("Вирт. камера", "monitor")
        h.addWidget(self.vcam_chip, 0, mid)
        self.connect_btn = QPushButton("  Подключить iPhone")
        self.connect_btn.setObjectName("primary")
        self.connect_btn.setIcon(icons.icon("phone", "#FFFFFF", 16))
        self.connect_btn.setIconSize(QSize(16, 16))
        self.connect_btn.setCursor(POINTER)
        self.connect_btn.clicked.connect(self.show_connect)
        h.addWidget(self.connect_btn, 0, mid)
        return w

    def _build_controls(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("controlBar")
        bar.setFixedHeight(58)
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 8, 10, 8)
        h.setSpacing(10)
        self.run_btn = QPushButton()
        self.run_btn.setCursor(POINTER)
        self.run_btn.setMinimumWidth(118)
        self.run_btn.setIconSize(QSize(16, 16))
        self.run_btn.clicked.connect(self.toggle_run)
        h.addWidget(self.run_btn)
        h.addSpacing(4)
        self.mode_seg = Segmented([("result", "", "face"), ("split", "", "split"), ("original", "", "eye"),
                                   ("mask", "", "mask")], expand=False,
                                  tooltips={"result": "Результат (Ctrl+1)", "split": "Сравнение до/после (Ctrl+2)",
                                            "original": "Оригинал (Ctrl+3)", "mask": "Маска замены (Ctrl+4)"})
        for b in self.mode_seg.buttons.values():
            b.setFixedSize(38, 30)
        self.mode_seg.changed.connect(lambda m: self.update_settings({"preview_mode": m}))
        h.addWidget(self.mode_seg)
        h.addStretch(1)
        self.swap_toggle = ToggleSwitch()
        self.swap_toggle.toggled.connect(lambda on: self.update_settings({"swap_enabled": on}))
        self.swap_label = label("Замена лица")
        self.vcam_toggle = ToggleSwitch()
        self.vcam_toggle.toggled.connect(lambda on: self.update_settings({"vcam_enabled": on}))
        self.vcam_label = label("Вирт. камера")
        for ic, lab, tog, tip in (("face", self.swap_label, self.swap_toggle, "Замена лица (Ctrl+E)"),
                                  ("monitor", self.vcam_label, self.vcam_toggle, "Отправлять видео в виртуальную камеру")):
            box = QHBoxLayout()
            box.setSpacing(8)
            icl = QLabel()
            icl.setPixmap(icons.pixmap(ic, theme.DIM, 18))
            icl.setToolTip(tip)
            tog.setToolTip(tip)
            box.addWidget(icl)
            box.addWidget(lab)
            box.addWidget(tog)
            h.addLayout(box)
            h.addSpacing(8)
        sep = QFrame()
        sep.setFixedSize(1, 28)
        sep.setStyleSheet(f"background: {theme.LINE};")
        h.addWidget(sep)
        self.snap_btn = IconButton("snapshot", "Снимок (Ctrl+S)")
        self.snap_btn.clicked.connect(self.snapshot)
        self.rec_btn = IconButton("record", "Запись видео (Ctrl+R)", checkable=True, active_color=theme.LIVE)
        self.rec_btn.setObjectName("rec")
        self.rec_btn.clicked.connect(self.toggle_record)
        self.full_btn = IconButton("expand", "Полный экран (F11)")
        self.full_btn.clicked.connect(self.toggle_fullscreen)
        for b in (self.snap_btn, self.rec_btn, self.full_btn):
            h.addWidget(b)
        return bar

    def _build_status(self) -> QWidget:
        w = QWidget()
        w.setObjectName("statusBar")
        w.setFixedHeight(24)
        h = QHBoxLayout(w)
        h.setContentsMargins(6, 0, 6, 0)
        h.setSpacing(4)
        self.src_dot = StatusDot(7, theme.FAINT)
        self.src_text = label("Источник не запущен", "faint")
        self.cam_dot = StatusDot(7, theme.FAINT)
        self.cam_text = label("Вирт. камера выкл.", "faint")
        self.perf_text = label("", "faint")
        h.addWidget(self.src_dot)
        h.addWidget(self.src_text)
        h.addSpacing(16)
        h.addWidget(self.cam_dot)
        h.addWidget(self.cam_text)
        h.addSpacing(16)
        h.addWidget(self.perf_text)
        h.addStretch(1)
        self.right_text = label(f"Mimiq {__version__}", "faint")
        h.addWidget(self.right_text)
        return w

    # ================================================================== wiring
    def _wire(self) -> None:
        e = self.engine
        e.frameReady.connect(self._on_frame)
        e.statsUpdated.connect(self._on_stats)
        e.stateChanged.connect(self._on_state)
        e.message.connect(lambda lvl, txt: self.toasts.show(lvl, txt))
        e.sourceStatus.connect(self._on_source_status)
        e.linkEvent.connect(self._on_link)
        e.vcamStatus.connect(self._on_vcam)
        e.modelProgress.connect(self._on_model_progress)
        e.modelsReady.connect(self._on_models_ready)
        e.recordingChanged.connect(self._on_recording)
        self.faces.faceSelected.connect(self.select_identity)
        self.faces.filesChosen.connect(self.create_face)
        self.faces.addPhotosRequested.connect(self.add_photos)
        self.faces.faceRenamed.connect(self._on_face_renamed)
        self.settings_panel.changed.connect(lambda kw: self.update_settings(kw, origin="panel"))
        self.settings_panel.connectRequested.connect(self.show_connect)
        self.settings_panel.openFolder.connect(self._open_folder)
        self.settings_panel.refreshCameras.connect(self.refresh_cameras)
        self.preview.toggleFullscreen.connect(self.toggle_fullscreen)
        self.preview.primaryClicked.connect(self._preview_action)
        self.preview.secondaryClicked.connect(self._preview_action)
        self._bridge = _Bridge()
        self._bridge.cameras.connect(self.settings_panel.set_cameras)
        self._bridge.gpu.connect(self._on_gpu_name)
        self._bridge.mcam.connect(self._on_mcam)
        self.settings_panel.vcamSetup.connect(self._vcam_setup)
        self.settings_panel.installTensorrt.connect(self._install_tensorrt)
        self.settings_panel.checkUpdates.connect(lambda: self.check_updates(manual=True))
        self._bridge.update.connect(self._on_update)

    def _shortcuts(self) -> None:
        for seq, fn in (("Space", self.toggle_run), ("F11", self.toggle_fullscreen), ("Ctrl+S", self.snapshot),
                        ("Ctrl+R", self.toggle_record), ("Ctrl+E", lambda: self.swap_toggle.toggle()),
                        ("Ctrl+1", lambda: self._set_mode("result")), ("Ctrl+2", lambda: self._set_mode("split")),
                        ("Ctrl+3", lambda: self._set_mode("original")), ("Ctrl+4", lambda: self._set_mode("mask")),
                        ("Ctrl+H", lambda: self.update_settings({"show_hud": not self.settings.show_hud})),
                        ("Escape", self._escape)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)

    def _apply_initial_state(self) -> None:
        s = self.settings
        self.preset_seg.set_value(s.preset if s.preset in PRESETS else None)
        self.custom_badge.setVisible(s.preset not in PRESETS)
        self.mode_seg.set_value(s.preview_mode)
        self.swap_toggle.set_value(s.swap_enabled)
        self.vcam_toggle.set_value(s.vcam_enabled)
        self.preview.set_mode(s.preview_mode)
        self.preview.set_swap_enabled(s.swap_enabled)
        self.preview.set_show_hud(s.show_hud)
        self.preview.set_source_kind(s.source_kind)
        self._update_run_button(False)
        self._update_vcam_chip(False, "")
        self.faces.reload(s.active_face)
        ident = self.library.get(s.active_face) if s.active_face else None
        if ident is not None:
            self.engine.set_identity(ident)
            self.preview.set_identity(ident.name, to_qimage(ident.thumbnail))
        elif s.active_face:
            self.settings.active_face = ""
        self._render_brand_frames()
        self.settings_panel.set_link_status("waiting", "Ожидание iPhone", "Нажмите «QR-код» и отсканируйте его")
        self._update_source_status_text()
        try:
            self.settings_panel.set_providers(models.available_providers(), models.resolve_provider(s.execution_provider))
        except Exception as exc:  # onnxruntime missing → the engine will report it too
            log.warning("providers unavailable: %s", exc)
        self.refresh_cameras()
        threading.Thread(target=lambda: self._bridge.gpu.emit(gpu_name()), daemon=True).start()
        self._refresh_mcam()
        if s.check_updates:
            QTimer.singleShot(5000, self.check_updates)   # after start-up, so it never slows the launch

    # ================================================================== settings
    def update_settings(self, kw: Dict, from_preset: bool = False, origin: str = "") -> None:
        old = self.settings
        new = old.copy().update(**kw)
        changed = [k for k in kw if getattr(old, k, None) != getattr(new, k, None)]
        if not changed:
            return
        if not from_preset and (set(changed) & PRESET_KEYS) and new.preset:
            new.preset = ""
            self.preset_seg.set_value(None)
            self.custom_badge.show()
        self.settings = new
        self.engine.apply(new.copy(), changed)
        ch = set(changed)
        if "preview_mode" in ch:
            self.preview.set_mode(new.preview_mode)
            self.mode_seg.set_value(new.preview_mode)
        if "swap_enabled" in ch:
            self.preview.set_swap_enabled(new.swap_enabled)
            self.swap_toggle.set_value(new.swap_enabled)
        if "vcam_enabled" in ch:
            self.vcam_toggle.set_value(new.vcam_enabled)
            self._update_vcam_chip(self.engine.vcam.active, "")
        if "show_hud" in ch:
            self.preview.set_show_hud(new.show_hud)
        if "source_kind" in ch:
            self.preview.set_source_kind(new.source_kind)
            self._update_source_status_text()
            if new.source_kind == "device":
                self.refresh_cameras()
        if ch & {"output_width", "output_height"}:
            self._render_brand_frames()
        if "check_updates" in ch and new.check_updates and self._release is None:
            self.check_updates()
        if origin == "panel":
            self.settings_panel.s = new.copy()
        else:
            self.settings_panel.load(new)
        if "vcam_enabled" in ch:
            self.settings_panel.set_vcam_status(self.engine.vcam.active, "")
        self._save_timer.start()

    def apply_preset(self, key: str) -> None:
        if key not in PRESETS:
            return
        kw = dict(PRESETS[key])
        kw["preset"] = key
        self.update_settings(kw, from_preset=True)
        self.custom_badge.hide()
        self.toasts.show("info", f"Режим «{PRESET_LABELS[key]}»: {PRESET_TIPS[key]}")

    def _save(self) -> None:
        try:
            self.settings.save()
        except Exception as exc:
            log.warning("could not save settings: %s", exc)

    def _set_mode(self, mode: str) -> None:
        self.update_settings({"preview_mode": mode})

    # ================================================================== identities
    def select_identity(self, ident: Optional[Identity]) -> None:
        self.engine.set_identity(ident)
        self.update_settings({"active_face": ident.id if ident else ""})
        if ident is None:
            self.preview.set_identity("", None)
        else:
            self.preview.set_identity(ident.name, to_qimage(ident.thumbnail))
            self.faces.set_active(ident.id)

    def create_face(self, files) -> None:
        files = expand_paths(list(files))
        if not files:
            self.toasts.show("warn", "Не нашёл подходящих изображений (JPG, PNG, HEIC, WEBP)")
            return
        n = len(self.library.list()) + 1
        dlg = NewFaceDialog(self.engine, files, None, f"Лицо {n}", self)
        if dlg.exec() and dlg.result_identity is not None:
            ident = dlg.result_identity
            self.faces.reload(ident.id)
            self.select_identity(ident)
            self.toasts.show("ok", f"Лицо «{ident.name}» готово и выбрано")

    def add_photos(self, ident: Identity) -> None:
        files = pick_photos(self, f"Добавить фото — {ident.name}")
        if not files:
            return
        dlg = NewFaceDialog(self.engine, files, ident, ident.name, self)
        if dlg.exec() and dlg.result_identity is not None:
            new = dlg.result_identity
            self.faces.reload()
            if self.settings.active_face == new.id:
                self.select_identity(new)
            self.toasts.show("ok", f"Профиль «{new.name}» обновлён · {new.photos} фото")

    def _on_face_renamed(self, ident: Identity) -> None:
        if self.settings.active_face == ident.id:
            self.engine.set_identity(ident)
            self.preview.set_identity(ident.name, to_qimage(ident.thumbnail))

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if files:
            self.create_face(files)

    # ================================================================== run control
    def toggle_run(self) -> None:
        if self.engine.running:
            self._manual_stop = True
            self.engine.stop()
        else:
            self._manual_stop = False
            self.engine.start()

    def _update_run_button(self, running: bool) -> None:
        if running:
            self.run_btn.setText("  Стоп")
            self.run_btn.setObjectName("danger")
            self.run_btn.setIcon(icons.icon("stop", "#FF7A70", 16))
        else:
            self.run_btn.setText("  Старт")
            self.run_btn.setObjectName("primary")
            self.run_btn.setIcon(icons.icon("play", "#FFFFFF", 16))
        self.run_btn.style().unpolish(self.run_btn)
        self.run_btn.style().polish(self.run_btn)
        self.rec_btn.setEnabled(running)
        self.snap_btn.setEnabled(running)

    def _on_state(self, state: str) -> None:
        running = state == "running"
        self._update_run_button(running)
        if running:
            self.preview.set_state("waiting", "", self.settings.source_kind)
        else:
            self.preview.set_state("idle", "", self.settings.source_kind)
            self.perf_text.setText("")
        self._update_source_status_text()

    def _preview_action(self, action: str) -> None:
        if action == "qr":
            self.show_connect()
        elif action == "start":
            if not self.engine.running:
                self._manual_stop = False
                self.engine.start()
        elif action == "retry":
            self.engine.stop()
            self.engine.start()
        elif action == "settings":
            self.settings_panel.show_tab(0)

    def snapshot(self) -> None:
        path = self.engine.snapshot()
        if path is None:
            self.toasts.show("warn", "Нет кадра для снимка — запустите обработку")
        else:
            self.toasts.show("ok", f"Снимок сохранён: {path.name}")

    def toggle_record(self) -> None:
        if self.engine.recording:
            self.engine.stop_recording()
        elif self.engine.running:
            self.engine.start_recording()
        else:
            self.rec_btn.setChecked(False)
            self.toasts.show("warn", "Сначала нажмите «Старт»")

    def _on_recording(self, on: bool, path: str) -> None:
        self.rec_btn.setChecked(on)
        self.preview.set_recording(on)
        if not on and path:
            self.toasts.show("ok", f"Видео сохранено: {path}")

    # ================================================================== engine events
    def _on_frame(self, d: dict) -> None:
        try:
            self.preview.set_frames(d.get("result"), d.get("original"))
        finally:
            self.engine.preview_consumed()

    def _on_stats(self, st: dict) -> None:
        w, h = st.get("input") or (None, None)
        res = f"{w}×{h}" if w and h else ""
        dpr = self.preview.devicePixelRatioF()
        self.engine.preview_size = (int(self.preview.width() * dpr), int(self.preview.height() * dpr))
        fluid = st.get("mode") == "fluid"
        likeness = st.get("likeness")
        self.preview.set_perf(st.get("face_fps", 0.0), fluid, likeness)
        self.preview.set_hud(st.get("fps", 0.0), st.get("latency", 0.0), st.get("source_name", ""), res,
                             st.get("faces", []))
        self.settings_panel.set_auto_status(self.engine.running, st.get("auto") or [], st.get("auto_exhausted", False))
        t = st.get("timings", {})
        if st.get("fps", 0) > 0:
            parts = [f"Видео {st['fps']:.0f} к/с"]
            if st.get("faces") and st.get("face_fps", 0) > 0:
                parts.append(f"лицо {rate_text(st['face_fps'])}")
            for key, name in STAGE_NAMES:
                if t.get(key, 0) > 0.5:
                    parts.append(f"{name} {t[key]:.0f} мс")
            if likeness is not None and st.get("faces"):
                parts.append(f"сходство {likeness:.2f} ({likeness_word(likeness)})")
            if st.get("auto"):
                parts.append("авто: " + ", ".join(st["auto"]))
            self.perf_text.setText("  ·  ".join(parts))
        if self.engine.running and st.get("fps", 0) == 0 and st.get("source_status") == "connecting":
            self.preview.set_state("waiting", "", self.settings.source_kind)

    def _on_source_status(self, status: str, text: str) -> None:
        if self.engine.running:
            if status == "connecting":
                self.preview.set_state("waiting", text, self.settings.source_kind)
            elif status == "error":
                self.preview.set_state("error", text, self.settings.source_kind)
        self._update_source_status_text(status, text)

    def _update_source_status_text(self, status: Optional[str] = None, text: str = "") -> None:
        s = self.settings
        if s.source_kind == "link":
            if self._link_connected:
                self.src_dot.set_color(theme.OK)
                self.src_text.setText("iPhone подключён · Mimiq Link")
            else:
                self.src_dot.set_color(theme.WARN, pulse=self.engine.running)
                self.src_text.setText("Ожидание iPhone · Mimiq Link")
        else:
            if s.source_kind == "device":
                name = s.device_name or f"Камера {s.device_index + 1}"
            else:
                name = s.stream_url or "Сетевой поток"
            if not self.engine.running:
                self.src_dot.set_color(theme.FAINT)
                self.src_text.setText(f"{name} · не запущено")
            elif status == "error":
                self.src_dot.set_color(theme.BAD)
                self.src_text.setText(f"{name} · {text or 'ошибка'}")
            elif status == "live":
                self.src_dot.set_color(theme.OK)
                self.src_text.setText(f"{name} · в эфире")
            else:
                self.src_dot.set_color(theme.WARN, pulse=True)
                self.src_text.setText(f"{name} · подключение…")

    def _on_link(self, kind: str, data: dict) -> None:
        if kind == "listening":
            ips = data.get("ips") or []
            self.settings_panel.set_link_status("waiting", "Ожидание iPhone",
                                                f"Сервер: {ips[0]}:{self.settings.link_port}" if ips else "")
        elif kind == "error":
            self.settings_panel.set_link_status("error", "Mimiq Link не запущен", data.get("message", ""))
        elif kind == "connected":
            self._link_connected = True
            self._manual_stop = False
            self.settings_panel.set_link_status("connected", "iPhone на связи", "Нажмите «Включить камеру» на телефоне")
            self._set_connect_button(True)
        elif kind == "hello":
            self._link_connected = True
            w, h = data.get("width") or 0, data.get("height") or 0
            dev = data.get("device") or "iPhone"
            cam = "фронтальная" if data.get("camera") == "user" else "основная"
            self.settings_panel.set_link_status("connected", f"{dev} подключён",
                                                f"{w}×{h} · {cam} камера" if w else "Камера ещё не включена")
            self._set_connect_button(True)
            if w and not self.engine.running and self.settings.source_kind == "link" and not self._manual_stop:
                self.engine.start()
            if w:
                self.toasts.show("ok", f"{dev} подключён по Wi-Fi")
        elif kind == "disconnected":
            self._link_connected = False
            self.settings_panel.set_link_status("waiting", "Ожидание iPhone", "Телефон отключился")
            self._set_connect_button(False)
            if self.settings.source_kind == "link":
                self.toasts.show("warn", "iPhone отключился. Mimiq ждёт повторного подключения")
        self._update_source_status_text()

    def _set_connect_button(self, connected: bool) -> None:
        if connected:
            self.connect_btn.setText("  iPhone подключён")
            self.connect_btn.setObjectName("")
            self.connect_btn.setIcon(icons.icon("check", theme.OK, 16))
        else:
            self.connect_btn.setText("  Подключить iPhone")
            self.connect_btn.setObjectName("primary")
            self.connect_btn.setIcon(icons.icon("phone", "#FFFFFF", 16))
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)

    def _on_vcam(self, active: bool, text: str) -> None:
        self.settings_panel.set_vcam_status(active, text)
        self._update_vcam_chip(active, text)
        if text and not active:
            self.toasts.show("warn", text, 6000)

    def _update_vcam_chip(self, active: bool, text: str) -> None:
        if active:
            self.vcam_chip.set("Вирт. камера · в эфире", theme.OK, self.engine.vcam.device)
            self.cam_dot.set_color(theme.OK)
            self.cam_text.setText(f"{self.engine.vcam.device or 'Виртуальная камера'} · транслируется")
        elif text:
            self.vcam_chip.set("Вирт. камера · ошибка", theme.BAD, text)
            self.cam_dot.set_color(theme.BAD)
            self.cam_text.setText("Вирт. камера недоступна")
        else:
            on = self.settings.vcam_enabled
            self.vcam_chip.set("Вирт. камера" + (" · ждёт старта" if on else " · выкл."), theme.DIM if on else theme.FAINT,
                               "Запустится вместе с обработкой" if on else "Выключена")
            self.cam_dot.set_color(theme.FAINT)
            self.cam_text.setText("Вирт. камера ждёт старта" if on else "Вирт. камера выкл.")

    def _on_model_progress(self, stage: str, frac: float, text: str) -> None:
        self.preview.set_loading(stage, frac, text)
        if stage == "error":
            QTimer.singleShot(8000, lambda: self.preview.set_loading(None))

    def _on_models_ready(self, ok: bool) -> None:
        self.preview.set_models_ready(ok)
        self.settings_panel.refresh_model_marks()
        try:
            self.settings_panel.set_providers(models.available_providers(), self.engine.provider, self._gpu_name)
        except Exception:
            pass
        self._update_provider_chip()
        if not ok:
            self.toasts.show("error", "Не удалось подготовить модели. Проверьте интернет и нажмите «Старт» снова "
                                      "или перезапустите Mimiq.", 8000)

    def _on_gpu_name(self, name: str) -> None:
        self._gpu_name = name
        self._update_provider_chip()

    def _update_provider_chip(self) -> None:
        if not self.engine.models_ready:
            return
        prov = self.engine.provider
        text = {"cuda": "CUDA", "tensorrt": "TensorRT", "directml": "DirectML", "cpu": "CPU"}.get(prov, prov)
        if self._gpu_name and prov in ("cuda", "tensorrt"):
            text += f" · {self._gpu_name}"
        self.provider_chip.set(text, theme.OK if prov != "cpu" else theme.WARN,
                               models.PROVIDER_LABELS.get(prov, prov))
        self.right_text.setText(f"{models.PROVIDER_LABELS.get(prov, prov)}  ·  Mimiq {__version__}")

    def _install_tensorrt(self) -> None:
        bat = paths.app_root() / "install.bat"
        if not bat.exists():
            self.toasts.show("error", "Не найден install.bat в папке Mimiq.", 6000)
            return
        try:
            os.startfile(str(bat), "open", "tensorrt", str(paths.app_root()))  # type: ignore[attr-defined]
        except Exception as exc:
            self.toasts.show("error", f"Не удалось открыть установщик: {exc}", 6000)
            return
        self.toasts.show("info", "Открылось окно установки TensorRT (≈1,9 ГБ). Когда оно закончит — "
                                 "перезапустите Mimiq.", 9000)

    # ================================================================== Mimiq Camera
    def _refresh_mcam(self) -> None:
        def job():
            try:
                st = vcam_setup.status()
            except Exception as exc:
                log.warning("Mimiq Camera status failed: %s", exc)
                st = {"supported": vcam_setup.supported(), "installed": False, "name": None, "other": False}
            self._bridge.mcam.emit(st)
        threading.Thread(target=job, daemon=True, name="mimiq-mcam").start()

    def _vcam_setup(self, action: str) -> None:
        if action == "uninstall":
            box = QMessageBox(QMessageBox.Icon.NoIcon, "Удалить Mimiq Camera",
                              "Камера «Mimiq Camera» пропадёт из списка камер во всех приложениях. "
                              "Вернуть её можно кнопкой «Установить».",
                              QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes, self)
            box.button(QMessageBox.StandardButton.Yes).setText("Удалить")
            box.button(QMessageBox.StandardButton.Yes).setObjectName("danger")
            box.button(QMessageBox.StandardButton.Cancel).setText("Отмена")
            if box.exec() != QMessageBox.StandardButton.Yes:
                return
        self.settings_panel.set_mcam_state(True, action == "uninstall",
                                           "Подтвердите запрос Windows о правах администратора…")
        if self.engine.vcam.backend == "mimiq":
            self.engine.restart_output()       # release the camera before it is re-registered

        def job():
            try:
                ok, msg = vcam_setup.run(action)
            except Exception as exc:
                log.exception("Mimiq Camera %s failed", action)
                ok, msg = False, str(exc)
            st = vcam_setup.status()
            st.update(result=msg, ok=ok, action=action)
            self._bridge.mcam.emit(st)
        threading.Thread(target=job, daemon=True, name="mimiq-mcam").start()

    def _on_mcam(self, st: dict) -> None:
        other = st.get("name") if st.get("other") else ""
        self.settings_panel.set_mcam_state(bool(st.get("supported")), bool(st.get("installed")), other=other or "")
        if "result" not in st:
            return
        if st.get("ok"):
            install = st.get("action") == "install"
            extra = (" Перезапустите браузер и приложения для звонков (Zoom, Teams, Discord, Telegram) — "
                     "только после перезапуска они увидят новую камеру. Chrome и Edge работают в фоне: "
                     "откройте chrome://restart или edge://restart." if install else "")
            self.toasts.show("ok", st["result"] + extra, 14000 if install else 7000)
            self.engine.restart_output()
        else:
            self.toasts.show("error", st["result"], 8000)

    # ================================================================== updates
    def check_updates(self, manual: bool = False) -> None:
        if self._update_busy:
            return
        self._update_busy = True
        if manual:
            self.settings_panel.set_update_status("Проверяю…", busy=True)

        def job():
            rel, err = updates.check()
            self._bridge.update.emit(rel, err, manual)
        threading.Thread(target=job, daemon=True, name="mimiq-update").start()

    def _on_update(self, rel: Optional[updates.Release], err: str, manual: bool) -> None:
        self._update_busy = False
        if rel is None:
            text = err or f"У вас последняя версия — {__version__}."
            self.settings_panel.set_update_status(text if manual or not err else "")
            if manual:
                self.toasts.show("error" if err else "ok", text, 5000)
            return
        first = self._release is None or self._release.version != rel.version
        self._release = rel
        self.update_chip.set(f"Доступна {rel.version}", theme.OK, "Новая версия Mimiq — нажмите, чтобы узнать, что нового")
        self.update_chip.show()
        self.settings_panel.set_update_status(f"Доступна версия {rel.version}.")
        if manual:
            self._show_update()
        elif first:
            self.toasts.show("info", f"Вышла Mimiq {rel.version}. Нажмите «Доступна {rel.version}» вверху, "
                                     f"чтобы узнать, что нового.", 9000)

    def _show_update(self) -> None:
        rel = self._release
        if rel is None:
            return
        text = f"Вышла Mimiq {rel.version} (у вас {__version__})."
        if rel.notes:
            text += "\n\n" + rel.notes
        text += ("\n\nКак обновиться: скачайте архив, распакуйте его поверх папки Mimiq (папки .venv и models "
                 "сохранятся) и запустите install.bat.")
        box = QMessageBox(QMessageBox.Icon.NoIcon, "Доступно обновление", text,
                          QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes, self)
        box.button(QMessageBox.StandardButton.Yes).setText("Скачать")
        box.button(QMessageBox.StandardButton.Yes).setObjectName("primary")
        box.button(QMessageBox.StandardButton.Cancel).setText("Позже")
        details = box.addButton("Страница релиза", QMessageBox.ButtonRole.ActionRole)
        details.setObjectName("ghost")
        res = box.exec()
        if box.clickedButton() is details:
            QDesktopServices.openUrl(QUrl(rel.url))
        elif res == QMessageBox.StandardButton.Yes:
            QDesktopServices.openUrl(QUrl(rel.download))

    # ================================================================== misc
    def refresh_cameras(self) -> None:
        def job():
            try:
                from ..io.sources import list_cameras
                cams = list_cameras()
            except Exception as exc:
                log.warning("camera enumeration failed: %s", exc)
                cams = []
            self._bridge.cameras.emit(cams)
        threading.Thread(target=job, daemon=True, name="mimiq-cams").start()

    def show_connect(self) -> None:
        if self.settings.source_kind != "link":
            self.update_settings({"source_kind": "link"})
        if self.engine.link is None:
            self.engine.start_link()
        if self._connect_dialog is None:
            self._connect_dialog = ConnectDialog(self.engine, self)
            self._connect_dialog.hostChanged.connect(lambda ip: self.update_settings({"link_host": ip}))
            self._connect_dialog.finished.connect(lambda *_: setattr(self, "_connect_dialog", None))
        self._connect_dialog.show()
        self._connect_dialog.raise_()
        self._connect_dialog.activateWindow()

    def _open_folder(self, key: str) -> None:
        folder = {"captures": paths.captures_dir, "models": paths.models_dir, "logs": paths.logs_dir}[key]()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _render_brand_frames(self) -> None:
        s = self.settings
        try:
            self.engine.watermark = brand.watermark_bgra(s.output_height)
            self.engine.placeholder = brand.placeholder_bgr(s.output_width, s.output_height)
        except Exception as exc:
            log.warning("brand frames failed: %s", exc)

    def toggle_fullscreen(self) -> None:
        full = not self.isFullScreen()
        for w in (self.header, self.faces, self.settings_panel, self.status, self.controls):
            w.setVisible(not full)
        lay = self.centralWidget().layout()
        if full:
            self._was_maximized = self.isMaximized()
            lay.setContentsMargins(0, 0, 0, 0)
            self.showFullScreen()
        else:
            lay.setContentsMargins(16, 10, 16, 6)
            self.showMaximized() if self._was_maximized else self.showNormal()

    def _escape(self) -> None:
        if self.isFullScreen():
            self.toggle_fullscreen()

    def resizeEvent(self, e):  # noqa: N802
        w = e.size().width()
        narrow = w < 1500
        self.faces.setFixedWidth(256 if narrow else 288)
        self.settings_panel.setFixedWidth(340 if narrow else 368)
        center = w - self.faces.width() - self.settings_panel.width() - 56
        roomy = center >= 930
        self.swap_label.setVisible(roomy)
        self.vcam_label.setVisible(roomy)
        super().resizeEvent(e)

    def closeEvent(self, e):  # noqa: N802
        self._save_timer.stop()
        self._save()
        try:
            self.engine.shutdown()
        except Exception:
            log.exception("shutdown failed")
        super().closeEvent(e)
