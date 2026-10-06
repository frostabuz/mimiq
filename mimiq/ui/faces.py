"""Face library panel (left column) and the "new face" dialog."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PySide6.QtCore import QObject, QPoint, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ..core.identity import Identity, IdentityLibrary
from ..imaging import IMAGE_EXTS, read_image, to_qimage
from . import icons, theme
from .widgets import POINTER, IconButton, Spinner, circle_pixmap, label

FILE_FILTER = "Фото (*.jpg *.jpeg *.png *.webp *.bmp *.heic *.heif *.tif *.tiff)"


def expand_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(str(f) for f in sorted(path.iterdir()) if f.suffix.lower() in IMAGE_EXTS)
        elif path.suffix.lower() in IMAGE_EXTS:
            out.append(str(path))
    return list(dict.fromkeys(out))


def pick_photos(parent: QWidget, title: str = "Выберите фото лица") -> List[str]:
    start = str(Path.home() / "Pictures") if (Path.home() / "Pictures").exists() else str(Path.home())
    files, _ = QFileDialog.getOpenFileNames(parent, title, start, FILE_FILTER)
    return files


# ====================================================================== drop zone
class DropZone(QFrame):
    filesDropped = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(POINTER)
        self.setMinimumHeight(96)
        self._hover = False
        self._drag = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(4)
        badge = QLabel()
        badge.setFixedSize(22, 22)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(icons.pixmap("upload", theme.DIM, 20, 1.8))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(badge)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addSpacing(2)
        t = label("Добавить лицо")
        t.setStyleSheet("font-weight: 600;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        s = label("Перетащите фото сюда или нажмите", "faint", wrap=True)
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(s)
        for w in (badge, t, s):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        active = self._drag or self._hover
        p.setBrush(QColor(255, 255, 255, 10 if active else 0))
        if self._drag:
            pen = QPen(QColor(theme.ACCENT), 1.5)
        else:
            pen = QPen(QColor("#505057" if active else "#3A3A40"), 1.2)
            pen.setDashPattern([4, 3])
        p.setPen(pen)
        p.drawRoundedRect(r, 6, 6)

    def enterEvent(self, e):  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            files = pick_photos(self)
            if files:
                self.filesDropped.emit(files)

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._drag = True
            self.update()

    def dragLeaveEvent(self, e):  # noqa: N802
        self._drag = False
        self.update()

    def dropEvent(self, e):  # noqa: N802
        self._drag = False
        self.update()
        files = expand_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
        if files:
            self.filesDropped.emit(files)


# ====================================================================== face card
class FaceCard(QFrame):
    clicked = Signal(str)
    menuRequested = Signal(str, QPoint)

    def __init__(self, ident: Identity, active: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ident = ident
        self._active = active
        self._hover = False
        self.setCursor(POINTER)
        self.setFixedHeight(60)
        self._thumb = to_qimage(ident.thumbnail)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 7, 4, 7)
        lay.setSpacing(10)
        self.avatar = QLabel()
        self.avatar.setFixedSize(44, 44)
        lay.addWidget(self.avatar)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addStretch(1)
        self.name = label(ident.name)
        self.name.setStyleSheet("font-weight: 600;")
        col.addWidget(self.name)
        q = int(round(ident.quality * 100))
        self.meta = label(f"{ident.photos} фото  ·  качество {q}%", "faint")
        col.addWidget(self.meta)
        col.addStretch(1)
        lay.addLayout(col, 1)
        self.more = IconButton("dots", "Действия", size=30, icon_size=18, flat=True, color=theme.DIM)
        self.more.clicked.connect(lambda: self.menuRequested.emit(
            self.ident.id, self.more.mapToGlobal(QPoint(0, self.more.height()))))
        lay.addWidget(self.more)
        for w in (self.avatar, self.name, self.meta):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._render_avatar()

    def _render_avatar(self) -> None:
        self.avatar.setPixmap(circle_pixmap(self._thumb, 44))

    def set_active(self, on: bool) -> None:
        if on != self._active:
            self._active = on
            self._render_avatar()
            self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._active:
            p.setBrush(QColor(theme.CARD_HI))
            p.setPen(QPen(QColor(theme.ACCENT), 1.2))
        else:
            p.setBrush(QColor(theme.CARD_HI if self._hover else theme.CARD))
            p.setPen(QPen(QColor(theme.LINE_HI if self._hover else theme.LINE), 1))
        p.drawRoundedRect(r, 6, 6)

    def enterEvent(self, e):  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.ident.id)
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, e):  # noqa: N802
        self.menuRequested.emit(self.ident.id, e.globalPos())


# ====================================================================== panel
class FacesPanel(QFrame):
    faceSelected = Signal(object)        # Identity | None
    filesChosen = Signal(list)           # create a new identity from these files
    addPhotosRequested = Signal(object)  # Identity
    faceDeleted = Signal(str)
    faceRenamed = Signal(object)

    def __init__(self, library: IdentityLibrary, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.library = library
        self.setObjectName("sidePanel")
        self.setAcceptDrops(True)
        self.cards: Dict[str, FaceCard] = {}
        self.active_id = ""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 12)
        outer.setSpacing(12)
        head = QHBoxLayout()
        head.addWidget(label("Лица", "h2"), 1)
        add = IconButton("plus", "Добавить лицо из фото", size=28, icon_size=16, flat=True)
        add.clicked.connect(self._pick)
        head.addWidget(add, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(head)
        self.drop = DropZone()
        self.drop.filesDropped.connect(self.filesChosen.emit)
        outer.addWidget(self.drop)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        self.list = QVBoxLayout(holder)
        self.list.setContentsMargins(0, 0, 2, 0)
        self.list.setSpacing(6)
        self.list.addStretch(1)
        self.scroll.setWidget(holder)
        outer.addWidget(self.scroll, 1)
        self.empty = label("Пока нет лиц. Добавьте 1–10 чётких фото одного человека — Mimiq соберёт из них "
                           "цифровой профиль лица.", "faint", wrap=True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setContentsMargins(8, 0, 8, 0)
        self.list.insertWidget(0, self.empty)
        tip = QFrame()
        tip.setObjectName("tip")
        tip.setStyleSheet(f"#tip {{ background: transparent; border-top: 1px solid {theme.LINE}; }}")
        tl = QHBoxLayout(tip)
        tl.setContentsMargins(0, 10, 0, 0)
        tl.setSpacing(8)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("shield", theme.DIM, 16))
        ic.setFixedSize(16, 16)
        tl.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
        tt = label("Используйте только лица людей, которые дали на это согласие.", wrap=True)
        tt.setStyleSheet(f"color: {theme.DIM}; font-size: 9pt;")
        tl.addWidget(tt, 1)
        outer.addWidget(tip)

    # ------------------------------------------------------------------
    def _pick(self) -> None:
        files = pick_photos(self)
        if files:
            self.filesChosen.emit(files)

    def reload(self, active_id: Optional[str] = None) -> None:
        if active_id is not None:
            self.active_id = active_id
        for card in self.cards.values():
            card.setParent(None)
            card.deleteLater()
        self.cards.clear()
        idents = self.library.list()
        for ident in idents:
            card = FaceCard(ident, ident.id == self.active_id)
            card.clicked.connect(self._select)
            card.menuRequested.connect(self._menu)
            self.list.insertWidget(self.list.count() - 1, card)
            self.cards[ident.id] = card
        self.empty.setVisible(not idents)

    def set_active(self, ident_id: str) -> None:
        self.active_id = ident_id
        for k, card in self.cards.items():
            card.set_active(k == ident_id)

    def _select(self, ident_id: str) -> None:
        card = self.cards.get(ident_id)
        if card is None:
            return
        self.set_active(ident_id)
        self.faceSelected.emit(card.ident)

    def _menu(self, ident_id: str, pos: QPoint) -> None:
        card = self.cards.get(ident_id)
        if card is None:
            return
        ident = card.ident
        m = QMenu(self)
        a_use = m.addAction(icons.icon("check", theme.TEXT, 16), "Использовать")
        a_add = m.addAction(icons.icon("image", theme.TEXT, 16), "Добавить фото…")
        a_ren = m.addAction(icons.icon("edit", theme.TEXT, 16), "Переименовать…")
        a_dir = m.addAction(icons.icon("folder", theme.TEXT, 16), "Открыть папку")
        m.addSeparator()
        a_del = m.addAction(icons.icon("trash", theme.BAD, 16), "Удалить")
        act = m.exec(pos)
        if act is a_use:
            self._select(ident_id)
        elif act is a_add:
            self.addPhotosRequested.emit(ident)
        elif act is a_ren:
            name, ok = QInputDialog.getText(self, "Переименовать", "Имя лица:", QLineEdit.EchoMode.Normal, ident.name)
            if ok and name.strip():
                self.library.rename(ident, name.strip())
                self.reload()
                self.faceRenamed.emit(ident)
        elif act is a_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.library.root / ident.id)))
        elif act is a_del:
            box = QMessageBox(QMessageBox.Icon.NoIcon, "Удалить лицо",
                              f"Удалить «{ident.name}»? Исходные фото не затрагиваются.",
                              QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes, self)
            box.button(QMessageBox.StandardButton.Yes).setText("Удалить")
            box.button(QMessageBox.StandardButton.Yes).setObjectName("danger")
            box.button(QMessageBox.StandardButton.Cancel).setText("Отмена")
            if box.exec() == QMessageBox.StandardButton.Yes:
                self.library.delete(ident.id)
                if self.active_id == ident.id:
                    self.active_id = ""
                    self.faceSelected.emit(None)
                self.reload()
                self.faceDeleted.emit(ident.id)

    # whole panel accepts drops too
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        files = expand_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
        if files:
            self.filesChosen.emit(files)


# ====================================================================== new face dialog
class _ThumbLoader(QObject):
    loaded = Signal(int, QImage)

    def start(self, files: List[str], size: int) -> None:
        def run():
            for i, f in enumerate(files):
                try:
                    img = read_image(f)
                    h, w = img.shape[:2]
                    s = size / min(h, w)
                    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                    self.loaded.emit(i, to_qimage(img))
                except Exception:
                    self.loaded.emit(i, QImage())
        threading.Thread(target=run, daemon=True, name="mimiq-thumbs").start()


class PhotoTile(QWidget):
    def __init__(self, name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = name
        self.setFixedSize(84, 84)
        self.image: Optional[QImage] = None
        self.state = "pending"   # pending | ok | bad
        self.setToolTip(name)

    def set_image(self, img: QImage) -> None:
        self.image = img if img is not None and not img.isNull() else None
        if self.image is None:
            self.state = "bad"
            self.setToolTip(f"{self.name}\nНе удалось открыть файл")
        self.update()

    def set_state(self, ok: bool, message: str) -> None:
        self.state = "ok" if ok else "bad"
        self.setToolTip(f"{self.name}\n{message}")
        self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(r, 6, 6)
        p.setClipPath(path)
        p.fillRect(r, QColor(theme.FIELD))
        if self.image is not None:
            img = self.image
            side = min(img.width(), img.height())
            src = QRectF((img.width() - side) / 2, (img.height() - side) / 2, side, side)
            p.drawImage(r, img, src)
            if self.state == "bad":
                p.fillRect(r, QColor(17, 17, 19, 150))
        p.setClipping(False)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(theme.BAD if self.state == "bad" else theme.LINE_HI), 1.2))
        p.drawRoundedRect(r, 6, 6)
        if self.state in ("ok", "bad"):
            c = QRectF(r.right() - 24, r.bottom() - 24, 20, 20)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.OK if self.state == "ok" else theme.BAD))
            p.drawEllipse(c)
            p.drawPixmap(c.adjusted(4, 4, -4, -4).toRect(), icons.pixmap("check" if self.state == "ok" else "close",
                                                                           "#111113", 12, 3.0))


class NewFaceDialog(QDialog):
    """Collects a name, shows the photos and builds the identity on the engine thread."""

    def __init__(self, engine, files: List[str], base: Optional[Identity] = None, default_name: str = "Лицо",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = engine
        self.files = files
        self.base = base
        self.result_identity: Optional[Identity] = None
        self._busy = False
        self.setWindowTitle("Добавить фото" if base else "Новое лицо")
        self.setModal(True)
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 24, 26, 22)
        lay.setSpacing(16)
        title = label(f"Добавить фото · {base.name}" if base else "Новое лицо", "h1")
        lay.addWidget(title)
        sub = label("Mimiq найдёт лицо на каждом фото и объединит их в один профиль. Чем больше разных "
                    "ракурсов и освещения — тем точнее сходство.", "dim", wrap=True)
        lay.addWidget(sub)
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setPlaceholderText("Имя, например «Алекс»")
        self.name_edit.setMaxLength(40)
        if base is None:
            lay.addWidget(label("Имя", "dim"))
            lay.addWidget(self.name_edit)
            lay.setSpacing(16)
        grid_box = QFrame()
        grid_box.setObjectName("box")
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setSpacing(10)
        self.tiles: Dict[str, List[PhotoTile]] = {}
        self._tile_list: List[PhotoTile] = []
        cols = 5
        shown = files[:15]
        for i, f in enumerate(shown):
            tile = PhotoTile(Path(f).name)
            self._tile_list.append(tile)
            self.tiles.setdefault(Path(f).name, []).append(tile)
            grid.addWidget(tile, i // cols, i % cols)
        if len(files) > len(shown):
            grid.addWidget(label(f"+ ещё {len(files) - len(shown)}", "dim"), len(shown) // cols, len(shown) % cols)
        lay.addWidget(grid_box)
        tips = label("Совет: 3–10 фото анфас и в лёгком повороте, лицо крупно, без сильных теней, "
                     "тёмных очков и фильтров. Поддерживаются JPG, PNG, HEIC.", "faint", wrap=True)
        lay.addWidget(tips)
        status = QHBoxLayout()
        status.setSpacing(10)
        self.spinner = Spinner(20, 2.5)
        self.spinner.hide()
        self.status = label("", "dim", wrap=True)
        status.addWidget(self.spinner)
        status.addWidget(self.status, 1)
        lay.addLayout(status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel = QPushButton("Отмена")
        self.cancel.setObjectName("ghost")
        self.cancel.setCursor(POINTER)
        self.cancel.clicked.connect(self.reject)
        self.ok = QPushButton("  Добавить фото" if base else "  Создать лицо")
        self.ok.setIcon(icons.icon("check", "#FFFFFF", 16))
        self.ok.setObjectName("primary")
        self.ok.setCursor(POINTER)
        self.ok.setDefault(True)
        self.ok.clicked.connect(self._build)
        buttons.addWidget(self.cancel)
        buttons.addWidget(self.ok)
        lay.addLayout(buttons)
        self.adjustSize()
        self._loader = _ThumbLoader(self)
        self._loader.loaded.connect(self._on_thumb)
        self._loader.start(shown, 168)
        engine.photoReport.connect(self._on_report)
        engine.identityBuilt.connect(self._on_built)
        engine.identityFailed.connect(self._on_failed)
        self.finished.connect(self._disconnect)

    def _disconnect(self, *_):
        for sig, slot in ((self.engine.photoReport, self._on_report), (self.engine.identityBuilt, self._on_built),
                          (self.engine.identityFailed, self._on_failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _on_thumb(self, i: int, img: QImage) -> None:
        if 0 <= i < len(self._tile_list):
            self._tile_list[i].set_image(img)

    def _build(self) -> None:
        if self._busy:
            return
        name = self.name_edit.text().strip() or "Лицо"
        self._busy = True
        self.ok.setEnabled(False)
        self.name_edit.setEnabled(False)
        self.spinner.show()
        self.status.setStyleSheet("")
        self.status.setText("Анализ фото…" if self.engine.models_ready else "Ждём загрузки нейросетей, затем анализ фото…")
        self.engine.build_identity(self.files, name, self.base)

    def _on_report(self, name: str, rep) -> None:
        for tile in self.tiles.get(name, []):
            tile.set_state(bool(rep.get("ok")), str(rep.get("message", "")))

    def _on_built(self, ident) -> None:
        if not self._busy:
            return
        self.result_identity = ident
        self.spinner.hide()
        bad = sum(1 for t in self._tile_list if t.state == "bad")
        self.status.setStyleSheet(f"color: {theme.OK};")
        msg = f"Готово: «{ident.name}» · {ident.photos} фото"
        if bad:
            msg += f" (пропущено: {bad})"
        self.status.setText(msg)
        QTimer.singleShot(900 if bad else 500, self.accept)

    def _on_failed(self, message: str) -> None:
        if not self._busy:
            return
        self._busy = False
        self.spinner.hide()
        self.status.setStyleSheet(f"color: {theme.BAD};")
        self.status.setText(message)
        self.ok.setEnabled(True)
        self.name_edit.setEnabled(True)
        self.cancel.setText("Закрыть")

    def reject(self) -> None:
        if self._busy:
            return  # wait for the build to finish
        super().reject()
