"""Mimiq application bootstrap: logging, Qt setup, single instance, consent, main window."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys

from . import __version__, paths

log = logging.getLogger("mimiq")
INSTANCE_KEY = "Mimiq-SingleInstance-v1"

# Must be set before OpenCV is imported: Media Foundation cameras otherwise take ~20 s to open,
# and OpenCV's probe warnings only clutter the log.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        fh = logging.handlers.RotatingFileHandler(paths.logs_dir() / "mimiq.log", maxBytes=2_000_000, backupCount=3,
                                                  encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None:  # pythonw.exe has no console
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    def excepthook(exc_type, exc, tb):
        logging.getLogger("mimiq.crash").error("unhandled exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = excepthook


def _stylesheet() -> str:
    from .ui import icons, theme
    d = paths.cache_dir() / "ui"
    d.mkdir(parents=True, exist_ok=True)
    arrow = icons.save_png("chevron", theme.DIM, 14, str(d / "chevron.png")).replace("\\", "/")
    check = icons.save_png("check", "#FFFFFF", 14, str(d / "check.png")).replace("\\", "/")
    return theme.STYLESHEET.replace("__ARROW__", arrow).replace("__CHECK__", check)


def _another_instance_running() -> bool:
    from PySide6.QtNetwork import QLocalSocket
    sock = QLocalSocket()
    sock.connectToServer(INSTANCE_KEY)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.flush()
        sock.waitForBytesWritten(300)
        sock.disconnectFromServer()
        return True
    return False


def _listen_for_instances(window) -> object:
    from PySide6.QtNetwork import QLocalServer
    QLocalServer.removeServer(INSTANCE_KEY)
    server = QLocalServer(window)

    def on_connection():
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.disconnectFromServer()
        window.showNormal() if window.isMinimized() else None
        window.raise_()
        window.activateWindow()
    server.newConnection.connect(on_connection)
    server.listen(INSTANCE_KEY)
    return server


def main(argv=None) -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication, QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from .ui.platform import ChromeFilter, set_app_user_model_id

    set_app_user_model_id()
    _setup_logging()
    log.info("Mimiq %s starting (Python %s)", __version__, sys.version.split()[0])
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Mimiq")
    app.setOrganizationName("Mimiq")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    from .ui import theme
    app.setPalette(theme.palette())
    app.setFont(theme.ui_font(10))
    app.setWindowIcon(QIcon(str(paths.package_dir() / "assets" / "mimiq.ico")))
    app.setStyleSheet(_stylesheet())
    chrome = ChromeFilter(app)
    app.installEventFilter(chrome)

    if _another_instance_running():
        log.info("another instance is running; asked it to come to front")
        return 0

    from .config import Settings
    from .core.identity import IdentityLibrary
    from .engine import Engine
    from .ui.dialogs import ConsentDialog
    from .ui.main_window import MainWindow

    settings = Settings.load()
    if not settings.consent_accepted:
        if ConsentDialog().exec() != ConsentDialog.DialogCode.Accepted:
            return 0
        settings.consent_accepted = True
        settings.save()

    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        QMessageBox.warning(None, "Mimiq", "Не найден ONNX Runtime — нейросети не запустятся.\n\n"
                                           "Запустите install.bat ещё раз.\n\n" + str(exc))

    library = IdentityLibrary(paths.faces_dir())
    engine = Engine(settings.copy(), library)
    window = MainWindow(settings, engine, library)
    screen = app.primaryScreen().availableGeometry()
    w = min(1720, int(screen.width() * 0.94))
    h = min(990, int(screen.height() * 0.92))
    window.resize(max(w, window.minimumWidth()), max(h, window.minimumHeight()))
    window.move(screen.center() - window.rect().center())
    if screen.width() < 1500:
        window.showMaximized()
    else:
        window.show()
    server = _listen_for_instances(window)  # noqa: F841 (kept alive by parent)

    QTimer.singleShot(120, engine.boot)
    if not settings.first_run_done:
        def welcome():
            window.toasts.show("info", "Добро пожаловать! Добавьте фото лица слева и подключите iPhone через QR-код. "
                                       "При первом запуске Mimiq скачает нейросети (~700 МБ).", 9000)
            window.update_settings({"first_run_done": True})
        QTimer.singleShot(700, welcome)
    code = app.exec()
    log.info("Mimiq exited with code %s", code)
    return code
