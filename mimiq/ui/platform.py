"""Windows integration: dark title bars that match the app, taskbar identity, GPU name lookup."""
from __future__ import annotations

import subprocess
import sys

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QWidget

APP_ID = "Mimiq.FaceStudio.1"


def set_app_user_model_id() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass


def apply_windows_chrome(widget: QWidget) -> None:
    """Dark title bar (Windows 10 20H1+/11) and caption colour equal to the app background (Windows 11)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        dwm = ctypes.windll.dwmapi
        on = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (new / old id)
            if dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
        caption = ctypes.c_int(0x00131111)   # COLORREF 0x00BBGGRR for #111113
        dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption))
        text = ctypes.c_int(0x00EEECEC)
        dwm.DwmSetWindowAttribute(hwnd, 36, ctypes.byref(text), ctypes.sizeof(text))
    except Exception:
        pass


class ChromeFilter(QObject):
    """Applies the dark window chrome to every top-level window (dialogs, message boxes) when first shown."""

    def eventFilter(self, obj, ev):  # noqa: N802
        if ev.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow() \
                and not obj.property("mimiqChrome"):
            obj.setProperty("mimiqChrome", True)
            apply_windows_chrome(obj)
        return False


def gpu_name() -> str:
    """Best-effort NVIDIA GPU name (empty string if unavailable)."""
    try:
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=4, creationflags=flags)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].replace("NVIDIA ", "").replace("GeForce ", "")
    except Exception:
        pass
    return ""
