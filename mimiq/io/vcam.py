"""Virtual camera output (OBS Virtual Camera / Unity Capture) via pyvirtualcam."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("mimiq.vcam")


def _friendly(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "not installed" in low or "obs" in low and "not" in low and "found" in low:
        return ("Виртуальная камера OBS не найдена. Установите OBS Studio 28+, один раз нажмите "
                "«Запустить виртуальную камеру», остановите её и закройте OBS.")
    if "in use" in low or "busy" in low or "already" in low:
        return "Виртуальная камера уже занята (закройте OBS или другое приложение, которое её использует)."
    if "no backend" in low or "v4l2" in low:
        return "Виртуальная камера недоступна на этой системе."
    return f"Виртуальная камера: {text}"


class VirtualCamera:
    def __init__(self):
        self.cam = None
        self.device = ""
        self.error: Optional[str] = None
        self.size = (0, 0)
        self.fps = 30

    @property
    def active(self) -> bool:
        return self.cam is not None

    def open(self, width: int, height: int, fps: int, backend: str = "auto") -> bool:
        self.close()
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat
        except Exception as exc:
            self.error = f"pyvirtualcam не установлен: {exc}"
            return False
        kw = dict(width=int(width), height=int(height), fps=float(fps), fmt=PixelFormat.BGR)
        if backend and backend != "auto":
            kw["backend"] = backend
        try:
            self.cam = pyvirtualcam.Camera(**kw)
        except Exception as exc:
            self.cam = None
            self.error = _friendly(exc)
            log.warning("virtual camera unavailable: %s", exc)
            return False
        self.device = getattr(self.cam, "device", "Virtual Camera")
        self.size, self.fps, self.error = (width, height), fps, None
        log.info("virtual camera started: %s %dx%d@%d", self.device, width, height, fps)
        return True

    def send(self, frame: np.ndarray) -> None:
        if self.cam is not None:
            self.cam.send(np.ascontiguousarray(frame))

    def wait(self) -> None:
        if self.cam is not None:
            self.cam.sleep_until_next_frame()

    def close(self) -> None:
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass
        self.cam = None
