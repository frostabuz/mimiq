"""Virtual camera output via pyvirtualcam: Mimiq Camera (own, Unity Capture filter) or OBS Virtual Camera."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import vcam_setup

log = logging.getLogger("mimiq.vcam")

NO_CAMERA = ("Камера «Mimiq Camera» не установлена: «Вывод» → «Виртуальная камера» → «Установить» "
             "(один раз, нужны права администратора).")


def _friendly(exc: Exception, backend: str) -> str:
    text = str(exc)
    low = text.lower()
    if backend == "mimiq" and ("no camera registered" in low or "not installed" in low):
        return NO_CAMERA
    if "not installed" in low or "obs" in low and "not" in low and "found" in low:
        return ("Виртуальная камера OBS не найдена. Установите OBS Studio 28+, один раз нажмите "
                "«Запустить виртуальную камеру», остановите её и закройте OBS — или выберите драйвер "
                "«Mimiq Camera».")
    if "in use" in low or "busy" in low or "already" in low:
        return "Виртуальная камера уже занята (закройте другое приложение, которое в неё транслирует)."
    if "no backend" in low or "v4l2" in low:
        return "Виртуальная камера недоступна на этой системе."
    return f"Виртуальная камера: {text}"


def resolve_backend(choice: str) -> str:
    """auto → Mimiq Camera when it is installed, otherwise OBS Virtual Camera."""
    if choice == "unitycapture":          # settings from Mimiq ≤ 1.1
        choice = "mimiq"
    if choice in ("mimiq", "obs"):
        return choice
    if vcam_setup.is_installed():
        return "mimiq"
    return "obs" if vcam_setup.supported() else "auto"


class VirtualCamera:
    def __init__(self):
        self.cam = None
        self.device = ""
        self.backend = ""
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
        kind = resolve_backend(backend)
        kw = dict(width=int(width), height=int(height), fps=float(fps), fmt=PixelFormat.BGR)
        if kind == "mimiq":
            if not vcam_setup.is_installed():
                self.error = NO_CAMERA
                return False
            kw.update(backend="unitycapture", device=vcam_setup.CAMERA_NAME)
        elif kind == "obs":
            kw["backend"] = "obs"
        try:
            self.cam = pyvirtualcam.Camera(**kw)
        except Exception as exc:
            self.cam = None
            self.error = _friendly(exc, kind)
            low = str(exc).lower()
            obs_missing = "not installed" in low or ("not" in low and "found" in low)
            if kind == "obs" and backend == "auto" and vcam_setup.supported() and obs_missing:
                self.error = NO_CAMERA
            log.warning("virtual camera unavailable: %s", exc)
            return False
        self.backend = kind
        self.device = vcam_setup.CAMERA_NAME if kind == "mimiq" else getattr(self.cam, "device", "Virtual Camera")
        self.size, self.fps, self.error = (width, height), fps, None
        log.info("virtual camera started: %s %dx%d@%d", self.device, width, height, fps)
        return True

    def send(self, frame: np.ndarray) -> None:
        if self.cam is not None:
            self.cam.send(np.ascontiguousarray(frame))

    def wait(self) -> None:
        if self.cam is not None:
            self.cam.sleep_until_next_frame()

    def close(self, farewell: Optional[np.ndarray] = None) -> None:
        """Stop the camera. Mimiq Camera keeps showing the last frame it got, so `farewell` (the pause card)
        is sent first — otherwise apps would keep showing a frozen face."""
        if self.cam is not None:
            if farewell is not None and self.backend == "mimiq":
                try:
                    w, h = self.size
                    if farewell.shape[1] == w and farewell.shape[0] == h:
                        self.cam.send(np.ascontiguousarray(farewell))
                except Exception:
                    pass
            try:
                self.cam.close()
            except Exception:
                pass
        self.cam = None
        self.backend = ""
