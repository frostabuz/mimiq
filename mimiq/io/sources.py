"""Video inputs: local cameras (incl. Camo / iVCam / Iriun / DroidCam drivers),
network streams (RTSP / HTTP MJPEG) and the built-in Mimiq Link (iPhone Safari)."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("mimiq.sources")

# names of virtual *output* devices that must never be used as input (feedback loop)
OUTPUT_DEVICE_HINTS = ("obs virtual camera", "unity video capture", "mimiq")
PHONE_HINTS = ("iphone", "camo", "ivcam", "e2esoft", "iriun", "droidcam", "epoccam", "continuity", "reincubate")


@dataclass
class CameraDevice:
    index: int
    name: str

    @property
    def is_phone(self) -> bool:
        n = self.name.lower()
        return any(h in n for h in PHONE_HINTS)


def list_cameras(max_probe: int = 6) -> List[CameraDevice]:
    names: List[str] = []
    if sys.platform == "win32":
        try:
            try:  # camera enumeration runs on a worker thread, which needs its own COM apartment
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass
            from pygrabber.dshow_graph import FilterGraph  # DirectShow order == OpenCV CAP_DSHOW order
            names = list(FilterGraph().get_input_devices())
        except Exception as exc:
            log.info("pygrabber unavailable (%s), probing indices", exc)
    if names:
        devs = [CameraDevice(i, n) for i, n in enumerate(names)]
    else:
        devs = []
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        for i in range(max_probe):
            cap = cv2.VideoCapture(i, api)
            ok = cap.isOpened()
            cap.release()
            if ok:
                devs.append(CameraDevice(i, f"Камера {i + 1}"))
    return [d for d in devs if not any(h in d.name.lower() for h in OUTPUT_DEVICE_HINTS)]


class FrameSource:
    """Base class: a background producer that always exposes the most recent frame."""

    kind = "base"

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._ts = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.status = "idle"            # idle | connecting | live | error
        self.message = ""
        self.info: Dict[str, object] = {}
        self.fps = 0.0
        self._fps_t = time.monotonic()
        self._fps_n = 0
        self.on_status: Optional[Callable[[str, str], None]] = None

    # -- producer side ------------------------------------------------------
    def _publish(self, frame: np.ndarray, ts: Optional[float] = None) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._ts = time.monotonic() if ts is None else ts
            self._cond.notify_all()
        self._fps_n += 1
        now = time.monotonic()
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0

    def _set_status(self, status: str, message: str = "") -> None:
        if status != self.status or message != self.message:
            self.status, self.message = status, message
            if self.on_status:
                try:
                    self.on_status(status, message)
                except Exception:
                    pass

    # -- consumer side ------------------------------------------------------
    def wait_frame(self, last_seq: int, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], int, float]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._frame, self._seq, self._ts

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_safe, name=f"src-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None
        self._set_status("idle")

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as exc:  # pragma: no cover
            log.exception("source crashed")
            self._set_status("error", str(exc))

    def _run(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def description(self) -> str:
        return self.kind


class DeviceSource(FrameSource):
    kind = "device"

    def __init__(self, index: int, name: str = "", width: int = 1280, height: int = 720, fps: int = 30,
                 backend: str = "dshow"):
        super().__init__()
        self.index, self.name = index, name or f"Камера {index + 1}"
        self.req = (width, height, fps)
        self.backend = backend

    @property
    def description(self) -> str:
        return self.name

    def _open(self):
        apis = []
        if sys.platform == "win32":
            if self.backend in ("dshow", "auto"):
                apis.append(cv2.CAP_DSHOW)
            if self.backend in ("msmf", "auto") or not apis:
                apis.append(cv2.CAP_MSMF)
        apis.append(cv2.CAP_ANY)
        for api in apis:
            cap = cv2.VideoCapture(self.index, api)
            if not cap.isOpened():
                cap.release()
                continue
            w, h, fps = self.req
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
            cap.release()
        return None, None

    def _run(self) -> None:
        backoff = 0.5
        while self._running:
            self._set_status("connecting", f"Подключение к «{self.name}»…")
            cap, frame = self._open()
            if cap is None:
                self._set_status("error", f"Камера «{self.name}» недоступна. Она занята другим приложением?")
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
                continue
            backoff = 0.5
            self.info = {"width": frame.shape[1], "height": frame.shape[0]}
            self._set_status("live", self.name)
            self._publish(frame)
            fails = 0
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails > 30:
                        break
                    time.sleep(0.01)
                    continue
                fails = 0
                self._publish(frame)
            cap.release()
            if self._running:
                self._set_status("connecting", "Камера отключилась, переподключение…")
                time.sleep(0.5)


class StreamSource(FrameSource):
    """RTSP / HTTP(S) MJPEG / any FFmpeg URL, e.g. DroidCam: http://PHONE_IP:4747/video"""

    kind = "url"

    def __init__(self, url: str):
        super().__init__()
        self.url = url.strip()

    @property
    def description(self) -> str:
        return self.url

    def _run(self) -> None:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                              "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0")
        while self._running:
            self._set_status("connecting", "Подключение к потоку…")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                self._set_status("error", "Поток недоступен. Проверьте адрес и Wi-Fi.")
                time.sleep(2.0)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._set_status("live", self.url)
            fails = 0
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails > 50:
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                self.info = {"width": frame.shape[1], "height": frame.shape[0]}
                self._publish(frame)
            cap.release()


class LinkSource(FrameSource):
    """Frames pushed by the Mimiq Link server (JPEG from iPhone Safari). Decoding is lazy:
    only the newest JPEG is decoded, on the processing thread, so nothing queues up."""

    kind = "link"

    def __init__(self):
        super().__init__()
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
        self._decoded_seq = 0
        self._decoded: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return str(self.info.get("device") or "iPhone · Mimiq Link")

    def push_jpeg(self, data: bytes) -> None:
        with self._cond:
            self._jpeg = data
            self._jpeg_seq += 1
            self._seq = self._jpeg_seq
            self._ts = time.monotonic()
            self._cond.notify_all()
        self._fps_n += 1
        now = time.monotonic()
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0

    def wait_frame(self, last_seq: int, timeout: float = 0.5):
        with self._cond:
            if self._jpeg_seq == last_seq:
                self._cond.wait(timeout)
            data, seq, ts = self._jpeg, self._jpeg_seq, self._ts
        if data is None:
            return None, seq, ts
        if seq != self._decoded_seq:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self._decoded, self._decoded_seq = img, seq
                self.info.update(width=img.shape[1], height=img.shape[0])
        return self._decoded, self._decoded_seq, ts

    def connected(self, info: Dict[str, object]) -> None:
        self.info.update(info)
        self._set_status("live", self.description)

    def disconnected(self) -> None:
        self._set_status("connecting", "Ожидание iPhone…")

    def start(self) -> None:
        self._running = True
        if self.status != "live":
            self._set_status("connecting", "Ожидание iPhone…")

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._set_status("idle")


def orient(frame: np.ndarray, rotate: int, mirror: bool) -> np.ndarray:
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror:
        frame = cv2.flip(frame, 1)
    return frame


def fit_frame(frame: np.ndarray, width: int, height: int, mode: str = "crop") -> np.ndarray:
    """Resize to the output resolution: `crop` fills (centre-crop), `fit` letterboxes."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    if mode == "fit":
        s = min(width / w, height / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out = np.zeros((height, width, 3), np.uint8)
        x, y = (width - nw) // 2, (height - nh) // 2
        out[y:y + nh, x:x + nw] = img
        return out
    s = max(width / w, height / h)
    nw, nh = max(width, int(round(w * s))), max(height, int(round(h * s)))
    img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x, y = (nw - width) // 2, (nh - height) // 2
    return img[y:y + height, x:x + width]
