"""Engine: capture → pipeline → virtual camera / preview / recorder, all off the UI thread.

Threads
-------
* analyse  — takes the newest camera frame, tracks faces (pipeline stage A) and publishes it right away
             when the face render can't keep up (fluid video);
* render   — swaps / enhances faces (stage B) and publishes its frame when it is still the newest one;
* output   — feeds the virtual camera and the recorder at a steady frame rate.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import paths
from .config import Settings
from .core import maskview, models
from .core.background import BackgroundCompositor, BackgroundMatter, Matte
from .core.identity import Identity, IdentityLibrary
from .core.pipeline import FrameJob, FrameResult, Pipeline
from .core.tuner import AutoTuner, rate_text
from .imaging import ascii_safe_dir, overlay_bgra, read_image, to_qimage, write_image
from .io.link import LinkServer
from .io.sources import DeviceSource, FrameSource, LinkSource, StreamSource, fit_frame, orient
from .io.vcam import VirtualCamera

log = logging.getLogger("mimiq.engine")

SOURCE_KEYS = {"source_kind", "device_index", "device_name", "device_backend", "stream_url", "capture_width",
               "capture_height", "capture_fps"}
MODEL_KEYS = {"swapper_model", "enhancer_model", "mask_occlusion", "occluder_model", "mask_region", "parser_model",
              "detector_model", "landmark_refine", "execution_provider", "gpu_device"}
OUTPUT_KEYS = {"vcam_enabled", "vcam_backend", "output_width", "output_height", "output_fps"}
LINK_KEYS = {"link_resolution", "link_fps", "link_quality", "link_camera"}
QUALITY_KEYS = {"swap_passes", "pixel_boost", "enhancer_model", "enhancer_blend", "mask_region", "landmark_refine",
                "mask_occlusion", "swapper_model", "auto_quality", "preset"}
FLUID_FACE_FPS = 15.0      # in fluid mode the face render aims for this many updates per second
PART_KEYS = ("swap", "enhance", "parser", "landmarks", "occlusion")
BG_KEYS = {"bg_mode", "bg_model"}


class Engine(QObject):
    frameReady = Signal(object)            # dict(result=QImage, original=QImage|None)
    statsUpdated = Signal(dict)
    stateChanged = Signal(str)             # idle | running
    message = Signal(str, str)             # level, text
    sourceStatus = Signal(str, str)        # status, text
    linkEvent = Signal(str, dict)
    vcamStatus = Signal(bool, str)
    modelProgress = Signal(str, float, str)  # stage, 0..1, text
    modelsReady = Signal(bool)
    identityBuilt = Signal(object)
    identityFailed = Signal(str)
    photoReport = Signal(str, object)
    recordingChanged = Signal(bool, str)

    def __init__(self, settings: Settings, library: IdentityLibrary):
        super().__init__()
        self.settings = settings
        self.library = library
        self.hub = models.SessionHub(settings.execution_provider, settings.gpu_device)
        self.pipeline = Pipeline(self.hub)
        self.models_ready = False
        self.identity: Optional[Identity] = None
        self.source: Optional[FrameSource] = None
        self.link: Optional[LinkServer] = None
        self.link_source = LinkSource()
        self.vcam = VirtualCamera()
        self.running = False
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.tuner = AutoTuner()
        self._slot: Optional[FrameJob] = None          # job waiting for the render stage
        self._slot_cond = threading.Condition()
        self._b_busy = False
        self._b_started = 0.0
        self._pub_lock = threading.Lock()
        self._pub_seq = 0
        self._seq = 0
        self._a_ms = 0.0                                # EMA per-frame cost of the analyse stage
        self._b_ms = 0.0                                # EMA per-face-frame cost of the render stage
        self._parts: Dict[str, float] = {}              # EMA of expensive extras (for the auto tuner)
        self._fluid = False
        self._face_times = deque(maxlen=40)
        self._last_tune = 0.0
        self._slow_since: Optional[float] = None
        self._hinted = False
        self.preview_size: Optional[Tuple[int, int]] = None   # physical px of the preview widget (set by UI)
        self._out_lock = threading.Lock()
        self._latest_out: Optional[np.ndarray] = None
        self._latest_ts = 0.0
        self._preview_busy = False
        self._reconfigure = threading.Event()
        self._output_restart = threading.Event()
        self._model_job: Optional[threading.Thread] = None
        self._cancel_dl = threading.Event()
        self._writer: Optional[cv2.VideoWriter] = None
        self._rec_path: Optional[Path] = None
        self._rec_tmp: Optional[Path] = None
        self._rec_lock = threading.Lock()
        self._times = deque(maxlen=60)
        self._lat = deque(maxlen=60)
        self._timings: Dict[str, deque] = {}
        self._faces: List[dict] = []
        self._last_stats = 0.0
        self._last_error = ""
        self.watermark: Optional[np.ndarray] = None
        self.placeholder: Optional[np.ndarray] = None
        self._pending_settings: Optional[Settings] = None
        self.matter: Optional[BackgroundMatter] = None     # background matting (loaded on demand)
        self.bgc = BackgroundCompositor()
        self._bg_job: Optional[threading.Thread] = None
        self._bg_on = False
        self._bg_msg = ""
        self.link_source.on_status = lambda st, msg: self.sourceStatus.emit(st, msg)

    # ================================================================== boot / models
    def boot(self) -> None:
        """Download (if needed) and load models in the background, start Mimiq Link."""
        if self.settings.source_kind == "link":
            self.start_link()
        self._load_models(self.settings.copy(), first=True)

    def _load_models(self, s: Settings, first: bool = False) -> None:
        if self._model_job and self._model_job.is_alive():
            self._cancel_dl.set()
            self._model_job.join(timeout=2)
        self._cancel_dl = threading.Event()
        cancel = self._cancel_dl
        self._model_job = threading.Thread(target=self._model_job_main, args=(s, first, cancel), daemon=True,
                                           name="mimiq-models")
        self._model_job.start()

    def _model_job_main(self, s: Settings, first: bool, cancel: threading.Event) -> None:
        try:
            need = models.missing(s.required_models())
            if need:
                total = sum(models.REGISTRY[k].size for k in need)
                base = 0

                def progress(key, done, size, speed):
                    frac = (base + done) / max(total, 1)
                    spec = models.REGISTRY[key]
                    self.modelProgress.emit("download", frac,
                                            f"Скачивание {spec.title} · {done / 1e6:.0f}/{size / 1e6:.0f} МБ · "
                                            f"{speed / 1e6:.1f} МБ/с")
                for key in need:
                    models.download(key, progress, cancel)
                    base += models.REGISTRY[key].size
            self.modelProgress.emit("load", 0.0, "Загрузка моделей на " + models.PROVIDER_LABELS.get(
                models.resolve_provider(s.execution_provider), "CPU") + "…")
            self.hub.configure(s.execution_provider, s.gpu_device)
            keys = list(dict.fromkeys(s.required_models()))
            trt = self.hub.provider == "tensorrt"
            for i, key in enumerate(keys):
                if cancel.is_set():
                    return
                title = models.REGISTRY[key].title
                build = trt and models.uses_tensorrt(key) and not models.tensorrt_cached(key)
                if build:
                    self.modelProgress.emit("load", i / len(keys),
                                            f"TensorRT: оптимизация «{title}» · один раз, 1–3 мин…")
                self.hub.get(key)
                if trt:
                    self.hub.prime(key)
                self.modelProgress.emit("load", (i + 1) / len(keys), f"Загрузка {title}…")
            if s.bg_mode != "off":
                self._make_matter(s.bg_model)
            failed = self.hub.trt_failed
            if failed:
                names = ", ".join(models.REGISTRY[k].title for k in failed)
                self.message.emit("warn", f"TensorRT не запустился для: {names} — они работают на CUDA. "
                                          "Подробности в логах.")
            self._pending_settings = s
            if self.running:
                self._reconfigure.set()
            else:
                self._apply_pipeline(s)
            if first and s.active_face:
                ident = self.library.get(s.active_face)
                if ident:
                    self.set_identity(ident)
            self.models_ready = True
            self.modelProgress.emit("done", 1.0, "Готово")
            self.modelsReady.emit(True)
        except models.DownloadCancelled:
            return
        except Exception as exc:
            log.exception("model setup failed")
            self.modelProgress.emit("error", 0.0, str(exc))
            self.message.emit("error", f"Ошибка загрузки моделей: {exc}")
            self.modelsReady.emit(False)

    # ================================================================== background
    def _make_matter(self, key: str) -> None:
        sess = self.hub.get(key)
        m = self.matter
        if m is not None and m.key == key and m.session is sess:
            return
        m = BackgroundMatter(key, sess)
        try:
            m.warmup()
        except Exception as exc:
            log.warning("background warm-up failed: %s", exc)
        self.matter = m
        log.info("background matting ready: %s", key)

    def _ensure_matter(self, s: Settings) -> None:
        """Download / load the matting model when the background is switched on (never blocks the video)."""
        if s.bg_mode == "off":
            return
        m = self.matter
        if m is not None and m.key == s.bg_model:
            return
        if self._bg_job is not None and self._bg_job.is_alive():
            return                                  # the running job re-checks the wanted model when done
        self._bg_job = threading.Thread(target=self._bg_job_main, args=(s.bg_model,), daemon=True, name="mimiq-bg")
        self._bg_job.start()

    def _bg_job_main(self, key: str) -> None:
        spec = models.REGISTRY[key]
        try:
            if not models.is_installed(key):
                def progress(_k, done, size, speed):
                    self.modelProgress.emit("download", done / max(size, 1),
                                            f"Скачивание {spec.title} · {done / 1e6:.0f}/{size / 1e6:.0f} МБ · "
                                            f"{speed / 1e6:.1f} МБ/с")
                try:
                    models.download(key, progress, self._cancel_dl)
                finally:
                    self.modelProgress.emit("done", 1.0, "Готово")
            self._make_matter(key)
        except models.DownloadCancelled:
            return
        except Exception as exc:
            log.exception("background model failed")
            self.message.emit("error", f"Не удалось подготовить замену фона: {exc}")
            return
        s = self.settings
        if s.bg_mode != "off" and s.bg_model != key:
            self._bg_job = None
            self._ensure_matter(s)

    def _matte(self, frame: np.ndarray, s: Settings) -> Optional[Matte]:
        m = self.matter
        on = s.bg_mode != "off" and m is not None
        if on and not self._bg_on and m is not None:
            m.reset()                               # fresh recurrent state after the background was off
        self._bg_on = on
        if not on:
            return None
        t0 = time.monotonic()
        try:
            mt = m.matte(frame, s.bg_stability)
        except Exception as exc:
            log.exception("background matting failed")
            self.matter = None
            self.message.emit("error", f"Замена фона остановлена: {exc}")
            return None
        self._record({"background": (time.monotonic() - t0) * 1000}, ("background",))
        return mt

    def _apply_background(self, img: np.ndarray, frame: np.ndarray, matte: Optional[Matte], s: Settings) -> np.ndarray:
        if matte is None or s.bg_mode == "off" or matte.size != (frame.shape[1], frame.shape[0]):
            return img
        try:
            out = self.bgc.apply(img, frame, matte, s.bg_mode, blur=s.bg_blur, image=s.bg_image,
                                 image_blur=s.bg_image_blur, color=s.bg_color)
        except Exception as exc:
            if str(exc) != self._bg_msg:
                self._bg_msg = str(exc)
                log.exception("background compositing failed")
                self.message.emit("error", f"Ошибка замены фона: {exc}")
            return img
        err = self.bgc.image_error if s.bg_mode == "image" else None
        if err and err != self._bg_msg:
            self.message.emit("warn", f"{err} — пока размываю фон. Выберите картинку во вкладке «Фон».")
        self._bg_msg = err or ""
        return out

    def reload_models(self) -> None:
        """Retry model download/loading (e.g. after a network error)."""
        self._load_models(self.settings.copy())

    def _apply_pipeline(self, s: Optional[Settings]) -> None:
        if s is None:
            return
        self.pipeline.configure(s)
        self._push_effective()                      # keep slider changes made while models were loading
        self.pipeline.want_mask = self.settings.preview_mode == "mask"
        self.pipeline.warmup()
        if self.hub.degraded:
            self.message.emit("warn", f"{models.PROVIDER_LABELS.get(self.hub.provider, self.hub.provider)} не запустилась — "
                                      "нейросети работают на CPU (медленно). Обновите драйвер NVIDIA и запустите "
                                      "install.bat ещё раз — подробности в README, раздел «Проблемы».")
        else:
            self.message.emit("ok", f"Модели готовы · {models.PROVIDER_LABELS.get(self.hub.effective, self.hub.effective)}")

    @property
    def provider(self) -> str:
        return self.hub.effective

    # ================================================================== settings
    def apply(self, new: Settings, changed: Sequence[str]) -> None:
        changed = set(changed)
        self.settings = new
        if changed & QUALITY_KEYS:
            self.tuner.reset()                      # an explicit choice: start from what the user asked for
        if changed & MODEL_KEYS:
            self._load_models(new.copy())
        if changed & BG_KEYS:
            self._ensure_matter(new)
        self._push_effective()
        self.pipeline.want_mask = new.preview_mode == "mask"
        if changed & SOURCE_KEYS and self.running:
            self._restart_source()
        if "source_kind" in changed:
            if new.source_kind == "link":
                self.start_link()
        if "link_port" in changed and self.link:
            self.stop_link()
            self.start_link()
        if changed & LINK_KEYS and self.link:
            self.send_link_config()
        if changed & OUTPUT_KEYS:
            self._output_restart.set()
        if "multi_face" in changed or "detector_model" in changed:
            self.pipeline.reset_tracking()

    def restart_output(self) -> None:
        """Re-open the virtual camera (e.g. after Mimiq Camera was installed)."""
        self._output_restart.set()

    def set_identity(self, ident: Optional[Identity]) -> None:
        self.identity = ident
        self.pipeline.set_identity(ident)

    # ================================================================== identities
    def build_identity(self, files: Sequence[str], name: str, base: Optional[Identity] = None) -> None:
        def job():
            try:
                deadline = time.time() + 120
                while not self.models_ready and time.time() < deadline:
                    time.sleep(0.2)
                builder = self.pipeline.builder()
                images = []
                for f in files:
                    try:
                        images.append((Path(f).name, read_image(f)))
                    except Exception as exc:
                        self.photoReport.emit(Path(f).name, {"ok": False, "message": f"Не удалось открыть: {exc}"})
                if not images:
                    raise RuntimeError("Нет подходящих изображений")

                def report(label, rep):
                    self.photoReport.emit(label, {"ok": rep.ok, "message": rep.message, "quality": rep.quality})
                if base is None:
                    ident = self.library.create(builder, images, name, report)
                else:
                    ident = self.library.add_photos(builder, base, images, report)
                self.identityBuilt.emit(ident)
            except Exception as exc:
                log.exception("identity build failed")
                self.identityFailed.emit(str(exc))
        threading.Thread(target=job, daemon=True, name="mimiq-identity").start()

    # ================================================================== link
    def start_link(self) -> None:
        if self.link and self.link.running:
            return
        self.link = LinkServer(self.settings.link_port, self.settings.link_token, self._on_link_event)
        self.link.sink = self.link_source
        threading.Thread(target=self._start_link_bg, daemon=True).start()

    def _start_link_bg(self) -> None:
        link = self.link
        if link and not link.start():
            self.message.emit("error", link.error or "Mimiq Link не запустился")

    def stop_link(self) -> None:
        if self.link:
            self.link.stop()
            self.link = None

    def _on_link_event(self, kind: str, data: Dict) -> None:
        if kind == "hello":
            self.send_link_config()
        self.linkEvent.emit(kind, data)

    def send_link_config(self) -> None:
        if self.link:
            s = self.settings
            self.link.send({"t": "config", "facing": s.link_camera, "res": s.link_resolution, "fps": s.link_fps,
                            "quality": s.link_quality})

    def link_urls(self) -> List[str]:
        if not self.link:
            return []
        urls = self.link.urls()
        pref = self.settings.link_host
        if pref:
            urls.sort(key=lambda u: 0 if f"//{pref}:" in u else 1)
        return urls

    # ================================================================== run control
    def _make_source(self) -> FrameSource:
        s = self.settings
        if s.source_kind == "device":
            src: FrameSource = DeviceSource(s.device_index, s.device_name, s.capture_width, s.capture_height,
                                            s.capture_fps, s.device_backend)
        elif s.source_kind == "url":
            src = StreamSource(s.stream_url)
        else:
            if not (self.link and self.link.running):
                self.start_link()
            src = self.link_source
        src.on_status = lambda st, msg: self.sourceStatus.emit(st, msg)
        return src

    def start(self) -> None:
        if self.running:
            return
        if self.settings.source_kind == "url" and not self.settings.stream_url.strip():
            self.message.emit("warn", "Укажите адрес потока в настройках источника")
            return
        self._stop.clear()
        self.source = self._make_source()
        self.source.start()
        self.running = True
        self.pipeline.reset_tracking()
        self.pipeline.clear_cache()
        self._reset_run_state()
        if self.matter is not None:
            self.matter.reset()
        self._threads = [threading.Thread(target=self._analyze_loop, daemon=True, name="mimiq-analyse"),
                         threading.Thread(target=self._render_loop, daemon=True, name="mimiq-render"),
                         threading.Thread(target=self._output_loop, daemon=True, name="mimiq-output")]
        for t in self._threads:
            t.start()
        self.stateChanged.emit("running")

    def _reset_run_state(self) -> None:
        with self._slot_cond:
            self._slot = None
            self._b_busy = False
        self._pub_seq = self._seq = 0
        self._a_ms = self._b_ms = 0.0
        self._parts.clear()
        self._fluid = False
        self._face_times.clear()
        self._times.clear()
        self._lat.clear()
        self._timings.clear()
        self._slow_since = None
        if self.tuner.applied:
            self.tuner.reset()
            self._push_effective()

    def stop(self) -> None:
        if not self.running:
            return
        self._stop.set()
        self.running = False
        with self._slot_cond:
            self._slot_cond.notify_all()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        if self.source is not None:
            self.source.stop()
            self.source = None
        self.stop_recording()
        with self._out_lock:
            self._latest_out = None
        self.pipeline.clear_cache()
        self.stateChanged.emit("idle")

    def _restart_source(self) -> None:
        old = self.source
        self.source = self._make_source()
        if old is not None and old is not self.source:
            old.stop()
        self.source.start()
        self.pipeline.reset_tracking()
        if self.matter is not None:
            self.matter.reset()

    def shutdown(self) -> None:
        self._cancel_dl.set()
        self.stop()
        self.stop_link()
        self.vcam.close()

    # ================================================================== auto quality
    def _push_effective(self) -> None:
        s = self.settings
        if not s.auto_quality and self.tuner.applied:
            self.tuner.reset()
        self.pipeline.update_light(self.tuner.effective(s, self._fluid) if s.auto_quality else s)
        self.pipeline.occ_interval = self.tuner.occ_interval if s.auto_quality else 1

    def _target_fps(self) -> float:
        src = self.source
        fps = src.fps if src is not None and src.fps > 1 else 30.0
        return float(np.clip(fps, 15.0, 30.0))

    def _tune(self) -> None:
        now = time.monotonic()
        if now - self._last_tune < 1.0:
            return
        self._last_tune = now
        s = self.settings
        if not s.auto_quality or not self.models_ready:
            return
        faces_live = self._faces_live(now)
        target = self._target_fps()
        budget_a = 1000.0 / target
        budget_b = 1000.0 / min(target, FLUID_FACE_FPS) if self._fluid else budget_a
        b_ms = self._b_ms if faces_live else 0.0       # nothing to render → nothing to judge
        if self.tuner.update(now, self._a_ms, b_ms, budget_a, budget_b, dict(self._parts), s, self._fluid):
            self._push_effective()
            log.info("auto quality: %s (A %.0f ms, B %.0f ms, %s)", self.tuner.labels(s) or "full",
                     self._a_ms, self._b_ms, "fluid" if self._fluid else "direct")
        self._maybe_hint(now, faces_live, target)

    def _maybe_hint(self, now: float, faces_live: bool, target: float) -> None:
        """One friendly hint per run when the PC is too slow even with every extra switched off."""
        if self._hinted or not faces_live:
            return
        face_fps = self._face_fps(now)
        slow = (self.tuner.exhausted or not self.settings.auto_quality) and (
            face_fps < 6.0 or self._video_fps(now) < 0.6 * target)
        if not slow:
            self._slow_since = None
            return
        if self._slow_since is None:
            self._slow_since = now
            return
        if now - self._slow_since < 10.0:
            return
        self._hinted = True
        rate = rate_text(face_fps, long=True)
        if self.hub.effective == "cpu":
            self.message.emit("warn", f"Нейросети работают на процессоре, поэтому лицо обновляется {rate}. "
                                      "Видео остаётся плавным, но для живой мимики нужна видеокарта: "
                                      "NVIDIA (CUDA) или любая с DirectML — см. README, раздел «Скорость».")
        else:
            self.message.emit("warn", f"Лицо обновляется {rate}. Выберите пресет «Скорость» "
                                      "или уменьшите разрешение камеры в настройках источника.")

    # ================================================================== stage A · analyse
    def _update_mode(self, s: Settings) -> bool:
        """Fluid video when the face render is slower than the camera (with hysteresis)."""
        if not s.fluid_video:
            self._fluid = False
            return False
        src = self.source
        fps = src.fps if src is not None and src.fps > 1 else 30.0
        interval = 1000.0 / float(np.clip(fps, 5.0, 60.0))
        if self._fluid and 0 < self._b_ms < interval * 0.8:
            self._fluid = False
        elif not self._fluid and self._b_ms > interval:
            self._fluid = True
        return self._fluid

    def _wait_render_free(self) -> bool:
        with self._slot_cond:
            while self._slot is not None and not self._stop.is_set() and not self._reconfigure.is_set():
                self._slot_cond.wait(0.1)
        return not self._stop.is_set()

    def _analyze_loop(self) -> None:
        last_seq = -1
        while not self._stop.is_set():
            if self._reconfigure.is_set():
                self._reconfigure.clear()
                try:
                    self._apply_pipeline(self._pending_settings)
                except Exception as exc:
                    self.message.emit("error", f"Не удалось применить модели: {exc}")
            src = self.source
            if src is None:
                time.sleep(0.05)
                continue
            s = self.settings
            ready = self.models_ready and self.pipeline.ready
            fluid = ready and self._update_mode(s)
            if ready and not fluid:
                # direct mode: prepare the next frame while the previous one renders, but start late enough
                # that it is still fresh when the render stage picks it up
                if not self._wait_render_free():
                    break
                delay = (self._b_ms - self._a_ms) * 0.85 / 1000.0 - (time.monotonic() - self._b_started)
                if delay > 0.003:
                    self._stop.wait(min(delay, 0.5))
            frame, seq, ts = src.wait_frame(last_seq, 0.25)
            if frame is None or seq == last_seq:
                self._maybe_stats()
                continue
            last_seq = seq
            t0 = time.monotonic()
            frame = orient(frame, s.rotate, s.mirror)
            self._seq += 1
            n = self._seq
            matte = self._matte(frame, s)
            if not ready:
                self._publish(n, FrameResult(frame), frame, s, ts, matte)
                self._maybe_stats()
                continue
            try:
                prepare = not fluid or (self._slot is None and not self._b_busy)
                job = self.pipeline.analyze(frame, t0, prepare=prepare, heavy=not fluid)
                job.seq, job.ts, job.matte = n, ts, matte
                if job.jobs:
                    with self._slot_cond:
                        self._slot = job
                        self._slot_cond.notify_all()
                if fluid or not job.jobs:
                    res = self.pipeline.compose(job)
                    self._publish(n, res, frame, s, ts, matte)
                    self._record(res.timings, ("track", "landmarks", "occlusion", "compose"))
                else:
                    self._record(job.timings, ("track", "landmarks", "occlusion"))
                self._last_error = ""
            except Exception as exc:
                self._report_error(exc)
                self._publish(n, FrameResult(frame), frame, s, ts, matte)
            dt = (time.monotonic() - t0) * 1000
            self._a_ms = dt if self._a_ms <= 0 else 0.85 * self._a_ms + 0.15 * dt
            self._tune()
            self._maybe_stats()

    # ================================================================== stage B · render
    def _render_loop(self) -> None:
        while not self._stop.is_set():
            with self._slot_cond:
                if self._slot is None:
                    self._slot_cond.wait(0.25)
                job = self._slot
                if job is None:
                    continue
                self._slot = None
                self._b_busy = True
                self._b_started = time.monotonic()
                self._slot_cond.notify_all()
            t0 = time.monotonic()
            try:
                res = self.pipeline.render(job)
            except Exception as exc:
                self._report_error(exc)
                res = FrameResult(job.frame, job.faces, dict(job.timings))
            self._publish(job.seq, res, job.frame, job.settings, job.ts, job.matte)
            now = time.monotonic()
            dt = (now - t0) * 1000
            if res.swapped:
                self._b_ms = dt if self._b_ms <= 0 else 0.8 * self._b_ms + 0.2 * dt
                self._face_times.append(now)
                self._record(res.timings, ("swap", "mask", "parser", "enhance", "occlusion", "identity", "render"))
                for k in PART_KEYS:
                    v = res.timings.get(k)
                    if v is not None:
                        old = self._parts.get(k)
                        self._parts[k] = v if old is None else 0.8 * old + 0.2 * v
            self._b_busy = False

    def _report_error(self, exc: Exception) -> None:
        if str(exc) != self._last_error:
            log.exception("processing failed")
            self._last_error = str(exc)
            self.message.emit("error", f"Ошибка обработки: {exc}")

    def _record(self, timings: Dict[str, float], keys: Sequence[str]) -> None:
        for k in keys:
            v = timings.get(k)
            if v is not None:
                self._timings.setdefault(k, deque(maxlen=30)).append(v)

    # ================================================================== publishing
    def _publish(self, seq: int, res: FrameResult, frame: np.ndarray, s: Settings, ts: float,
                 matte: Optional[Matte] = None) -> bool:
        """Make `res` the current output frame unless a newer frame was already published."""
        if seq <= self._pub_seq:
            return False
        img = self._apply_background(res.frame, frame, matte, self.settings)
        out = fit_frame(img, s.output_width, s.output_height, s.output_fit)
        if s.watermark and self.watermark is not None:
            if out is res.frame or out is frame:
                out = out.copy()
            wm = self.watermark
            overlay_bgra(out, wm, out.shape[1] - wm.shape[1] - 18, out.shape[0] - wm.shape[0] - 18)
        with self._pub_lock:
            if seq <= self._pub_seq:
                return False
            self._pub_seq = seq
            with self._out_lock:
                self._latest_out = out
                self._latest_ts = ts
            now = time.monotonic()
            self._times.append(now)
            self._lat.append((now - ts) * 1000)
            self._faces = [dict(id=f.id, status=f.status, alpha=f.alpha, score=f.score) for f in res.faces]
        if not self._preview_busy:
            self._emit_preview(out, frame, res, s)
        return True

    def _fit_preview(self, img: np.ndarray) -> np.ndarray:
        size = self.preview_size
        if not size:
            return img
        h, w = img.shape[:2]
        k = min(size[0] / w, size[1] / h)
        if k >= 0.95:
            return img
        return cv2.resize(img, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA)

    def _emit_preview(self, out: np.ndarray, frame: np.ndarray, res: FrameResult, s: Settings) -> None:
        mode = s.preview_mode
        original = None
        view = out
        if mode in ("split", "original"):
            original = self._fit_preview(fit_frame(frame, s.output_width, s.output_height, s.output_fit))
        if mode == "mask":
            try:
                pad = (s.mask_padding_top, s.mask_padding_sides, s.mask_padding_bottom, s.mask_padding_sides)
                view = maskview.render(frame, res.marks, (s.output_width, s.output_height), s.output_fit,
                                       self.preview_size, s.mask_blur, pad)
            except Exception as exc:   # a debug view must never stop the video
                log.debug("mask view failed: %s", exc)
        view = self._fit_preview(view)
        self._preview_busy = True
        self.frameReady.emit({"result": to_qimage(view), "original": to_qimage(original) if original is not None else None})

    def preview_consumed(self) -> None:
        self._preview_busy = False

    def _video_fps(self, now: float) -> float:
        with self._pub_lock:
            times = list(self._times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        interval = span / (len(times) - 1)
        if span <= 0 or now - times[-1] > max(1.0, 2.5 * interval):
            return 0.0
        return (len(times) - 1) / span

    def _face_fps(self, now: float) -> float:
        """Face updates per second — also meaningful when they are rarer than once a second (CPU)."""
        times = list(self._face_times)
        if len(times) < 2:
            return 0.0
        recent = [t for t in times if now - t < 4.0]
        if len(recent) >= 3:
            return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-3)
        last = times[-5:]
        interval = (last[-1] - last[0]) / (len(last) - 1)
        if now - times[-1] > max(2.0, 2.5 * interval):
            return 0.0          # no face / stalled
        return 1.0 / max(interval, now - times[-1], 1e-3)

    def _faces_live(self, now: float) -> bool:
        if self._b_busy:
            return True
        return bool(self._face_times) and now - self._face_times[-1] < max(2.0, 3.0 * self._b_ms / 1000.0)

    def _maybe_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats < 0.5:
            return
        self._last_stats = now
        fps = self._video_fps(now)
        src = self.source
        s = self.settings
        with self._pub_lock:
            lat = list(self._lat)
            faces = list(self._faces)
        stats = {
            "fps": fps,
            "face_fps": self._face_fps(now),
            "mode": "fluid" if self._fluid else "direct",
            "latency": float(np.mean(lat)) if lat and fps > 0 else 0.0,
            "timings": {k: float(np.mean(v)) for k, v in list(self._timings.items()) if v},
            "a_ms": self._a_ms,
            "b_ms": self._b_ms,
            "auto": self.tuner.labels(s) if s.auto_quality else [],
            "auto_exhausted": bool(s.auto_quality and self.tuner.exhausted),
            "likeness": self.pipeline.likeness.value,
            "faces": faces if fps > 0 else [],
            "source_fps": src.fps if src else 0.0,
            "source_status": src.status if src else "idle",
            "source_name": src.description if src else "",
            "input": (src.info.get("width"), src.info.get("height")) if src else (None, None),
            "provider": self.hub.effective,
            "vcam": self.vcam.active,
            "identity": self.identity.name if self.identity else "",
            "link": dict(self.link.stats) if self.link and self.link.connected else None,
        }
        self.statsUpdated.emit(stats)

    # ================================================================== output
    def _output_loop(self) -> None:
        retry_at = 0.0
        last_err = None
        while not self._stop.is_set():
            s = self.settings
            if self._output_restart.is_set():
                self._output_restart.clear()
                self.vcam.close(self._farewell(s))
                retry_at = 0.0
                self.vcamStatus.emit(False, "")
            if s.vcam_enabled and not self.vcam.active and time.monotonic() >= retry_at:
                if self.vcam.open(s.output_width, s.output_height, s.output_fps, s.vcam_backend):
                    self.vcamStatus.emit(True, self.vcam.device)
                    last_err = None
                else:
                    retry_at = time.monotonic() + 5.0
                    if self.vcam.error != last_err:
                        last_err = self.vcam.error
                        self.vcamStatus.emit(False, self.vcam.error or "")
            if not s.vcam_enabled and self.vcam.active:
                self.vcam.close(self._farewell(s))
                self.vcamStatus.emit(False, "")
            with self._out_lock:
                frame = self._latest_out
            if frame is None:
                frame = self.placeholder if self.placeholder is not None else np.zeros(
                    (s.output_height, s.output_width, 3), np.uint8)
            if frame.shape[1] != s.output_width or frame.shape[0] != s.output_height:
                frame = fit_frame(frame, s.output_width, s.output_height, "fit")
            if self.vcam.active:
                try:
                    self.vcam.send(frame)
                except Exception as exc:
                    self.vcamStatus.emit(False, f"Виртуальная камера: {exc}")
                    self.vcam.close()
                    retry_at = time.monotonic() + 3.0
            with self._rec_lock:
                if self._writer is not None:
                    self._writer.write(frame)
            if self.vcam.active:
                self.vcam.wait()
            else:
                time.sleep(1.0 / max(s.output_fps, 1))
        self.vcam.close(self._farewell(self.settings))
        self.vcamStatus.emit(False, "")

    def _farewell(self, s: Settings) -> Optional[np.ndarray]:
        """Pause card for the virtual camera when the stream stops (Mimiq Camera keeps the last frame)."""
        card = self.placeholder
        if card is None or not self.vcam.active or self.vcam.backend != "mimiq":
            return None
        w, h = self.vcam.size
        if card.shape[1] != w or card.shape[0] != h:
            card = fit_frame(card, w, h, "fit")
        return card

    # ================================================================== captures
    def snapshot(self) -> Optional[Path]:
        with self._out_lock:
            frame = None if self._latest_out is None else self._latest_out.copy()
        if frame is None:
            return None
        path = paths.captures_dir() / f"Mimiq_{datetime.now():%Y%m%d_%H%M%S}.png"
        write_image(path, frame)
        return path

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def start_recording(self) -> Optional[Path]:
        if not self.running:
            return None
        s = self.settings
        videos = Path.home() / "Videos"
        folder = (videos if videos.exists() else Path.home()) / "Mimiq"
        folder.mkdir(parents=True, exist_ok=True)
        name = f"Mimiq_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        path = folder / name
        tmp = ascii_safe_dir(folder) / name     # OpenCV can't open non-ASCII paths on Windows
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), float(s.output_fps),
                                 (s.output_width, s.output_height))
        if not writer.isOpened():
            self.message.emit("error", "Не удалось начать запись видео")
            return None
        with self._rec_lock:
            self._writer, self._rec_path, self._rec_tmp = writer, path, tmp
        self.recordingChanged.emit(True, str(path))
        return path

    def stop_recording(self) -> Optional[Path]:
        with self._rec_lock:
            writer, path, tmp = self._writer, self._rec_path, self._rec_tmp
            self._writer, self._rec_path, self._rec_tmp = None, None, None
        if writer is not None:
            writer.release()
            if tmp is not None and path is not None and tmp != path:
                try:
                    if not (path.exists() and os.path.samefile(tmp, path)):
                        shutil.move(str(tmp), str(path))
                except OSError as exc:
                    log.warning("could not move recording %s → %s: %s", tmp, path, exc)
                    path = tmp
            self.recordingChanged.emit(False, str(path))
        return path
