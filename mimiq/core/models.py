"""Model registry, downloader and ONNX Runtime session factory."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .. import paths

log = logging.getLogger("mimiq.models")

HF = "https://huggingface.co/facefusion/{repo}/resolve/main/{file}"
GH = "https://github.com/facefusion/facefusion-assets/releases/download/{repo}/{file}"
RVM = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/{file}"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    file: str
    repo: str
    size: int
    kind: str
    title: str
    note: str = ""
    url: str = ""                 # own download address (models that are not in the FaceFusion assets)

    @property
    def urls(self) -> List[str]:
        if self.url:
            return [self.url]
        return [HF.format(repo=self.repo, file=self.file), GH.format(repo=self.repo, file=self.file)]

    @property
    def path(self) -> Path:
        return paths.models_dir() / self.file

    @property
    def size_mb(self) -> int:
        return round(self.size / 1_048_576)


_SPECS = [
    # detection / landmarks / identity
    ModelSpec("retinaface_10g", "retinaface_10g.onnx", "models-3.0.0", 16926877, "detector", "RetinaFace 10G"),
    ModelSpec("scrfd_2.5g", "scrfd_2.5g.onnx", "models-3.0.0", 3295067, "detector", "SCRFD 2.5G (быстрый)"),
    ModelSpec("2dfan4", "2dfan4.onnx", "models-3.0.0", 97904803, "landmarker", "2DFAN4 · 68 точек"),
    ModelSpec("arcface_w600k_r50", "arcface_w600k_r50.onnx", "models-3.0.0", 174388474, "recognizer", "ArcFace W600K R50"),
    # swappers
    ModelSpec("inswapper_128_fp16", "inswapper_128_fp16.onnx", "models-3.0.0", 277680829, "swapper",
              "InSwapper 128 · FP16", "Лучшая похожесть. Некоммерческая лицензия."),
    ModelSpec("inswapper_128", "inswapper_128.onnx", "models-3.0.0", 555303150, "swapper",
              "InSwapper 128 · FP32", "Та же модель в полной точности."),
    ModelSpec("hyperswap_1a_256", "hyperswap_1a_256.onnx", "models-3.3.0", 402742682, "swapper",
              "HyperSwap 1A · 256", "Родное разрешение 256 px."),
    ModelSpec("hyperswap_1b_256", "hyperswap_1b_256.onnx", "models-3.3.0", 402742682, "swapper",
              "HyperSwap 1B · 256", "Родное разрешение 256 px."),
    ModelSpec("hyperswap_1c_256", "hyperswap_1c_256.onnx", "models-3.3.0", 402742682, "swapper",
              "HyperSwap 1C · 256", "Родное разрешение 256 px."),
    # enhancers
    ModelSpec("gpen_bfr_256", "gpen_bfr_256.onnx", "models-3.0.0", 75792988, "enhancer", "GPEN 256 (быстрый)"),
    ModelSpec("gpen_bfr_512", "gpen_bfr_512.onnx", "models-3.0.0", 284340240, "enhancer", "GPEN 512"),
    ModelSpec("gfpgan_1.4", "gfpgan_1.4.onnx", "models-3.0.0", 340299087, "enhancer", "GFPGAN 1.4"),
    ModelSpec("codeformer", "codeformer.onnx", "models-3.0.0", 376951650, "enhancer", "CodeFormer"),
    ModelSpec("restoreformer_plus_plus", "restoreformer_plus_plus.onnx", "models-3.0.0", 294264232, "enhancer",
              "RestoreFormer++"),
    # masks
    ModelSpec("xseg_1", "xseg_1.onnx", "models-3.1.0", 70324286, "occluder", "XSeg 1"),
    ModelSpec("xseg_2", "xseg_2.onnx", "models-3.1.0", 70324286, "occluder", "XSeg 2"),
    ModelSpec("xseg_3", "xseg_3.onnx", "models-3.2.0", 70327709, "occluder", "XSeg 3"),
    ModelSpec("bisenet_resnet_18", "bisenet_resnet_18.onnx", "models-3.1.0", 53205356, "parser", "BiSeNet R18"),
    ModelSpec("bisenet_resnet_34", "bisenet_resnet_34.onnx", "models-3.0.0", 93632546, "parser", "BiSeNet R34"),
    # background matting (Robust Video Matting, GPL-3.0)
    ModelSpec("rvm_mobilenetv3", "rvm_mobilenetv3_fp32.onnx", "rvm-1.0.0", 14975696, "matting",
              "RVM MobileNetV3 · быстро", "Плавно даже на ноутбуке. Волосы и плечи без мерцания.",
              RVM.format(file="rvm_mobilenetv3_fp32.onnx")),
    ModelSpec("rvm_resnet50", "rvm_resnet50_fp32.onnx", "rvm-1.0.0", 107479165, "matting",
              "RVM ResNet-50 · максимум", "Точнее по прядям волос и пальцам. Нужна видеокарта RTX.",
              RVM.format(file="rvm_resnet50_fp32.onnx")),
]

REGISTRY: Dict[str, ModelSpec] = {s.key: s for s in _SPECS}
BASE_MODELS = ["retinaface_10g", "arcface_w600k_r50", "inswapper_128_fp16", "xseg_1", "gpen_bfr_256", "2dfan4"]


def by_kind(kind: str) -> List[ModelSpec]:
    return [s for s in _SPECS if s.kind == kind]


def is_installed(key: str) -> bool:
    spec = REGISTRY[key]
    p = spec.path
    return p.exists() and p.stat().st_size == spec.size


def missing(keys: Iterable[str]) -> List[str]:
    return [k for k in dict.fromkeys(keys) if k and not is_installed(k)]


ProgressFn = Callable[[str, int, int, float], None]  # key, done, total, bytes/sec


class DownloadCancelled(Exception):
    pass


def download(key: str, progress: Optional[ProgressFn] = None, cancel: Optional[threading.Event] = None) -> Path:
    """Download one model with resume support and size verification."""
    spec = REGISTRY[key]
    target = spec.path
    if is_installed(key):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    last_error: Optional[Exception] = None
    for url in spec.urls:
        for attempt in range(3):
            try:
                _download_url(url, part, spec, progress, cancel)
                if part.stat().st_size != spec.size:
                    raise IOError(f"size mismatch {part.stat().st_size} != {spec.size}")
                os.replace(part, target)
                log.info("downloaded %s", spec.file)
                return target
            except DownloadCancelled:
                raise
            except Exception as exc:  # network hiccup → retry / next mirror
                last_error = exc
                log.warning("download %s from %s failed (%s), attempt %d", spec.file, url, exc, attempt + 1)
                if part.exists() and part.stat().st_size > spec.size:
                    part.unlink()
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Не удалось скачать {spec.file}: {last_error}")


def _download_url(url: str, part: Path, spec: ModelSpec, progress, cancel) -> None:
    have = part.stat().st_size if part.exists() else 0
    if have >= spec.size:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mimiq/1.0", "Range": f"bytes={have}-"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if have and resp.status != 206:  # server ignored the range → restart
            have = 0
        mode = "ab" if have else "wb"
        t0, done0 = time.monotonic(), have
        with open(part, mode) as fh:
            done = have
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled()
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    dt = max(time.monotonic() - t0, 1e-6)
                    progress(spec.key, done, spec.size, (done - done0) / dt)


def ensure(keys: Iterable[str], progress: Optional[ProgressFn] = None,
           cancel: Optional[threading.Event] = None) -> None:
    for key in missing(keys):
        download(key, progress, cancel)


# --------------------------------------------------------------------------------------
# ONNX Runtime
# --------------------------------------------------------------------------------------

PROVIDER_IDS = {
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
PROVIDER_LABELS = {
    "auto": "Авто",
    "tensorrt": "NVIDIA TensorRT",
    "cuda": "NVIDIA CUDA",
    "directml": "DirectML",
    "coreml": "Apple CoreML",
    "cpu": "CPU",
}

_ort = None
_ort_lock = threading.Lock()


def ort():
    """Import onnxruntime lazily and preload CUDA/cuDNN DLLs shipped via pip."""
    global _ort
    with _ort_lock:
        if _ort is None:
            import onnxruntime as _rt
            _rt.set_default_logger_severity(3)
            cuda = "CUDAExecutionProvider" in _rt.get_available_providers()
            if sys.platform == "win32" and hasattr(_rt, "preload_dlls"):
                try:
                    if cuda:
                        _rt.preload_dlls()
                except Exception as exc:  # pragma: no cover - depends on host
                    log.warning("preload_dlls failed: %s", exc)
            if cuda and "TensorrtExecutionProvider" in _rt.get_available_providers():
                _setup_tensorrt(_rt)
            _ort = _rt
    return _ort


# --------------------------------------------------------------------------------------
# TensorRT (optional, NVIDIA RTX): libraries come from the pip wheel tensorrt-cu13-libs / tensorrt-cu12-libs
# --------------------------------------------------------------------------------------

TRT_KINDS = {"swapper", "enhancer", "occluder", "parser", "landmarker"}
TRT_SKIP = {"codeformer"}       # float64 "fidelity" input, TensorRT can't take it
# FP16 only where it is known to be safe: the FP16 swapper already computes in half precision, masks and
# landmarks are robust. GAN enhancers / HyperSwap stay in FP32 (TF32) — no risk of overflow artefacts.
TRT_FP16 = {"inswapper_128_fp16", "inswapper_128", "xseg_1", "xseg_2", "xseg_3", "bisenet_resnet_18",
            "bisenet_resnet_34", "2dfan4"}
TRT_DISTS = ("tensorrt-cu13-libs", "tensorrt-cu12-libs", "tensorrt-libs", "tensorrt")
_DLL_DIRS: list = []
_trt: Dict[str, object] = {"ok": False, "reason": "нужна видеокарта NVIDIA с CUDA", "version": "", "dir": ""}


def ort_tensorrt_major(rt=None) -> Optional[int]:
    """TensorRT major version the installed onnxruntime-gpu was built for (nvinfer_10.dll → 10)."""
    import re
    try:
        if rt is not None:
            capi = Path(rt.__file__).resolve().parent / "capi"
        else:
            import importlib.util
            spec = importlib.util.find_spec("onnxruntime")
            if spec is None or not spec.submodule_search_locations:
                return None
            capi = Path(list(spec.submodule_search_locations)[0]) / "capi"
        for name in ("onnxruntime_providers_tensorrt.dll", "libonnxruntime_providers_tensorrt.so"):
            f = capi / name
            if f.exists():
                m = re.search(rb"(?:nvinfer_(\d+)\.dll|libnvinfer\.so\.(\d+))", f.read_bytes())
                if m:
                    return int(m.group(1) or m.group(2))
    except Exception as exc:
        log.debug("TensorRT major lookup failed: %s", exc)
    return None


def _tensorrt_libs_dir() -> Optional[Path]:
    import importlib.util
    try:
        spec = importlib.util.find_spec("tensorrt_libs")   # not imported: its __init__ loads ~2 GB of DLLs
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def _tensorrt_version() -> str:
    import importlib.metadata as md
    for dist in TRT_DISTS:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
    return ""


def _setup_tensorrt(rt) -> None:
    """Make the TensorRT DLLs findable for ONNX Runtime's TensorRT provider and check that they load."""
    import ctypes
    major = ort_tensorrt_major(rt)
    folder = _tensorrt_libs_dir()
    if folder is None:
        _trt.update(ok=False, reason="библиотеки TensorRT не установлены")
        return
    if major is None:
        _trt.update(ok=False, reason="эта сборка ONNX Runtime без TensorRT")
        return
    win = sys.platform == "win32"
    names = ([f"nvinfer_{major}.dll", f"nvinfer_plugin_{major}.dll", f"nvonnxparser_{major}.dll"] if win else
             [f"libnvinfer.so.{major}", f"libnvinfer_plugin.so.{major}", f"libnvonnxparser.so.{major}"])
    if not (folder / names[0]).exists():
        _trt.update(ok=False, reason=f"нужен TensorRT {major}.x — запустите install.bat tensorrt")
        return
    try:
        if win:
            _DLL_DIRS.append(os.add_dll_directory(str(folder)))
        # nvinfer loads its per-GPU builder resources by name later on → keep the folder on PATH
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        for name in names:
            if (folder / name).exists():
                if win:
                    ctypes.WinDLL(str(folder / name))
                else:
                    ctypes.CDLL(str(folder / name), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        _trt.update(ok=False, reason=f"не загрузились библиотеки TensorRT: {exc}")
        log.warning("TensorRT libraries failed to load: %s", exc)
        return
    _trt.update(ok=True, reason="", version=_tensorrt_version(), dir=str(folder))
    log.info("TensorRT %s ready (%s)", _trt["version"], folder)


def tensorrt_status() -> dict:
    """{"ok": bool, "reason": str, "version": str} — whether the TensorRT provider can really be used."""
    try:
        ort()
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "version": ""}
    return dict(_trt)


def tensorrt_cache_dir() -> Path:
    """Engine cache. TensorRT/ONNX Runtime can't handle non-ASCII paths on Windows (Cyrillic user names)."""
    from ..imaging import ascii_safe_dir
    tag = "trt-" + (str(_trt.get("version") or "x").replace("/", "_"))
    folder = paths.cache_dir() / "tensorrt" / tag
    folder.mkdir(parents=True, exist_ok=True)
    safe = ascii_safe_dir(folder)
    if safe.name == "Mimiq-temp":                 # public fallback folder → keep versions apart
        safe = safe / "tensorrt" / tag
        safe.mkdir(parents=True, exist_ok=True)
    return safe


def uses_tensorrt(key: str) -> bool:
    spec = REGISTRY.get(key)
    return spec is not None and spec.kind in TRT_KINDS and key not in TRT_SKIP


def _cache_prefix(key: str) -> str:
    # trailing "_x": no key's prefix is a prefix of another key's (inswapper_128 vs inswapper_128_fp16)
    return "mimiq_" + "".join(c if c.isalnum() else "_" for c in key) + "_x"


def tensorrt_cached(key: str) -> bool:
    try:
        return any(tensorrt_cache_dir().glob(_cache_prefix(key) + "*.engine"))
    except Exception:
        return False


def available_providers() -> List[str]:
    ids = set(ort().get_available_providers())
    out = [k for k, v in PROVIDER_IDS.items() if v in ids]
    if "tensorrt" in out and not _trt.get("ok"):
        out.remove("tensorrt")                       # the provider is compiled in, but its libraries are missing
    return out


def resolve_provider(choice: str) -> str:
    try:
        avail = available_providers()
    except Exception as exc:  # onnxruntime missing or broken
        log.error("onnxruntime unavailable: %s", exc)
        return "cpu"
    if choice != "auto" and choice in avail:
        return choice
    for k in ("tensorrt", "cuda", "directml", "coreml", "cpu"):
        if k in avail:
            return k
    return "cpu"


def _cuda_options(device_id: int) -> dict:
    return {"device_id": device_id, "cudnn_conv_algo_search": "EXHAUSTIVE", "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": True}


def _provider_chain(kind: str, device_id: int, key: str = "") -> list:
    cpu = "CPUExecutionProvider"
    if kind == "tensorrt":
        if not uses_tensorrt(key):          # detector (changing input sizes) & co. → plain CUDA
            return [("CUDAExecutionProvider", _cuda_options(device_id)), cpu]
        trt_cache = str(tensorrt_cache_dir())
        return [
            ("TensorrtExecutionProvider", {
                "device_id": device_id,
                "trt_fp16_enable": key in TRT_FP16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache,
                "trt_engine_cache_prefix": _cache_prefix(key),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": trt_cache,
                "trt_builder_optimization_level": 3,
                "trt_max_workspace_size": 2 << 30,
            }),
            ("CUDAExecutionProvider", _cuda_options(device_id)),
            cpu,
        ]
    if kind == "cuda":
        return [("CUDAExecutionProvider", _cuda_options(device_id)), cpu]
    if kind == "directml":
        return [("DmlExecutionProvider", {"device_id": device_id}), cpu]
    if kind == "coreml":
        return ["CoreMLExecutionProvider", cpu]
    return [cpu]


class LockedSession:
    """InferenceSession wrapper that serialises run().

    DirectML does not allow concurrent runs on one session, and with the two-stage pipeline a background job
    (photo analysis, likeness probe) may use the same model as the video at the same moment."""

    __slots__ = ("_sess", "_lock")

    def __init__(self, sess):
        self._sess = sess
        self._lock = threading.Lock()

    def run(self, *args, **kwargs):
        with self._lock:
            return self._sess.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._sess, name)


_RANK = {"TensorrtExecutionProvider": 0, "CUDAExecutionProvider": 1, "DmlExecutionProvider": 1,
         "CoreMLExecutionProvider": 1, "CPUExecutionProvider": 2}
_BY_ID = {v: k for k, v in PROVIDER_IDS.items()}


class SessionHub:
    """Creates and caches InferenceSessions for the active execution provider."""

    def __init__(self, provider: str = "auto", device_id: int = 0):
        self._lock = threading.RLock()
        self._sessions: Dict[str, object] = {}
        self._actual: Dict[str, str] = {}
        self.provider = resolve_provider(provider)
        self.device_id = device_id

    @property
    def effective(self) -> str:
        """Provider really used by the loaded sessions.

        ONNX Runtime silently falls back to CPU when, for example, CUDA/cuDNN DLLs are missing,
        so the requested provider alone is not a reliable indicator. With TensorRT some models
        (the detector) intentionally run on CUDA — that still counts as TensorRT."""
        with self._lock:
            acts = set(self._actual.values())
        if not acts:
            return self.provider
        if "CPUExecutionProvider" in acts and self.provider != "cpu":
            return "cpu"
        if self.provider == "tensorrt":
            return "tensorrt" if "TensorrtExecutionProvider" in acts else "cuda"
        worst = max(acts, key=lambda a: _RANK.get(a, 9))
        return _BY_ID.get(worst, "cpu")

    @property
    def trt_failed(self) -> List[str]:
        """Models that should have run on TensorRT but fell back to CUDA."""
        with self._lock:
            return [k for k, a in self._actual.items()
                    if self.provider == "tensorrt" and uses_tensorrt(k) and a != "TensorrtExecutionProvider"]

    @property
    def degraded(self) -> bool:
        return self.provider != "cpu" and self.effective == "cpu"

    def configure(self, provider: str, device_id: int) -> bool:
        resolved = resolve_provider(provider)
        with self._lock:
            if resolved == self.provider and device_id == self.device_id:
                return False
            self._sessions.clear()
            self._actual.clear()
            self.provider, self.device_id = resolved, device_id
            return True

    def get(self, key: str):
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                spec = REGISTRY[key]
                if not is_installed(key):
                    raise FileNotFoundError(f"Модель {spec.file} не скачана")
                sess = LockedSession(self._create(spec.path, key))
                self._sessions[key] = sess
                try:
                    self._actual[key] = sess.get_providers()[0]
                except Exception:
                    self._actual[key] = "CPUExecutionProvider"
            return sess

    def release(self, keep: Iterable[str]) -> None:
        keep = set(keep)
        with self._lock:
            for k in list(self._sessions):
                if k not in keep:
                    del self._sessions[k]
                    self._actual.pop(k, None)

    def prime(self, key: str) -> None:
        """Run a model once with dummy inputs. With TensorRT this builds the engine now (minutes on the very first
        start, then loaded from the cache) instead of freezing the video later. If TensorRT fails at run time the
        model is reloaded on CUDA."""
        import numpy as np
        sess = self.get(key)
        if self._actual.get(key) != "TensorrtExecutionProvider":
            return
        dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
                  "tensor(int64)": np.int64, "tensor(int32)": np.int32}
        try:
            feed = {}
            for inp in sess.get_inputs():
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                feed[inp.name] = np.zeros(shape, dtypes.get(inp.type, np.float32))
            sess.run(None, feed)
        except Exception as exc:
            log.error("TensorRT failed for %s (%s) — using CUDA for it", key, exc)
            with self._lock:
                spec = REGISTRY[key]
                cuda = LockedSession(self._create(spec.path, key, force="cuda"))
                self._sessions[key] = cuda
                self._actual[key] = cuda.get_providers()[0]

    def _create(self, path: Path, key: str = "", force: str = ""):
        rt = ort()
        so = rt.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Two pipeline stages run in parallel: spinning worker threads would only steal CPU from each other
        # (and from Python) — on GPU providers they are pure waste.
        try:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception:
            pass
        if self.provider == "directml":
            so.enable_mem_pattern = False
            so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        kind = force or self.provider
        chain = _provider_chain(kind, self.device_id, key)
        t0 = time.monotonic()
        try:
            sess = rt.InferenceSession(str(path), sess_options=so, providers=chain)
        except Exception as exc:
            if kind == "cpu":
                raise
            sess = None
            if kind == "tensorrt":
                log.error("TensorRT failed for %s (%s); trying CUDA", path.name, exc)
                try:
                    sess = rt.InferenceSession(str(path), sess_options=so,
                                               providers=_provider_chain("cuda", self.device_id, key))
                except Exception as exc2:
                    exc = exc2
            if sess is None:
                log.error("provider %s failed for %s (%s); falling back to CPU", kind, path.name, exc)
                sess = rt.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        active = sess.get_providers()[0]
        if self.provider != "cpu" and active == "CPUExecutionProvider":
            log.warning("%s: %s is unavailable, ONNX Runtime fell back to CPU", path.name, self.provider)
        if kind == "tensorrt" and uses_tensorrt(key) and active != "TensorrtExecutionProvider":
            log.warning("%s: TensorRT unavailable, running on %s", path.name, active)
        log.info("loaded %s on %s in %.1f s", path.name, active, time.monotonic() - t0)
        return sess
