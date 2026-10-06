"""Pick the best ONNX Runtime build for this PC.

    python tools/select_runtime.py            → prints the recommendation
    python tools/select_runtime.py --json     → machine-readable

Rules (2025+ wheels on PyPI):
  * NVIDIA RTX / GTX 16xx (compute ≥ 7.5) + driver ≥ 580  → onnxruntime-gpu (CUDA 13) + pip CUDA/cuDNN
  * other NVIDIA (GTX 9xx/10xx, older drivers ≥ 528)       → onnxruntime-gpu 1.26 (last CUDA 12 build)
  * AMD / Intel GPU on Windows, or very old NVIDIA driver   → onnxruntime-directml (DirectX 12)
  * everything else                                         → onnxruntime (CPU)
Override with  MIMIQ_RUNTIME=cuda|cuda12|directml|cpu  or  --force <name>.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional

CUDA13_SPEC = "onnxruntime-gpu[cuda,cudnn]"
CUDA12_SPEC = "onnxruntime-gpu[cuda,cudnn]==1.26.0"
RUNTIMES = {
    "cuda": (CUDA13_SPEC, "onnxruntime-gpu", "NVIDIA CUDA 13"),
    "cuda12": (CUDA12_SPEC, "onnxruntime-gpu", "NVIDIA CUDA 12"),
    "directml": ("onnxruntime-directml", "onnxruntime-directml", "DirectML (DirectX 12)"),
    "cpu": ("onnxruntime", "onnxruntime", "CPU"),
}
ALL_DISTS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class Gpu:
    name: str
    driver: str = ""
    compute: float = 0.0
    vendor: str = "nvidia"


@dataclass
class Choice:
    runtime: str
    spec: str
    dist: str
    label: str
    reason: str
    gpus: List[Gpu] = field(default_factory=list)
    warning: str = ""


def _run(cmd: List[str], timeout: float = 20) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW,
                             encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for cand in (r"C:\Windows\System32\nvidia-smi.exe",
                 r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.exists(cand):
            return cand
    return None


def _guess_compute(name: str) -> float:
    n = name.upper()
    if re.search(r"RTX\s*50|BLACKWELL", n):
        return 12.0
    if re.search(r"RTX\s*40|ADA|L4\b|L40", n):
        return 8.9
    if re.search(r"RTX\s*30|A\d{3,4}\b|A40|A100", n):
        return 8.6
    if re.search(r"RTX\s*20|GTX\s*16|TITAN RTX|QUADRO RTX|T4\b|T\d{3,4}\b", n):
        return 7.5
    if re.search(r"GTX\s*10|TITAN X|QUADRO P|P\d{3,4}\b|MX\s*[1-4]", n):
        return 6.1
    if re.search(r"GTX\s*9|GTX\s*7[45]0|QUADRO M|M\d{3,4}\b", n):
        return 5.2
    return 0.0


def nvidia_gpus() -> List[Gpu]:
    exe = _nvidia_smi()
    if not exe:
        return []
    out = _run([exe, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"])
    with_cc = out is not None
    if out is None:  # old drivers don't know compute_cap
        out = _run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        cc = 0.0
        if with_cc and len(parts) > 2:
            try:
                cc = float(parts[2])
            except ValueError:
                cc = 0.0
        gpus.append(Gpu(parts[0], parts[1] if len(parts) > 1 else "", cc or _guess_compute(parts[0])))
    return gpus


def other_gpus() -> List[Gpu]:
    if sys.platform != "win32":
        return []
    out = _run(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_VideoController).Name"])
    gpus = []
    for line in (out or "").splitlines():
        name = line.strip()
        if not name or "NVIDIA" in name.upper() or "BASIC" in name.upper() or "MIRROR" in name.upper():
            continue
        vendor = "amd" if re.search(r"AMD|RADEON", name, re.I) else "intel" if "INTEL" in name.upper() else "other"
        gpus.append(Gpu(name, vendor=vendor))
    return gpus


def _driver_major(driver: str) -> int:
    try:
        return int(driver.split(".")[0])
    except (ValueError, IndexError):
        return 0


def choose(force: Optional[str] = None) -> Choice:
    force = (force or os.environ.get("MIMIQ_RUNTIME") or "").strip().lower() or None
    nv = nvidia_gpus()
    others = other_gpus() if not nv else []
    gpus = nv + others

    def make(rt: str, reason: str, warning: str = "") -> Choice:
        spec, dist, label = RUNTIMES[rt]
        return Choice(rt, spec, dist, label, reason, gpus, warning)

    if force:
        if force not in RUNTIMES:
            raise SystemExit(f"Неизвестный вариант '{force}'. Доступно: {', '.join(RUNTIMES)}")
        return make(force, "выбрано вручную")

    if nv:
        best = max(nv, key=lambda g: g.compute)
        drv = _driver_major(best.driver)
        if best.compute >= 7.5 and drv >= 580:
            return make("cuda", f"{best.name}, драйвер {best.driver}")
        if best.compute >= 7.5:
            return make("cuda12", f"{best.name}, драйвер {best.driver}",
                        "Обновите драйвер NVIDIA до версии 580+ — станет доступна новейшая сборка CUDA 13.")
        if best.compute >= 5.0 and drv >= 528:
            return make("cuda12", f"{best.name} (compute {best.compute:.1f}), драйвер {best.driver}")
        if sys.platform == "win32":
            return make("directml", f"{best.name}, драйвер {best.driver or '?'}",
                        "Драйвер NVIDIA слишком старый для CUDA 12 — используется DirectML. "
                        "Обновите драйвер и запустите install.bat снова.")
        return make("cpu", "NVIDIA без подходящего драйвера")

    if sys.platform == "win32":
        if others:
            return make("directml", ", ".join(g.name for g in others))
        return make("directml", "видеокарта не определена — DirectML работает с любой DirectX 12 GPU")
    return make("cpu", "GPU-ускорение для этой ОС не настроено")


def main(argv: List[str]) -> int:
    force = None
    if "--force" in argv:
        i = argv.index("--force")
        force = argv[i + 1] if i + 1 < len(argv) else None
    c = choose(force)
    if "--json" in argv:
        d = asdict(c)
        print(json.dumps(d, ensure_ascii=False))
        return 0
    print(f"Видеокарта: {', '.join(g.name for g in c.gpus) or 'не найдена'}")
    print(f"Сборка ONNX Runtime: {c.label}  →  pip install \"{c.spec}\"")
    print(f"Почему: {c.reason}")
    if c.warning:
        print(f"Внимание: {c.warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
