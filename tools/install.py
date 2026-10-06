"""Mimiq installer step 2 (runs inside .venv, started by install.bat).

    install.bat                 → full install / repair
    install.bat directml        → force a runtime: cuda | cuda12 | directml | cpu
    install.bat --no-models     → skip model download (Mimiq downloads them on first start)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))

import select_runtime  # noqa: E402

PY = sys.executable
WIN = sys.platform == "win32"

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

C = {"ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m", "dim": "\033[90m", "acc": "\033[95m", "end": "\033[0m"}
if WIN:
    os.system("")  # enable ANSI colours in the Windows console


def say(text: str = "", kind: str = "") -> None:
    print(f"{C.get(kind, '')}{text}{C['end'] if kind else ''}", flush=True)


def step(n: int, total: int, text: str) -> None:
    say()
    say(f"  [{n}/{total}] {text}", "acc")


def pip(*args: str, quiet: bool = False) -> int:
    cmd = [PY, "-m", "pip", *args, "--disable-pip-version-check"]
    if quiet:
        cmd.append("-q")
    return subprocess.call(cmd)


def _has_dist(name: str) -> bool:
    import importlib.metadata as md
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


def installed_dists() -> dict:
    import importlib.metadata as md
    found = {}
    for name in select_runtime.ALL_DISTS:
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return found


PROBE = r'''
import json, sys
import numpy as np
want = sys.argv[1]
res = {"ok": False}
try:
    import onnxruntime as ort
    res["version"] = ort.__version__
    if want == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:
            res["preload"] = str(exc)
    res["available"] = ort.get_available_providers()
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    w = nh.from_array(np.ones((4, 3, 3, 3), np.float32), "w")
    g = h.make_graph([h.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])], "probe",
                     [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 16, 16])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 4, 16, 16])], [w])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if want == "DmlExecutionProvider":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=[want, "CPUExecutionProvider"])
    y = s.run(None, {"x": np.ones((1, 3, 16, 16), np.float32)})[0]
    res["active"] = s.get_providers()[0]
    res["ok"] = res["active"] == want and abs(float(y[0, 0, 8, 8]) - 27.0) < 1e-3
except Exception as exc:
    res["error"] = f"{type(exc).__name__}: {exc}"
print("PROBE" + json.dumps(res))
'''

EP = {"cuda": "CUDAExecutionProvider", "cuda12": "CUDAExecutionProvider",
      "directml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}


def probe(runtime: str) -> dict:
    try:
        out = subprocess.run([PY, "-c", PROBE, EP[runtime]], capture_output=True, text=True, timeout=300,
                             encoding="utf-8", errors="replace").stdout
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    for line in out.splitlines():
        if line.startswith("PROBE"):
            return json.loads(line[5:])
    return {"ok": False, "error": out.strip()[-400:] or "нет ответа"}


def want_version_ok(runtime: str, version: str) -> bool:
    if runtime == "cuda12":
        return version == "1.26.0"
    if runtime == "cuda":
        try:
            major, minor = (int(x) for x in version.split(".")[:2])
            return (major, minor) >= (1, 27)
        except ValueError:
            return False
    return True


def install_runtime(choice) -> bool:
    have = installed_dists()
    current = have.get(choice.dist)
    if current and want_version_ok(choice.runtime, current) and len(have) == 1:
        say(f"  Уже установлено: {choice.dist} {current} — проверяю…", "dim")
        res = probe(choice.runtime)
        if res.get("ok"):
            return True
        say("  Проверка не прошла — переустанавливаю.", "warn")
    conflicting = [d for d in have if d != choice.dist or not want_version_ok(choice.runtime, have[d])]
    if conflicting:
        say(f"  Удаляю другие сборки: {', '.join(conflicting)}", "dim")
        pip("uninstall", "-y", *conflicting, quiet=True)
    if choice.runtime in ("cuda", "cuda12"):
        say("  Скачиваю ONNX Runtime GPU с библиотеками CUDA и cuDNN (~1.5 ГБ, один раз)…", "dim")
    rc = pip("install", "--upgrade", choice.spec)
    return rc == 0


def report_probe(choice, res) -> None:
    if res.get("ok"):
        say(f"  ✓ {choice.label} работает · onnxruntime {res.get('version', '?')}", "ok")
        return
    say(f"  ✗ {choice.label} не запустилась", "err")
    detail = res.get("error") or f"активный провайдер: {res.get('active', '?')}"
    say(f"    {detail[:400]}", "dim")
    if "DLL load failed" in detail or "msvcp" in detail.lower() or "vcruntime" in detail.lower():
        say("    Похоже, не хватает Microsoft Visual C++ Redistributable:", "warn")
        say("    https://aka.ms/vs/17/release/vc_redist.x64.exe — установите и запустите install.bat снова.", "warn")


def ask(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"  {question} {hint}: ").strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in ("y", "yes", "д", "да")


def create_shortcuts() -> None:
    if not WIN:
        return
    pythonw = Path(PY).with_name("pythonw.exe")
    ico = ROOT / "mimiq" / "assets" / "mimiq.ico"

    def q(p) -> str:
        return str(p).replace("'", "''")
    ps = f"""
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {{
  if (-not $dir) {{ continue }}
  $s = $ws.CreateShortcut((Join-Path $dir 'Mimiq.lnk'))
  $s.TargetPath = '{q(pythonw)}'
  $s.Arguments = '-m mimiq'
  $s.WorkingDirectory = '{q(ROOT)}'
  $s.IconLocation = '{q(ico)},0'
  $s.Description = 'Mimiq - face swap studio'
  $s.Save()
}}
"""
    rc = subprocess.call(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 0:
        say("  ✓ Ярлык «Mimiq» создан на рабочем столе и в меню «Пуск»", "ok")
    else:
        say("  Не удалось создать ярлык — запускайте Mimiq.bat", "warn")


def main(argv) -> int:
    force = next((a for a in argv if a in select_runtime.RUNTIMES), None)
    total = 4
    say()
    say("  ███╗   ███╗██╗███╗   ███╗██╗ ██████╗ ", "acc")
    say("  ████╗ ████║██║████╗ ████║██║██╔═══██╗", "acc")
    say("  ██╔████╔██║██║██╔████╔██║██║██║   ██║", "acc")
    say("  ██║╚██╔╝██║██║██║╚██╔╝██║██║██║▄▄ ██║", "acc")
    say("  ██║ ╚═╝ ██║██║██║ ╚═╝ ██║██║╚██████╔╝", "acc")
    say("  ╚═╝     ╚═╝╚═╝╚═╝     ╚═╝╚═╝ ╚══▀▀═╝ ", "acc")
    say(f"  Установка · Python {sys.version.split()[0]} · {ROOT}", "dim")

    step(1, total, "Библиотеки интерфейса и обработки видео")
    clash = [d for d in ("opencv-python", "opencv-contrib-python") if _has_dist(d)]
    if clash:  # they ship the same cv2 module as opencv-python-headless
        pip("uninstall", "-y", *clash, quiet=True)
    if pip("install", "--upgrade", "-r", str(ROOT / "requirements.txt")) != 0:
        say("  Не удалось установить зависимости. Проверьте интернет и запустите install.bat ещё раз.", "err")
        return 1
    say("  ✓ Готово", "ok")

    step(2, total, "Ускорение нейросетей (ONNX Runtime)")
    choice = select_runtime.choose(force)
    gpus = ", ".join(g.name for g in choice.gpus) or "не найдена"
    say(f"  Видеокарта: {gpus}")
    say(f"  Выбрано: {choice.label}  ({choice.reason})")
    if choice.warning:
        say(f"  ! {choice.warning}", "warn")
    if not install_runtime(choice):
        say("  Не удалось установить ONNX Runtime.", "err")
        return 1
    res = probe(choice.runtime)
    report_probe(choice, res)
    if not res.get("ok") and choice.runtime in ("cuda", "cuda12") and WIN:
        alt = select_runtime.choose("cuda12" if choice.runtime == "cuda" else "directml")
        say(f"  Можно попробовать {alt.label}.", "warn")
        if ask(f"Установить {alt.label} вместо этого?"):
            if install_runtime(alt):
                res = probe(alt.runtime)
                report_probe(alt, res)
                choice = alt
            if not res.get("ok") and alt.runtime != "directml":
                dml = select_runtime.choose("directml")
                if ask("Последний вариант — DirectML (работает на любой видеокарте DirectX 12). Установить?"):
                    if install_runtime(dml):
                        res = probe(dml.runtime)
                        report_probe(dml, res)
                        choice = dml
    if choice.runtime == "cpu" or not res.get("ok"):
        say("  Mimiq будет работать на процессоре — это очень медленно (около 1 кадра в секунду или меньше). Для реального времени нужна "
            "видеокарта, лучше NVIDIA RTX.", "warn")

    step(3, total, "Нейросети (≈680 МБ, один раз)")
    if "--no-models" in argv:
        say("  Пропущено — Mimiq скачает модели при первом запуске.", "dim")
    else:
        import download_models
        rc = download_models.main([])
        if rc != 0:
            say("  Модели не скачались полностью — Mimiq докачает их при запуске.", "warn")

    step(4, total, "Ярлыки")
    if "--no-shortcut" in argv:
        say("  Пропущено", "dim")
    else:
        create_shortcuts()

    say()
    say("  ✓ Mimiq установлен!", "ok")
    say("  Запуск: ярлык «Mimiq» на рабочем столе или Mimiq.bat", "dim")
    say("  Виртуальная камера: установите OBS Studio 28+ и один раз нажмите в нём «Запустить виртуальную камеру».",
        "dim")
    if WIN and "--no-launch" not in argv and ask("Запустить Mimiq сейчас?"):
        subprocess.Popen([str(Path(PY).with_name("pythonw.exe")), "-m", "mimiq"], cwd=str(ROOT),
                         creationflags=0x00000008)  # DETACHED_PROCESS
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        say("\n  Прервано.", "warn")
        sys.exit(130)
