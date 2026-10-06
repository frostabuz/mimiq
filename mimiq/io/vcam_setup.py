"""Install / remove "Mimiq Camera" — Mimiq's own DirectShow virtual camera (no OBS needed).

The camera is the open-source Unity Capture filter (MIT, https://github.com/schellingb/UnityCapture),
registered under the fixed name "Mimiq Camera". The DLLs are copied to Program Files so the camera
keeps working when the Mimiq folder is moved or updated. Registration writes to HKLM, so it needs
administrator rights: the work is done by a short elevated copy of this module (one UAC prompt).

    python -m mimiq.io.vcam_setup status | install | uninstall
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("mimiq.vcam_setup")

CAMERA_NAME = "Mimiq Camera"       # fixed: apps must see an honestly named virtual camera
DLL64 = "UnityCaptureFilter64.dll"
DLL32 = "UnityCaptureFilter32.dll"
CLSID = "{5C2CD55C-92AD-4999-8666-912BD3E70010}"     # device #1 of the 64-bit filter
CLSID32 = "{5C2CD55C-92AD-4999-8666-912BD3E70020}"   # device #1 of the 32-bit filter
WIN = sys.platform == "win32"
_NO_WINDOW = 0x08000000 if WIN else 0


def supported() -> bool:
    return WIN


def source_dir() -> Path:
    from .. import paths
    return paths.package_dir() / "vcam"


def install_dir() -> Path:
    base = os.environ.get("ProgramW6432") or os.environ.get("ProgramFiles") or r"C:\Program Files"
    return Path(base) / CAMERA_NAME


def _reg_value(clsid: str, wow32: bool = False, sub: str = "") -> Optional[str]:
    if not WIN:
        return None
    import winreg
    view = winreg.KEY_WOW64_32KEY if wow32 else winreg.KEY_WOW64_64KEY
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf"CLSID\{clsid}" + (rf"\{sub}" if sub else ""), 0,
                            winreg.KEY_READ | view) as key:
            value, _ = winreg.QueryValueEx(key, "")
            return str(value).strip()
    except OSError:
        return None


def registered_name() -> Optional[str]:
    """Name of the first Unity Capture device, or None when no such camera is registered."""
    return _reg_value(CLSID)


def is_installed() -> bool:
    if registered_name() != CAMERA_NAME:
        return False
    dll = _reg_value(CLSID, sub="InprocServer32")
    return bool(dll) and os.path.exists(os.path.expandvars(dll))


def status() -> dict:
    name = registered_name()
    return {"supported": supported(), "installed": is_installed(), "name": name,
            "other": bool(name) and name != CAMERA_NAME}


# ------------------------------------------------------------------ elevated part
def _is_admin() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _regsvr(exe: Path, dll: Path, register: bool) -> int:
    args = [str(exe), "/s"]
    if register:
        args += [str(dll), f"/i:UnityCaptureName={CAMERA_NAME}"]
    else:
        args += ["/u", str(dll)]
    return subprocess.call(args, creationflags=_NO_WINDOW)


def _copy(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size and dst.read_bytes() == src.read_bytes():
            return
        try:
            dst.unlink()
        except OSError:          # in use by an app that has the camera open → move it out of the way
            old = dst.with_suffix(f".old{int(time.time())}")
            os.replace(dst, old)
    shutil.copy2(src, dst)


def _windir() -> Path:
    return Path(os.environ.get("SystemRoot") or r"C:\Windows")


def _do_install() -> Tuple[bool, str]:
    src = source_dir()
    for name in (DLL64, DLL32):
        if not (src / name).exists():
            return False, f"Не найден файл {name} в папке Mimiq — распакуйте архив Mimiq заново."
    dest = install_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for name in (DLL64, DLL32, "LICENSE.txt"):
        if (src / name).exists():
            _copy(src / name, dest / name)
    for old in dest.glob("*.old*"):
        try:
            old.unlink()
        except OSError:
            pass
    rc = _regsvr(_windir() / "System32" / "regsvr32.exe", dest / DLL64, True)
    if rc != 0:
        return False, f"Windows не зарегистрировала камеру (regsvr32, код {rc})."
    wow = _windir() / "SysWOW64" / "regsvr32.exe"
    if wow.exists():                       # 32-bit apps (some older messengers) see the camera too
        _regsvr(wow, dest / DLL32, True)
    if registered_name() != CAMERA_NAME:
        return False, "Камера зарегистрирована, но Windows не видит её имя — перезагрузите ПК и попробуйте снова."
    return True, f"Камера «{CAMERA_NAME}» установлена."


def _do_uninstall() -> Tuple[bool, str]:
    dest = install_dir()
    wow = _windir() / "SysWOW64" / "regsvr32.exe"
    if (dest / DLL32).exists() and wow.exists() and _reg_value(CLSID32, wow32=True) == CAMERA_NAME:
        _regsvr(wow, dest / DLL32, False)
    if (dest / DLL64).exists() and registered_name() == CAMERA_NAME:
        _regsvr(_windir() / "System32" / "regsvr32.exe", dest / DLL64, False)
    shutil.rmtree(dest, ignore_errors=True)
    if registered_name() == CAMERA_NAME:
        return False, "Не удалось удалить камеру — закройте приложения, которые её используют, и попробуйте снова."
    return True, f"Камера «{CAMERA_NAME}» удалена."


def _run_local(action: str) -> Tuple[bool, str]:
    try:
        return _do_install() if action == "install" else _do_uninstall()
    except Exception as exc:
        log.exception("vcam %s failed", action)
        return False, f"Ошибка: {exc}"


# ------------------------------------------------------------------ public API
def run(action: str) -> Tuple[bool, str]:
    """Install or uninstall the camera, asking for administrator rights (UAC) when needed. Blocking."""
    if action not in ("install", "uninstall"):
        raise ValueError(action)
    if not WIN:
        return False, "Mimiq Camera работает только в Windows."
    if _is_admin():
        return _run_local(action)
    import ctypes
    from ctypes import wintypes

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("fMask", ctypes.c_ulong), ("hwnd", wintypes.HWND),
                    ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
                    ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
                    ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
                    ("dwHotKey", wintypes.DWORD), ("hIconOrMonitor", wintypes.HANDLE),
                    ("hProcess", wintypes.HANDLE)]

    from .. import paths
    fd, result = tempfile.mkstemp(prefix="mimiq-vcam-", suffix=".json")
    os.close(fd)
    exe = Path(sys.executable)
    if exe.name.lower() == "python.exe" and exe.with_name("pythonw.exe").exists():
        exe = exe.with_name("pythonw.exe")         # no console window flashing up
    # The elevated process may start in System32, so the package root is passed explicitly.
    boot = "import sys; sys.path.insert(0, sys.argv[1]); from mimiq.io.vcam_setup import main; sys.exit(main(sys.argv[2:]))"
    params = subprocess.list2cmdline(["-c", boot, str(paths.app_root()), action, "--elevated", "--result", result])
    info = SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = 0x00000040 | 0x00000400            # SEE_MASK_NOCLOSEPROCESS | SEE_MASK_FLAG_NO_UI
    info.lpVerb = "runas"
    info.lpFile = str(exe)
    info.lpParameters = params
    info.lpDirectory = str(paths.app_root())
    info.nShow = 0
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        if not shell32.ShellExecuteExW(ctypes.byref(info)):
            err = ctypes.get_last_error()
            if err == 1223:
                return False, "Отменено: без прав администратора Windows не даёт добавить камеру."
            return False, f"Не удалось запросить права администратора (код {err})."
        kernel32.WaitForSingleObject(info.hProcess, 120_000)
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
        kernel32.CloseHandle(info.hProcess)
        try:
            data = json.loads(Path(result).read_text(encoding="utf-8"))
            return bool(data["ok"]), str(data["message"])
        except Exception:
            ok = is_installed() if action == "install" else not is_installed()
            return ok, ("Готово." if ok else f"Не получилось (код {code.value}).")
    finally:
        try:
            os.unlink(result)
        except OSError:
            pass


def main(argv) -> int:
    action = argv[0] if argv else "status"
    if action == "status":
        print(json.dumps(status(), ensure_ascii=False))
        return 0
    if action not in ("install", "uninstall"):
        print("usage: python -m mimiq.io.vcam_setup status|install|uninstall")
        return 2
    if "--elevated" in argv:
        ok, msg = _run_local(action)
        if "--result" in argv:
            i = argv.index("--result")
            try:
                Path(argv[i + 1]).write_text(json.dumps({"ok": ok, "message": msg}, ensure_ascii=False),
                                             encoding="utf-8")
            except Exception:
                pass
    else:
        ok, msg = run(action)
    try:
        print(msg)
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
