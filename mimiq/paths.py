"""Filesystem locations used by Mimiq."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Mimiq"


def app_root() -> Path:
    """Folder that contains the `mimiq` package (or the frozen executable)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def package_dir() -> Path:
    """Read-only resources bundled with the package (web app, assets)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / "mimiq"
    return Path(__file__).resolve().parent


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return _ensure(base / APP_NAME)


def _is_writable(path: Path) -> bool:
    try:
        _ensure(path)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def models_dir() -> Path:
    env = os.environ.get("MIMIQ_MODELS_DIR")
    if env:
        return _ensure(Path(env))
    portable = app_root() / "models"
    if _is_writable(portable):
        return portable
    return _ensure(user_data_dir() / "models")


def faces_dir() -> Path:
    return _ensure(user_data_dir() / "faces")


def backgrounds_dir() -> Path:
    return _ensure(user_data_dir() / "backgrounds")


def certs_dir() -> Path:
    return _ensure(user_data_dir() / "certs")


def cache_dir() -> Path:
    return _ensure(user_data_dir() / "cache")


def logs_dir() -> Path:
    return _ensure(user_data_dir() / "logs")


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def captures_dir() -> Path:
    pictures = Path.home() / "Pictures"
    if not pictures.exists():
        pictures = Path.home()
    return _ensure(pictures / APP_NAME)
