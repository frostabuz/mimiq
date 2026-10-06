"""Image I/O helpers (Unicode paths, EXIF rotation, HEIC) and Qt conversions."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def read_image(path: str | Path) -> np.ndarray:
    """Read any photo as BGR uint8, honouring EXIF orientation (iPhone photos) and HEIC."""
    path = Path(path)
    try:
        from PIL import Image, ImageOps
        if path.suffix.lower() in (".heic", ".heif"):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Для HEIC установите пакет pillow-heif") from exc
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        data = np.fromfile(str(path), dtype=np.uint8)  # works with Cyrillic paths on Windows
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Не удалось открыть {path.name}")
        return img


def decode_file(path: str | Path):
    """cv2.imread replacement that works with Cyrillic/Unicode paths on Windows; None on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def ascii_safe_dir(folder: Path) -> Path:
    """Directory OpenCV's VideoWriter can open.

    On Windows OpenCV cannot open files whose path contains non-ASCII characters (for example a
    Cyrillic user name). Prefer the 8.3 short alias of the folder, otherwise a public temp folder;
    the finished file is moved to `folder` afterwards."""
    text = str(folder)
    if sys.platform != "win32" or text.isascii():
        return folder
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(text, buf, 32768) and buf.value.isascii():
            return Path(buf.value)
    except Exception:
        pass
    for base in (os.environ.get("PUBLIC"), os.environ.get("SystemDrive", "C:") + "\\Users\\Public",
                 tempfile.gettempdir(), os.environ.get("SystemDrive", "C:") + "\\"):
        if base and base.isascii():
            cand = Path(base) / "Mimiq-temp"
            try:
                cand.mkdir(parents=True, exist_ok=True)
                return cand
            except OSError:
                continue
    return folder


def write_image(path: str | Path, img: np.ndarray) -> None:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("encode failed")
    buf.tofile(str(path))


def to_qimage(img: np.ndarray):
    from PySide6.QtGui import QImage
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    if img.shape[2] == 4:
        return QImage(img.data, w, h, 4 * w, QImage.Format.Format_ARGB32).copy()
    return QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


def qimage_to_bgra(qimg) -> np.ndarray:
    from PySide6.QtGui import QImage
    q = qimg.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = q.width(), q.height()
    ptr = q.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=q.sizeInBytes()).reshape(h, q.bytesPerLine() // 4, 4)[:, :w]
    return arr.copy()  # B, G, R, A on little-endian


def overlay_bgra(frame: np.ndarray, badge: np.ndarray, x: int, y: int) -> None:
    h, w = badge.shape[:2]
    fh, fw = frame.shape[:2]
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        return
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    a = badge[..., 3:4].astype(np.float32) / 255.0
    frame[y:y + h, x:x + w] = (roi * (1 - a) + badge[..., :3].astype(np.float32) * a).astype(np.uint8)
