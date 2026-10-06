"""Update check: asks GitHub Releases whether a newer Mimiq exists. Only the version is requested —
nothing about the user, their video or photos is sent."""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

from . import __version__

log = logging.getLogger("mimiq.updates")

REPO = "frostabuz/mimiq"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"


@dataclass
class Release:
    version: str
    url: str              # release page
    download: str         # Mimiq-X.Y.Z-windows.zip (or the release page when there is no asset)
    notes: str            # "Что нового" section as plain text


def parse_version(text: str) -> Tuple[int, ...]:
    """'v1.2.10' → (1, 2, 10); junk → ()."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", text or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def is_newer(remote: str, local: str = __version__) -> bool:
    r, l = parse_version(remote), parse_version(local)
    if not r or not l:
        return False
    n = max(len(r), len(l))
    return r + (0,) * (n - len(r)) > l + (0,) * (n - len(l))


def whats_new(body: str, limit: int = 900) -> str:
    """First «Что нового» section of the release notes, without Markdown."""
    lines = (body or "").replace("\r\n", "\n").split("\n")
    out, inside = [], False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = "нового" in line.lower() or "исправлено" in line.lower()
            continue
        if inside:
            out.append(line)
    text = "\n".join(out).strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^- ", "• ", text, flags=re.M)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def fetch_latest(timeout: float = 8.0) -> Release:
    req = urllib.request.Request(API_LATEST, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"Mimiq/{__version__}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tag = str(data.get("tag_name") or "")
    page = str(data.get("html_url") or RELEASES_PAGE)
    download = page
    for asset in data.get("assets") or []:
        name = str(asset.get("name") or "")
        if name.lower().endswith("-windows.zip"):
            download = str(asset.get("browser_download_url") or page)
            break
    return Release(version=tag.lstrip("vV"), url=page, download=download, notes=whats_new(str(data.get("body") or "")))


def check(timeout: float = 8.0) -> Tuple[Optional[Release], str]:
    """(release, "") when a newer version exists, (None, "") when up to date, (None, error) on failure."""
    try:
        rel = fetch_latest(timeout)
    except Exception as exc:
        log.info("update check failed: %s", exc)
        return None, "Не удалось проверить обновления — нет связи с GitHub."
    if is_newer(rel.version):
        log.info("update available: %s (installed %s)", rel.version, __version__)
        return rel, ""
    return None, ""
