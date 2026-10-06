"""Download Mimiq neural network models (resumable, size-verified).

    python tools/download_models.py            → base set (~680 MB)
    python tools/download_models.py --all      → every model in the registry
    python tools/download_models.py --list     → show what is installed
    python tools/download_models.py hyperswap_1a_256 gfpgan_1.4
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mimiq import paths  # noqa: E402
from mimiq.core import models  # noqa: E402


class Bar:
    def __init__(self, keys):
        self.total = sum(models.REGISTRY[k].size for k in keys)
        self.base = 0
        self.key = None
        self.last = 0.0

    def __call__(self, key, done, size, speed):
        if key != self.key:
            if self.key is not None:
                self.base += models.REGISTRY[self.key].size
                print()
            self.key = key
        now = time.monotonic()
        if now - self.last < 0.15 and done < size:
            return
        self.last = now
        frac = (self.base + done) / max(self.total, 1)
        width = 28
        fill = int(frac * width)
        title = models.REGISTRY[key].title[:26]
        sys.stdout.write(f"\r  [{'█' * fill}{'·' * (width - fill)}] {frac * 100:5.1f}%  {title:<26} "
                         f"{done / 1e6:6.0f}/{size / 1e6:.0f} МБ  {speed / 1e6:5.1f} МБ/с ")
        sys.stdout.flush()


def main(argv) -> int:
    if "--list" in argv:
        print(f"Папка моделей: {paths.models_dir()}")
        for spec in models.REGISTRY.values():
            mark = "✓" if models.is_installed(spec.key) else " "
            print(f"  [{mark}] {spec.key:<26} {spec.size_mb:>4} МБ  {spec.title}")
        return 0
    if "--all" in argv:
        keys = list(models.REGISTRY)
    else:
        keys = [a for a in argv if not a.startswith("-")] or list(models.BASE_MODELS)
    unknown = [k for k in keys if k not in models.REGISTRY]
    if unknown:
        print("Неизвестные модели:", ", ".join(unknown))
        return 2
    need = models.missing(keys)
    print(f"  Папка моделей: {paths.models_dir()}")
    if not need:
        print("  Все нужные модели уже скачаны ✓")
        return 0
    mb = sum(models.REGISTRY[k].size for k in need) / 1_048_576
    print(f"  Нужно скачать {len(need)} файл(ов), ~{mb:.0f} МБ. Загрузку можно прервать — она продолжится с места остановки.")
    bar = Bar(need)
    try:
        for key in need:
            models.download(key, bar)
    except KeyboardInterrupt:
        print("\n  Остановлено. Запустите снова, чтобы докачать.")
        return 130
    except Exception as exc:
        print(f"\n  Ошибка: {exc}")
        print("  Проверьте интернет (Hugging Face / GitHub должны открываться) и запустите ещё раз.")
        return 1
    print("\n  Модели готовы ✓")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
