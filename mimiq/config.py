"""User settings with JSON persistence and quality presets."""
from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

from . import paths

log = logging.getLogger("mimiq.config")

# 2: Pixel Boost 128 by default (4× less swapper work), identity strength, auto quality, fluid video
SETTINGS_VERSION = 2


@dataclass
class Settings:
    # ---- input ---------------------------------------------------------------------
    source_kind: str = "link"            # link | device | url
    device_index: int = 0
    device_name: str = ""
    device_backend: str = "dshow"        # dshow | msmf | auto
    stream_url: str = ""
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 30
    rotate: int = 0                      # 0 / 90 / 180 / 270
    mirror: bool = False
    # ---- Mimiq Link (iPhone over Wi-Fi) -------------------------------------------
    link_port: int = 8443
    link_token: str = field(default_factory=lambda: secrets.token_urlsafe(6))
    link_host: str = ""                  # preferred LAN IP ("" = auto)
    link_resolution: str = "1280x720"
    link_fps: int = 30
    link_quality: float = 0.82
    link_camera: str = "user"            # user (front) | environment (back)
    # ---- swap ------------------------------------------------------------------------
    swap_enabled: bool = True
    swapper_model: str = "inswapper_128_fp16"
    pixel_boost: int = 128               # 128 / 256 / 512 (each step = 4× more swapper work)
    identity_strength: int = 60          # 0..100 — pushes the result away from the camera face, towards the photo
    swap_passes: int = 1                 # 1..3 — re-run the swapper on its own output (more likeness, slower)
    multi_face: bool = False
    # ---- enhancer --------------------------------------------------------------------
    enhancer_model: str = "gpen_bfr_256"  # "" = off
    enhancer_blend: float = 0.7
    enhancer_fidelity: float = 0.75       # CodeFormer only
    # ---- mask ------------------------------------------------------------------------
    mask_occlusion: bool = True
    occluder_model: str = "xseg_1"
    mask_region: bool = False
    parser_model: str = "bisenet_resnet_18"
    mask_blur: float = 0.3
    mask_padding_top: int = 0
    mask_padding_bottom: int = 0
    mask_padding_sides: int = 0
    mask_temporal: float = 0.35           # temporal smoothing of the mask 0..0.9
    color_match: float = 0.35             # 0 = off
    # ---- tracking --------------------------------------------------------------------
    detector_model: str = "retinaface_10g"
    detector_size: int = 640
    detector_score: float = 0.5
    landmark_refine: bool = True
    smoothing: float = 0.5
    hold_ms: int = 700
    fade_ms: int = 250
    # ---- output ----------------------------------------------------------------------
    vcam_enabled: bool = True
    vcam_backend: str = "auto"           # auto (Mimiq Camera, else OBS) | mimiq | obs
    output_width: int = 1280
    output_height: int = 720
    output_fps: int = 30
    output_fit: str = "crop"             # crop | fit
    watermark: bool = True
    # ---- performance -----------------------------------------------------------------
    execution_provider: str = "auto"
    gpu_device: int = 0
    auto_quality: bool = True            # lower the load automatically when the PC can't keep up
    fluid_video: bool = True             # video never waits for the face: re-use the newest face between renders
    # ---- ui / misc ---------------------------------------------------------------------
    preview_mode: str = "result"         # result | split | original | mask
    show_hud: bool = True
    active_face: str = ""
    preset: str = "balanced"
    consent_accepted: bool = False
    first_run_done: bool = False
    check_updates: bool = True           # ask GitHub Releases for a newer version at start-up
    settings_version: int = SETTINGS_VERSION

    # ------------------------------------------------------------------
    def copy(self) -> "Settings":
        return Settings(**asdict(self))

    def update(self, **kw: Any) -> "Settings":
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def save(self, path: Path | None = None) -> None:
        path = path or paths.settings_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or paths.settings_path()
        s = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                names = {f.name: f for f in fields(cls)}
                for k, v in data.items():
                    if k in names:
                        default = getattr(s, k)
                        if isinstance(default, bool):
                            v = bool(v)
                        elif isinstance(default, int) and not isinstance(default, bool):
                            v = int(v)
                        elif isinstance(default, float):
                            v = float(v)
                        setattr(s, k, v)
                if int(data.get("settings_version", 1)) < SETTINGS_VERSION:
                    s.migrate(int(data.get("settings_version", 1)))
                if s.vcam_backend not in ("auto", "mimiq", "obs"):
                    s.vcam_backend = "mimiq" if s.vcam_backend == "unitycapture" else "auto"
            except Exception as exc:
                log.warning("settings unreadable, using defaults: %s", exc)
        return s

    def migrate(self, old: int) -> None:
        """Bring settings saved by an older Mimiq up to date (new, faster defaults)."""
        if old < 2:
            if self.preset in PRESETS:
                self.update(**PRESETS[self.preset])
            elif self.pixel_boost > 128 and self.swapper_model.startswith("inswapper"):
                self.pixel_boost = 128
            if abs(self.smoothing - 0.6) < 1e-6:
                self.smoothing = 0.5
            log.info("settings migrated from v%d to v%d", old, SETTINGS_VERSION)
        self.settings_version = SETTINGS_VERSION

    def required_models(self) -> List[str]:
        keys = [self.detector_model, "arcface_w600k_r50", self.swapper_model]
        if self.landmark_refine:
            keys.append("2dfan4")
        if self.enhancer_model:
            keys.append(self.enhancer_model)
        if self.mask_occlusion:
            keys.append(self.occluder_model)
        if self.mask_region:
            keys.append(self.parser_model)
        return keys


# The swapper (InSwapper, ~175 GMAC per 128 px tile) dominates the cost: Pixel Boost 256 = 4 tiles,
# 512 = 16 tiles, every extra pass doubles it. Presets are ceilings — with auto_quality on, Mimiq steps
# down on its own when the PC can't keep up and climbs back when there is headroom.
PRESETS: Dict[str, Dict[str, Any]] = {
    "speed": dict(swapper_model="inswapper_128_fp16", pixel_boost=128, swap_passes=1, enhancer_model="",
                  mask_occlusion=True, mask_region=False, landmark_refine=False, detector_size=480, color_match=0.25),
    "balanced": dict(swapper_model="inswapper_128_fp16", pixel_boost=128, swap_passes=1,
                     enhancer_model="gpen_bfr_256", enhancer_blend=0.7, mask_occlusion=True, mask_region=False,
                     landmark_refine=True, detector_size=640, color_match=0.35),
    "quality": dict(swapper_model="inswapper_128_fp16", pixel_boost=256, swap_passes=1,
                    enhancer_model="gpen_bfr_512", enhancer_blend=0.7, mask_occlusion=True, mask_region=True,
                    landmark_refine=True, detector_size=640, color_match=0.4),
    "ultra": dict(swapper_model="inswapper_128_fp16", pixel_boost=256, swap_passes=2,
                  enhancer_model="gpen_bfr_512", enhancer_blend=0.75, mask_occlusion=True, mask_region=True,
                  landmark_refine=True, detector_size=640, color_match=0.4),
}

PRESET_LABELS = {"speed": "Скорость", "balanced": "Баланс", "quality": "Качество", "ultra": "Ультра"}
