"""Face tracker that keeps the swap locked on, even when the detector briefly misses.

Strategy
--------
* Every frame each track is re-detected inside a ROI around its last position.
  The ROI is rotated by the head's roll so tilted heads stay upright for the detector,
  and it runs at a small input size, so it is both more robust and cheaper than a
  full-frame pass.
* Full-frame detection only runs to acquire faces or recover a missed track.
* Hysteresis: a high score is required to start a track, a low one to keep it.
* When a face is momentarily lost (motion blur, hand, strong turn) the track is held
  with damped motion for `hold` seconds and then faded out — the mask never "blinks".
* Landmarks go through a One Euro filter (steady at rest, no lag when moving).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import geometry as geo
from .analysis import Detection, FaceDetector, Landmarker68
from .filters import OneEuroFilter, smoothing_to_params


@dataclass
class TrackerConfig:
    detector_size: int = 640
    roi_size: int = 320
    acquire_score: float = 0.5
    keep_score: float = 0.3
    hold_seconds: float = 0.7
    fade_seconds: float = 0.25
    fade_in_seconds: float = 0.12
    smoothing: float = 0.6
    multi_face: bool = False
    max_faces: int = 4
    refine_landmarks: bool = True
    full_scan_interval: int = 12


@dataclass
class Track:
    id: int
    lm5: np.ndarray
    box: np.ndarray
    score: float
    last_seen: float
    filter: OneEuroFilter
    alpha: float = 0.0
    visible_now: bool = True
    hits: int = 1
    state: Dict[str, object] = field(default_factory=dict)

    @property
    def size(self) -> float:
        return float(max(self.box[2] - self.box[0], self.box[3] - self.box[1]))

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0])

    @property
    def status(self) -> str:
        return "tracking" if self.visible_now else "holding"


class FaceTracker:
    def __init__(self, detector: FaceDetector, landmarker: Optional[Landmarker68] = None,
                 config: Optional[TrackerConfig] = None):
        self.detector = detector
        self.landmarker = landmarker
        self.cfg = config or TrackerConfig()
        self.tracks: List[Track] = []
        self._ids = itertools.count(1)
        self._frame = 0
        self._last_t: Optional[float] = None
        self.refine_ms = 0.0        # time spent in the 68-point landmarker during the last update()

    def configure(self, cfg: TrackerConfig) -> None:
        smoothing_changed = cfg.smoothing != self.cfg.smoothing
        self.cfg = cfg
        if smoothing_changed:
            mc, beta = smoothing_to_params(cfg.smoothing)
            for t in self.tracks:
                t.filter.min_cutoff, t.filter.beta = mc, beta
        if not cfg.multi_face and len(self.tracks) > 1:
            self.tracks = self.tracks[:1]

    def reset(self) -> None:
        self.tracks.clear()
        self._last_t = None

    # ------------------------------------------------------------------
    def _new_filter(self) -> OneEuroFilter:
        mc, beta = smoothing_to_params(self.cfg.smoothing)
        return OneEuroFilter(min_cutoff=mc, beta=beta)

    def _refine(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        if self.landmarker is None or not self.cfg.refine_landmarks:
            return det.kps
        t0 = time.perf_counter()
        try:
            lm68, score = self.landmarker.detect(frame, det.box, geo.roll_degrees(det.kps))
        except Exception:
            return det.kps
        finally:
            self.refine_ms += (time.perf_counter() - t0) * 1000
        if score < 0.45:
            return det.kps
        five = Landmarker68.to_five(lm68)
        # guard against a landmarker that drifted onto something else
        if np.linalg.norm(five - det.kps, axis=1).mean() > 0.12 * max(det.size, 1.0):
            return det.kps
        return five

    def _apply(self, track: Track, frame: np.ndarray, det: Detection, t: float) -> None:
        raw = self._refine(frame, det)
        if self.cfg.smoothing > 0.01:
            track.lm5 = track.filter(raw, t, scale=det.size)
        else:
            track.lm5 = raw
        track.box = det.box.copy()
        track.score = det.score
        track.last_seen = t
        track.visible_now = True
        track.hits += 1

    def _roi_detect(self, frame: np.ndarray, track: Track) -> Optional[Detection]:
        roll = geo.roll_degrees(track.lm5)
        side = track.size * 2.1
        dets = self.detector.detect_roi(frame, track.center, side, roll, self.cfg.roi_size, self.cfg.keep_score)
        if not dets:
            return None
        # best = high score, close to the predicted centre
        def cost(d: Detection) -> float:
            dist = np.linalg.norm(d.center - track.center) / max(track.size, 1.0)
            return dist - d.score
        best = min(dets, key=cost)
        if np.linalg.norm(best.center - track.center) > 0.9 * track.size:
            return None
        return best

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, t: float) -> List[Track]:
        cfg = self.cfg
        dt = 0.033 if self._last_t is None else float(np.clip(t - self._last_t, 1e-3, 0.5))
        self._last_t = t
        self._frame += 1
        self.refine_ms = 0.0

        for tr in self.tracks:
            tr.visible_now = False
            det = self._roi_detect(frame, tr)
            if det is not None:
                self._apply(tr, frame, det, t)

        missed = [tr for tr in self.tracks if not tr.visible_now]
        want_new = not self.tracks or (cfg.multi_face and len(self.tracks) < cfg.max_faces
                                       and self._frame % cfg.full_scan_interval == 0)
        if missed or want_new:
            thr = min(cfg.keep_score, cfg.acquire_score) if missed else cfg.acquire_score
            dets = self.detector.detect(frame, cfg.detector_size, thr)
            used = set()
            for tr in missed:  # re-associate missed tracks
                best, best_cost = None, 1e9
                for j, d in enumerate(dets):
                    if j in used:
                        continue
                    overlap = geo.iou(tr.box, d.box)
                    dist = np.linalg.norm(d.center - tr.center) / max(tr.size, 1.0)
                    if overlap < 0.15 and dist > 0.8:
                        continue
                    c = dist - overlap
                    if c < best_cost:
                        best, best_cost = j, c
                if best is not None:
                    used.add(best)
                    self._apply(tr, frame, dets[best], t)
            if want_new or (not cfg.multi_face and not self.tracks):
                fresh = [d for j, d in enumerate(dets) if j not in used and d.score >= cfg.acquire_score
                         and all(geo.iou(d.box, tr.box) < 0.3 for tr in self.tracks)]
                if not cfg.multi_face:
                    fresh = [] if self.tracks else sorted(fresh, key=lambda d: (d.size, d.score), reverse=True)[:1]
                for d in fresh[: max(0, cfg.max_faces - len(self.tracks))]:
                    tr = Track(next(self._ids), d.kps.copy(), d.box.copy(), d.score, t, self._new_filter())
                    self._apply(tr, frame, d, t)
                    tr.hits = 1
                    self.tracks.append(tr)

        alive = []
        for tr in self.tracks:
            if tr.visible_now:
                step = dt / max(cfg.fade_in_seconds, 1e-3)
                tr.alpha = min(1.0, tr.alpha + step)
                alive.append(tr)
                continue
            lost = t - tr.last_seen
            if lost > cfg.hold_seconds + cfg.fade_seconds:
                continue  # drop
            vel = tr.filter.velocity
            if vel is not None and lost < 0.25:  # short, damped extrapolation while holding
                damp = max(0.0, 1.0 - lost / 0.25)
                shift = vel.reshape(-1, 2).mean(0) * dt * damp
                tr.lm5 = tr.lm5 + shift
                tr.box = tr.box + np.tile(shift, 2)
            if lost > cfg.hold_seconds:
                tr.alpha = max(0.0, 1.0 - (lost - cfg.hold_seconds) / max(cfg.fade_seconds, 1e-3))
            alive.append(tr)
        self.tracks = alive
        if not cfg.multi_face and len(self.tracks) > 1:
            self.tracks.sort(key=lambda tr: (tr.visible_now, tr.hits), reverse=True)
            self.tracks = self.tracks[:1]
        return [tr for tr in self.tracks if tr.alpha > 0.001]
