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

Occlusions (glasses, caps, headphones, bandanas, masks, a hand or a microphone)
-------------------------------------------------------------------------------
* The 68 points are fused with a per-face shape prior (`landmarks.ShapePrior`): points that disagree
  with the rest of the face are treated as hidden and replaced by the prior, so the mask does not
  follow the edge of the object in front of the face.
* Landmark lock: when the detector loses an established face (hand over the mouth, mask plus
  sunglasses), the 68-point network is run on the predicted position; if the shape still fits, the
  face keeps being tracked for up to `lock_seconds` without the detector.
* Misses inside the ROI are retried with local contrast (shadow of a visor, backlight), and the
  full-frame search adds a coarse pass for faces right in front of the camera.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import geometry as geo
from .analysis import Detection, FaceDetector, Landmarker68
from .filters import OneEuroFilter, smoothing_to_params
from .landmarks import ShapePrior, five_from_68, weighted_similarity


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
    lock_seconds: float = 2.0


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
    prior: ShapePrior = field(default_factory=ShapePrior)
    lm68: Optional[np.ndarray] = None       # smoothed 68 points (same filter as lm5) — for the mask view
    hidden: Optional[np.ndarray] = None     # 68 flags: point hidden by glasses / mask / hand …
    last_detected: float = 0.0              # last time the detector itself confirmed the face
    locked: bool = False                    # this frame was tracked by landmark lock

    @property
    def size(self) -> float:
        return float(max(self.box[2] - self.box[0], self.box[3] - self.box[1]))

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0])

    @property
    def status(self) -> str:
        if not self.visible_now:
            return "holding"
        if self.locked or (self.hidden is not None and float(np.mean(self.hidden)) > 0.2):
            return "covered"      # glasses / mask / hand: tracked, partly by the shape prior
        return "tracking"


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
        self.last_path = ""         # how the last followed face was found (diagnostics)
        # newest occlusion mask (XSeg) of every face, set by the pipeline: track id → (mask, frame→mask matrix, t)
        self.hints: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
        self._now = 0.0
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
        self.hints = {}
        self._last_t = None

    # ------------------------------------------------------------------
    def _new_filter(self) -> OneEuroFilter:
        mc, beta = smoothing_to_params(self.cfg.smoothing)
        return OneEuroFilter(min_cutoff=mc, beta=beta)

    def _landmarks(self, frame: np.ndarray, box: np.ndarray, angle: float, track: Optional[Track]):
        """Run the 68-point network and fuse it with the face's shape prior → (lm68, hidden, quality, score)."""
        t0 = time.perf_counter()
        try:
            lm68, score, conf = self.landmarker.detect_full(frame, box, angle)
        except Exception:
            return None
        finally:
            self.refine_ms += (time.perf_counter() - t0) * 1000
        if track is not None:
            conf = self._occlusion_conf(track, lm68, conf)
        prior = track.prior if track is not None else ShapePrior()
        fused, hidden, quality = prior.fuse(lm68, conf, update=False)
        return fused, hidden, quality, score

    def _occlusion_conf(self, track: Track, lm68: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """The landmark network also "sees" points through a hand or a mask. The occlusion mask of the
        previous frame says which of them are covered: their confidence drops, so the shape prior takes over."""
        hint = self.hints.get(track.id)
        if hint is None or self._now - hint[2] > 0.5:
            return conf
        mask, m, _ = hint
        # jaw points lie on the face border — probe a little inside the face
        probe = lm68.copy()
        probe[:17] += (lm68[30] - lm68[:17]) * 0.15
        q = geo.transform_points(probe, m)
        h, w = mask.shape[:2]
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
        inside = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        v = np.where(inside, mask[yi, xi], 1.0)
        return conf * np.clip((v - 0.15) / 0.45, 0.0, 1.0)

    def _refine(self, frame: np.ndarray, det: Detection, track: Track):
        """Landmarks for a detection → (lm5, lm68 or None, hidden or None)."""
        if self.landmarker is None or not self.cfg.refine_landmarks:
            return det.kps, None, None
        r = self._landmarks(frame, det.box, geo.roll_degrees(det.kps), track)
        if r is None:
            return det.kps, None, None
        lm68, hidden, quality, score = r
        if score < 0.45 or quality < 0.3:
            return det.kps, None, None
        five = five_from_68(lm68)
        # guard against a landmarker that drifted onto something else
        if np.linalg.norm(five - det.kps, axis=1).mean() > 0.12 * max(det.size, 1.0):
            return det.kps, None, None
        track.prior.fuse(lm68, np.where(hidden, 0.0, 1.0), update=True)
        return five, lm68, hidden

    def _set(self, track: Track, box: np.ndarray, score: float, raw: np.ndarray, lm68, hidden, t: float,
             detected: bool) -> None:
        size = float(max(box[2] - box[0], box[3] - box[1]))
        prev5 = track.lm5
        if self.cfg.smoothing > 0.01:
            track.lm5 = track.filter(raw, t, scale=size)
        else:
            track.lm5 = raw
        if lm68 is not None:
            # carry the 68 points along with the filtered five (same smoothing, no extra lag)
            track.lm68 = geo.transform_points(lm68, geo.umeyama(raw, track.lm5))
            track.hidden = hidden
        elif track.lm68 is not None and self.landmarker is not None and self.cfg.refine_landmarks:
            # detector-only frame: move the last shape along (keeps the face "established")
            track.lm68 = geo.transform_points(track.lm68, geo.umeyama(prev5, track.lm5))
            track.hidden = None
        else:
            track.lm68, track.hidden = None, None
        track.box = np.asarray(box, np.float64).copy()
        track.score = float(score)
        track.last_seen = t
        if detected:
            track.last_detected = t
        track.locked = not detected
        track.visible_now = True
        track.hits += 1

    def _apply(self, track: Track, frame: np.ndarray, det: Detection, t: float) -> None:
        raw, lm68, hidden = self._refine(frame, det, track)
        self._set(track, det.box, det.score, raw, lm68, hidden, t, True)

    @staticmethod
    def _box_like(track: Track, lm68: np.ndarray) -> np.ndarray:
        """Detector-style box for landmarks found without the detector (same scale as the track's box)."""
        five = five_from_68(lm68)
        shift = five.mean(0) - track.lm5.mean(0)
        spread_new = float(np.sqrt(((lm68 - lm68.mean(0)) ** 2).sum(1).mean()))
        spread_old = float(np.sqrt(((track.lm68 - track.lm68.mean(0)) ** 2).sum(1).mean()))
        k = float(np.clip(spread_new / max(spread_old, 1e-6), 0.8, 1.25))
        c = track.center + shift
        half = np.array([track.box[2] - track.box[0], track.box[3] - track.box[1]]) * k / 2.0
        return np.array([c[0] - half[0], c[1] - half[1], c[0] + half[0], c[1] + half[1]])

    def _follow(self, frame: np.ndarray, tr: Track, t: float) -> bool:
        """Update an existing track; False = not found in this frame."""
        det = self._roi_detect(frame, tr)
        self.last_path = "lost" if det is None else "detector"
        if self.landmarker is None or not self.cfg.refine_landmarks:
            if det is None:
                return False
            self._apply(tr, frame, det, t)
            return True
        size = max(tr.size, 1.0)
        established = tr.hits >= 4 and tr.lm68 is not None and t - tr.last_detected <= self.cfg.lock_seconds

        def jump(five: np.ndarray) -> float:
            return float(np.linalg.norm(five - tr.lm5, axis=1).mean()) / size

        # where the detector says the face is, with the face's last shape moved rigidly onto it (eyes count
        # most: a hand or a mask usually covers the lower half) — robust to a fooled detector
        rigid = None
        if det is not None and established:
            m = weighted_similarity(tr.lm5, det.kps, np.array([1.0, 1.0, 0.6, 0.35, 0.35]))
            rigid = geo.transform_points(tr.lm5, m), m
            if jump(rigid[0]) > 0.6:
                det, rigid = None, None          # not our face

        def near(five: np.ndarray, limit: float) -> bool:
            ref = rigid[0] if rigid is not None else tr.lm5
            return float(np.linalg.norm(five - ref, axis=1).mean()) / size <= limit

        cands = []
        if det is not None:
            r = self._landmarks(frame, det.box, geo.roll_degrees(det.kps), tr)
            if r is not None and r[3] >= 0.45 and r[2] >= 0.3:
                five = five_from_68(r[0])
                if np.linalg.norm(five - det.kps, axis=1).mean() <= 0.12 * max(det.size, 1.0) and \
                        (not established or near(five, 0.45)):
                    cands.append((r[2], det, r))
        if established and (not cands or cands[0][0] < 0.75):
            # landmark lock: look for the face where it was — the detector may be fooled by a hand or a mask
            r = self._landmarks(frame, tr.box, geo.roll_degrees(tr.lm5), tr)
            need = 0.3 if tr.id in self.hints else 0.45
            if r is not None and r[3] >= 0.5 and r[2] >= need and near(five_from_68(r[0]), 0.25):
                cands.append((r[2] + 1e-3, None, r))
        if cands:
            _q, d, (lm68, hidden, _quality, _score) = max(cands, key=lambda c: c[0])
            tr.prior.fuse(lm68, np.where(hidden, 0.0, 1.0), update=True)
            self.last_path = "landmarks" if d is not None else "lock"
            if d is not None:
                self._set(tr, d.box, d.score, five_from_68(lm68), lm68, hidden, t, True)
            else:
                self._set(tr, self._box_like(tr, lm68), max(self.cfg.keep_score, tr.score * 0.9),
                          five_from_68(lm68), lm68, hidden, t, False)
            return True
        if det is None:
            return False
        if rigid is None:
            self.last_path = "detector"
            self._set(tr, det.box, det.score, det.kps, None, None, t, True)
            return True
        # the face is there but its points can't be trusted (covered): follow the detector rigidly
        self.last_path = "covered"
        lm68 = geo.transform_points(tr.lm68, rigid[1]) if tr.lm68 is not None else None
        self._set(tr, det.box, det.score, rigid[0], lm68, None if lm68 is None else np.ones(68, bool), t, True)
        tr.locked = True
        return True

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
        self._now = t
        self._frame += 1
        self.refine_ms = 0.0

        for tr in self.tracks:
            tr.visible_now = False
            self._follow(frame, tr, t)

        missed = [tr for tr in self.tracks if not tr.visible_now]
        want_new = not self.tracks or (cfg.multi_face and len(self.tracks) < cfg.max_faces
                                       and self._frame % cfg.full_scan_interval == 0)
        if missed or want_new:
            thr = min(cfg.keep_score, cfg.acquire_score) if missed else cfg.acquire_score
            dets = self.detector.detect_all(frame, cfg.detector_size, thr, multi=cfg.multi_face)
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
                if tr.lm68 is not None:
                    tr.lm68 = tr.lm68 + shift
            if lost > cfg.hold_seconds:
                tr.alpha = max(0.0, 1.0 - (lost - cfg.hold_seconds) / max(cfg.fade_seconds, 1e-3))
            alive.append(tr)
        self.tracks = alive
        if self.hints:
            ids = {tr.id for tr in alive}
            for k in list(self.hints):
                if k not in ids:
                    self.hints.pop(k, None)
        if not cfg.multi_face and len(self.tracks) > 1:
            self.tracks.sort(key=lambda tr: (tr.visible_now, tr.hits), reverse=True)
            self.tracks = self.tracks[:1]
        return [tr for tr in self.tracks if tr.alpha > 0.001]
