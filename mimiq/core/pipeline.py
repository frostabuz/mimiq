"""Two-stage face swap pipeline.

Stage A · analyse  (camera thread): track faces → align crops (→ occlusion masks, identity of the camera face).
Stage B · render   (render thread): swap → masks → colour → paste back → enhance.

The engine runs the stages on two threads, so stage A of frame N+1 overlaps stage B of frame N and the frame
rate is set by the slower stage instead of the sum of both.

Fluid video: when stage B can't keep up with the camera, the engine does not wait for it. Every camera frame
is published right after stage A by `compose()`, which re-uses the newest rendered face: the face is stored
"un-blended" in face-aligned space and pasted with the *current* landmarks, so head motion stays at the full
camera frame rate and only the expression updates at the speed of the PC.

Each stage only touches its own models and state; settings travel with every job, so a slider change never
tears a frame.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import paths
from ..config import Settings
from . import geometry as geo
from . import models
from .analysis import (FaceDetector, FaceEnhancer, FaceParser, FaceRecognizer, FaceSwapper, Landmarker68, Occluder,
                       apply_color_transfer, box_mask, lab_stats)
from .identity import Identity, IdentityBuilder
from .tracker import FaceTracker, TrackerConfig

log = logging.getLogger("mimiq.pipeline")


@dataclass
class FaceInfo:
    id: int
    box: List[float]
    status: str
    alpha: float
    score: float


@dataclass
class FaceJob:
    tid: int
    lm5: np.ndarray
    alpha: float
    visible: bool
    matrix: np.ndarray                    # frame → swapper crop
    crop: np.ndarray                      # uint8 crop, pixel_boost × pixel_boost
    swapper: str
    occlusion: Optional[np.ndarray] = None
    latent: Optional[np.ndarray] = None   # None → computed by stage B
    need_occlusion: bool = False          # occlusion mask still to be computed (by stage B)


@dataclass
class FrameJob:
    frame: np.ndarray
    t: float
    settings: Settings
    faces: List[FaceInfo] = field(default_factory=list)
    lite: List[Tuple[int, np.ndarray, float]] = field(default_factory=list)   # (track id, lm5, alpha)
    jobs: List[FaceJob] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    want_mask: bool = False
    seq: int = 0                # set by the engine
    ts: float = 0.0             # capture time (set by the engine)


@dataclass
class FrameResult:
    frame: np.ndarray
    faces: List[FaceInfo] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    mask: Optional[np.ndarray] = None   # full-frame mask (debug view)
    swapped: bool = False               # a face was rendered by stage B for this very frame


@dataclass
class FaceCache:
    """Newest rendered face of a track, in face-aligned space, *before* blending (see compose)."""
    tid: int
    face: np.ndarray            # uint8 size × size × 3
    mask: np.ndarray            # float32 size × size (without the track's fade alpha)
    template: str
    size: int
    t: float


def _tracker_config(s: Settings) -> TrackerConfig:
    return TrackerConfig(detector_size=s.detector_size, acquire_score=s.detector_score,
                         keep_score=max(0.2, s.detector_score - 0.22), hold_seconds=s.hold_ms / 1000.0,
                         fade_seconds=s.fade_ms / 1000.0, smoothing=s.smoothing, multi_face=s.multi_face,
                         refine_landmarks=s.landmark_refine)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-9)


class LikenessProbe:
    """How close the output face is to the chosen photo (ArcFace cosine), measured about once a second
    on a short-lived background thread so it never delays a frame."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.value: Optional[float] = None
        self._busy = False
        self._t = -1e9
        self._ema: Optional[float] = None

    def reset(self) -> None:
        self.value, self._ema, self._t = None, None, -1e9

    def due(self, t: float) -> bool:
        return not self._busy and t - self._t >= self.interval

    def submit(self, recognizer: FaceRecognizer, crop112: np.ndarray, identity: np.ndarray, t: float) -> None:
        self._busy, self._t = True, t

        def run():
            try:
                sim = float(_unit(recognizer.embed_crop(crop112)) @ _unit(identity))
                self._ema = sim if self._ema is None else 0.6 * self._ema + 0.4 * sim
                self.value = self._ema
            except Exception as exc:  # metrics must never disturb the video
                log.debug("likeness probe failed: %s", exc)
            finally:
                self._busy = False
        threading.Thread(target=run, daemon=True, name="mimiq-likeness").start()


class Pipeline:
    def __init__(self, hub: models.SessionHub):
        self.hub = hub
        self.lock_a = threading.RLock()     # tracker, detector, landmarker (+ occluder/recogniser in direct mode)
        self.lock_b = threading.RLock()     # swapper, parser, enhancer, render state
        self.settings = Settings()
        self.detector: Optional[FaceDetector] = None
        self.landmarker: Optional[Landmarker68] = None
        self.recognizer: Optional[FaceRecognizer] = None
        self.swapper: Optional[FaceSwapper] = None
        self.enhancer: Optional[FaceEnhancer] = None
        self.occluder: Optional[Occluder] = None
        self.parser: Optional[FaceParser] = None
        self.tracker: Optional[FaceTracker] = None
        self.identity: Optional[Identity] = None
        self._identity_gen = 0
        self._light: Optional[Settings] = None
        self._reset = False
        self._render_state: Dict[int, dict] = {}
        self._cache: Dict[int, FaceCache] = {}
        self._cache_latest: Optional[FaceCache] = None
        self.want_mask = False
        self.occ_interval = 1               # 2 = occlusion mask every other job (auto quality)
        self.likeness = LikenessProbe()

    # ------------------------------------------------------------------ configuration
    def configure(self, s: Settings) -> None:
        """(Re)build components. Models must already be downloaded. Blocks both stages briefly."""
        with self.lock_a, self.lock_b:
            self.hub.configure(s.execution_provider, s.gpu_device)
            self.detector = FaceDetector(self.hub.get(s.detector_model))
            self.recognizer = FaceRecognizer(self.hub.get("arcface_w600k_r50"))
            self.landmarker = Landmarker68(self.hub.get("2dfan4")) if s.landmark_refine else None
            if self.swapper is None or self.swapper.key != s.swapper_model or \
                    self.swapper.session is not self.hub.get(s.swapper_model):
                spec = models.REGISTRY[s.swapper_model]
                self.swapper = FaceSwapper(s.swapper_model, self.hub.get(s.swapper_model), spec.path,
                                           paths.cache_dir())
            self.enhancer = FaceEnhancer(s.enhancer_model, self.hub.get(s.enhancer_model)) if s.enhancer_model else None
            self.occluder = Occluder(self.hub.get(s.occluder_model)) if s.mask_occlusion else None
            self.parser = FaceParser(self.hub.get(s.parser_model)) if s.mask_region else None
            if self.tracker is None:
                self.tracker = FaceTracker(self.detector, self.landmarker, _tracker_config(s))
            else:
                self.tracker.detector, self.tracker.landmarker = self.detector, self.landmarker
                self.tracker.configure(_tracker_config(s))
                for tr in self.tracker.tracks:
                    tr.state.clear()
            self.settings = s.copy()
            self._light = None
            self._render_state.clear()
            self.hub.release(s.required_models())
            log.info("pipeline configured: %s on %s", s.swapper_model, self.hub.provider)

    def update_light(self, s: Settings) -> None:
        """Apply settings that need no new models (sliders, auto quality). Never blocks: picked up by stage A
        at the start of the next frame and carried to stage B inside the job."""
        self._light = s.copy()

    def _take_light(self) -> None:
        s = self._light
        if s is None:
            return
        self._light = None
        self.settings = s
        if self.tracker is not None:
            self.tracker.configure(_tracker_config(s))

    def builder(self) -> IdentityBuilder:
        if self.detector is None or self.recognizer is None:
            raise RuntimeError("Модели ещё загружаются")
        return IdentityBuilder(self.detector, self.recognizer, self.landmarker)

    def set_identity(self, ident: Optional[Identity]) -> None:
        self.identity = ident
        self._identity_gen += 1
        self.likeness.reset()
        if ident is None:
            self.clear_cache()

    def clear_cache(self) -> None:
        self._cache = {}
        self._cache_latest = None

    def reset_tracking(self) -> None:
        self._reset = True

    @property
    def ready(self) -> bool:
        return self.tracker is not None

    def warmup(self) -> None:
        """Run every model once so the first real frame is not slow (CUDA/cuDNN autotune)."""
        dummy = np.full((720, 1280, 3), 127, np.uint8)
        with self.lock_a, self.lock_b:
            try:
                self.detector.detect(dummy, self.settings.detector_size, 0.9)
                self.detector.detect(dummy[:320, :320], self.tracker.cfg.roi_size, 0.9)
                self.recognizer.embed_crop(np.zeros((112, 112, 3), np.uint8))
                if self.swapper is not None:
                    n = self.swapper.size
                    self.swapper.swap(np.zeros((n, n, 3), np.uint8), np.zeros((1, 512), np.float32))
                if self.enhancer is not None:
                    self.enhancer.enhance(np.zeros((self.enhancer.size, self.enhancer.size, 3), np.uint8))
                if self.occluder is not None:
                    self.occluder.mask(np.zeros((256, 256, 3), np.uint8))
                if self.parser is not None:
                    self.parser.mask(np.zeros((256, 256, 3), np.uint8))
                if self.landmarker is not None:
                    self.landmarker.detect(dummy, np.array([500, 200, 780, 520], np.float64))
            except Exception as exc:  # warm-up must never break start-up
                log.warning("warm-up failed: %s", exc)

    def swap_active(self, s: Settings) -> bool:
        return bool(s.swap_enabled and self.identity is not None and self.swapper is not None)

    # ------------------------------------------------------------------ stage A
    def analyze(self, frame: np.ndarray, t: Optional[float] = None, prepare: bool = True,
                heavy: bool = True) -> FrameJob:
        """Track faces. With `prepare`, also cut the crops stage B needs; with `heavy`, compute the occlusion
        masks and the swap conditioning here too (direct mode) instead of leaving them to stage B."""
        t = time.monotonic() if t is None else t
        with self.lock_a:
            self._take_light()
            if self._reset:
                self._reset = False
                if self.tracker is not None:
                    self.tracker.reset()
            s = self.settings
            job = FrameJob(frame=frame, t=t, settings=s, want_mask=self.want_mask)
            if self.tracker is None:
                return job
            t0 = time.perf_counter()
            tracks = self.tracker.update(frame, t)
            t1 = time.perf_counter()
            job.timings["track"] = (t1 - t0) * 1000
            job.timings["landmarks"] = self.tracker.refine_ms
            job.faces = [FaceInfo(tr.id, [float(v) for v in tr.box], tr.status, float(tr.alpha), float(tr.score))
                         for tr in tracks]
            job.lite = [(tr.id, tr.lm5.copy(), float(tr.alpha)) for tr in tracks]
            sw, ident = self.swapper, self.identity
            if not (prepare and tracks and self.swap_active(s)):
                job.timings["analyze"] = (time.perf_counter() - t0) * 1000
                return job
            boost = max(sw.size, int(s.pixel_boost) // sw.size * sw.size)
            occ_ms = 0.0
            for tr in tracks:
                m = geo.face_matrix(tr.lm5, sw.template, boost)
                crop = geo.warp_face(frame, m, boost)
                fj = FaceJob(tr.id, tr.lm5.copy(), float(tr.alpha), bool(tr.visible_now), m, crop, sw.key)
                if heavy:
                    o0 = time.perf_counter()
                    fj.occlusion = self._occlusion(tr.state, crop, s)
                    occ_ms += (time.perf_counter() - o0) * 1000
                    fj.latent = self._latent(tr.state, frame, tr.lm5, tr.visible_now, s, t, sw, ident)
                else:
                    fj.need_occlusion = self.occluder is not None and s.mask_occlusion
                job.jobs.append(fj)
            if heavy:
                job.timings["occlusion"] = occ_ms
            job.timings["analyze"] = (time.perf_counter() - t0) * 1000
            return job

    def _occlusion(self, st: dict, crop: np.ndarray, s: Settings) -> Optional[np.ndarray]:
        if self.occluder is None or not s.mask_occlusion:
            return None
        n = int(st.get("occ_n", 0))
        st["occ_n"] = n + 1
        prev = st.get("occ")
        if self.occ_interval > 1 and n % self.occ_interval and isinstance(prev, np.ndarray) \
                and prev.shape == crop.shape[:2]:
            return prev     # crops are face-aligned, so the last mask is still a good fit
        mask = self.occluder.mask(crop)
        st["occ"] = mask
        return mask

    def _latent(self, st: dict, frame: np.ndarray, lm5: np.ndarray, visible: bool, s: Settings, t: float,
                sw: FaceSwapper, ident: Identity) -> np.ndarray:
        base = ident.embedding if sw.spec["type"] == "inswapper" else ident.embedding_norm
        k = float(np.clip(s.identity_strength / 100.0, 0.0, 1.5))
        temb = st.get("temb")
        if k > 1e-3 and self.recognizer is not None and visible and \
                (temb is None or t - float(st.get("temb_t", -1e9)) > 1.0):
            # who is in front of the camera — refreshed about once a second, smoothed
            try:
                e = _unit(self.recognizer.embed(frame, lm5))
                temb = e if temb is None else _unit(0.75 * temb + 0.25 * e)
                st["temb"], st["temb_t"] = temb, t
                st["temb_v"] = int(st.get("temb_v", 0)) + 1
            except Exception as exc:
                log.debug("target embedding failed: %s", exc)
        use_k = k if temb is not None else 0.0
        key = (self._identity_gen, sw.key, round(use_k, 3), st.get("temb_v", 0) if use_k else 0)
        if st.get("lat_key") != key:
            st["lat"] = sw.latent(base, temb, use_k)
            st["lat_key"] = key
        return st["lat"]

    def compose(self, job: FrameJob) -> FrameResult:
        """Fluid video: paste the newest rendered face of every track onto this frame (current landmarks)."""
        s = job.settings
        res = FrameResult(frame=job.frame, faces=job.faces, timings=dict(job.timings))
        full = np.zeros(job.frame.shape[:2], np.float32) if job.want_mask else None
        res.mask = full
        if not job.lite or not self.swap_active(s):
            return res
        t0 = time.perf_counter()
        out = None
        cache, latest = self._cache, self._cache_latest
        for tid, lm5, alpha in job.lite:
            c = cache.get(tid)
            if c is None and not s.multi_face:
                c = latest              # e.g. the face was re-acquired under a new track id
            if c is None or alpha <= 0.001:
                continue
            if out is None:
                out = job.frame.copy()
            m = geo.face_matrix(lm5, c.template, c.size)
            mask = c.mask * float(alpha)
            geo.paste(out, c.face, mask, m)
            if full is not None:
                geo.paste_mask(full, mask, m)
        if out is not None:
            res.frame = out
        res.timings["compose"] = (time.perf_counter() - t0) * 1000
        return res

    # ------------------------------------------------------------------ stage B
    def render(self, job: FrameJob) -> FrameResult:
        with self.lock_b:
            s = job.settings
            res = FrameResult(frame=job.frame, faces=job.faces, timings=dict(job.timings))
            full_mask = np.zeros(job.frame.shape[:2], np.float32) if job.want_mask else None
            res.mask = full_mask
            self._prune(job)
            sw, ident = self.swapper, self.identity
            if not job.jobs or sw is None or ident is None:
                return res
            t0 = time.perf_counter()
            out = job.frame.copy()
            acc = {"swap": 0.0, "mask": 0.0, "parser": 0.0, "enhance": 0.0}
            primary = None
            done = []
            for fj in job.jobs:
                if fj.swapper != sw.key:
                    continue        # job prepared for a model that was just replaced
                st = self._render_state.setdefault(fj.tid, {})
                if fj.need_occlusion:
                    p0 = time.perf_counter()
                    fj.occlusion = self._occlusion(st, fj.crop, s)
                    acc["occlusion"] = acc.get("occlusion", 0.0) + (time.perf_counter() - p0) * 1000
                if fj.latent is None:
                    p0 = time.perf_counter()
                    fj.latent = self._latent(st, job.frame, fj.lm5, fj.visible, s, job.t, sw, ident)
                    acc["identity"] = acc.get("identity", 0.0) + (time.perf_counter() - p0) * 1000
                final, mask = self._render_one(out, fj, s, acc, full_mask, job.t, st)
                done.append((fj, final, mask))
                if primary is None or fj.alpha > primary.alpha:
                    primary = fj
            c0 = time.perf_counter()
            for fj, final, mask in done:
                self._store(job, out, fj, final, mask)
            acc["cache"] = (time.perf_counter() - c0) * 1000
            res.timings.update(acc)
            res.timings["render"] = (time.perf_counter() - t0) * 1000
            res.frame = out
            res.swapped = bool(done)
            if primary is not None and self.recognizer is not None and primary.alpha > 0.9 \
                    and self.likeness.due(job.t):
                self.likeness.submit(self.recognizer, FaceRecognizer.align(out, primary.lm5), ident.embedding_norm,
                                     job.t)
            return res

    def _prune(self, job: FrameJob) -> None:
        live = {fj.tid for fj in job.jobs} | {tid for tid, _, _ in job.lite}
        for tid in [k for k, v in self._render_state.items() if k not in live and job.t - v.get("t", 0) > 2.0]:
            del self._render_state[tid]
        stale = [k for k, c in self._cache.items() if k not in live and job.t - c.t > 5.0]
        if stale:
            self._cache = {k: c for k, c in self._cache.items() if k not in stale}

    def _store(self, job: FrameJob, out: np.ndarray, fj: FaceJob, final: np.ndarray, mask: np.ndarray) -> None:
        """Keep the rendered face for compose(): un-blend it in face space at ~1:1 screen resolution, so that
        pasting it again with the same mask reproduces this frame exactly."""
        sw = self.swapper
        boost = fj.crop.shape[0]
        footprint = boost / max(geo.matrix_scale(fj.matrix), 1e-6)
        size = int(np.clip(round(footprint / 32.0) * 32, 128, 512))
        m = geo.face_matrix(fj.lm5, sw.template, size)
        a = geo.warp_face(out, m, size).astype(np.float32)
        b = geo.warp_face(job.frame, m, size).astype(np.float32)
        fm = cv2.resize(final, (size, size), interpolation=cv2.INTER_LINEAR)
        weight = np.maximum(fm, 0.02)[..., None]
        face = np.where(fm[..., None] > 0.02, b + (a - b) / weight, a)
        entry = FaceCache(fj.tid, np.clip(face + 0.5, 0, 255).astype(np.uint8),
                          cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR), sw.template, size, job.t)
        cache = dict(self._cache)
        cache[fj.tid] = entry
        self._cache = cache
        self._cache_latest = entry

    def _render_one(self, out: np.ndarray, fj: FaceJob, s: Settings, acc, full_mask, t: float, st: dict):
        sw = self.swapper
        boost = fj.crop.shape[0]
        m = fj.matrix
        st["t"] = t

        t1 = time.perf_counter()
        swapped = sw.swap(fj.crop, fj.latent, int(np.clip(s.swap_passes, 1, 3)))
        t2 = time.perf_counter()

        pad = (s.mask_padding_top, s.mask_padding_sides, s.mask_padding_bottom, s.mask_padding_sides)
        masks = [box_mask(boost, s.mask_blur, pad)]
        if fj.occlusion is not None and fj.occlusion.shape == (boost, boost):
            masks.append(fj.occlusion)
        p_ms = 0.0
        if self.parser is not None and s.mask_region:
            p0 = time.perf_counter()
            masks.append(self.parser.mask(np.clip(swapped, 0, 255).astype(np.uint8)))
            p_ms = (time.perf_counter() - p0) * 1000
        mask = np.minimum.reduce(masks) if len(masks) > 1 else masks[0].copy()

        # temporal smoothing of the mask in face space → no flicker at the edges
        k = float(np.clip(s.mask_temporal, 0.0, 0.9))
        prev = st.get("mask")
        if k > 0 and isinstance(prev, np.ndarray) and prev.shape == mask.shape:
            mask = cv2.addWeighted(prev, k, mask, 1.0 - k, 0.0)
        st["mask"] = mask

        # colour harmonisation with smoothed statistics
        if s.color_match > 0.01:
            st_src = lab_stats(swapped, mask)
            st_ref = lab_stats(fj.crop, mask)
            if st_src is not None and st_ref is not None:
                ema = st.get("color")
                if ema is not None:
                    st_src = tuple(0.7 * a + 0.3 * b for a, b in zip(ema[0], st_src))
                    st_ref = tuple(0.7 * a + 0.3 * b for a, b in zip(ema[1], st_ref))
                st["color"] = (st_src, st_ref)
                swapped = apply_color_transfer(swapped, st_src, st_ref, s.color_match)
        t3 = time.perf_counter()

        final = mask * float(fj.alpha)
        geo.paste(out, swapped, final, m)

        if self.enhancer is not None and s.enhancer_blend > 0.01:
            en = self.enhancer
            me = geo.face_matrix(fj.lm5, en.template, en.size)
            ecrop = geo.warp_face(out, me, en.size)
            enhanced = en.enhance(ecrop, s.enhancer_fidelity)
            to_e = geo.compose(me, geo.invert(m))
            emask = cv2.warpAffine(final, to_e, (en.size, en.size), flags=cv2.INTER_LINEAR, borderValue=0)
            emask = np.minimum(emask, box_mask(en.size, 0.25)) * float(s.enhancer_blend)
            geo.paste(out, enhanced, emask, me)
        t4 = time.perf_counter()

        if full_mask is not None:
            geo.paste_mask(full_mask, final, m)

        acc["swap"] += (t2 - t1) * 1000
        acc["mask"] += (t3 - t2) * 1000 - p_ms
        acc["parser"] += p_ms
        acc["enhance"] += (t4 - t3) * 1000
        return final, mask

    # ------------------------------------------------------------------ single-threaded convenience
    def process(self, frame: np.ndarray, t: Optional[float] = None) -> FrameResult:
        return self.render(self.analyze(frame, t))
