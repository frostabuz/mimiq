"""Auto quality («Авто-плавность»).

Keeps the video fluid on any PC. Each pipeline stage has a time budget per frame:

* stage A (tracking) decides the frame rate of the video → budget = 1 / target fps;
* stage B (face render) decides how often the face itself is refreshed → the same budget in direct mode,
  a relaxed one (≈15 face updates per second) in fluid mode, where the video does not wait for it.

When a stage is over its budget, its most expensive extra that is actually worth switching off goes first
(an extra that saves only a few percent is kept — quality is not thrown away for nothing). When there is
headroom for the measured cost of that extra, it comes back. The user's settings are never modified:
the tuner only produces an "effective" copy for the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..config import Settings

Parts = Dict[str, float]


@dataclass(frozen=True)
class Step:
    key: str
    stage: str                                   # "a" (tracking) | "b" (render)
    label: str                                   # what is reduced — shown to the user
    apply: Callable[[Settings], None]
    saving: Callable[[Parts, Settings], Optional[float]]   # expected ms saved per frame (None = unknown)


def _set(**kw) -> Callable[[Settings], None]:
    def fn(s: Settings) -> None:
        for k, v in kw.items():
            setattr(s, k, v)
    return fn


def _part(name: str, factor: float = 1.0) -> Callable[[Parts, Settings], Optional[float]]:
    def fn(parts: Parts, s: Settings) -> Optional[float]:
        v = parts.get(name)
        return None if v is None else v * factor
    return fn


def candidate_steps(s: Settings, fluid: bool = False) -> List[Step]:
    """Extras that can be reduced, in the order they are given up within each stage."""
    out: List[Step] = []
    if s.swap_passes > 1:
        p = int(s.swap_passes)
        out.append(Step("passes", "b", "1 проход замены", _set(swap_passes=1), _part("swap", (p - 1) / p)))
    if s.pixel_boost >= 512:
        out.append(Step("boost512", "b", "детализация 256", _set(pixel_boost=256), _part("swap", 0.75)))
    if s.pixel_boost >= 256:
        out.append(Step("boost256", "b", "детализация 128", _set(pixel_boost=128), _part("swap", 0.75)))
    if s.enhancer_model and s.enhancer_blend > 0.01:
        out.append(Step("enhancer", "b", "без улучшения лица", _set(enhancer_blend=0.0), _part("enhance")))
    if s.mask_region:
        out.append(Step("parser", "b", "без точной формы маски", _set(mask_region=False), _part("parser")))
    if s.landmark_refine:
        out.append(Step("landmarks", "a", "без 68 точек", _set(landmark_refine=False), _part("landmarks")))
    if s.mask_occlusion:
        # in fluid mode the occlusion mask is computed by the render stage
        out.append(Step("occlusion", "b" if fluid else "a", "маска перекрытий реже", _set(),
                        _part("occlusion", 0.5)))
    return out


def rate_text(fps: float, long: bool = False) -> str:
    """How often the face is refreshed, in words: «12/с» / «раз в 3 с» (long: «~12 раз в секунду»)."""
    if fps <= 0:
        return "—"
    if fps >= 1.5:
        return f"~{fps:.0f} раз в секунду" if long else f"{fps:.0f}/с"
    every = 1.0 / fps
    if every < 1.5:
        return "~1 раз в секунду" if long else "1/с"
    return f"примерно раз в {every:.0f} с" if long else f"раз в {every:.0f} с"


class AutoTuner:
    DOWN = 1.10          # stage time above budget × DOWN → reduce
    UP = 0.85            # stage time + cost of the extra below budget × UP → restore
    STABLE_UP = 3.0      # seconds of headroom before restoring
    BLOCK = 45.0         # an extra that did not fit right after coming back stays off this long
    WORTH = 0.12         # an extra is only dropped if it saves at least this share of the stage time

    def __init__(self):
        self.applied: List[str] = []
        self.cost: Dict[str, float] = {}         # expected ms per frame of each extra that is off
        self.blocked: Dict[str, float] = {}
        self._restored: Dict[str, float] = {}
        self._changed_at = -1e9
        self._settle = 1.5
        self._good_since: Optional[float] = None
        self.exhausted = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.applied.clear()
        self._good_since = None
        self._changed_at = -1e9
        self.exhausted = False

    def effective(self, s: Settings, fluid: bool = False) -> Settings:
        eff = s.copy()
        steps = {st.key: st for st in candidate_steps(s, fluid)}
        for key in self.applied:
            st = steps.get(key)
            if st is not None:
                st.apply(eff)
        return eff

    @property
    def occ_interval(self) -> int:
        return 2 if "occlusion" in self.applied else 1

    def labels(self, s: Settings) -> List[str]:
        steps = {st.key: st for st in candidate_steps(s)}
        return [steps[k].label for k in self.applied if k in steps]

    # ------------------------------------------------------------------
    def update(self, now: float, a_ms: float, b_ms: float, budget_a: float, budget_b: float, parts: Parts,
               s: Settings, fluid: bool = False) -> bool:
        """Feed the current per-frame cost of both stages; returns True when the effective settings changed."""
        steps = candidate_steps(s, fluid)
        by_key = {st.key: st for st in steps}
        if any(k not in by_key for k in self.applied):          # the user switched an extra off meanwhile
            self.applied = [k for k in self.applied if k in by_key]
            self._good_since = None
            return True
        if now - self._changed_at < self._settle or (a_ms <= 0 and b_ms <= 0):
            return False
        load = {"a": a_ms / max(budget_a, 1.0), "b": b_ms / max(budget_b, 1.0)}
        stage_ms = {"a": a_ms, "b": b_ms}
        budget = {"a": budget_a, "b": budget_b}

        over = [g for g in sorted(load, key=load.get, reverse=True) if load[g] > self.DOWN]
        if over:
            for g in over:
                step = self._next_down(steps, g, parts, s, stage_ms[g], budget[g])
                if step is not None:
                    saving = step.saving(parts, s)
                    self.cost[step.key] = saving if saving is not None else 0.3 * stage_ms[g]
                    if now - self._restored.get(step.key, -1e9) < 12.0:
                        self.blocked[step.key] = now + self.BLOCK   # it came back and did not fit
                    self.applied.append(step.key)
                    self._mark(now, max(a_ms, b_ms))
                    self.exhausted = False
                    return True
            self.exhausted = True
            return False

        self.exhausted = False
        if self.applied:
            key = self.applied[-1]
            st = by_key[key]
            g = st.stage
            cost = self.cost.get(key, budget[g])
            fits = stage_ms[g] + cost < budget[g] * self.UP
            if fits and now >= self.blocked.get(key, 0.0):
                if self._good_since is None:
                    self._good_since = now
                elif now - self._good_since >= self.STABLE_UP:
                    self.applied.pop()
                    self._restored[key] = now
                    self._mark(now, max(a_ms, b_ms))
                    return True
            else:
                self._good_since = None
        return False

    def _next_down(self, steps: List[Step], stage: str, parts: Parts, s: Settings, stage_ms: float,
                   budget: float) -> Optional[Step]:
        for st in steps:
            if st.stage != stage or st.key in self.applied:
                continue
            saving = st.saving(parts, s)
            if saving is None:
                return st
            if saving >= self.WORTH * stage_ms or stage_ms - saving <= budget * self.DOWN:
                return st
        return None

    def _mark(self, now: float, frame_ms: float) -> None:
        self._changed_at = now
        # let the timing averages catch up: at least ~8 frames
        self._settle = max(1.5, min(6.0, 8 * frame_ms / 1000.0))
        self._good_since = None
