"""Simulated patient: scenario-driven vital signs and real-time waveforms.

Two clocks run here. Vital signs evolve in *simulation* time (which can run faster than
real time). Waveforms are synthesised in *wall-clock* time from the current vitals, so
the sweep always moves at the true paper speed. Motion artifacts are injected into the
waveforms and corrupt the measured numerics, which the AI layer must detect.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

import numpy as np

from .config import AUTO_CYCLE, BASE, SCENARIOS

ECG_FS = 50.0        # Hz
PLETH_FS = 25.0
RESP_FS = 12.5


@dataclass
class Phase:
    minutes: float
    label: str
    targets: dict


def build_phases(name: str) -> List[Phase]:
    names = list(SCENARIOS) if name == AUTO_CYCLE else [name]
    out = []
    for n in names:
        tgt = dict(BASE)
        for minutes, spec in SCENARIOS[n]:
            tgt = {**tgt, **{k: v for k, v in spec.items() if k != "label"}}
            out.append(Phase(minutes, f"{n}: {spec['label']}" if name == AUTO_CYCLE
                             else spec["label"], dict(tgt)))
        if name == AUTO_CYCLE:      # recovery between scenarios
            out.append(Phase(30, f"{n}: recovery", dict(BASE)))
    return out


@dataclass
class Vitals:
    time: datetime
    hr: float
    sbp: float
    dbp: float
    rr: float
    spo2: float
    temp: float
    af: bool
    avpu: str
    perf: float              # perfusion index proxy, 1 = normal
    o2: bool                 # supplemental oxygen
    artifact: bool = False   # measured values corrupted by motion
    rr_intervals: List[float] = field(default_factory=list)   # recent beat-to-beat, s

    @property
    def map(self) -> float:
        return (self.sbp + 2 * self.dbp) / 3


class PatientSimulator:
    KEYS = ("hr", "sbp", "dbp", "rr", "spo2", "temp", "perf")
    TAU_S = {"hr": 240, "sbp": 300, "dbp": 300, "rr": 300, "spo2": 240, "temp": 900, "perf": 300}
    NOISE = {"hr": 1.6, "sbp": 2.5, "dbp": 1.6, "rr": 0.8, "spo2": 0.35, "temp": 0.02,
             "perf": 0.02}

    def __init__(self, scenario: str, start: Optional[datetime] = None):
        self.time = start or datetime.now(timezone.utc).replace(microsecond=0)
        self.o2 = False
        self.state = {k: float(BASE[k]) for k in self.KEYS}
        self.set_scenario(scenario)
        self.motion_start = self.motion_until = 0.0     # wall-clock seconds (wave time)
        self.wave = WaveformGenerator(self)

    # scenario --------------------------------------------------------------------------
    def set_scenario(self, name: str):
        self.scenario = name
        self.phases = build_phases(name)
        self.phase_idx = 0
        self.phase_elapsed = 0.0

    @property
    def phase(self) -> Phase:
        return self.phases[self.phase_idx]

    def targets(self) -> dict:
        t = dict(self.phase.targets)
        if self.o2:
            t["spo2"] = min(99.0, t["spo2"] + 4.0)
            t["rr"] = t["rr"] - 1.5 if t["rr"] > 20 else t["rr"]
        return t

    # vitals ----------------------------------------------------------------------------
    def step(self, dt: float) -> Vitals:
        """Advance simulation time by dt seconds (in 5 s sub-steps)."""
        n = max(1, int(round(dt / 5)))
        h = dt / n
        for _ in range(n):
            tgt = self.targets()
            for k in self.KEYS:
                a = 1 - math.exp(-h / self.TAU_S[k])
                self.state[k] += a * (tgt[k] - self.state[k]) \
                    + self.NOISE[k] * math.sqrt(h / 60) * random.gauss(0, 1)
            self.state["spo2"] = min(100.0, self.state["spo2"])
            self.phase_elapsed += h
            if self.phase_elapsed >= self.phase.minutes * 60:
                self.phase_elapsed = 0.0
                if self.phase_idx < len(self.phases) - 1:
                    self.phase_idx += 1
                elif self.scenario == AUTO_CYCLE:
                    self.phase_idx = 0
            self.time += timedelta(seconds=h)
        return self.measure()

    def measure(self) -> Vitals:
        """Monitor readings; motion corrupts SpO2 and HR (as a real monitor would)."""
        s, tgt = self.state, self.phase.targets
        # the monitor's averaging delays the effect of motion on the numerics by ~3 s
        art = self.motion_active() and self.wave.t - self.motion_start >= 3.0
        hr, spo2 = s["hr"], s["spo2"]
        if art:
            spo2 = spo2 - random.uniform(6, 14)
            hr = hr + random.uniform(-25, 45)
        return Vitals(time=self.time, hr=hr, sbp=s["sbp"], dbp=s["dbp"], rr=s["rr"],
                      spo2=max(50.0, spo2), temp=s["temp"], af=tgt["af"], avpu=tgt["avpu"],
                      perf=max(0.1, s["perf"]), o2=self.o2, artifact=art,
                      rr_intervals=list(self.wave.rr_hist))

    # artifacts -------------------------------------------------------------------------
    def start_motion(self, seconds: float = 15.0):
        self.motion_start = self.wave.t
        self.motion_until = self.wave.t + seconds

    def motion_active(self) -> bool:
        return self.wave.t < self.motion_until


class WaveformGenerator:
    """Synthesises ECG lead II, plethysmogram and respiration at fixed sample rates."""

    def __init__(self, pt: PatientSimulator):
        self.pt = pt
        self.t = 0.0                       # wave clock, seconds
        self.next_beat = 0.5
        self.beats: Deque[float] = deque(maxlen=4)
        self.rr_hist: Deque[float] = deque(maxlen=40)
        self.resp_phase = 0.0
        self.noise_lp = 0.0
        self._n = {"ecg": 0, "pleth": 0, "resp": 0}

    def _beat_interval(self) -> float:
        s, af = self.pt.state, self.pt.phase.targets["af"]
        mean = 60.0 / max(25.0, s["hr"])
        if af:   # irregularly irregular
            return max(0.28, mean * random.uniform(0.6, 1.45))
        resp_mod = 0.03 * math.sin(self.resp_phase)        # sinus arrhythmia
        return mean * (1 + resp_mod + random.gauss(0, 0.012))

    def advance(self, seconds: float) -> Dict[str, np.ndarray]:
        """Generate samples for the next `seconds` of wall time."""
        t_end = self.t + seconds
        out = {"ecg": [], "pleth": [], "resp": []}
        fs = {"ecg": ECG_FS, "pleth": PLETH_FS, "resp": RESP_FS}
        # schedule beats up to t_end (+ lookahead), snapped to the ECG grid so the
        # R peak is always sampled
        while self.next_beat <= t_end + 0.3:
            self.beats.append(self.next_beat)
            iv = self._beat_interval()
            if len(self.beats) > 1:
                self.rr_hist.append(self.beats[-1] - self.beats[-2])
            self.next_beat = round((self.next_beat + iv) * ECG_FS) / ECG_FS
        for key in out:
            while True:
                ts = self._n[key] / fs[key]
                if ts >= t_end:
                    break
                out[key].append(getattr(self, f"_{key}")(ts))
                self._n[key] += 1
        self.t = t_end
        return {k: np.array(v) for k, v in out.items()}

    # waveform models -------------------------------------------------------------------
    @staticmethod
    def _g(x, mu, sd):
        return math.exp(-0.5 * ((x - mu) / sd) ** 2)

    def _ecg(self, ts: float) -> float:
        af = self.pt.phase.targets["af"]
        v = 0.0
        for b in self.beats:
            x = ts - b
            if -0.3 < x < 0.5:
                if not af:
                    v += 0.12 * self._g(x, -0.16, 0.025)
                v += -0.10 * self._g(x, -0.03, 0.010) + 1.0 * self._g(x, 0.0, 0.011) \
                    - 0.25 * self._g(x, 0.03, 0.011) + 0.28 * self._g(x, 0.26, 0.05)
        if af:
            v += 0.05 * math.sin(2 * math.pi * 6.3 * ts) + 0.03 * math.sin(2 * math.pi * 8.9 * ts)
        v += 0.015 * random.gauss(0, 1)
        if ts < self.pt.motion_until:
            self.noise_lp = 0.9 * self.noise_lp + 0.25 * random.gauss(0, 1)
            v += self.noise_lp + (random.uniform(-0.8, 0.8) if random.random() < 0.04 else 0)
        return v

    def _pleth(self, ts: float) -> float:
        perf = self.pt.state["perf"]
        v = 0.0
        for b in self.beats:
            x = ts - b - 0.2
            if -0.1 < x < 0.9:
                v += self._g(x, 0.12, 0.055) + 0.35 * self._g(x, 0.36, 0.08)
        v = v * (0.35 + 0.65 * min(1.2, perf))
        if ts < self.pt.motion_until:
            v += 0.7 * math.sin(2 * math.pi * 1.3 * ts + random.uniform(0, 0.6)) \
                + 0.35 * random.gauss(0, 1)
        return v

    def _resp(self, ts: float) -> float:
        self.resp_phase += 2 * math.pi * self.pt.state["rr"] / 60 / RESP_FS
        return math.sin(self.resp_phase) + 0.04 * random.gauss(0, 1)
