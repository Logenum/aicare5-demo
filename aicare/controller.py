"""GUI-independent monitor state: simulation cycle, waveform buffers and alarms.

The UI layer reads from a MonitorController and calls its command methods; nothing here
depends on a GUI toolkit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from .ai_engine import AIEngine, AIResult
from .config import ALARM_LIMITS, AUTO_CYCLE, CRITICAL_LIMITS
from .patient import ECG_FS, PLETH_FS, RESP_FS, PatientSimulator

SPEEDS = [("Real time", 1), ("x10", 10), ("x60", 60), ("x300", 300)]
WARMUP_MINUTES = 120
SWEEP_S = {"ecg": 6.0, "pleth": 6.0, "resp": 12.0}
FS = {"ecg": ECG_FS, "pleth": PLETH_FS, "resp": RESP_FS}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
PARAM_LABEL = {"hr": "HR", "spo2": "SpO₂", "sbp": "NIBP sys", "map": "MAP", "rr": "RR",
               "temp": "Temp"}
UNITS = {"hr": "bpm", "spo2": "%", "sbp": "mmHg", "map": "mmHg", "rr": "/min", "temp": "°C"}


@dataclass
class Alarm:
    key: str
    priority: str        # high / medium / low / info
    text: str
    raised: datetime
    active: bool = True
    acked: bool = False
    suppressed: bool = False
    cleared: Optional[datetime] = None


class Sweep:
    """Fixed-length circular display buffer with a write cursor (monitor 'erase bar')."""

    def __init__(self, n: int):
        self.y = np.full(n, np.nan)
        self.i = 0

    def push(self, samples: np.ndarray):
        for s in samples:
            self.y[self.i] = s
            self.i = (self.i + 1) % len(self.y)

    def recent(self, n: int) -> np.ndarray:
        idx = (self.i - n + np.arange(n)) % len(self.y)
        return self.y[idx]


class MonitorController:
    def __init__(self):
        self.dt = 60
        self.paused = False
        self.silenced_until: Optional[datetime] = None
        self.alarms: List[Alarm] = []
        self.active_alarms: Dict[str, Alarm] = {}
        self.alarm_version = 0
        self.suppressed_count = 0
        self.result: Optional[AIResult] = None
        self.warming_up = False
        self.battery = 87.0
        start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=WARMUP_MINUTES)
        self.pt = PatientSimulator(AUTO_CYCLE, start)
        self.ai = AIEngine()
        self.sweeps = {k: Sweep(int(SWEEP_S[k] * FS[k])) for k in SWEEP_S}
        self.recent = {k: deque(maxlen=int(3 * FS[k])) for k in ("ecg", "pleth")}

    # =====================================================================================
    # Commands
    # =====================================================================================
    def set_scenario(self, name: str):
        self.pt.set_scenario(name)
        self.raise_info(f"Scenario changed to '{name}'")

    def set_o2(self, on: bool):
        self.pt.o2 = on
        self.raise_info("Supplemental O₂ started (4 L/min)" if on else "Supplemental O₂ stopped")

    def patient_movement(self):
        self.pt.start_motion(15.0)
        self.raise_info("Demo: patient movement - motion artifact on ECG and SpO₂ probe")

    def silence(self, minutes: int = 2):
        self.silenced_until = self.pt.time + timedelta(minutes=minutes)
        for a in self.alarms:
            a.acked = True
        self.alarm_version += 1

    # =====================================================================================
    # Cycle
    # =====================================================================================
    def warmup_steps(self) -> int:
        target = datetime.now(timezone.utc)
        steps = int((target - self.pt.time).total_seconds() // 60)
        if steps < 60:
            self.pt.time = target - timedelta(minutes=WARMUP_MINUTES)
            steps = WARMUP_MINUTES
        return min(steps, 150)

    def warm_up(self):
        self.warming_up = True
        for _ in range(self.warmup_steps()):
            self.cycle(60)
        self.warming_up = False

    def advance_waves(self, seconds: float) -> Dict[str, np.ndarray]:
        """Generate the next wall-clock slice of waveforms and push it to the sweeps."""
        new = self.pt.wave.advance(seconds)
        for k, s in new.items():
            self.sweeps[k].push(s)
            if k in self.recent:
                self.recent[k].extend(s)
        return new

    def tick(self):
        """Once per wall-clock second."""
        self.battery = max(5.0, self.battery - 0.004)
        if not self.paused:
            self.cycle(self.dt)

    def cycle(self, dt: float):
        v = self.pt.step(dt)
        if not self.warming_up:
            self.ai.update_sqi(np.array(self.recent["ecg"]), np.array(self.recent["pleth"]))
        self.result = self.ai.update(v)
        if not self.warming_up:
            self._evaluate_alarms(self.result)

    # =====================================================================================
    # Alarms
    # =====================================================================================
    def raise_info(self, text: str):
        t = self.pt.time
        self.alarms.insert(0, Alarm("info", "info", text, t, active=False, acked=True, cleared=t))
        del self.alarms[500:]
        self.alarm_version += 1

    def unacked(self) -> List[Alarm]:
        return [a for a in self.alarms if a.active and not a.acked and not a.suppressed]

    def audio_paused(self) -> bool:
        return self.silenced_until is not None and self.pt.time < self.silenced_until

    def top_alarm(self) -> Optional[Alarm]:
        act = [a for a in self.active_alarms.values() if not a.suppressed]
        return min(act, key=lambda a: PRIORITY_ORDER[a.priority]) if act else None

    def _evaluate_alarms(self, r: AIResult):
        val = r.valid
        cond = {}
        for k, (lo, hi) in ALARM_LIMITS.items():
            raw = getattr(r.raw, k) if k != "map" else r.raw.map
            x = raw if k not in r.unreliable else val[k]
            clo, chi = CRITICAL_LIMITS.get(k, (None, None))
            low = lo is not None and x < lo
            high = hi is not None and x > hi
            crit = (clo is not None and x < clo) or (chi is not None and x > chi)
            txt = f"{PARAM_LABEL[k]} {'LOW' if low else 'HIGH'} {x:.{1 if k == 'temp' else 0}f} {UNITS[k]}"
            cond[f"phys:{k}"] = (low or high, "high" if crit else "medium", txt)
            # the raw (unvalidated) reading would have alarmed: AI suppression
            raw_lo = lo is not None and raw < lo
            raw_hi = hi is not None and raw > hi
            cond[f"supp:{k}"] = (k in r.unreliable and (raw_lo or raw_hi) and not (low or high),
                                 "suppressed",
                                 f"AI suppressed false {PARAM_LABEL[k]} alarm "
                                 f"({raw:.0f} {UNITS[k]}: {r.unreliable.get(k, '')})")
        cond["rhythm"] = (r.af_prob > 0.7, "medium",
                          f"AI: atrial fibrillation suspected ({r.af_prob * 100:.0f}%)")
        cond["news2"] = (r.news2 >= 5, "high" if r.news2 >= 7 else "medium",
                         f"NEWS2 {r.news2} - {r.news2_response.lower()}")
        cond["ai_news"] = (r.news2_eta is not None, "low",
                           f"AI: NEWS2 ≥5 predicted in ~{r.news2_eta} min")
        for name, p in r.risks.items():
            if name.startswith("Arrhythmia"):
                continue
            cond[f"ai:{name}"] = (p > 0.6, "medium", f"AI: {name.lower()} risk {p * 100:.0f}%")
        for k, why in r.unreliable.items():
            cond[f"tech:{k}"] = (True, "low", f"{PARAM_LABEL[k]} signal unreliable: {why}")
        for k in ("hr", "spo2"):
            cond.setdefault(f"tech:{k}", (False, "low", ""))
        cond["battery"] = (self.battery < 20, "low", f"Battery low ({self.battery:.0f}%)")

        changed = False
        for key, (on, prio, text) in cond.items():
            cur = self.active_alarms.get(key)
            if on and cur is None:
                supp = prio == "suppressed"
                a = Alarm(key, "low" if supp else prio, text, r.raw.time, suppressed=supp,
                          acked=supp)
                if supp:
                    self.suppressed_count += 1
                self.alarms.insert(0, a)
                self.active_alarms[key] = a
                changed = True
            elif on and cur is not None and prio != "suppressed" and \
                    (cur.text, cur.priority) != (text, prio):
                if PRIORITY_ORDER[prio] < PRIORITY_ORDER[cur.priority]:
                    cur.acked = False        # escalation needs a fresh acknowledgement
                cur.text, cur.priority = text, prio
                changed = True
            elif not on and cur is not None:
                cur.active = False
                cur.cleared = r.raw.time
                del self.active_alarms[key]
                changed = True
        del self.alarms[500:]
        if changed:
            self.alarm_version += 1
