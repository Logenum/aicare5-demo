"""AIcare AI layer.

* Signal quality (SQI): kurtosis of ECG and skewness of the plethysmogram detect motion
  artifacts. Readings from a poor-quality signal are replaced by the last validated value
  and their alarms are suppressed.
* Rhythm: atrial fibrillation detection from beat-to-beat (RR) irregularity.
* NEWS2: the Royal College of Physicians National Early Warning Score 2.
* Risk models: sepsis, respiratory failure and haemodynamic instability probabilities
  (next 60 min) with their top contributing factors.
* Forecast: damped-trend regression of each vital sign to +60 min with 80% bands, and the
  NEWS2 those forecasts imply.

The risk models are hand-calibrated logistic models for demonstration, not trained or
validated on patient data.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .patient import Vitals

HISTORY_SECONDS = 6 * 3600
FORECAST_MIN = list(range(5, 61, 5))
SERIES = ("hr", "spo2", "sbp", "map", "rr", "temp", "news2", "index")


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


# --------------------------------------------------------------------------------------
# NEWS2
# --------------------------------------------------------------------------------------
def news2(rr, spo2, o2, sbp, hr, avpu, temp) -> Tuple[int, Dict[str, int]]:
    def band(v, table):
        for upper, pts in table:
            if v <= upper:
                return pts
        return table[-1][1]

    c = {
        "Resp. rate": band(round(rr), [(8, 3), (11, 1), (20, 0), (24, 2), (999, 3)]),
        "SpO₂": band(round(spo2), [(91, 3), (93, 2), (95, 1), (999, 0)]),
        "Air or O₂": 2 if o2 else 0,
        "Systolic BP": band(round(sbp), [(90, 3), (100, 2), (110, 1), (219, 0), (999, 3)]),
        "Pulse": band(round(hr), [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (999, 3)]),
        "Consciousness": 0 if avpu == "A" else 3,
        "Temperature": band(round(temp, 1), [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1),
                                             (99, 2)]),
    }
    return sum(c.values()), c


def news2_band(total: int, comps: Dict[str, int]) -> Tuple[str, str]:
    """(clinical risk, response) per RCP NEWS2 chart."""
    if total >= 7:
        return "HIGH", "Emergency response"
    if total >= 5:
        return "MEDIUM", "Urgent review"
    if any(v == 3 for v in comps.values()):
        return "LOW-MEDIUM", "Urgent ward review"
    return "LOW", "Routine monitoring"


# --------------------------------------------------------------------------------------
# Signal quality and rhythm
# --------------------------------------------------------------------------------------
def ecg_sqi(x: np.ndarray) -> float:
    """Kurtosis SQI: clean ECG is spiky (kurtosis well above 5); noise is not."""
    if len(x) < 100:
        return 1.0
    x = x - x.mean()
    sd = x.std()
    if sd < 1e-6:
        return 0.0
    k = float(np.mean(x ** 4) / sd ** 4)
    return float(np.clip((k - 4.0) / 6.0, 0, 1))


def pleth_sqi(x: np.ndarray) -> float:
    """Skewness SQI: arterial pulses are positively skewed; motion artifact is not."""
    if len(x) < 50:
        return 1.0
    x = x - x.mean()
    sd = x.std()
    if sd < 1e-6:
        return 0.0
    s = float(np.mean(x ** 3) / sd ** 3)
    return float(np.clip((s - 0.1) / 0.6, 0, 1))


def af_probability(rr: List[float]) -> Tuple[float, str]:
    """Irregularity of successive RR intervals (normalised RMSSD and pNN50-like share)."""
    if len(rr) < 12:
        return 0.0, "insufficient beats"
    r = np.asarray(rr[-30:])
    d = np.diff(r)
    nrmssd = float(np.sqrt(np.mean(d ** 2)) / r.mean())
    p_big = float(np.mean(np.abs(d) > 0.1 * r.mean()))
    p = sigmoid(-6.0 + 28 * nrmssd + 4 * p_big)
    return p, f"RR irregularity {nrmssd * 100:.0f}%, {p_big * 100:.0f}% of beats vary >10%"


# --------------------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------------------
@dataclass
class Forecast:
    minutes: float
    mean: float
    sd: float

    def interval(self, z: float = 1.28) -> Tuple[float, float]:
        return self.mean - z * self.sd, self.mean + z * self.sd


def damped_trend(t: np.ndarray, y: np.ndarray, horizons_min: List[float],
                 window_s: float = 2700, damp_tau_min: float = 45) -> List[Forecast]:
    """Weighted linear trend over the last window, damped into the future."""
    m = np.isfinite(y) & (t >= t[-1] - window_s)
    tt, yy = t[m], y[m]
    if len(tt) < 6:
        return [Forecast(h, float(y[-1]), 0.0) for h in horizons_min]
    x = (tt - tt[-1]) / 60.0
    w = np.exp(x / 20.0)                        # favour recent data
    A = np.vstack([np.ones_like(x), x]).T * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A, yy * np.sqrt(w), rcond=None)
    level, slope = coef
    resid = yy - (level + slope * x)
    sd0 = float(np.sqrt(np.average(resid ** 2, weights=w)))
    out = []
    for h in horizons_min:
        eff = damp_tau_min * (1 - math.exp(-h / damp_tau_min))
        out.append(Forecast(h, float(level + slope * eff),
                            sd0 * math.sqrt(1 + h / 10) + abs(slope) * eff * 0.25))
    return out


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------
@dataclass
class AIResult:
    raw: Vitals
    valid: Dict[str, float]                # validated values shown to the clinician
    unreliable: Dict[str, str]             # parameter -> reason
    sqi: Dict[str, float]
    rhythm: str
    af_prob: float
    af_note: str
    news2: int
    news2_parts: Dict[str, int]
    news2_risk: str
    news2_response: str
    risks: Dict[str, float]
    drivers: Dict[str, str]
    index: float
    level: str
    forecasts: Dict[str, List[Forecast]]
    curves: Dict[str, List[Tuple[datetime, float, float, float]]]
    news2_forecast: List[Tuple[int, int]]   # (minutes, NEWS2)
    news2_eta: Optional[int]                # minutes until NEWS2 >= 5 predicted
    confidence: float
    summary: List[Tuple[str, str]] = field(default_factory=list)   # (bold lead, text)


class AIEngine:
    def __init__(self):
        self.hist: Dict[str, deque] = {k: deque() for k in ("t",) + SERIES}
        self.last_valid: Dict[str, float] = {}
        self.sqi = {"ECG": 1.0, "Pleth": 1.0}
        self.suppressed = 0

    def reset(self):
        self.__init__()

    def series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.hist["t"]), np.array(self.hist[key])

    def update_sqi(self, ecg: np.ndarray, pleth: np.ndarray):
        self.sqi = {"ECG": ecg_sqi(ecg), "Pleth": pleth_sqi(pleth)}

    # -----------------------------------------------------------------------------------
    def update(self, v: Vitals) -> AIResult:
        # ---- signal quality: hold last validated value for a poor-quality channel
        unreliable = {}
        valid = dict(hr=v.hr, spo2=v.spo2, sbp=v.sbp, dbp=v.dbp, map=v.map, rr=v.rr, temp=v.temp)
        if self.sqi["Pleth"] < 0.5:
            unreliable["spo2"] = f"motion artifact on pleth (SQI {self.sqi['Pleth']:.2f})"
        if self.sqi["ECG"] < 0.5:
            unreliable["hr"] = f"noisy ECG (SQI {self.sqi['ECG']:.2f})"
        for k in valid:
            if k in unreliable and k in self.last_valid:
                valid[k] = self.last_valid[k]
            else:
                self.last_valid[k] = valid[k]

        # ---- rhythm
        af_p, af_note = af_probability(v.rr_intervals)
        if not v.rr_intervals:            # no waveform data yet (e.g. during warm-up)
            af_p, af_note = (0.9, "scenario rhythm") if v.af else (0.02, "scenario rhythm")
        rhythm = "AF suspected" if af_p > 0.7 else "Irregular?" if af_p > 0.4 else "Sinus rhythm"
        if "hr" in unreliable:
            rhythm = "Rhythm: signal noisy"

        # ---- NEWS2 on validated values
        total, parts = news2(valid["rr"], valid["spo2"], v.o2, valid["sbp"], valid["hr"],
                             v.avpu, valid["temp"])
        risk_band, response = news2_band(total, parts)

        # ---- history
        t = v.time.timestamp()
        h = self.hist
        for k in ("hr", "spo2", "sbp", "map", "rr", "temp"):
            h[k].append(valid[k])
        h["t"].append(t)
        h["news2"].append(total)

        def slope_per_h(key, window=1800):
            tt, yy = np.array(h["t"]), np.array(h[key], dtype=float)
            m = tt >= t - window
            if m.sum() < 5:
                return 0.0
            return float(np.polyfit((tt[m] - t) / 3600, yy[m], 1)[0])

        # ---- risk models (logistic, with per-feature contributions for explanation)
        def model(bias, feats):
            contrib = {name: w * x for name, (w, x) in feats.items()}
            p = sigmoid(bias + sum(contrib.values()))
            top = [n for n, c in sorted(contrib.items(), key=lambda kv: -kv[1]) if c > 0.25][:2]
            return p, ", ".join(top) if top else "no significant factors"

        hr_tr, map_tr, spo2_tr, rr_tr = (slope_per_h("hr"), slope_per_h("map"),
                                         slope_per_h("spo2"), slope_per_h("rr"))
        T, HR, RR, SBP, MAP, SPO2 = (valid["temp"], valid["hr"], valid["rr"], valid["sbp"],
                                     valid["map"], valid["spo2"])
        altered = 1.0 if v.avpu != "A" else 0.0
        risks, drivers = {}, {}
        risks["Sepsis"], drivers["Sepsis"] = model(-5.0, {
            "fever" if T > 37 else "hypothermia": (2.0, max(0, T - 37.8) + max(0, 35.8 - T)),
            "tachycardia": (0.35, max(0, HR - 90) / 10),
            "tachypnoea": (0.3, max(0, RR - 20) / 2),
            "falling BP": (0.4, max(0, 110 - SBP) / 10),
            "rising HR trend": (0.3, min(2.0, max(0, hr_tr) / 10)),
            "altered consciousness": (1.0, altered),
        })
        risks["Respiratory failure"], drivers["Respiratory failure"] = model(-4.6, {
            "low SpO₂": (0.6, max(0, 95 - SPO2)),
            "tachypnoea": (0.5, max(0, RR - 20) / 2),
            "falling SpO₂ trend": (0.6, min(3.0, max(0, -spo2_tr))),
            "rising RR trend": (0.3, max(0, rr_tr) / 2),
            "O₂ requirement": (1.0, 1.0 if v.o2 else 0.0),
            "altered consciousness": (1.2, altered),
        })
        si = HR / max(SBP, 1)
        risks["Haemodynamic instability"], drivers["Haemodynamic instability"] = model(-4.6, {
            f"shock index {si:.2f}": (0.7, max(0, si - 0.7) * 10),
            "low MAP": (0.7, max(0, 72 - MAP) / 4),
            "falling MAP trend": (0.5, min(3.0, max(0, -map_tr) / 5)),
            "poor perfusion": (3.0, max(0, 0.85 - v.perf)),
        })
        risks["Arrhythmia (AF)"], drivers["Arrhythmia (AF)"] = af_p, af_note
        index = 100 * (1 - math.prod(1 - p * w for p, w in zip(
            risks.values(), (1.0, 1.0, 1.0, 0.5))))
        index = max(index, min(100.0, total * 9.0))
        level = ("LOW" if index < 25 else "MODERATE" if index < 50 else
                 "HIGH" if index < 75 else "CRITICAL")
        h["index"].append(index)

        # trim history
        while h["t"] and h["t"][0] < t - HISTORY_SECONDS:
            for k in h:
                h[k].popleft()

        # ---- forecasts and the NEWS2 they imply
        tt = np.array(h["t"])
        fc, curves = {}, {}
        for k in ("hr", "spo2", "sbp", "map", "rr", "temp"):
            fc[k] = damped_trend(tt, np.array(h[k], dtype=float), FORECAST_MIN)
            curves[k] = [(v.time + timedelta(minutes=f.minutes), f.mean, *f.interval())
                         for f in fc[k]]
            curves[k].insert(0, (v.time, valid[k], valid[k], valid[k]))
        news_fc = []
        for i, mins in enumerate(FORECAST_MIN):
            n, _ = news2(fc["rr"][i].mean, min(100, fc["spo2"][i].mean), v.o2, fc["sbp"][i].mean,
                         fc["hr"][i].mean, v.avpu, fc["temp"][i].mean)
            news_fc.append((mins, n))
        eta = None
        if total < 5:
            eta = next((m for m, n in news_fc if n >= 5), None)
        curves["news2"] = [(v.time, total, total, total)] + [
            (v.time + timedelta(minutes=m), n, n, n) for m, n in news_fc]

        conf = 92.0 - 35 * len(unreliable) - (25.0 if len(h["t"]) < 30 else 0.0)
        res = AIResult(raw=v, valid=valid, unreliable=unreliable, sqi=dict(self.sqi),
                       rhythm=rhythm, af_prob=af_p, af_note=af_note, news2=total,
                       news2_parts=parts, news2_risk=risk_band, news2_response=response,
                       risks=risks, drivers=drivers, index=index, level=level, forecasts=fc,
                       curves=curves, news2_forecast=news_fc, news2_eta=eta,
                       confidence=max(30.0, conf))
        res.summary = self._summary(res)
        return res

    # -----------------------------------------------------------------------------------
    def _summary(self, r: AIResult) -> List[Tuple[str, str]]:
        v, val = r.raw, r.valid
        out = [(f"{v.time:%H:%M} status. ",
                f"HR {val['hr']:.0f} ({r.rhythm.lower()}), BP {val['sbp']:.0f}/{val['dbp']:.0f} "
                f"(MAP {val['map']:.0f}), SpO₂ {val['spo2']:.0f}%"
                f"{' on O₂' if v.o2 else ' on air'}, RR {val['rr']:.0f}, T {val['temp']:.1f} °C. "
                f"NEWS2 {r.news2} ({r.news2_risk.lower()} risk): {r.news2_response.lower()}.")]
        n60 = r.news2_forecast[-1][1]
        if r.news2_eta is not None:
            out.append(("Prediction. ", f"NEWS2 expected to reach 5 in about {r.news2_eta} min "
                                        f"({n60} at +60 min) if current trends continue."))
        else:
            word = "rise" if n60 > r.news2 else "fall" if n60 < r.news2 else "stay"
            out.append(("Prediction. ", f"NEWS2 expected to {word} "
                                        f"{'at' if word == 'stay' else 'to'} {n60} within 60 min."))
        top = sorted(((p, k) for k, p in r.risks.items()), reverse=True)
        hi = [(p, k) for p, k in top if p >= 0.3]
        if hi:
            out.append(("Risks. ", "; ".join(f"{k} {p * 100:.0f}% ({r.drivers[k]})"
                                             for p, k in hi) + "."))
        else:
            out.append(("Risks. ", "No elevated deterioration risk in the next 60 min."))
        if r.unreliable:
            out.append(("Signal quality. ", "; ".join(
                f"{k.upper()} {why} - showing last valid value, alarms suppressed"
                for k, why in r.unreliable.items()) + "."))
        if r.af_prob > 0.7:
            out.append(("Rhythm. ", "Irregularly irregular rhythm consistent with atrial "
                                    "fibrillation. Confirm with 12-lead ECG."))
        return out
