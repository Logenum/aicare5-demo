"""Static configuration: patient, alarm limits, scenarios and the monitor colour theme."""

APP_NAME = "AIcare 5"
APP_VERSION = "1.0.0"

PATIENT = {"name": "DEMO, Patient", "id": "MRN 000-DEMO", "age": 67, "sex": "M",
           "weight_kg": 82, "bed": "ICU · Bed 4"}

# Physiological alarm limits (adult ICU defaults): (low, high); None = no limit.
ALARM_LIMITS = {
    "hr": (45, 130),
    "spo2": (90, None),
    "sbp": (90, 180),
    "map": (65, None),
    "rr": (8, 28),
    "temp": (35.5, 38.5),
}
# Limits that make an alarm high priority instead of medium.
CRITICAL_LIMITS = {"hr": (40, 150), "spo2": (85, None), "sbp": (80, None), "map": (55, None),
                   "rr": (6, 35)}

# Scenario phases: (minutes, targets). Missing keys keep the previous phase's value.
BASE = dict(hr=78, sbp=126, dbp=72, rr=15, spo2=97, temp=36.9, af=False, avpu="A", perf=1.0)
SCENARIOS = {
    "Stable post-op": [(240, dict(label="Stable"))],
    "Early sepsis": [
        (30, dict(label="Stable")),
        (60, dict(label="SIRS developing", hr=98, sbp=114, dbp=64, rr=20, spo2=95, temp=38.3)),
        (60, dict(label="Sepsis", hr=118, sbp=98, dbp=56, rr=24, spo2=94, temp=38.9, perf=0.7)),
        (60, dict(label="Septic shock", hr=134, sbp=84, dbp=46, rr=29, spo2=92, temp=39.4,
                  avpu="C", perf=0.45)),
    ],
    "Respiratory deterioration": [
        (30, dict(label="Stable")),
        (45, dict(label="Increasing work of breathing", hr=90, rr=22, spo2=94)),
        (45, dict(label="Hypoxaemia", hr=104, rr=27, spo2=90, sbp=134)),
        (60, dict(label="Respiratory failure", hr=118, rr=33, spo2=85, sbp=142, avpu="C")),
    ],
    "New-onset atrial fibrillation": [
        (30, dict(label="Sinus rhythm")),
        (90, dict(label="AF with rapid ventricular response", hr=138, sbp=106, dbp=68, af=True,
                  perf=0.75)),
        (60, dict(label="AF, rate controlled", hr=96, sbp=118, dbp=70, af=True, perf=0.9)),
    ],
    "Post-op bleeding": [
        (30, dict(label="Stable")),
        (60, dict(label="Compensated blood loss", hr=96, sbp=112, dbp=68, rr=18, temp=36.6)),
        (45, dict(label="Haemorrhagic shock developing", hr=116, sbp=94, dbp=58, rr=22,
                  temp=36.3, perf=0.6)),
        (45, dict(label="Decompensated shock", hr=136, sbp=78, dbp=46, rr=27, temp=36.0,
                  perf=0.35, avpu="V")),
    ],
}
AUTO_CYCLE = "Auto cycle (all scenarios)"

# --- Theme ---------------------------------------------------------------------------
# Patient-monitor conventions: black background, one fixed colour per parameter.
# Alarm colours follow IEC 60601-1-8 (red high, yellow medium, cyan low priority).
C = {
    "page": "#000000",
    "surface": "#0e0f11",
    "surface2": "#17191c",
    "border": "#2a2d31",
    "grid": "#1f2225",
    "axis": "#34383d",
    "text": "#ffffff",
    "text2": "#c3c6cb",
    "muted": "#868a91",
    "brand": "#29c4b4",
    "accent": "#8f7cf7",       # AI features
    "ecg": "#3ddc84",
    "spo2": "#2ed3e6",
    "bp": "#ff5a5f",
    "resp": "#ffd23f",
    "temp": "#f2f2f2",
    "good": "#2fb344",
    "warning": "#f5b70a",
    "serious": "#f08c3a",
    "critical": "#e5383b",
    "alarm_high": "#e5383b",
    "alarm_medium": "#f5b70a",
    "alarm_low": "#2ed3e6",
}
