# AIcare 5

Demo GUI of a portable intensive care monitor with a visible AI layer, written in
Python with [Flet](https://flet.dev). It runs as a desktop app, or as a fully static
web app where Python runs in the browser via Pyodide (no server).

**Live demo:** https://logenum.github.io/aicare5-demo/

The patient is **simulated**. This is a technology demonstrator, **not a medical
device**: the AI models are hand-calibrated for demonstration and have not been
trained or validated on patient data.

## Run

Desktop window:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "flet[all]==1.1.3" numpy
python3 main.py
```

In a browser instead (local dev server): `flet run --web main.py`

## Publish as a static website

```bash
flet publish main.py --assets assets --distpath dist --base-url aicare5-demo --route-url-strategy hash
ghp-import -n -p -f dist      # pip install ghp-import; -n adds .nojekyll
```

GitHub Pages serves the `gh-pages` branch (Settings → Pages → Deploy from a branch →
`gh-pages` / root). The first visit downloads Pyodide and numpy from a CDN
(about 10-15 s); after that everything runs in the visitor's browser.

## What's on screen

**Monitor**
- Live sweeping waveforms: ECG lead II, SpO₂ plethysmogram, impedance respiration
- Numerics: HR, SpO₂ (with perfusion index), NIBP systolic/diastolic (MAP), RR,
  core temperature, each with its alarm limits
- Alarm bar with IEC 60601-1-8 priorities (red !!! high, yellow !! medium,
  cyan ! low), audio pause, alarm log
- Patient banner, battery and connectivity (it is a portable monitor)

**AIcare AI** (`aicare/ai_engine.py`)
- **Artifact detection**: signal-quality indices (ECG kurtosis, pleth skewness). When a
  signal is corrupted the AI holds the last valid value (`AI HOLD`) and suppresses the
  false alarm, and the alarm bar shows how many it has suppressed.
- **Rhythm analysis**: atrial fibrillation detection from RR-interval irregularity
- **NEWS2** early warning score with its components and clinical response
- **Risk models** for the next 60 min: sepsis, respiratory failure, haemodynamic
  instability and AF, each with its top contributing factors, combined into a 0-100
  deterioration index
- **Forecast**: damped-trend forecasts of every vital sign to +60 min with 80% bands,
  the NEWS2 they imply, and a warning such as "NEWS2 ≥5 predicted in ~25 min"
- **AI summary** in plain language

## Demo controls

- **Scenario**: Stable post-op, Early sepsis, Respiratory deterioration, New-onset
  atrial fibrillation, Post-op bleeding, or Auto cycle (all in turn)
- **Trend simulation**: real time, ×10, ×60 or ×300. Waveforms always run in real time.
- **Patient movement**: 15 s motion artifact on the ECG and SpO₂ probe
- **O₂ 4 L/min**: supplemental oxygen (raises SpO₂, adds 2 NEWS2 points)
- **Audio pause 2 min** acknowledges the alarms

## Layout

```
main.py                entry point (ft.run)
assets/fonts/          JetBrains Mono (OFL) for numerics
aicare/config.py       patient, alarm limits, scenarios, colour theme
aicare/patient.py      vital-sign simulator and real-time waveform synthesis
aicare/ai_engine.py    SQI, AF detection, NEWS2, risk models, forecasts, summary
aicare/controller.py   GUI-free state: simulation cycle, sweep buffers, alarms
aicare/widgets.py      Flet widgets: parameter tiles, waveforms, gauge, tables
aicare/charts.py       canvas trend chart with forecast band and hover readout
aicare/app.py          Flet page layout, controls, waveform and refresh loops
```

Only `widgets.py`, `charts.py` and `app.py` depend on Flet. The simulation and AI
code is plain Python + numpy, which Pyodide ships.
