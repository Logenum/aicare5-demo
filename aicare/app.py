"""AIcare 5 Flet user interface. Runs on the desktop or fully in the browser (Pyodide)."""

from __future__ import annotations

import asyncio
import time

import flet as ft
import numpy as np

from .charts import TrendChart
from .config import ALARM_LIMITS, APP_NAME, APP_VERSION, AUTO_CYCLE, C, PATIENT, SCENARIOS
from .controller import SPEEDS, MonitorController
from .widgets import (MONO, STATUS_ICON, Panel, ParamTile, RiskBar, ScoreGauge, StatusBadge,
                      Table, Waveform, caption, muted)

FRAME_S = 0.1            # waveform refresh period
PRIO_MARK = {"high": "!!!", "medium": "!!", "low": "!", "info": ""}
PRIO_COLOR = {"high": C["alarm_high"], "medium": C["alarm_medium"], "low": C["alarm_low"],
              "info": C["muted"]}
TAB_NAMES = ["Trends & AI forecast", "NEWS2", "Alarm log", "About"]
NEWS_STATUS = {"LOW": "good", "LOW-MEDIUM": "warning", "MEDIUM": "serious", "HIGH": "critical"}
LEVEL_STATUS = {"LOW": "good", "MODERATE": "warning", "HIGH": "serious", "CRITICAL": "critical"}


def limits(key: str) -> str:
    lo, hi = ALARM_LIMITS[key]
    return f"{hi if hi is not None else '--'}\n{lo if lo is not None else '--'}"


def dropdown(label: str, options, value: str, on_select, width: float) -> ft.Dropdown:
    return ft.Dropdown(label=label, value=value, width=width, dense=True, text_size=13,
                       options=[ft.DropdownOption(key=k, text=t) for k, t in options],
                       on_select=on_select, border_color=C["border"], bgcolor=C["surface2"],
                       enable_search=False)


class MonitorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.ctl = MonitorController()
        self.busy = True
        self.flash = False
        self.tab = 0
        self.alarm_version_shown = -1
        self._build()

    # =====================================================================================
    # UI construction
    # =====================================================================================
    def _build(self):
        self.progress = ft.ProgressBar(value=0, bar_height=3, color=C["brand"],
                                       bgcolor=C["surface"])
        main = ft.ResponsiveRow([self._build_monitor(), self._build_ai()], spacing=8,
                                run_spacing=8)
        self.root = ft.Column([
            self.progress,
            self._build_header(),
            self._build_alarm_bar(),
            self._build_controls(),
            main,
            self._build_tabs(),
            muted(f"{APP_NAME} {APP_VERSION} · simulated patient · technology demonstrator, "
                  f"not a medical device - not for clinical use", text_align=ft.TextAlign.CENTER),
        ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def _build_header(self) -> ft.Control:
        logo = ft.Text(spans=[
            ft.TextSpan("AI", ft.TextStyle(color=C["accent"], weight=ft.FontWeight.W_800,
                                           font_family=MONO)),
            ft.TextSpan("care", ft.TextStyle(color=C["brand"], weight=ft.FontWeight.W_800)),
            ft.TextSpan(" 5", ft.TextStyle(color=C["text"], weight=ft.FontWeight.W_300)),
        ], size=26)
        p = PATIENT
        patient = ft.Column([
            ft.Text(spans=[ft.TextSpan(p["name"], ft.TextStyle(weight=ft.FontWeight.BOLD,
                                                               color=C["text"])),
                           ft.TextSpan(f"   {p['sex']} · {p['age']} y · {p['weight_kg']} kg",
                                       ft.TextStyle(color=C["text2"]))], size=14),
            muted(f"{p['id']}  ·  {p['bed']}  ·  Adult"),
        ], spacing=0)
        self.b_news = StatusBadge("NEWS2 0", "National Early Warning Score 2 (RCP)")
        self.b_index = StatusBadge("AI LOW", "AI deterioration index, next 60 min")
        self.b_rhythm = StatusBadge("Sinus", "AI rhythm analysis")
        self.lbl_battery = ft.Text(size=12, color=C["text2"])
        self.ic_battery = ft.Icon(ft.Icons.BATTERY_5_BAR, size=18, color=C["good"])
        self.lbl_clock = ft.Text("00:00:00", size=22, weight=ft.FontWeight.BOLD,
                                 font_family=MONO, color=C["text"])
        self.lbl_date = muted()
        badges = ft.Row([b.view for b in (self.b_news, self.b_index, self.b_rhythm)],
                        spacing=6, tight=True)
        right = ft.Row([
            ft.Row([ft.Icon(ft.Icons.WIFI, size=16, color=C["text2"]), self.ic_battery,
                    self.lbl_battery], spacing=4, tight=True),
            ft.Column([self.lbl_clock, self.lbl_date], spacing=0,
                      horizontal_alignment=ft.CrossAxisAlignment.END)],
            spacing=16, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        return Panel(controls=[ft.Row([logo, patient, badges, right], wrap=True, spacing=20,
                                      run_spacing=8,
                                      alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                      vertical_alignment=ft.CrossAxisAlignment.CENTER)]).view

    def _build_alarm_bar(self) -> ft.Control:
        self.lbl_alarm = ft.Text("No active alarms", size=16, weight=ft.FontWeight.BOLD,
                                 color=C["text2"])
        self.lbl_alarm_more = ft.Text("", size=12, color=C["text2"])
        self.lbl_suppressed = ft.Text("", size=12, weight=ft.FontWeight.BOLD, color="#000000")
        self.chip_suppressed = ft.Container(
            self.lbl_suppressed, bgcolor=C["accent"], border_radius=12, visible=False,
            padding=ft.Padding.symmetric(horizontal=10, vertical=3),
            tooltip="Alarms the AI identified as caused by motion artifact")
        self.alarm_bar = ft.Container(
            ft.Row([ft.Row([self.lbl_alarm, self.lbl_alarm_more], spacing=12, wrap=True,
                           expand=True),
                    self.chip_suppressed], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=C["surface2"], border_radius=6, on_click=lambda _: self._select_tab(2),
            padding=ft.Padding.symmetric(horizontal=14, vertical=8),
            border=ft.Border.all(1, C["border"]))
        return self.alarm_bar

    def _build_controls(self) -> ft.Control:
        self.dd_scen = dropdown("Scenario", [(s, s) for s in [AUTO_CYCLE, *SCENARIOS]],
                                AUTO_CYCLE, self._on_scenario, 280)
        self.dd_speed = dropdown("Trend simulation", [(str(v), n) for n, v in SPEEDS],
                                 str(self.ctl.dt), self._on_speed, 160)
        self.btn_pause = ft.OutlinedButton("Pause", icon=ft.Icons.PAUSE, on_click=self._on_pause)
        self.btn_move = ft.OutlinedButton(
            "Patient movement", icon=ft.Icons.DIRECTIONS_RUN, on_click=self._on_move,
            tooltip="Demo: motion artifact on ECG and SpO₂ - watch the AI suppress false alarms")
        self.sw_o2 = ft.Switch(label="O₂ 4 L/min", value=False, on_change=self._on_o2,
                               active_color=C["spo2"])
        self.btn_silence = ft.OutlinedButton("Audio pause 2 min",
                                             icon=ft.Icons.NOTIFICATIONS_PAUSED,
                                             on_click=self._on_silence)
        self.lbl_phase = muted()
        return ft.Row([self.dd_scen, self.dd_speed, self.btn_pause, self.btn_move, self.sw_o2,
                       self.btn_silence, self.lbl_phase],
                      wrap=True, spacing=8, run_spacing=8,
                      vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _build_monitor(self) -> ft.Control:
        wave_col, tile_col = {"xs": 7, "sm": 8}, {"xs": 5, "sm": 4}
        self.w_ecg = Waveform("II", C["ecg"], (-0.45, 1.15), scale_label="25 mm/s  1 mV")
        self.w_pleth = Waveform("Pleth", C["spo2"], (-0.25, 1.35))
        self.w_resp = Waveform("Resp", C["resp"], (-1.3, 1.3), scale_label="12.5 mm/s")
        self.t_hr = ParamTile("HR", "bpm", "ecg", limits("hr"))
        self.t_spo2 = ParamTile("SpO₂", "%", "spo2", limits("spo2"))
        self.t_rr = ParamTile("RR", "rpm", "resp", limits("rr"))
        self.t_bp = ParamTile("NIBP", "mmHg", "bp", limits("sbp"), big=44, height=112,
                              col={"xs": 12, "sm": 8})
        self.t_temp = ParamTile("Temp", "°C", "temp", limits("temp"), big=44, height=112,
                                col={"xs": 12, "sm": 4})
        rows = []
        for wave, tile in ((self.w_ecg, self.t_hr), (self.w_pleth, self.t_spo2),
                           (self.w_resp, self.t_rr)):
            wave.view.col, tile.view.col = wave_col, tile_col
            rows.append(ft.ResponsiveRow([wave.view, tile.view], spacing=6, run_spacing=6))
        rows.append(ft.ResponsiveRow([self.t_bp.view, self.t_temp.view], spacing=6,
                                     run_spacing=6))
        return ft.Column(rows, spacing=6, col={"xs": 12, "lg": 8})

    def _build_ai(self) -> ft.Control:
        self.lbl_conf = muted()
        head = ft.Row([ft.Text(spans=[
            ft.TextSpan("◆ ", ft.TextStyle(color=C["accent"])),
            ft.TextSpan("AI", ft.TextStyle(color=C["accent"], weight=ft.FontWeight.W_800,
                                           font_family=MONO)),
            ft.TextSpan("care decision support", ft.TextStyle(color=C["accent"],
                                                              weight=ft.FontWeight.W_800)),
            ], size=14),
            self.lbl_conf], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True)
        self.gauge = ScoreGauge()
        self.lbl_news = ft.Text(size=12)
        self.lbl_pred = ft.Text(size=12)
        top = ft.Row([ft.Column([caption("Deterioration index (60 min)"), self.gauge.view],
                                spacing=2),
                      ft.Column([self.lbl_news, self.lbl_pred], spacing=8, expand=True)],
                     vertical_alignment=ft.CrossAxisAlignment.START)
        self.risk_bars = {n: RiskBar(n) for n in ("Sepsis", "Respiratory failure",
                                                  "Haemodynamic instability", "Arrhythmia (AF)")}
        self.sqi_bars = {n: ft.ProgressBar(value=1, bar_height=6, border_radius=3,
                                           bgcolor=C["grid"], color=C["good"], expand=True)
                         for n in ("ECG", "Pleth")}
        self.sqi_lbls = {n: ft.Text("", size=11, color=C["text2"], width=88,
                                    text_align=ft.TextAlign.RIGHT) for n in self.sqi_bars}
        sqi = ft.Column([ft.Row([ft.Text(n, size=12, color=C["text2"], width=44),
                                 self.sqi_bars[n], self.sqi_lbls[n]],
                                vertical_alignment=ft.CrossAxisAlignment.CENTER)
                         for n in self.sqi_bars], spacing=4)
        self.summary = ft.Text(size=12.5, color=C["text2"], selectable=True)
        part = {"xs": 12, "md": 6, "lg": 12}
        a = ft.Column([top, caption("Risk of deterioration (next 60 min)")]
                      + [rb.view for rb in self.risk_bars.values()], spacing=6, col=part)
        b = ft.Column([caption("Signal quality (AI artifact detection)"), sqi,
                       caption("AI summary"), self.summary], spacing=6, col=part)
        return Panel(controls=[head, ft.ResponsiveRow([a, b], spacing=16, run_spacing=10)],
                     col={"xs": 12, "lg": 4}).view

    def _build_tabs(self) -> ft.Control:
        chart_col = {"xs": 12, "md": 6, "xl": 4}
        self.ch_hr = TrendChart("Heart rate", "bpm", y_floor=0, col=chart_col)
        self.ch_spo2 = TrendChart("SpO₂", "%", decimals=0, y_ceil=100, col=chart_col)
        self.ch_bp = TrendChart("Blood pressure", "mmHg", y_floor=0, col=chart_col)
        self.ch_rr = TrendChart("Respiratory rate", "/min", y_floor=0, col=chart_col)
        self.ch_temp = TrendChart("Temperature", "°C", decimals=1, col=chart_col)
        self.ch_news = TrendChart("NEWS2 (forecast from predicted vitals)", "", y_floor=0,
                                  col=chart_col)
        self.charts = [self.ch_hr, self.ch_spo2, self.ch_bp, self.ch_rr, self.ch_temp,
                       self.ch_news]
        self.dd_window = dropdown("History window", [("1", "1 h"), ("3", "3 h"), ("6", "6 h")],
                                  "3", self._on_window, 150)
        trends = ft.Column([ft.Row([self.dd_window], alignment=ft.MainAxisAlignment.END),
                            ft.ResponsiveRow([c.view for c in self.charts], spacing=6,
                                             run_spacing=6)], spacing=6)
        self.tbl_news = Table(["Parameter", "Value", "Score"], [4, 3, 2], ["l", "r", "c"])
        self.tbl_news_fc = Table(["", "Now", "+15", "+30", "+45", "+60 min"], [3, 2, 2, 2, 2, 2],
                                 ["l", "c", "c", "c", "c", "c"])
        self.lbl_news_total = ft.Text(size=14)
        news = ft.ResponsiveRow([
            ft.Column([caption("NEWS2 components (validated values)"), self.tbl_news.view,
                       self.lbl_news_total], spacing=6, col={"xs": 12, "md": 6}),
            ft.Column([caption("AI forecast of NEWS2"), self.tbl_news_fc.view,
                       muted("Score computed from the AI-forecast vital signs. NEWS2 thresholds: "
                             "0-4 low, 3 in one parameter low-medium, 5-6 medium, ≥7 high.")],
                      spacing=6, col={"xs": 12, "md": 6}),
        ], spacing=16, run_spacing=12)
        self.lbl_alarm_sum = ft.Text(color=C["text2"])
        self.tbl_alarm = Table(["Time", "Priority", "Message", "State"], [2, 2, 8, 3],
                               scroll=True, expand=True)
        alarms = ft.Column([self.lbl_alarm_sum, ft.Container(self.tbl_alarm.view, height=420)],
                           spacing=6)
        about = ft.Text(spans=[
            ft.TextSpan("AIcare 5 ", ft.TextStyle(weight=ft.FontWeight.BOLD, color=C["text"])),
            ft.TextSpan(
                "is a demonstrator of a portable intensive care monitor with an AI layer. The "
                "patient, waveforms and vital signs are simulated.\n\n"
                "AI features: signal-quality analysis (ECG kurtosis, pleth skewness) that detects "
                "motion artifact, holds the last valid value and suppresses false alarms; "
                "atrial fibrillation detection from RR-interval irregularity; NEWS2 early warning "
                "score; sepsis, respiratory failure and haemodynamic instability risk models "
                "with their top contributing factors; 60-minute vital-sign forecasts and the "
                "NEWS2 they imply; and a plain-language summary.\n\n"),
            ft.TextSpan("Not a medical device. ", ft.TextStyle(weight=ft.FontWeight.BOLD,
                                                               color=C["warning"])),
            ft.TextSpan("The risk models are hand-calibrated for demonstration and have not been "
                        "trained or validated on patient data. Do not use for clinical decisions."),
        ], size=13, color=C["text2"])

        self.tab_bodies = [trends, news, alarms, ft.Container(about, padding=8)]
        for i, body in enumerate(self.tab_bodies):
            body.visible = i == 0
        self.tabs = ft.Tabs(
            length=len(TAB_NAMES), selected_index=0, on_change=self._on_tab,
            content=ft.TabBar(tabs=[ft.Tab(label=n) for n in TAB_NAMES], scrollable=True,
                              tab_alignment=ft.TabAlignment.START,
                              label_color=C["text"], unselected_label_color=C["text2"],
                              indicator_color=C["brand"], divider_color=C["border"]))
        return Panel(controls=[self.tabs] + self.tab_bodies).view

    # =====================================================================================
    # Controls
    # =====================================================================================
    def _on_scenario(self, e):
        self.ctl.set_scenario(e.control.value)
        self.refresh()

    def _on_speed(self, e):
        self.ctl.dt = int(e.control.value)
        self.refresh()

    def _on_pause(self, _):
        self.ctl.paused = not self.ctl.paused
        self.btn_pause.content = "Resume" if self.ctl.paused else "Pause"
        self.btn_pause.icon = ft.Icons.PLAY_ARROW if self.ctl.paused else ft.Icons.PAUSE
        self.refresh()

    def _on_move(self, _):
        self.ctl.patient_movement()
        self.refresh()

    def _on_o2(self, e):
        self.ctl.set_o2(bool(e.control.value))
        self.refresh()

    def _on_silence(self, _):
        self.ctl.silence(2)
        self.refresh()

    def _on_window(self, e):
        for ch in self.charts:
            ch.set_window(int(e.control.value))
        self.page.update()

    def _on_tab(self, e):
        self._select_tab(int(e.control.selected_index), from_tabs=True)

    def _select_tab(self, i: int, from_tabs: bool = False):
        self.tab = i
        if not from_tabs:
            self.tabs.selected_index = i
        for k, body in enumerate(self.tab_bodies):
            body.visible = k == i
        self.refresh()

    # =====================================================================================
    # Lifecycle
    # =====================================================================================
    async def start(self):
        self.page.add(self.root)
        await self.warm_up()
        last = time.monotonic()
        next_tick = last + 1.0
        while True:
            await asyncio.sleep(FRAME_S)
            now = time.monotonic()
            self.ctl.advance_waves(min(0.5, now - last))
            last = now
            if now >= next_tick:
                next_tick = max(next_tick + 1.0, now - 0.5)
                self.ctl.tick()
                self.refresh(waves=False)
            self._draw_waves()

    async def warm_up(self):
        """Run the warm-up in slices so the browser stays responsive and shows progress."""
        self.busy = True
        self.ctl.warming_up = True
        steps = self.ctl.warmup_steps()
        self.lbl_phase.value = "Loading two hours of patient history and AI models…"
        for i in range(steps):
            self.ctl.cycle(60)
            if i % 10 == 0:
                self.progress.value = i / steps
                self.page.update()
                await asyncio.sleep(0)
        self.ctl.warming_up = False
        self.progress.visible = False
        self.busy = False
        self.ctl.advance_waves(1.0)
        self.refresh()

    def _draw_waves(self):
        r = self.ctl.result
        sw = self.ctl.sweeps
        art = r is not None and r.unreliable
        self.w_ecg.set_data(sw["ecg"].y.copy(), sw["ecg"].i,
                            "AI: ECG ARTIFACT" if art and "hr" in r.unreliable else "")
        self.w_pleth.set_data(sw["pleth"].y.copy(), sw["pleth"].i,
                              "AI: MOTION ARTIFACT" if art and "spo2" in r.unreliable else "")
        self.w_resp.set_data(sw["resp"].y.copy(), sw["resp"].i)
        for w in (self.w_ecg, self.w_pleth, self.w_resp):
            if w.w:
                w.canvas.update()

    # =====================================================================================
    # UI refresh
    # =====================================================================================
    def refresh(self, waves: bool = True):
        ctl, r = self.ctl, self.ctl.result
        if r is None:
            return
        v, val = r.raw, r.valid
        self.flash = not self.flash

        # ---- header
        self.lbl_clock.value = f"{v.time:%H:%M:%S}"
        spd = next(n for n, s in SPEEDS if s == ctl.dt)
        self.lbl_date.value = (f"{v.time:%a %d %b %Y}  ·  trends {spd}"
                               + ("  ·  PAUSED" if ctl.paused else ""))
        self.lbl_battery.value = f"{ctl.battery:.0f}%"
        self.ic_battery.color = C["good"] if ctl.battery > 20 else C["critical"]
        ph = ctl.pt.phase
        left = ph.minutes - ctl.pt.phase_elapsed / 60
        self.lbl_phase.value = f"Phase: {ph.label}  ({left:.0f} min left)"
        self.b_news.set(f"NEWS2 {r.news2}", NEWS_STATUS[r.news2_risk])
        self.b_index.set(r.level, LEVEL_STATUS[r.level], prefix="AI ")
        self.b_rhythm.set(r.rhythm, "warning" if r.af_prob > 0.4 or "noisy" in r.rhythm
                          else "good")

        # ---- alarm bar
        top = ctl.top_alarm()
        active = [a for a in ctl.active_alarms.values() if not a.suppressed]
        if top:
            col = PRIO_COLOR[top.priority]
            on = self.flash or top.acked or ctl.audio_paused() or top.priority == "low"
            self.alarm_bar.bgcolor = col if on else C["surface2"]
            fg = "#000000" if on else col
            self.lbl_alarm.value = f"{PRIO_MARK[top.priority]}  {top.text}"
            self.lbl_alarm.color = self.lbl_alarm_more.color = fg
            more = len(active) - 1
            self.lbl_alarm_more.value = ((f"+{more} more" if more else "")
                                         + ("   🔕 audio paused" if ctl.audio_paused() else ""))
        else:
            self.alarm_bar.bgcolor = C["surface2"]
            self.lbl_alarm.value = "No active alarms"
            self.lbl_alarm.color = C["text2"]
            self.lbl_alarm_more.value = "🔕 audio paused" if ctl.audio_paused() else ""
        self.chip_suppressed.visible = ctl.suppressed_count > 0
        self.lbl_suppressed.value = f"◆ AI suppressed {ctl.suppressed_count} false alarm" \
                                    + ("s" if ctl.suppressed_count != 1 else "")

        # ---- parameter tiles
        def alarm_for(*keys):
            prios = [ctl.active_alarms[f"phys:{k}"].priority for k in keys
                     if f"phys:{k}" in ctl.active_alarms]
            return min(prios, key=["high", "medium", "low"].index) if prios else None

        hr_bad, spo2_bad = "hr" in r.unreliable, "spo2" in r.unreliable
        self.t_hr.set(f"{val['hr']:.0f}", "", f"{r.rhythm}" + (f" · AF {r.af_prob * 100:.0f}%"
                                                              if r.af_prob > 0.4 else ""),
                      "AI HOLD" if hr_bad else "", alarm_for("hr"), self.flash, dim=hr_bad)
        self.t_spo2.set(f"{val['spo2']:.0f}", "", f"PI {2.8 * v.perf:.1f}  ·  "
                        f"{'O₂ 4 L/min' if v.o2 else 'room air'}"
                        + (f"  ·  raw {v.spo2:.0f}" if spo2_bad else ""),
                        "AI HOLD" if spo2_bad else "", alarm_for("spo2"), self.flash,
                        dim=spo2_bad)
        self.t_rr.set(f"{val['rr']:.0f}", "", "Impedance", "", alarm_for("rr"), self.flash)
        self.t_bp.set(f"{val['sbp']:.0f}/{val['dbp']:.0f}", f"({val['map']:.0f})",
                      f"MAP {val['map']:.0f}  ·  shock index {val['hr'] / max(1, val['sbp']):.2f}"
                      f"  ·  auto 15 min", "", alarm_for("sbp", "map"), self.flash)
        self.t_temp.set(f"{val['temp']:.1f}", "", "Core (T1)", "", alarm_for("temp"), self.flash)

        self._refresh_ai(r)
        if self.tab == 0:
            self._refresh_charts(r)
        elif self.tab == 1:
            self._refresh_news(r)
        if ctl.alarm_version != self.alarm_version_shown:
            self.alarm_version_shown = ctl.alarm_version
            self._refresh_alarms()
        if waves:
            self._draw_waves()
        self.page.update()

    def _refresh_ai(self, r):
        self.lbl_conf.value = f"confidence {r.confidence:.0f}%"
        self.gauge.set_value(r.index, r.level)
        st = NEWS_STATUS[r.news2_risk]
        self.lbl_news.spans = [
            ft.TextSpan("NEWS2\n", ft.TextStyle(color=C["muted"], size=11)),
            ft.TextSpan(f"{r.news2}  ", ft.TextStyle(size=22, weight=ft.FontWeight.BOLD,
                                                     color=C[st])),
            ft.TextSpan(f"{r.news2_risk} · {r.news2_response}", ft.TextStyle(color=C["text2"]))]
        n60 = r.news2_forecast[-1][1]
        if r.news2_eta is not None:
            self.lbl_pred.spans = [ft.TextSpan(f"▲ NEWS2 ≥5 predicted in ~{r.news2_eta} min",
                                               ft.TextStyle(color=C["warning"],
                                                            weight=ft.FontWeight.BOLD))]
        else:
            arrow = "↑" if n60 > r.news2 else "↓" if n60 < r.news2 else "→"
            self.lbl_pred.spans = [ft.TextSpan("Forecast +60 min  ", ft.TextStyle(color=C["muted"])),
                                   ft.TextSpan(f"NEWS2 {arrow} {n60}",
                                               ft.TextStyle(color=C["text"],
                                                            weight=ft.FontWeight.BOLD))]
        for k, rb in self.risk_bars.items():
            rb.set_value(r.risks[k], r.drivers.get(k, ""))
        for k, bar in self.sqi_bars.items():
            q = r.sqi[k]
            bar.value = max(0.02, q)
            bar.color = C["good"] if q >= 0.7 else C["warning"] if q >= 0.5 else C["critical"]
            self.sqi_lbls[k].value = f"{q:.2f} " + ("good" if q >= 0.7 else "fair" if q >= 0.5
                                                    else "ARTIFACT")
        spans = []
        for i, (lead, text) in enumerate(r.summary):
            spans += [ft.TextSpan(("\n\n" if i else "") + lead,
                                  ft.TextStyle(color=C["text"], weight=ft.FontWeight.BOLD)),
                      ft.TextSpan(text)]
        self.summary.spans = spans

    def _refresh_news(self, r):
        val, raw = r.valid, r.raw
        values = {"Resp. rate": f"{val['rr']:.0f} /min", "SpO₂": f"{val['spo2']:.0f} %",
                  "Air or O₂": "O₂" if raw.o2 else "Air",
                  "Systolic BP": f"{val['sbp']:.0f} mmHg", "Pulse": f"{val['hr']:.0f} /min",
                  "Consciousness": {"A": "Alert", "C": "New confusion", "V": "Voice",
                                    "P": "Pain", "U": "Unresponsive"}[raw.avpu],
                  "Temperature": f"{val['temp']:.1f} °C"}
        sc_col = {0: C["text2"], 1: C["warning"], 2: C["serious"], 3: C["critical"]}
        self.tbl_news.set_rows([[k, values[k], (str(p), sc_col[p], True)]
                                for k, p in r.news2_parts.items()])
        st = NEWS_STATUS[r.news2_risk]
        self.lbl_news_total.spans = [
            ft.TextSpan(f"Total {r.news2}  ", ft.TextStyle(weight=ft.FontWeight.BOLD,
                                                           color=C[st], size=18)),
            ft.TextSpan(f"{r.news2_risk} clinical risk · {r.news2_response}",
                        ft.TextStyle(color=C["text2"]))]
        fc = dict(r.news2_forecast)
        cells = [str(r.news2)] + [str(fc[m]) for m in (15, 30, 45, 60)]
        vit = r.forecasts

        def at(key, m, fmt):
            return fmt.format(vit[key][m // 5 - 1].mean)

        self.tbl_news_fc.set_rows([
            [("NEWS2", C["text"], True)] + [(c, C["accent"] if i else C["text"], True)
                                            for i, c in enumerate(cells)],
            ["HR", f"{val['hr']:.0f}"] + [at("hr", m, "{:.0f}") for m in (15, 30, 45, 60)],
            ["SpO₂", f"{val['spo2']:.0f}"] + [at("spo2", m, "{:.0f}") for m in (15, 30, 45, 60)],
            ["Sys BP", f"{val['sbp']:.0f}"] + [at("sbp", m, "{:.0f}") for m in (15, 30, 45, 60)],
            ["RR", f"{val['rr']:.0f}"] + [at("rr", m, "{:.0f}") for m in (15, 30, 45, 60)],
            ["Temp", f"{val['temp']:.1f}"] + [at("temp", m, "{:.1f}") for m in (15, 30, 45, 60)],
        ])

    def _refresh_alarms(self):
        ctl = self.ctl
        rows = []
        for a in ctl.alarms[:200]:
            if a.priority == "info":
                prio, state = ("ℹ EVENT", C["muted"]), "EVENT"
            else:
                prio = (f"{PRIO_MARK[a.priority]} {a.priority.upper()}", PRIO_COLOR[a.priority])
                state = ("ACTIVE" if a.active else "CLEARED") + ("" if a.acked else " • UNACK")
            if a.suppressed:
                prio, state = ("◆ AI", C["accent"]), "SUPPRESSED"
            rows.append([a.raised.strftime("%H:%M:%S"), (prio[0], prio[1], True),
                         (a.text, C["text"] if a.active and not a.suppressed else C["text2"]),
                         (state, C["text"] if a.active and not a.acked else C["muted"])])
        self.tbl_alarm.set_rows(rows)
        self.lbl_alarm_sum.value = (f"{len([a for a in ctl.active_alarms.values() if not a.suppressed])}"
                                    f" active, {len(ctl.unacked())} unacknowledged, "
                                    f"{ctl.suppressed_count} false alarms suppressed by AI")

    def _refresh_charts(self, r):
        ai = self.ctl.ai
        now = r.raw.time.timestamp()

        def fc(key, color):
            pts = r.curves[key]
            return dict(color=color, t=[p[0].timestamp() for p in pts],
                        mean=[p[1] for p in pts], lo=[p[2] for p in pts], hi=[p[3] for p in pts])

        t, hr = ai.series("hr")
        lim = lambda k: [(x, f"Alarm {x}") for x in ALARM_LIMITS[k] if x is not None]
        self.ch_hr.set_data(now, [dict(name="HR", color=C["ecg"], t=t, y=hr)],
                            [fc("hr", C["ecg"])], lim("hr"))
        _, sp = ai.series("spo2")
        self.ch_spo2.set_data(now, [dict(name="SpO₂", color=C["spo2"], t=t, y=sp)],
                              [fc("spo2", C["spo2"])], lim("spo2"))
        _, sbp = ai.series("sbp")
        _, mp = ai.series("map")
        self.ch_bp.set_data(now, [dict(name="Systolic", color=C["bp"], t=t, y=sbp),
                                  dict(name="MAP", color="#ffa3a6", t=t, y=mp)],
                            [fc("sbp", C["bp"]), fc("map", "#ffa3a6")],
                            [(90, "Sys 90"), (65, "MAP 65")])
        _, rr = ai.series("rr")
        self.ch_rr.set_data(now, [dict(name="RR", color=C["resp"], t=t, y=rr)],
                            [fc("rr", C["resp"])], lim("rr"))
        _, te = ai.series("temp")
        self.ch_temp.set_data(now, [dict(name="Temp", color=C["temp"], t=t, y=te)],
                              [fc("temp", C["temp"])], lim("temp"))
        _, nw = ai.series("news2")
        self.ch_news.set_data(now, [dict(name="NEWS2", color=C["accent"], t=t,
                                         y=np.asarray(nw, dtype=float))],
                              [fc("news2", C["accent"])], [(5, "Urgent 5"), (7, "Emergency 7")])


async def main(page: ft.Page):
    page.title = f"{APP_NAME} – Portable ICU Monitor (demo)"
    page.theme_mode = ft.ThemeMode.DARK
    page.dark_theme = ft.Theme(color_scheme=ft.ColorScheme(primary=C["brand"],
                                                           surface=C["surface"]))
    page.bgcolor = C["page"]
    page.padding = 10
    page.scroll = ft.ScrollMode.AUTO
    page.fonts = {MONO: "fonts/JetBrainsMono.ttf"}
    await MonitorApp(page).start()
