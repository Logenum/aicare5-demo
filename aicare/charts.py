"""Lightweight time-series chart with history, AI forecast band and hover crosshair."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import flet as ft
import flet.canvas as cv
import numpy as np

from .config import C
from .widgets import label, paint

MAX_POINTS = 240          # per history series; keeps each canvas update small


def nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 1)]


class TrendChart:
    """Two stacked canvases: the plot, and a light overlay for the hover readout."""

    def __init__(self, title: str, unit: str, log: bool = False,
                 y_floor: Optional[float] = None, y_ceil: Optional[float] = None,
                 decimals: int = 0, height: float = 230, col=None):
        self.title, self.unit, self.log = title, unit, log
        self.y_floor, self.y_ceil, self.decimals = y_floor, y_ceil, decimals
        self.series: List[dict] = []
        self.forecasts: List[dict] = []
        self.thresholds: List[Tuple[float, str]] = []
        self.now: Optional[float] = None
        self.window_s = 3 * 3600
        self.horizon_s = 3600          # AI forecast horizon shown right of "now"
        self.hover_x: Optional[float] = None
        self.w = self.h = 0.0
        self.plot = cv.Canvas(left=0, top=0, right=0, bottom=0, on_resize=self._on_resize,
                              resize_interval=50)
        self.overlay = cv.Canvas(left=0, top=0, right=0, bottom=0)
        self.view = ft.Container(
            ft.GestureDetector(ft.Stack([self.plot, self.overlay]), hover_interval=40,
                               on_hover=self._on_hover, on_exit=self._on_exit,
                               on_tap_down=self._on_hover),
            height=height, col=col, bgcolor=C["surface"], border_radius=6,
            border=ft.Border.all(1, C["border"]))

    # data -----------------------------------------------------------------------------
    def set_data(self, now: float, series: List[dict], forecasts: List[dict],
                 thresholds: List[Tuple[float, str]] = ()):
        """series: [{name, color, t, y}], forecasts: [{color, t, mean, lo, hi}]"""
        self.now, self.series, self.forecasts = now, series, forecasts
        self.thresholds = list(thresholds)
        self.redraw()

    def set_window(self, hours: float):
        self.window_s = hours * 3600
        self.redraw()

    # geometry ---------------------------------------------------------------------------
    def _plot_rect(self):
        """(left, top, right, bottom)"""
        return 52, 30, self.w - 12, self.h - 24

    def _tf(self, v):
        return math.log10(max(v, 1e-9)) if self.log else v

    def _y_range(self):
        t0 = self.now - self.window_s
        vals = []
        for s in self.series:
            y = np.asarray(s["y"])[np.asarray(s["t"]) >= t0]
            vals.extend(y[np.isfinite(y)].tolist())
        for f in self.forecasts:
            vals.extend(f["lo"])
            vals.extend(f["hi"])
        if not vals:
            return 0, 1
        lo, hi = min(vals), max(vals)
        if self.y_floor is not None:
            lo = max(lo, self.y_floor)
        if self.y_ceil is not None:
            hi = min(hi, self.y_ceil)
        if self.log:
            lo, hi = math.log10(max(lo, 1)), math.log10(max(hi, 1))
            lo, hi = math.floor(lo * 2) / 2, math.ceil(hi * 2) / 2
            if hi - lo < 1:
                hi = lo + 1
            return lo, hi
        pad = max((hi - lo) * 0.12, 10 ** -self.decimals)
        lo, hi = lo - pad, hi + pad
        if self.y_floor is not None:
            lo = max(lo, self.y_floor)
        return lo, hi

    def _scales(self):
        L, T, R, B = self._plot_rect()
        t0, t1 = self.now - self.window_s, self.now + self.horizon_s
        ylo, yhi = self._y_range()

        def X(t):
            return L + (t - t0) / (t1 - t0) * (R - L)

        def Y(v):
            y = B - (self._tf(v) - ylo) / (yhi - ylo) * (B - T)
            return min(B + 2, max(T - 2, y))           # clip to the plot area

        return X, Y, t0, t1, ylo, yhi

    # drawing ----------------------------------------------------------------------------
    def _on_resize(self, e: cv.CanvasResizeEvent):
        self.w, self.h = e.width, e.height
        self.redraw()
        self.plot.update()

    def redraw(self):
        self.plot.shapes = self._draw() if self.w > 60 and self.h > 60 else []
        self.overlay.shapes = self._draw_hover() if self.plot.shapes else []

    def _draw(self) -> List[cv.Shape]:
        unit = "" if len(self.series) >= 2 else f"   {self.unit}"
        s: List[cv.Shape] = [cv.Text(10, 14, alignment=ft.Alignment.CENTER_LEFT, spans=[
            ft.TextSpan(self.title, ft.TextStyle(size=12.5, color=C["text"],
                                                 weight=ft.FontWeight.BOLD)),
            ft.TextSpan(unit, ft.TextStyle(size=12, color=C["muted"]))])]
        if self.now is None:
            return s
        L, T, R, B = self._plot_rect()
        X, Y, t0, t1, ylo, yhi = self._scales()
        xn = X(self.now)

        # forecast region wash
        s.append(cv.Rect(xn, T, R - xn, B - T, paint=paint("text", fill=True, alpha=8)))
        s.append(label(xn + 6, T + 2, "AI FORECAST  (80% band)", "muted", 9.5,
                       align=ft.Alignment.TOP_LEFT))

        # grid + y labels
        if self.log:
            ticks = [10 ** e * m for e in range(int(ylo), int(math.ceil(yhi)) + 1) for m in (1, 3)
                     if ylo <= math.log10(10 ** e * m) <= yhi]
        else:
            ticks = nice_ticks(ylo, yhi, 4)
            if self.decimals == 0:          # integer scales: no repeated rounded labels
                ticks = [tv for tv in ticks if abs(tv - round(tv)) < 1e-9]
        for tv in ticks:
            y = Y(tv)
            s.append(cv.Line(L, y, R, y, paint("grid")))
            lab = (f"{tv / 1000:g}k" if self.log and tv >= 1000 else
                   f"{tv:.{self.decimals}f}" if not self.log else f"{tv:g}")
            s.append(label(L - 6, y, lab, "muted", 10, align=ft.Alignment.CENTER_RIGHT))

        # x labels (thinned on narrow charts)
        step = 1800 if self.window_s <= 3 * 3600 else 3600
        while (R - L) / ((t1 - t0) / step) < 46:
            step *= 2
        t = math.ceil(t0 / step) * step
        while t <= t1:
            s.append(label(X(t), B + 4, datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M"),
                           "muted", 10, align=ft.Alignment.TOP_CENTER))
            t += step
        s.append(cv.Line(L, B, R, B, paint("axis")))

        # thresholds
        for val, lab in self.thresholds:
            if ylo <= self._tf(val) <= yhi:
                y = Y(val)
                s.append(cv.Line(L, y, R, y, paint("critical", alpha=160)))
                s.append(label(L + 4, y - 2, lab, "text2", 10, align=ft.Alignment.BOTTOM_LEFT))

        # forecast band + mean
        for fc in self.forecasts:
            if not fc["t"]:
                continue
            hi = [(X(t), Y(v)) for t, v in zip(fc["t"], fc["hi"])]
            lo = [(X(t), Y(v)) for t, v in zip(fc["t"], fc["lo"])]
            band = [cv.Path.MoveTo(*hi[0])] + [cv.Path.LineTo(*q) for q in hi[1:] + lo[::-1]]
            s.append(cv.Path(band + [cv.Path.Close()], paint=paint(fc["color"], fill=True, alpha=45)))
            s.append(cv.Points([ft.Offset(X(t), Y(v)) for t, v in zip(fc["t"], fc["mean"])],
                               point_mode=cv.PointMode.POLYGON,
                               paint=paint(fc["color"], 2, cap=ft.StrokeCap.ROUND, dash=[6, 4])))

        # history lines, split at gaps
        for ser in self.series:
            tt, yy = np.asarray(ser["t"]), np.asarray(ser["y"], dtype=float)
            m = tt >= t0 - 60
            tt, yy = tt[m], yy[m]
            if len(tt) > MAX_POINTS:
                k = int(math.ceil(len(tt) / MAX_POINTS))
                keep = np.r_[np.arange(0, len(tt) - 1, k), len(tt) - 1]
                tt, yy = tt[keep], yy[keep]
            seg: List[ft.Offset] = []
            for t, v in zip(tt, yy):
                if np.isfinite(v):
                    seg.append(ft.Offset(X(t), Y(v)))
                    continue
                if len(seg) > 1:
                    s.append(cv.Points(seg, point_mode=cv.PointMode.POLYGON,
                                       paint=paint(ser["color"], 2)))
                seg = []
            if len(seg) > 1:
                s.append(cv.Points(seg, point_mode=cv.PointMode.POLYGON,
                                   paint=paint(ser["color"], 2)))
            if len(tt) and np.isfinite(yy[-1]):
                s.append(cv.Circle(X(tt[-1]), Y(yy[-1]), 5, paint("surface", fill=True)))
                s.append(cv.Circle(X(tt[-1]), Y(yy[-1]), 4, paint(ser["color"], fill=True)))

        # now line
        s.append(cv.Line(xn, T, xn, B, paint("text2", alpha=140)))

        # legend (>= 2 series), right-aligned in the title row
        if len(self.series) >= 2:
            x = R
            for ser in reversed(self.series):
                x -= len(ser["name"]) * 6.2
                s.append(label(x, 14, ser["name"], "text2", 10.5, align=ft.Alignment.CENTER_LEFT))
                x -= 14
                s.append(cv.Rect(x, 10, 10, 8, border_radius=2,
                                 paint=paint(ser["color"], fill=True)))
                x -= 14
        return s

    # hover ----------------------------------------------------------------------------
    def _value_at(self, t: float):
        rows = []
        if t <= self.now:
            for s in self.series:
                tt = np.asarray(s["t"])
                if not len(tt):
                    continue
                i = int(np.argmin(np.abs(tt - t)))
                rows.append((s["name"], s["color"], s["y"][i], None))
        else:
            for s, fc in zip(self.series, self.forecasts):
                tt = np.asarray(fc["t"])
                i = int(np.argmin(np.abs(tt - t)))
                rows.append((s["name"] + " (AI)", fc["color"], fc["mean"][i],
                             (fc["lo"][i], fc["hi"][i])))
        return rows

    def _fmt(self, v):
        if self.log:
            return f"{v:,.0f}".replace(",", " ")
        return f"{v:.{self.decimals}f}"

    def _draw_hover(self) -> List[cv.Shape]:
        L, T, R, B = self._plot_rect()
        hx = self.hover_x
        if hx is None or self.now is None or not (L <= hx <= R):
            return []
        _, _, t0, t1, _, _ = self._scales()
        t = t0 + (hx - L) / (R - L) * (t1 - t0)
        rows = self._value_at(t)
        if not rows:
            return []
        lines = [datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M") +
                 ("  forecast" if t > self.now else "")]
        for name, _, v, band in rows:
            txt = f"{name}: {self._fmt(v)} {self.unit}"
            if band:
                txt += f"  [{self._fmt(band[0])}-{self._fmt(band[1])}]"
            lines.append(txt)
        bw = max(len(l) for l in lines) * 6.6 + 30
        bh = 18 * len(lines) + 8
        bx = hx + 12 if hx + 12 + bw < self.w else hx - 12 - bw
        by = T + 8
        s: List[cv.Shape] = [
            cv.Line(hx, T, hx, B, paint("text2", alpha=120)),
            cv.Rect(bx, by, bw, bh, border_radius=6, paint=paint("surface2", fill=True, alpha=245)),
            cv.Rect(bx, by, bw, bh, border_radius=6, paint=paint("border")),
        ]
        for i, l in enumerate(lines):
            y = by + 4 + 18 * i + 9
            if i > 0:
                s.append(cv.Rect(bx + 8, y - 4, 10, 8, border_radius=2,
                                 paint=paint(rows[i - 1][1], fill=True)))
            s.append(label(bx + (24 if i else 8), y, l, "text" if i else "text2", 11,
                           align=ft.Alignment.CENTER_LEFT))
        return s

    def _on_hover(self, e):
        self.hover_x = e.local_position.x
        self.overlay.shapes = self._draw_hover()
        self.overlay.update()

    def _on_exit(self, _):
        self.hover_x = None
        self.overlay.shapes = []
        self.overlay.update()
