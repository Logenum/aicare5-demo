"""Flet widgets: panels, badges, tables, parameter tiles, sweep waveforms and a gauge.

Each widget wraps a Flet control tree in `.view`. Setters change control properties
only; the app calls `page.update()` once per refresh so Flet sends a single diff.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import flet as ft
import flet.canvas as cv
import numpy as np

from .config import C

STATUS_ICON = {"good": "●", "warning": "▲", "serious": "▲", "critical": "■"}
MONO = "JetBrains Mono"


def rgba(name: str, alpha: int = 255) -> str:
    """Theme colour (or #rrggbb) with alpha, as Flet's #aarrggbb."""
    c = C.get(name, name).lstrip("#")
    return f"#{alpha:02x}{c}"


def paint(color: str, width: float = 1.0, fill: bool = False, alpha: int = 255,
          cap: ft.StrokeCap = ft.StrokeCap.BUTT, dash: Optional[List[float]] = None) -> ft.Paint:
    return ft.Paint(color=rgba(color, alpha), stroke_width=width, stroke_cap=cap,
                    stroke_dash_pattern=dash, anti_alias=True, stroke_join=ft.StrokeJoin.ROUND,
                    style=ft.PaintingStyle.FILL if fill else ft.PaintingStyle.STROKE)


def label(x, y, text, color="text2", size=10.0, bold=False, align=ft.Alignment.CENTER,
          family: Optional[str] = None) -> cv.Text:
    return cv.Text(x, y, text, alignment=align,
                   style=ft.TextStyle(size=size, color=C.get(color, color), font_family=family,
                                      weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL))


def caption(text: str, color: str = "muted") -> ft.Text:
    return ft.Text(text.upper(), size=11, weight=ft.FontWeight.W_600, color=C[color])


def muted(text: str = "", **kw) -> ft.Text:
    return ft.Text(text, size=12, color=C["muted"], **kw)


class Panel:
    """Rounded surface with a small caps title."""

    def __init__(self, title: str = "", controls: Sequence[ft.Control] = (), col=None,
                 padding=None, title_color: str = "muted"):
        self.body = ft.Column(([caption(title, title_color)] if title else []) + list(controls),
                              spacing=6)
        self.view = ft.Container(self.body, bgcolor=C["surface"], border_radius=6,
                                 border=ft.Border.all(1, C["border"]), col=col,
                                 padding=padding or ft.Padding.symmetric(horizontal=12,
                                                                         vertical=10))


class StatusBadge:
    """Pill with icon + label (status colour never carries meaning alone)."""

    def __init__(self, text: str, tooltip: str = ""):
        self.text = ft.Text(text, size=12, weight=ft.FontWeight.BOLD)
        self.view = ft.Container(self.text, border_radius=12, tooltip=tooltip,
                                 padding=ft.Padding.symmetric(horizontal=10, vertical=4))
        self.set(text, "good")

    def set(self, text: str, status: str, prefix: str = ""):
        self.text.value = f"{STATUS_ICON.get(status, '')} {prefix}{text}"
        self.text.color = "#000000" if status in ("warning", "serious", "good") else "#ffffff"
        self.view.bgcolor = C.get(status, C["muted"])


class Table:
    """Compact read-only table. Cells are text or (text, color, bold, tooltip) tuples.
    Controls are reused when the row count is unchanged so updates stay small."""

    def __init__(self, headers: List[str], weights: List[float], aligns: Optional[List[str]] = None,
                 scroll: bool = False, expand=False):
        self.weights = weights
        self.aligns = aligns or ["l"] * len(headers)
        head = ft.Container(ft.Row([self._cell(h, i, header=True) for i, h in enumerate(headers)],
                                   spacing=8),
                            bgcolor=C["surface2"], padding=ft.Padding.symmetric(horizontal=6,
                                                                                vertical=4),
                            border=ft.Border.only(bottom=ft.BorderSide(1, C["border"])))
        self.rows = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO if scroll else None,
                              expand=expand)
        self.view = ft.Column([head, self.rows], spacing=0, expand=expand)

    @staticmethod
    def _spec(spec):
        return (spec, None, False, None) if isinstance(spec, str) else (tuple(spec) + (None,) * 4)[:4]

    def _cell(self, spec, i, header=False) -> ft.Container:
        text, color, bold, tip = self._spec(spec)
        align = {"l": ft.TextAlign.LEFT, "r": ft.TextAlign.RIGHT, "c": ft.TextAlign.CENTER}[
            self.aligns[i]]
        t = ft.Text(text, size=11 if header else 12, text_align=align,
                    color=C["muted"] if header else (color or C["text"]),
                    weight=ft.FontWeight.W_600 if header or bold else ft.FontWeight.NORMAL)
        return ft.Container(t, expand=self.weights[i], tooltip=tip)

    def set_rows(self, rows: List[List]):
        if len(rows) != len(self.rows.controls):
            self.rows.controls = [
                ft.Container(ft.Row([self._cell(c, i) for i, c in enumerate(r)], spacing=8),
                             padding=ft.Padding.symmetric(horizontal=6, vertical=4),
                             bgcolor=C["surface2"] if k % 2 else None)
                for k, r in enumerate(rows)]
            return
        for row_ctl, r in zip(self.rows.controls, rows):
            for cell, spec in zip(row_ctl.content.controls, r):
                text, color, bold, tip = self._spec(spec)
                cell.tooltip = tip
                cell.content.value = text
                cell.content.color = color or C["text"]
                cell.content.weight = ft.FontWeight.W_600 if bold else ft.FontWeight.NORMAL


class RiskBar:
    """Labelled probability bar with the model's top drivers underneath."""

    def __init__(self, name: str):
        self.value = ft.Text("", size=12, weight=ft.FontWeight.BOLD, color=C["text"])
        self.bar = ft.ProgressBar(value=0, bar_height=8, border_radius=4, bgcolor=C["grid"],
                                  color=C["good"])
        self.why = ft.Text("", size=11, color=C["muted"], max_lines=1,
                           overflow=ft.TextOverflow.ELLIPSIS)
        self.view = ft.Column([
            ft.Row([ft.Text(name, size=12, color=C["text2"]), self.value],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            self.bar, self.why], spacing=2)

    @staticmethod
    def level(p):
        if p < 0.25:
            return "LOW", "good"
        if p < 0.5:
            return "MOD", "warning"
        if p < 0.75:
            return "HIGH", "serious"
        return "SEVERE", "critical"

    def set_value(self, p: float, why: str = ""):
        p = max(0.0, min(1.0, p))
        lvl, st = self.level(p)
        self.value.value = f"{p * 100:.0f}%  {lvl}"
        self.bar.value = max(p, 0.02)
        self.bar.color = C[st]
        self.why.value = why if p >= 0.25 else ""
        self.why.visible = p >= 0.25
        self.view.tooltip = why or None


class ParamTile:
    """Monitor-style numeric: label + unit top, big value, alarm limits, an extra line
    and an AI quality badge. Flashes its background while its alarm is active."""

    def __init__(self, name: str, unit: str, color: str, limits: str = "", big: int = 52,
                 height: float = 128, col=None):
        self.color = C[color]
        self.value = ft.Text("--", size=big, weight=ft.FontWeight.W_500, color=self.color,
                             font_family=MONO)
        self.sub = ft.Text("", size=13, color=self.color, font_family=MONO)
        self.extra = ft.Text("", size=11, color=C["text2"], max_lines=2)
        self.badge = ft.Container(ft.Text("", size=10, weight=ft.FontWeight.BOLD,
                                          color="#000000"),
                                  bgcolor=C["accent"], border_radius=3, visible=False,
                                  padding=ft.Padding.symmetric(horizontal=5, vertical=1))
        self.view = ft.Container(
            ft.Column([
                ft.Row([ft.Text(name, size=13, weight=ft.FontWeight.BOLD, color=self.color),
                        muted(unit), ft.Container(expand=True), self.badge,
                        ft.Text(limits, size=10, color=C["muted"], font_family=MONO)],
                       spacing=6),
                ft.Row([self.value, self.sub], vertical_alignment=ft.CrossAxisAlignment.END,
                       spacing=8, wrap=True),
                self.extra,
            ], spacing=0, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            bgcolor=C["surface"], height=height, col=col,
            border=ft.Border.all(1, C["border"]), border_radius=6,
            padding=ft.Padding.symmetric(horizontal=10, vertical=6))

    def set(self, value: str, sub: str = "", extra: str = "", badge: str = "",
            alarm: Optional[str] = None, flash: bool = False, dim: bool = False):
        self.value.value = value
        self.value.color = C["muted"] if dim else self.color
        self.sub.value = sub
        self.extra.value = extra
        self.badge.content.value = badge
        self.badge.visible = bool(badge)
        if alarm and flash:
            self.view.bgcolor = C[f"alarm_{alarm}"]
            self.value.color = self.sub.color = "#000000"
        else:
            self.view.bgcolor = C["surface"]
            self.sub.color = self.color
            self.view.border = ft.Border.all(2 if alarm else 1,
                                             C[f"alarm_{alarm}"] if alarm else C["border"])


# --------------------------------------------------------------------------------------

class CanvasWidget:
    """Base for instruments drawn on a canvas that tracks its own size."""

    def __init__(self, height: Optional[float] = None, width: Optional[float] = None):
        self.w = self.h = 0.0
        self.canvas = cv.Canvas(expand=True, on_resize=self._on_resize, resize_interval=50)
        self.view = ft.Container(self.canvas, height=height, width=width)

    def _on_resize(self, e: cv.CanvasResizeEvent):
        self.w, self.h = e.width, e.height
        self.redraw()
        self.canvas.update()

    def redraw(self):
        self.canvas.shapes = self.draw() if self.w > 10 and self.h > 10 else []

    def draw(self) -> List[cv.Shape]:
        raise NotImplementedError


class Waveform(CanvasWidget):
    """Sweeping trace: the buffer is drawn left to right with a gap at the write cursor,
    like the erase bar of a bedside monitor."""

    def __init__(self, name: str, color: str, y_range=(-0.5, 1.2), height: float = 128,
                 scale_label: str = ""):
        super().__init__(height)
        self.name, self.color, self.y_range = name, color, y_range
        self.scale_label = scale_label
        self.y: Optional[np.ndarray] = None
        self.cursor = 0
        self.note = ""
        self.view.bgcolor = C["page"]
        self.view.border = ft.Border.all(1, C["border"])
        self.view.border_radius = 6

    def set_data(self, y: np.ndarray, cursor: int, note: str = ""):
        self.y, self.cursor, self.note = y, cursor, note
        self.redraw()

    def draw(self):
        w, h = self.w, self.h
        s: List[cv.Shape] = [label(8, 10, self.name, self.color, 12, bold=True,
                                   align=ft.Alignment.CENTER_LEFT)]
        if self.scale_label:
            s.append(label(w - 8, 10, self.scale_label, "muted", 10,
                           align=ft.Alignment.CENTER_RIGHT))
        if self.note:
            s.append(label(w / 2, 10, self.note, "accent", 11, bold=True))
        if self.y is None:
            return s
        n = len(self.y)
        lo, hi = self.y_range
        top, bottom = 22, h - 6
        xs = np.arange(n) * (w - 4) / (n - 1) + 2
        ys = bottom - (np.clip(self.y, lo - 0.3, hi + 0.3) - lo) / (hi - lo) * (bottom - top)
        gap = max(2, n // 40)
        segments = [(self.cursor + gap, n), (0, self.cursor)]
        for a, b in segments:
            pts = [ft.Offset(float(xs[i]), float(ys[i])) for i in range(a, b)
                   if np.isfinite(ys[i])]
            if len(pts) > 1:
                s.append(cv.Points(pts, point_mode=cv.PointMode.POLYGON,
                                   paint=paint(self.color, 1.8)))
        return s


class ScoreGauge(CanvasWidget):
    """Semicircular 0-100 deterioration index gauge."""

    def __init__(self, width=170, height=110):
        super().__init__(height, width)
        self.value, self.level = 0.0, "LOW"

    def set_value(self, v: float, level: str):
        self.value, self.level = v, level
        self.redraw()

    def draw(self):
        w, h = self.w, self.h
        r = min(w / 2 - 10, h - 14)
        cx, cy = w / 2, h - 4
        st = {"LOW": "good", "MODERATE": "warning", "HIGH": "serious"}.get(self.level, "critical")
        box = (cx - r, cy - r, 2 * r, 2 * r)
        return [
            cv.Arc(*box, math.pi, math.pi, paint=paint("grid", 14)),
            cv.Arc(*box, math.pi, math.pi * max(0.0, min(100.0, self.value)) / 100,
                   paint=paint(st, 14)),
            label(cx, cy - r * 0.5, f"{self.value:.0f}", "text", 28, bold=True),
            label(cx, cy, f"{STATUS_ICON[st]} {self.level}", st, 12, bold=True,
                  align=ft.Alignment.BOTTOM_CENTER),
        ]
