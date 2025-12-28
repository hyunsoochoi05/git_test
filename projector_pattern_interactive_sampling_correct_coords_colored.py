"""
projector_pattern_interactive_sampling.py

Interactive projector-warp tuning + Gaussian sampling points.

- Draws the outer table border (140 x 75; same units as your sketch).
- A1, A2 positions follow the sketch:
    A1: x=56 from left, y=31 from bottom
    A2: x=87 from left, y=23.5 from bottom
- No concentric circles.
- Samples points from 2D isotropic Gaussians (cov = sigma^2 I) centered at:
    A1: sigma = 5 cm
    A2: sigma = 3 cm
  One sample per center, re-sampled on demand.
- Warp (tilt) is tunable live via sliders:
    * Side tilt (좌/우 기울기)
    * Front/back tilt (앞/뒤 기울기)
    * X bend (optional mild nonlinear x compression)

Controls
- Button "Sample A1": resample A1 point
- Button "Sample A2": resample A2 point
- Button "Reset": reset warp sliders
- Button "Save": save PNG
- Keys:
    * '1' : resample A1
    * '2' : resample A2
    * 's' : save PNG
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# -------------------------
# Specs
# -------------------------

@dataclass
class TableSpec:
    width: float = 140.0
    height: float = 75.0

    # From sketch (from left, from bottom)
    a1_x_from_left: float = 56.0
    a1_y_from_bottom: float = 31.0

    a2_x_from_left: float = 87.0
    a2_y_from_bottom: float = 23.5


@dataclass
class WarpSpec:
    side_tilt: float = 0.0
    fb_tilt: float = 0.0
    x_bend: float = 0.17


@dataclass
class SampleSpec:
    a1_sigma_cm: float = 2.0
    a2_sigma_cm: float = 1.5
    point_size: float = 80.0
    center_size: float = 80.0


# -------------------------
# Warp
# -------------------------

def two_axis_trapezoid_transform(
    x: float,
    y: float,
    w: float,
    h: float,
    warp: WarpSpec,
) -> tuple[float, float]:
    """
    Simple 2-axis keystone-like warp (not full homography).
    Tunable and stable for projector alignment testing.
    """
    xn = (x - w / 2.0) / (w / 2.0)
    yn = (y - h / 2.0) / (h / 2.0)

    # Vertical keystone across X
    k_side = 0.45 * float(warp.side_tilt)
    scale_y = 1.0 - k_side * xn
    y_center = h / 2.0
    y_warped = (y - y_center) * scale_y + y_center

    # Horizontal keystone across Y
    k_fb = 0.45 * float(warp.fb_tilt)
    scale_x = 1.0 - k_fb * yn
    x_center = w / 2.0
    x_warped = (x - x_center) * scale_x + x_center

    # Mild nonlinear x bend (optional)
    if warp.x_bend != 0.0:
        t = (x_warped / w)
        x_warped = x_warped * (1.0 - float(warp.x_bend) * t)

    return x_warped, y_warped


def warp_polyline(xs: np.ndarray, ys: np.ndarray, table: TableSpec, warp: WarpSpec) -> tuple[np.ndarray, np.ndarray]:
    xw, yw = [], []
    for x, y in zip(xs, ys):
        xt, yt = two_axis_trapezoid_transform(float(x), float(y), table.width, table.height, warp)
        xw.append(xt); yw.append(yt)
    return np.array(xw), np.array(yw)


# -------------------------
# Drawing helpers
# -------------------------

def setup_axes(ax):
    """Reuse helper.setup_axes if it exists; fallback otherwise."""
    try:
        import helper  # type: ignore
        if hasattr(helper, "setup_axes"):
            helper.setup_axes(ax, height=0)
            return
    except Exception:
        pass

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)


def compute_view_limits(table: TableSpec, warp: WarpSpec, pad: float = 7.0):
    corners = np.array([
        two_axis_trapezoid_transform(0, 0, table.width, table.height, warp),
        two_axis_trapezoid_transform(table.width, 0, table.width, table.height, warp),
        two_axis_trapezoid_transform(0, table.height, table.width, table.height, warp),
        two_axis_trapezoid_transform(table.width, table.height, table.width, table.height, warp),
    ])
    minx, miny = corners.min(axis=0)
    maxx, maxy = corners.max(axis=0)
    return (minx - pad, maxx + pad, miny - pad, maxy + pad)


def sample_point(center_xy: tuple[float, float], sigma: float, rng: np.random.Generator) -> tuple[float, float]:
    cx, cy = center_xy
    dx, dy = rng.normal(0.0, sigma, size=2)
    return (cx + float(dx), cy + float(dy))


# -------------------------
# Interactive app
# -------------------------

def run_interactive(seed: int | None = None):
    table = TableSpec()
    warp = WarpSpec(side_tilt=0.0, fb_tilt=0.0, x_bend=0.17)
    spec = SampleSpec(a1_sigma_cm=5.0, a2_sigma_cm=3.0, point_size=26.0, center_size=60.0)

    rng = np.random.default_rng(seed)

    # Convert "from bottom" to "from top" because we invert y-axis
    a1 = (table.a1_x_from_left, table.height - table.a1_y_from_bottom)
    a2 = (table.a2_x_from_left, table.height - table.a2_y_from_bottom)

    # Initial samples
    p1 = sample_point(a1, spec.a1_sigma_cm, rng)
    p2 = sample_point(a2, spec.a2_sigma_cm, rng)

    fig = plt.figure(figsize=(12.8, 7.6))
    ax = fig.add_axes([0.06, 0.10, 0.72, 0.86])
    setup_axes(ax)

    # Sliders
    ax_side = fig.add_axes([0.82, 0.78, 0.15, 0.03])
    ax_fb   = fig.add_axes([0.82, 0.71, 0.15, 0.03])
    ax_bend = fig.add_axes([0.82, 0.64, 0.15, 0.03])
    ax_ps   = fig.add_axes([0.82, 0.54, 0.15, 0.03])

    s_side = Slider(ax_side, "Side tilt", -1.0, 1.0, valinit=warp.side_tilt, valstep=0.01)
    s_fb   = Slider(ax_fb,   "F/B tilt",  -1.0, 1.0, valinit=warp.fb_tilt,   valstep=0.01)
    s_bend = Slider(ax_bend, "X bend",     0.0, 0.40, valinit=warp.x_bend,    valstep=0.005)
    s_ps   = Slider(ax_ps,   "Point size", 3.0, 80.0, valinit=spec.point_size, valstep=1.0)

    # Buttons
    ax_a1   = fig.add_axes([0.82, 0.42, 0.15, 0.055])
    ax_a2   = fig.add_axes([0.82, 0.34, 0.15, 0.055])
    ax_reset= fig.add_axes([0.82, 0.22, 0.07, 0.06])
    ax_save = fig.add_axes([0.90, 0.22, 0.07, 0.06])

    b_a1 = Button(ax_a1, "Sample A1")
    b_a2 = Button(ax_a2, "Sample A2")
    b_reset = Button(ax_reset, "Reset")
    b_save  = Button(ax_save,  "Save")

    artists = []

    def redraw():
        nonlocal artists
        for a in artists:
            try:
                a.remove()
            except Exception:
                pass
        artists = []

        warp.side_tilt = float(s_side.val)
        warp.fb_tilt   = float(s_fb.val)
        warp.x_bend    = float(s_bend.val)
        spec.point_size = float(s_ps.val)

        # Outer border (black)
        n = 500
        # Top
        xs = np.linspace(0, table.width, n); ys = np.zeros(n)
        xw, yw = warp_polyline(xs, ys, table, warp)
        artists += ax.plot(xw, yw, linewidth=5.0, color="black")
        # Bottom
        ys = np.ones(n) * table.height
        xw, yw = warp_polyline(xs, ys, table, warp)
        artists += ax.plot(xw, yw, linewidth=5.0, color="black")
        # Left
        ys = np.linspace(0, table.height, n); xs = np.zeros(n)
        xw, yw = warp_polyline(xs, ys, table, warp)
        artists += ax.plot(xw, yw, linewidth=5.0, color="black")
        # Right
        xs = np.ones(n) * table.width
        xw, yw = warp_polyline(xs, ys, table, warp)
        artists += ax.plot(xw, yw, linewidth=5.0, color="black")

        # Centers (black)
        a1w = two_axis_trapezoid_transform(a1[0], a1[1], table.width, table.height, warp)
        a2w = two_axis_trapezoid_transform(a2[0], a2[1], table.width, table.height, warp)
        artists.append(ax.scatter([a1w[0], a2w[0]], [a1w[1], a2w[1]],
                                  s=spec.center_size, color="black"))

        # Labels (black)
        artists.append(ax.text(a1w[0] + 1.2, a1w[1] - 1.2, "A1",
                               fontsize=14, fontweight="bold", color="black"))
        artists.append(ax.text(a2w[0] + 1.2, a2w[1] - 1.2, "A2",
                               fontsize=14, fontweight="bold", color="black"))

        # Sampled points (black)
        p1w = two_axis_trapezoid_transform(p1[0], p1[1], table.width, table.height, warp)
        p2w = two_axis_trapezoid_transform(p2[0], p2[1], table.width, table.height, warp)
        artists.append(ax.scatter([p1w[0]], [p1w[1]], s=spec.point_size, color="red"))
        artists.append(ax.scatter([p2w[0]], [p2w[1]], s=spec.point_size, color="blue"))

        # Limits
        minx, maxx, miny, maxy = compute_view_limits(table, warp, pad=7.0)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(maxy, miny)  # because invert_yaxis

        fig.canvas.draw_idle()

    def on_change(_):
        redraw()

    def on_reset(_):
        s_side.reset(); s_fb.reset(); s_bend.reset(); s_ps.reset()
        redraw()

    def do_save(_=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(f"projector_pattern_sampling_{ts}.png")
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"Saved: {out.resolve()}")

    def resample_a1(_=None):
        nonlocal p1
        p1 = sample_point(a1, spec.a1_sigma_cm, rng)
        redraw()

    def resample_a2(_=None):
        nonlocal p2
        p2 = sample_point(a2, spec.a2_sigma_cm, rng)
        redraw()

    def on_key(event):
        if not event.key:
            return
        k = event.key.lower()
        if k == "s":
            do_save()
        elif k == "1":
            resample_a1()
        elif k == "2":
            resample_a2()

    s_side.on_changed(on_change)
    s_fb.on_changed(on_change)
    s_bend.on_changed(on_change)
    s_ps.on_changed(on_change)

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(do_save)
    b_a1.on_clicked(resample_a1)
    b_a2.on_clicked(resample_a2)

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()


if __name__ == "__main__":
    run_interactive()
