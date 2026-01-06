"""
proj_minimal_bottom_v2.py

Minimal tilt controls with bottom UI and last_warp.json autoload.
Fixed scale/translation keeps the 140x75 aspect ratio.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
    a1_y_from_bottom: float = 21.1

    a2_x_from_left: float = 104.6
    a2_y_from_bottom: float = 6.4


@dataclass
class WarpSpec:
    # 8-DoF homography-like parameters (only some are controlled)
    sx: float = 1.0
    sy: float = 1.0
    rot_deg: float = 0.0
    shear: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    px: float = 0.0
    py: float = 0.0


@dataclass
class SampleSpec:
    a1_sigma_cm: float = 8.75
    a2_sigma_cm: float = 5.25
    center_size: float = 60.0
    a1_point_diam: float = 30
    a2_point_diam: float = 13


# -------------------------
# Colors
# -------------------------

BG_COLOR = "#ffffff"
LINE_COLOR = "#000000"
LABEL_COLOR = "#000000"
CENTER_FILL = "#fff4d6"
CENTER_EDGE = "#000000"
P1_COLOR = "#00e5ff"
# P2_COLOR = "#ff5a1f"
P2_COLOR = "#f65ce6"


# -------------------------
# Homography-like warp
# -------------------------

def _build_H(warp: WarpSpec) -> np.ndarray:
    th = math.radians(float(warp.rot_deg))
    ct, st = math.cos(th), math.sin(th)

    S = np.array([[float(warp.sx), 0.0],
                  [0.0, float(warp.sy)]], dtype=float)
    Sh = np.array([[1.0, float(warp.shear)],
                   [0.0, 1.0]], dtype=float)
    R = np.array([[ct, -st],
                  [st,  ct]], dtype=float)

    M2 = R @ Sh @ S
    H = np.array([
        [M2[0, 0], M2[0, 1], float(warp.tx)],
        [M2[1, 0], M2[1, 1], float(warp.ty)],
        [float(warp.px), float(warp.py), 1.0],
    ], dtype=float)
    return H


def homography_transform(
    x: float,
    y: float,
    w: float,
    h: float,
    warp: WarpSpec,
) -> tuple[float, float]:
    cx, cy = w / 2.0, h / 2.0
    X = np.array([x - cx, y - cy, 1.0], dtype=float)
    H = _build_H(warp)
    Xp = H @ X

    denom = float(Xp[2])
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6

    xw = float(Xp[0] / denom) + cx
    yw = float(Xp[1] / denom) + cy
    return xw, yw


def warp_polyline(xs: np.ndarray, ys: np.ndarray, table: TableSpec, warp: WarpSpec) -> tuple[np.ndarray, np.ndarray]:
    xw, yw = [], []
    for x, y in zip(xs, ys):
        xt, yt = homography_transform(float(x), float(y), table.width, table.height, warp)
        xw.append(xt); yw.append(yt)
    return np.array(xw), np.array(yw)


# -------------------------
# Drawing helpers
# -------------------------

def setup_axes(ax):
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
        homography_transform(0, 0, table.width, table.height, warp),
        homography_transform(table.width, 0, table.width, table.height, warp),
        homography_transform(0, table.height, table.width, table.height, warp),
        homography_transform(table.width, table.height, table.width, table.height, warp),
    ])
    minx, miny = corners.min(axis=0)
    maxx, maxy = corners.max(axis=0)
    return (minx - pad, maxx + pad, miny - pad, maxy + pad)


def sample_point(center_xy: tuple[float, float], sigma: float, rng: np.random.Generator) -> tuple[float, float]:
    cx, cy = center_xy
    dx, dy = rng.normal(0.0, sigma, size=2)
    return (cx + float(dx), cy + float(dy))


def draw_pattern(
    ax,
    table: TableSpec,
    warp: WarpSpec,
    spec: SampleSpec,
    a1: tuple[float, float],
    a2: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
):
    artists = []

    n = 600
    xs = np.linspace(0, table.width, n); ys = np.zeros(n)
    xw, yw = warp_polyline(xs, ys, table, warp)
    artists += ax.plot(xw, yw, linewidth=5.0, color=LINE_COLOR)

    ys = np.ones(n) * table.height
    xw, yw = warp_polyline(xs, ys, table, warp)
    artists += ax.plot(xw, yw, linewidth=5.0, color=LINE_COLOR)

    ys = np.linspace(0, table.height, n); xs = np.zeros(n)
    xw, yw = warp_polyline(xs, ys, table, warp)
    artists += ax.plot(xw, yw, linewidth=5.0, color=LINE_COLOR)

    xs = np.ones(n) * table.width
    xw, yw = warp_polyline(xs, ys, table, warp)
    artists += ax.plot(xw, yw, linewidth=5.0, color=LINE_COLOR)

    a1w = homography_transform(a1[0], a1[1], table.width, table.height, warp)
    a2w = homography_transform(a2[0], a2[1], table.width, table.height, warp)
    artists.append(ax.scatter([a1w[0], a2w[0]], [a1w[1], a2w[1]],
                              s=spec.center_size, color=CENTER_FILL, edgecolor=CENTER_EDGE, linewidth=2.0))

    artists.append(ax.text(a1w[0] + 1.2, a1w[1] - 1.2, "A1",
                           fontsize=14, fontweight="bold", color=LABEL_COLOR))
    artists.append(ax.text(a2w[0] + 1.2, a2w[1] - 1.2, "A2",
                           fontsize=14, fontweight="bold", color=LABEL_COLOR))

    p1w = homography_transform(p1[0], p1[1], table.width, table.height, warp)
    p2w = homography_transform(p2[0], p2[1], table.width, table.height, warp)
    a1_r = spec.a1_point_diam / 2.0
    a2_r = spec.a2_point_diam / 2.0
    artists.append(ax.add_patch(Circle((p1w[0], p1w[1]), a1_r, facecolor=P1_COLOR, edgecolor="none")))
    artists.append(ax.add_patch(Circle((p2w[0], p2w[1]), a2_r, facecolor=P2_COLOR, edgecolor="none")))

    return artists


def render_frame(
    out_path: Path,
    table: TableSpec,
    warp: WarpSpec,
    spec: SampleSpec,
    a1: tuple[float, float],
    a2: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    bg_color: str,
    draw_marks: bool = True,
):
    fig = plt.figure(figsize=(12.8, 7.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    setup_axes(ax)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    minx, maxx, miny, maxy = compute_view_limits(table, warp, pad=9.0)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(maxy, miny)

    if draw_marks:
        draw_pattern(ax, table, warp, spec, a1, a2, p1, p2)

    fig.savefig(out_path, dpi=220, pad_inches=0)
    plt.close(fig)


# -------------------------
# Persistence
# -------------------------

def load_last_warp(path: Path) -> WarpSpec | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return WarpSpec(
            rot_deg=float(data.get("rot_deg", 0.0)),
            shear=float(data.get("shear", 0.0)),
            px=float(data.get("px", 0.0)),
            py=float(data.get("py", 0.0)),
        )
    except Exception:
        return None


def save_last_warp(path: Path, warp: WarpSpec) -> None:
    data = {
        "rot_deg": warp.rot_deg,
        "shear": warp.shear,
        "px": warp.px,
        "py": warp.py,
    }
    path.write_text(json.dumps(data, indent=2))


# -------------------------
# Interactive app
# -------------------------

def run_interactive(seed: int | None = None):
    table = TableSpec()
    warp = WarpSpec()
    spec = SampleSpec()

    last_warp_path = Path(__file__).with_name("last_warp.json")
    last_warp = load_last_warp(last_warp_path)
    if last_warp is not None:
        warp = last_warp

    rng = np.random.default_rng(seed)

    a1 = (table.a1_x_from_left, table.height - table.a1_y_from_bottom)
    a2 = (table.a2_x_from_left, table.height - table.a2_y_from_bottom)

    p1 = sample_point(a1, spec.a1_sigma_cm, rng)
    p2 = sample_point(a2, spec.a2_sigma_cm, rng)

    fig = plt.figure(figsize=(12.8, 7.6))
    ax = fig.add_axes([0.005, 0.19, 0.99, 0.80])
    setup_axes(ax)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # --- Sliders (compact, bottom) ---
    h0 = 0.016
    w0 = 0.26
    y1 = 0.11
    y2 = 0.06

    ax_rot   = fig.add_axes([0.05, y1, w0, h0])
    ax_shear = fig.add_axes([0.37, y1, w0, h0])
    ax_px    = fig.add_axes([0.69, y1, w0, h0])
    ax_py    = fig.add_axes([0.05, y2, w0, h0])
    ax_zoom  = fig.add_axes([0.37, y2, w0, h0])

    s_rot   = Slider(ax_rot,   "rot°", -25.0, 25.0, valinit=warp.rot_deg, valstep=0.1)
    s_shear = Slider(ax_shear, "shear", -0.50, 0.50, valinit=warp.shear,  valstep=0.005)
    s_px    = Slider(ax_px,    "px",  -0.010, 0.010, valinit=warp.px,     valstep=0.0001)
    s_py    = Slider(ax_py,    "py",  -0.010, 0.010, valinit=warp.py,     valstep=0.0001)
    s_zoom  = Slider(ax_zoom,  "zoom", 0.5, 2.5, valinit=1.0, valstep=0.01)

    # --- Buttons ---
    by = 0.015
    bw = 0.08
    bh = 0.03
    ax_a1    = fig.add_axes([0.05, by, bw, bh])
    ax_a2    = fig.add_axes([0.15, by, bw, bh])
    ax_save  = fig.add_axes([0.25, by, bw, bh])
    ax_batch = fig.add_axes([0.35, by, bw, bh])
    ax_reset = fig.add_axes([0.45, by, bw, bh])

    b_a1 = Button(ax_a1, "Sample A1")
    b_a2 = Button(ax_a2, "Sample A2")
    b_batch = Button(ax_batch, "Save 100")
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

        warp.rot_deg = float(s_rot.val)
        warp.shear   = float(s_shear.val)
        warp.px      = float(s_px.val)
        warp.py      = float(s_py.val)
        artists = draw_pattern(ax, table, warp, spec, a1, a2, p1, p2)

        minx, maxx, miny, maxy = compute_view_limits(table, warp, pad=9.0)
        zoom = float(s_zoom.val)
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        half_w = (maxx - minx) / 2.0 / zoom
        half_h = (maxy - miny) / 2.0 / zoom
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy + half_h, cy - half_h)

        fig.canvas.draw_idle()

    def on_change(_):
        redraw()

    def on_reset(_):
        s_rot.reset(); s_shear.reset(); s_px.reset(); s_py.reset(); s_zoom.reset()
        redraw()

    def do_save(_=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(f"projector_pattern_sampling_homography_wood_{ts}.png")
        warp_now = WarpSpec(
            rot_deg=float(s_rot.val),
            shear=float(s_shear.val),
            px=float(s_px.val),
            py=float(s_py.val),
        )
        render_frame(out, table, warp_now, spec, a1, a2, p1, p2, BG_COLOR, draw_marks=True)
        save_last_warp(last_warp_path, warp_now)
        print(f"Saved: {out.resolve()}")
        print(f"warp rot_deg={warp_now.rot_deg:.4f} shear={warp_now.shear:.4f} px={warp_now.px:.6f} py={warp_now.py:.6f}")

    def save_sequence(_=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(ts)
        out_dir.mkdir(parents=True, exist_ok=False)

        warp_now = WarpSpec(
            rot_deg=float(s_rot.val),
            shear=float(s_shear.val),
            px=float(s_px.val),
            py=float(s_py.val),
        )

        total_frames = 100 * 2 - 1
        digits = len(str(total_frames))
        frame_idx = 0

        for i in range(100):
            p1_local = sample_point(a1, spec.a1_sigma_cm, rng)
            p2_local = sample_point(a2, spec.a2_sigma_cm, rng)
            out = out_dir / f"{frame_idx:0{digits}d}_sample.png"
            render_frame(out, table, warp_now, spec, a1, a2, p1_local, p2_local, BG_COLOR, draw_marks=True)
            frame_idx += 1

            if i < 99:
                out = out_dir / f"{frame_idx:0{digits}d}_black.png"
                render_frame(out, table, warp_now, spec, a1, a2, p1_local, p2_local, "#000000", draw_marks=False)
                frame_idx += 1

        save_last_warp(last_warp_path, warp_now)
        print(f"Saved sequence: {out_dir.resolve()}")
        print(f"warp rot_deg={warp_now.rot_deg:.4f} shear={warp_now.shear:.4f} px={warp_now.px:.6f} py={warp_now.py:.6f}")

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
        elif k == "g":
            save_sequence()
        elif k == "1":
            resample_a1()
        elif k == "2":
            resample_a2()

    for s in [s_rot, s_shear, s_px, s_py, s_zoom]:
        s.on_changed(on_change)

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(do_save)
    b_batch.on_clicked(save_sequence)
    b_a1.on_clicked(resample_a1)
    b_a2.on_clicked(resample_a2)

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()


if __name__ == "__main__":
    run_interactive()
