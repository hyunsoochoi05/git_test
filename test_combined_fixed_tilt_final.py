"""
test_combined_fixed_tilt.py

Fixed-tilt dataset generator:
- Draw warped table border (140 x 75)
- A1/A2 centers at sketch coords
- Sample points from 2D isotropic Gaussians around A1/A2
- Repeat N times (default 100)
- Save each generated image
- Insert a black frame after each image (for projector sequence)

Usage examples:
  python test_combined_fixed_tilt.py
  python test_combined_fixed_tilt.py --side_tilt 0.20 --fb_tilt -0.10 --x_bend 0.17 --num 100
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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
    side_tilt: float = 0.49    # 좌/우 기울기
    fb_tilt: float = 0.07      # 앞/뒤 기울기
    x_bend: float = 0.17      # optional mild nonlinear x compression


@dataclass
class RenderSpec:
    a1_sigma_cm: float = 5.0
    a2_sigma_cm: float = 3.0
    point_size: float = 80.0
    center_size: float = 30.0
    border_lw: float = 5.0

    fig_size: tuple[float, float] = (12.8, 7.6)
    dpi: int = 220
    pad: float = 7.0


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
        xw.append(xt)
        yw.append(yt)
    return np.array(xw), np.array(yw)


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
# Drawing helpers
# -------------------------

def setup_axes(ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)


def draw_one(
    ax,
    table: TableSpec,
    warp: WarpSpec,
    render: RenderSpec,
    p1: tuple[float, float],
    p2: tuple[float, float],
):
    setup_axes(ax)

    # Centers: convert "from bottom" to "from top" because we invert y-axis
    a1 = (table.a1_x_from_left, table.height - table.a1_y_from_bottom)
    a2 = (table.a2_x_from_left, table.height - table.a2_y_from_bottom)

    # Border
    n = 500
    xs = np.linspace(0, table.width, n)

    # Top
    ys = np.zeros(n)
    xw, yw = warp_polyline(xs, ys, table, warp)
    ax.plot(xw, yw, linewidth=render.border_lw, color="black")

    # Bottom
    ys = np.ones(n) * table.height
    xw, yw = warp_polyline(xs, ys, table, warp)
    ax.plot(xw, yw, linewidth=render.border_lw, color="black")

    # Left
    ys = np.linspace(0, table.height, n)
    xs0 = np.zeros(n)
    xw, yw = warp_polyline(xs0, ys, table, warp)
    ax.plot(xw, yw, linewidth=render.border_lw, color="black")

    # Right
    xs1 = np.ones(n) * table.width
    xw, yw = warp_polyline(xs1, ys, table, warp)
    ax.plot(xw, yw, linewidth=render.border_lw, color="black")

    # Warp centers
    a1w = two_axis_trapezoid_transform(a1[0], a1[1], table.width, table.height, warp)
    a2w = two_axis_trapezoid_transform(a2[0], a2[1], table.width, table.height, warp)

    ax.scatter([a1w[0], a2w[0]], [a1w[1], a2w[1]], s=render.center_size, color="black")
    ax.text(a1w[0] + 1.2, a1w[1] - 1.2, "A1", fontsize=14, fontweight="bold", color="black")
    ax.text(a2w[0] + 1.2, a2w[1] - 1.2, "A2", fontsize=14, fontweight="bold", color="black")

    # Warp sampled points
    p1w = two_axis_trapezoid_transform(p1[0], p1[1], table.width, table.height, warp)
    p2w = two_axis_trapezoid_transform(p2[0], p2[1], table.width, table.height, warp)

    ax.scatter([p1w[0]], [p1w[1]], s=render.point_size, color="red")
    ax.scatter([p2w[0]], [p2w[1]], s=render.point_size, color="blue")

    # Limits
    minx, maxx, miny, maxy = compute_view_limits(table, warp, pad=render.pad)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(maxy, miny)  # because invert_yaxis


# -------------------------
# IO helpers
# -------------------------

def make_output_dir(base_dir: Path, warp: WarpSpec) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"tilt_side{warp.side_tilt:+.3f}_fb{warp.fb_tilt:+.3f}_bend{warp.x_bend:+.3f}_{ts}"
    out = base_dir / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_black_frame(path: Path, render: RenderSpec):
    # black frame: (H, W, 3)
    h = int(render.fig_size[1] * render.dpi)
    w = int(render.fig_size[0] * render.dpi)
    black = np.zeros((h, w, 3), dtype=np.uint8)
    plt.imsave(str(path), black)


# -------------------------
# Main
# -------------------------

def run_dataset(
    side_tilt: float = 0.20,
    fb_tilt: float = -0.10,
    x_bend: float = 0.17,
    num: int = 100,
    seed: int | None = None,
):
    table = TableSpec()
    warp = WarpSpec(side_tilt=side_tilt, fb_tilt=fb_tilt, x_bend=x_bend)
    render = RenderSpec()

    rng = np.random.default_rng(seed)

    # Centers (top-origin in drawing coords)
    a1 = (table.a1_x_from_left, table.height - table.a1_y_from_bottom)
    a2 = (table.a2_x_from_left, table.height - table.a2_y_from_bottom)

    # Output folder (next to this script): ./rnd_img/<tilt_name_timestamp>/
    base_dir = Path(__file__).resolve().parent / "rnd_img"
    out_dir = make_output_dir(base_dir, warp)

    frame_idx = 1

    for i in range(num):
        # sample points in *unwarped* table coords
        p1 = sample_point(a1, render.a1_sigma_cm, rng)
        p2 = sample_point(a2, render.a2_sigma_cm, rng)

        # draw & save
        fig, ax = plt.subplots(1, 1, figsize=render.fig_size)
        draw_one(ax, table, warp, render, p1, p2)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = out_dir / f"episode_{frame_idx:05d}_{ts}.png"
        fig.savefig(img_path, dpi=render.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"[{i+1}/{num}] saved: {img_path.name}")
        frame_idx += 1

        # black frame right after
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        black_path = out_dir / f"episode_{frame_idx:05d}_{ts}.png"
        save_black_frame(black_path, render)
        print(f"           black: {black_path.name}")
        frame_idx += 1

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--side_tilt", type=float, default=0.20)
    parser.add_argument("--fb_tilt", type=float, default=-0.10)
    parser.add_argument("--x_bend", type=float, default=0.17)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dataset(
        side_tilt=args.side_tilt,
        fb_tilt=args.fb_tilt,
        x_bend=args.x_bend,
        num=args.num,
        seed=args.seed,
    )
