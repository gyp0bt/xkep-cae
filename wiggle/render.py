"""PyVista による 3D アニメーション。MP4/GIF を吐く。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv

from wiggle.kinematics import (
    StranderConfig,
    bobbin_axis,
    bobbin_position,
    core_strand_centerline,
    outer_strand_centerline,
)

DEFAULT_PALETTE: tuple[str, ...] = (
    "#e76f51",
    "#f4a261",
    "#e9c46a",
    "#2a9d8f",
    "#5e8ac1",
    "#9d4edd",
)


def _polyline(points: np.ndarray) -> pv.PolyData:
    n = len(points)
    cells = np.empty(n + 1, dtype=np.int64)
    cells[0] = n
    cells[1:] = np.arange(n, dtype=np.int64)
    poly = pv.PolyData(points)
    poly.lines = cells
    return poly


def _tube(points: np.ndarray, radius: float, n_sides: int = 18) -> pv.PolyData:
    return _polyline(points).tube(radius=radius, n_sides=n_sides)


def _build_frame(
    plotter: pv.Plotter,
    cfg: StranderConfig,
    t: float,
    palette: Sequence[str],
) -> None:
    plotter.clear_actors()

    core_pts = core_strand_centerline(cfg, t)
    plotter.add_mesh(
        _tube(core_pts, cfg.core_radius),
        color="#9aa0a6",
        smooth_shading=True,
        specular=0.75,
        specular_power=25,
        ambient=0.18,
    )

    for i in range(cfg.n_outer):
        pts = outer_strand_centerline(cfg, i, t)
        plotter.add_mesh(
            _tube(pts, cfg.outer_radius),
            color=palette[i % len(palette)],
            smooth_shading=True,
            specular=0.85,
            specular_power=30,
            ambient=0.2,
        )

    for i in range(cfg.n_outer):
        pos = bobbin_position(cfg, i, t)
        axis = bobbin_axis(cfg, i, t)
        bobbin = pv.Cylinder(
            center=pos + axis * 0.6,
            direction=axis,
            radius=0.65,
            height=1.2,
            resolution=28,
        )
        flange_a = pv.Cylinder(
            center=pos + axis * 0.05,
            direction=axis,
            radius=0.95,
            height=0.12,
            resolution=28,
        )
        flange_b = pv.Cylinder(
            center=pos + axis * 1.15,
            direction=axis,
            radius=0.95,
            height=0.12,
            resolution=28,
        )
        plotter.add_mesh(bobbin, color="#3b2a1d", smooth_shading=True, specular=0.25)
        plotter.add_mesh(flange_a, color="#2a1d12", smooth_shading=True, specular=0.3)
        plotter.add_mesh(flange_b, color="#2a1d12", smooth_shading=True, specular=0.3)

    lay_plate = pv.Disc(
        center=(0, 0, cfg.z_lay),
        inner=cfg.R_layer * 1.2,
        outer=cfg.R_bobbin * 0.8,
        normal=(0, 0, 1),
        r_res=60,
        c_res=60,
    )
    plotter.add_mesh(lay_plate, color="#2f3640", opacity=0.55, smooth_shading=True)

    takeup = pv.Cylinder(
        center=(0, 0, cfg.z1 + 0.6),
        direction=(1, 0, 0),
        radius=1.0,
        height=0.5,
        resolution=40,
    )
    plotter.add_mesh(takeup, color="#3c4148", smooth_shading=True, specular=0.4)

    z_mid = 0.5 * (cfg.z0 + cfg.z1)
    grid_extent = max(cfg.R_bobbin * 1.8, 6.0)
    floor = pv.Plane(
        center=(0, -grid_extent, z_mid),
        direction=(0, 1, 0),
        i_size=grid_extent * 2.5,
        j_size=cfg.L_tail_in + cfg.L_tail_out + 4,
    )
    plotter.add_mesh(floor, color="#15171b", smooth_shading=False, ambient=0.4)


def setup_plotter(cfg: StranderConfig, off_screen: bool = True) -> pv.Plotter:
    pv.global_theme.background = "#0b0d10"
    pv.global_theme.font.color = "#cccccc"
    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=(1280, 720),
        lighting="light_kit",
    )
    z_mid = 0.5 * (cfg.z0 + cfg.z1)
    machine_length = cfg.L_tail_in + cfg.L_tail_out
    cam_distance = max(machine_length * 1.05, 18.0)
    plotter.camera_position = [
        (-cam_distance, cfg.R_bobbin * 1.3, z_mid),
        (0.0, 0.0, z_mid),
        (0.0, 1.0, 0.0),
    ]
    plotter.camera.view_angle = 32.0
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    return plotter


def render_animation(
    cfg: StranderConfig,
    out_path: str | Path,
    duration: float = 6.0,
    fps: int = 24,
    off_screen: bool = True,
    palette: Sequence[str] = DEFAULT_PALETTE,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = setup_plotter(cfg, off_screen=off_screen)

    is_gif = out_path.suffix.lower() == ".gif"
    if is_gif:
        plotter.open_gif(str(out_path), fps=fps)
    else:
        plotter.open_movie(str(out_path), framerate=fps, quality=7)

    n_frames = max(1, int(round(duration * fps)))
    for k in range(n_frames):
        t = duration * k / n_frames
        _build_frame(plotter, cfg, t, palette)
        plotter.write_frame()
    plotter.close()
    return out_path


def render_still(
    cfg: StranderConfig,
    out_path: str | Path,
    t: float = 0.0,
    off_screen: bool = True,
    palette: Sequence[str] = DEFAULT_PALETTE,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = setup_plotter(cfg, off_screen=off_screen)
    _build_frame(plotter, cfg, t, palette)
    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path
