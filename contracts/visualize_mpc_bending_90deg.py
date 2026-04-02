"""MPC θ_y 90度曲げの3Dパイプレンダリング（時系列5フレーム）.

status-280: MPC + θ_y + 接触なしで90度曲げ完走の変形形状確認。

Usage:
    python contracts/visualize_mpc_bending_90deg.py 2>&1 | tee /tmp/log-$(date +%s).log

[← README](../README.md)
"""

import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

_STRAND_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def _draw_tube(ax, p0, p1, radius, color, n_seg=8, alpha=0.6):
    """2ノード間にチューブ（円筒側面）を描画."""
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-15:
        return
    axis_dir = axis / length
    if abs(axis_dir[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis_dir, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_dir, e1)
    theta = np.linspace(0, 2 * np.pi, n_seg + 1)
    circle = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    ring0 = p0 + radius * circle
    ring1 = p1 + radius * circle
    for j in range(n_seg):
        verts = [[ring0[j], ring0[j + 1], ring1[j + 1], ring1[j]]]
        poly = Poly3DCollection(verts, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)


def main():
    kappa_90 = math.pi / 2.0 / 100.0
    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=kappa_90,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=False,  # MPC
        contact_enabled=False,
    )

    print("Solving MPC + θ_y + no contact (90° bending)...")
    result = StrandBendingOscillationProcess().process(cfg)
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    print(f"frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}")

    if not sr.displacement_history:
        print("ERROR: no displacement history")
        return

    mesh = result.mesh
    coords_ref = mesh.node_coords
    conn = mesh.connectivity
    strand_ids = mesh.strand_ids
    n_strand_nodes = result.n_strand_nodes
    wire_radius = cfg.wire_radius

    # 5フレーム等間隔選択
    n_snaps = len(sr.displacement_history)
    snap_indices = np.linspace(0, n_snaps - 1, 5, dtype=int)

    # --- 5フレーム横並び ---
    fig = plt.figure(figsize=(30, 6))
    fig.suptitle(
        f"MPC + θ_y bending (90°, no contact) — frac={frac:.4f}\n"
        f"κ={kappa_90:.5f} [1/mm], θ_target={math.degrees(kappa_90 * 100):.1f}°, "
        f"incr={sr.n_increments}, cutback={sr.n_cutbacks}",
        fontsize=14,
    )

    for panel_idx, snap_idx in enumerate(snap_indices):
        u = sr.displacement_history[snap_idx]
        f_val = sr.load_history[snap_idx] if snap_idx < len(sr.load_history) else 0.0

        coords_cur = coords_ref.copy()
        for nd in range(n_strand_nodes):
            coords_cur[nd, 0] += u[6 * nd + 0]
            coords_cur[nd, 1] += u[6 * nd + 1]
            coords_cur[nd, 2] += u[6 * nd + 2]

        ax = fig.add_subplot(1, 5, panel_idx + 1, projection="3d")
        for e in range(len(conn)):
            n0, n1 = int(conn[e, 0]), int(conn[e, 1])
            if n0 >= n_strand_nodes or n1 >= n_strand_nodes:
                continue
            p0, p1 = coords_cur[n0], coords_cur[n1]
            sid = int(strand_ids[e]) if strand_ids is not None else 0
            color = _STRAND_COLORS[sid % len(_STRAND_COLORS)]
            _draw_tube(ax, p0, p1, wire_radius, color, n_seg=8, alpha=0.7)

        angle_deg = f_val * math.degrees(kappa_90 * 100)
        ax.set_title(f"frac={f_val:.3f}\n({angle_deg:.1f}°)", fontsize=10)
        ax.view_init(elev=20, azim=-60)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 全フレームで同じスケール
        all_x = coords_cur[:n_strand_nodes, 0]
        all_y = coords_cur[:n_strand_nodes, 1]
        all_z = coords_cur[:n_strand_nodes, 2]
        max_range = (
            max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min())
            / 2
            * 1.1
        )
        mid_x = (all_x.max() + all_x.min()) / 2
        mid_y = (all_y.max() + all_y.min()) / 2
        mid_z = (all_z.max() + all_z.min()) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    out_path = "docs/verification/mpc_bending_90deg_pipe.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
