"""7本ヘリカル素線 接触あり Hertz型ペナルティ 90度曲げ 3Dパイプレンダリング.

status-285: Hertz型ペナルティ（α=1.5）で frac=0.998 達成。
変形形状をパイプレンダリングで物理的妥当性を確認する。

Usage:
    python contracts/visualize_hertz_90deg.py

[← README](../README.md)
"""

import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from xkep_cae.numerical_tests.strand_bending_oscillation import (  # noqa: E402
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _draw_tube(ax, p0, p1, radius, color, n_seg=8, alpha=0.6):
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


STRAND_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#e377c2"]


def main():
    print("Solving 7-strand helical 90deg bending (Hertz penalty, α=1.5, contact ON)...")
    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=math.pi / 200.0,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=50,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=True,
        loading_mode="rotation",
        penalty_exponent=1.5,
    )
    result = StrandBendingOscillationProcess().process(cfg)
    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    print(f"frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}")

    u = sr.u
    coords = result.mesh.node_coords
    conn = result.mesh.connectivity
    sid = result.mesh.strand_ids
    n_nodes = len(coords)

    # 変形座標
    coords_def = coords.copy()
    for nd in range(n_nodes):
        coords_def[nd, 0] += u[6 * nd + 0]
        coords_def[nd, 1] += u[6 * nd + 1]
        coords_def[nd, 2] += u[6 * nd + 2]

    # --- 3Dパイプレンダリング ---
    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        f"7-Strand Helical Wire 90° Bending (Contact ON, Hertz α=1.5)\n"
        f"frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}",
        fontsize=13,
    )

    # 左パネル: 初期配置
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for i, e in enumerate(conn):
        n0, n1 = int(e[0]), int(e[1])
        _draw_tube(ax1, coords[n0], coords[n1], 0.5, STRAND_COLORS[sid[i]], alpha=0.7)
    ax1.set_title("Initial Configuration")
    ax1.view_init(elev=15, azim=-80)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(-5, 75)
    ax1.set_ylim(-5, 5)
    ax1.set_zlim(-5, 105)

    # 右パネル: 変形後
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for i, e in enumerate(conn):
        n0, n1 = int(e[0]), int(e[1])
        _draw_tube(ax2, coords_def[n0], coords_def[n1], 0.5, STRAND_COLORS[sid[i]], alpha=0.7)
    # エラスティカ理論線
    r_theory = 100.0 / (math.pi / 2.0)
    theta_arr = np.linspace(0, frac * math.pi / 2, 100)
    z_th = r_theory * (1 - np.cos(theta_arr))
    x_th = r_theory * np.sin(theta_arr)
    ax2.plot(x_th, np.zeros_like(x_th), z_th, "k--", linewidth=2, label="Elastica")
    ax2.set_title(f"Deformed (frac={frac:.4f}, θ≈{frac*90:.1f}°)")
    ax2.view_init(elev=15, azim=-80)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(-5, 75)
    ax2.set_ylim(-5, 5)
    ax2.set_zlim(-5, 105)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = "docs/verification/7strand_90deg_hertz_contact_pipe.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
