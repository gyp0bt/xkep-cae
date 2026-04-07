"""変形メッシュ 2D投影スナップショット.

status-300: 3D梁形状の2D投影（XZ平面 + XY平面）で物理的妥当性を目視確認。
90度曲げ + 揺動解析の変形形状を初期配置と比較して描画する。

Usage:
    # 90度曲げのみ
    python contracts/visualize_deformed_mesh_2d.py
    # 90度曲げ + ±48mm揺動
    python contracts/visualize_deformed_mesh_2d.py --oscillation 48

[← README](../README.md)
"""

import math
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from xkep_cae.numerical_tests.strand_bending_oscillation import (  # noqa: E402
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

STRAND_COLORS = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#e377c2",
]


def plot_2d_projection(
    coords: np.ndarray,
    coords_def: np.ndarray,
    conn: np.ndarray,
    strand_ids: np.ndarray,
    frac: float,
    n_increments: int,
    n_cutbacks: int,
    title_suffix: str = "",
    out_path: str = "docs/verification/deformed_mesh_2d.png",
    bending_length: float = 100.0,
    bending_angle: float = math.pi / 2.0,
):
    """変形メッシュの2D投影図を描画・保存.

    4パネル構成:
      - 左上: 初期配置 XZ平面
      - 右上: 変形後 XZ平面（+ エラスティカ理論線）
      - 左下: 初期配置 XY平面（断面図）
      - 右下: 変形後 XY平面（断面図）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"7-Strand Deformed Mesh 2D Projection{title_suffix}\n"
        f"frac={frac:.4f}, incr={n_increments}, cutback={n_cutbacks}",
        fontsize=13,
    )

    # --- パネル定義 ---
    panels = [
        (axes[0, 0], coords, "Initial — XZ Plane (Side View)", 0, 2),
        (axes[0, 1], coords_def, "Deformed — XZ Plane (Side View)", 0, 2),
        (axes[1, 0], coords, "Initial — XY Plane (End View)", 0, 1),
        (axes[1, 1], coords_def, "Deformed — XY Plane (End View)", 0, 1),
    ]

    axis_labels = {
        (0, 2): ("X [mm]", "Z [mm]"),
        (0, 1): ("X [mm]", "Y [mm]"),
    }

    for ax, c, title, ix, iy in panels:
        # 素線ごとに描画
        for i, e in enumerate(conn):
            n0, n1 = int(e[0]), int(e[1])
            color = STRAND_COLORS[strand_ids[i] % len(STRAND_COLORS)]
            ax.plot(
                [c[n0, ix], c[n1, ix]],
                [c[n0, iy], c[n1, iy]],
                color=color,
                linewidth=1.2,
                alpha=0.8,
            )
        ax.set_title(title, fontsize=11)
        xlabel, ylabel = axis_labels[(ix, iy)]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # エラスティカ理論線（変形後XZ平面に重ねる）
    if frac > 0.01:
        ax_xz = axes[0, 1]
        r_theory = bending_length / bending_angle
        theta_arr = np.linspace(0, frac * bending_angle, 200)
        x_th = r_theory * np.sin(theta_arr)
        z_th = r_theory * (1 - np.cos(theta_arr))
        ax_xz.plot(x_th, z_th, "k--", linewidth=2, alpha=0.7, label="Elastica")
        ax_xz.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_2d_snapshots(
    coords: np.ndarray,
    conn: np.ndarray,
    strand_ids: np.ndarray,
    displacement_history: tuple,
    load_history: tuple,
    n_snapshots: int = 6,
    title_suffix: str = "",
    out_path: str = "docs/verification/deformed_mesh_2d_snapshots.png",
):
    """変形過程のXZ平面スナップショット（等間隔フレーム抽出）."""
    n_hist = len(displacement_history)
    if n_hist == 0:
        print("No displacement history available for snapshots.")
        return

    # 等間隔でフレームを選択（最初と最後を必ず含む）
    indices = np.linspace(0, n_hist - 1, min(n_snapshots, n_hist), dtype=int)
    n_actual = len(indices)
    n_cols = min(n_actual, 3)
    n_rows = (n_actual + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Deformation Snapshots — XZ Plane{title_suffix}",
        fontsize=13,
    )

    n_nodes = len(coords)
    for idx_panel, hist_idx in enumerate(indices):
        row, col = divmod(idx_panel, n_cols)
        ax = axes[row, col]

        u = displacement_history[hist_idx]
        frac_i = load_history[hist_idx] if hist_idx < len(load_history) else 0.0
        coords_def = coords.copy()
        for nd in range(n_nodes):
            coords_def[nd, 0] += u[6 * nd + 0]
            coords_def[nd, 1] += u[6 * nd + 1]
            coords_def[nd, 2] += u[6 * nd + 2]

        for i, e in enumerate(conn):
            n0, n1 = int(e[0]), int(e[1])
            color = STRAND_COLORS[strand_ids[i] % len(STRAND_COLORS)]
            ax.plot(
                [coords_def[n0, 0], coords_def[n1, 0]],
                [coords_def[n0, 2], coords_def[n1, 2]],
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )
        ax.set_title(f"frac={frac_i:.4f}", fontsize=10)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Z [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # 余白パネルを非表示
    for idx_panel in range(len(indices), n_rows * n_cols):
        row, col = divmod(idx_panel, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    """メイン: ソルバー実行 + 2D投影可視化."""
    # コマンドライン引数解析
    osc_amp = 0.0
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--oscillation" and i < len(sys.argv) - 1:
            osc_amp = float(sys.argv[i + 1])

    kappa_90 = math.pi / 2.0 / 100.0

    cfg_kwargs = dict(
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
        free_end_mode=True,
        contact_enabled=True,
        penalty_exponent=1.5,
    )

    if osc_amp > 0:
        cfg_kwargs["n_oscillation_cycles"] = 2
        cfg_kwargs["oscillation_amplitude"] = osc_amp
        suffix = f" (90° + ±{osc_amp:.0f}mm osc)"
        tag = f"90deg_osc{osc_amp:.0f}mm"
    else:
        suffix = " (90° bend)"
        tag = "90deg_bend"

    cfg = StrandBendingOscillationConfig(**cfg_kwargs)

    print(f"Solving: 7-strand Hertz{suffix}...")
    t0 = time.time()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.time() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    print(
        f"frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, elapsed={elapsed:.1f}s"
    )

    # 変形座標計算
    coords = result.mesh.node_coords
    conn = result.mesh.connectivity
    sid = result.mesh.strand_ids
    n_nodes = len(coords)
    u = sr.u

    coords_def = coords.copy()
    for nd in range(n_nodes):
        coords_def[nd, 0] += u[6 * nd + 0]
        coords_def[nd, 1] += u[6 * nd + 1]
        coords_def[nd, 2] += u[6 * nd + 2]

    # 2D投影図（初期 vs 変形後）
    plot_2d_projection(
        coords=coords,
        coords_def=coords_def,
        conn=conn,
        strand_ids=sid,
        frac=frac,
        n_increments=sr.n_increments,
        n_cutbacks=sr.n_cutbacks,
        title_suffix=suffix,
        out_path=f"docs/verification/deformed_mesh_2d_{tag}.png",
    )

    # 変形過程スナップショット
    if sr.displacement_history:
        plot_2d_snapshots(
            coords=coords,
            conn=conn,
            strand_ids=sid,
            displacement_history=sr.displacement_history,
            load_history=sr.load_history,
            n_snapshots=6,
            title_suffix=suffix,
            out_path=f"docs/verification/deformed_mesh_2d_{tag}_snapshots.png",
        )

    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
