"""ストランド曲げ揺動の3Dパイプレンダリング（貫入検査用）.

各ワイヤを円筒パイプとして3D描画し、
ワイヤ同士の貫入（交差）がないか視覚確認する。

Usage:
    python contracts/visualize_strand_bending_3d.py 2>&1 | tee /tmp/log-$(date +%s).log

[← README](../README.md)
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from xkep_cae.numerical_tests.strand_bending_oscillation import (  # noqa: E402
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

# ストランドごとの色
_STRAND_COLORS = [
    "#1f77b4",  # 青
    "#ff7f0e",  # オレンジ
    "#2ca02c",  # 緑
    "#d62728",  # 赤
    "#9467bd",  # 紫
    "#8c564b",  # 茶
    "#e377c2",  # ピンク
]


def _draw_tube(
    ax,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    color: str,
    n_seg: int = 8,
    alpha: float = 0.6,
) -> None:
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


def _draw_cap(
    ax,
    center: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    color: str,
    n_seg: int = 8,
    alpha: float = 0.6,
) -> None:
    """円筒の端面キャップを描画."""
    if abs(axis_dir[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis_dir, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_dir, e1)

    theta = np.linspace(0, 2 * np.pi, n_seg + 1)
    ring = center + radius * (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2))

    for j in range(n_seg):
        verts = [[center, ring[j], ring[j + 1]]]
        poly = Poly3DCollection(verts, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)


def main() -> None:
    # --- 解析実行 ---
    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.001,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=50,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        smoothing_delta=1000.0,  # δ=1000 で完走実績あり
    )

    print("Solving strand bending oscillation (7-wire, 1 pitch)...")
    proc = StrandBendingOscillationProcess()
    result = proc.process(cfg)
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    print(f"Result: converged={sr.converged}, frac={frac:.4f}")
    print(f"  increments={sr.n_increments}, cutbacks={sr.n_cutbacks}")

    if not sr.displacement_history:
        print("ERROR: no displacement history")
        return

    # --- メッシュ情報 ---
    mesh = result.mesh
    coords_ref = mesh.node_coords  # (n_nodes, 3)
    conn = mesh.connectivity  # (n_elems, 2)
    strand_ids = mesh.strand_ids  # (n_elems,) 各要素のストランドID
    n_strand_nodes = result.n_strand_nodes
    wire_radius = cfg.wire_radius

    n_snaps = len(sr.displacement_history)
    # 最終スナップショット + 中間2点 + 初期 = 4フレーム
    snap_indices = np.linspace(0, n_snaps - 1, min(4, n_snaps), dtype=int)

    # --- 3Dパイプレンダリング（2視点 × 4フレーム）---
    n_frames = len(snap_indices)
    fig = plt.figure(figsize=(6 * n_frames, 12))
    fig.suptitle(
        f"Strand Bending 3D Pipe Rendering (penetration check)\n"
        f"frac={frac:.4f}, δ={cfg.smoothing_delta}, "
        f"r={wire_radius}mm, n_strands={cfg.n_strands}",
        fontsize=14,
    )

    for panel_idx, snap_idx in enumerate(snap_indices):
        u = sr.displacement_history[snap_idx]
        f_val = sr.load_history[snap_idx] if snap_idx < len(sr.load_history) else 0.0

        # 変位から現在座標（ストランドノードのみ）
        coords_cur = coords_ref.copy()
        for nd in range(n_strand_nodes):
            coords_cur[nd, 0] += u[6 * nd + 0]
            coords_cur[nd, 1] += u[6 * nd + 1]
            coords_cur[nd, 2] += u[6 * nd + 2]

        # 上段: 側面図（XY平面を見る）
        ax1 = fig.add_subplot(2, n_frames, panel_idx + 1, projection="3d")
        # 下段: 端面図（XZ平面、Z方向から見る）
        ax2 = fig.add_subplot(2, n_frames, n_frames + panel_idx + 1, projection="3d")

        for ax in [ax1, ax2]:
            for e in range(len(conn)):
                n0, n1 = int(conn[e, 0]), int(conn[e, 1])
                if n0 >= n_strand_nodes or n1 >= n_strand_nodes:
                    continue  # 参照点ノードはスキップ
                p0 = coords_cur[n0]
                p1 = coords_cur[n1]
                sid = int(strand_ids[e]) if strand_ids is not None else 0
                color = _STRAND_COLORS[sid % len(_STRAND_COLORS)]
                _draw_tube(ax, p0, p1, wire_radius, color, n_seg=10, alpha=0.7)

        ax1.set_title(f"Side view: frac={f_val:.3f}", fontsize=10)
        ax1.view_init(elev=10, azim=-80)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        ax2.set_title(f"End view: frac={f_val:.3f}", fontsize=10)
        ax2.view_init(elev=0, azim=0)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

    plt.tight_layout()
    out_path = "docs/verification/strand_bending_3d_pipe.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # --- 端面断面拡大図（最終フレーム、Z=中央 付近のスライス）---
    u_final = sr.displacement_history[-1]
    coords_final = coords_ref.copy()
    for nd in range(n_strand_nodes):
        coords_final[nd, 0] += u_final[6 * nd + 0]
        coords_final[nd, 1] += u_final[6 * nd + 1]
        coords_final[nd, 2] += u_final[6 * nd + 2]

    # Z中央付近の要素を特定
    z_mid = (coords_final[:n_strand_nodes, 2].min() + coords_final[:n_strand_nodes, 2].max()) / 2
    z_range = coords_final[:n_strand_nodes, 2].max() - coords_final[:n_strand_nodes, 2].min()

    fig2, ax_cs = plt.subplots(1, 1, figsize=(8, 8))
    fig2.suptitle(
        f"Cross-section at Z≈{z_mid:.1f}mm (final frac={frac:.4f})\n"
        f"Wire radius={wire_radius}mm — circles should NOT overlap",
        fontsize=13,
    )

    # 各ストランドの中央付近のノードをプロット
    for s in range(cfg.n_strands):
        # このストランドの要素
        s_mask = strand_ids == s if strand_ids is not None else np.ones(len(conn), dtype=bool)
        s_elems = np.where(s_mask)[0]
        if len(s_elems) == 0:
            continue

        # Z方向中央に最も近いノードを各ストランドから取得
        s_nodes = set()
        for e_idx in s_elems:
            s_nodes.add(int(conn[e_idx, 0]))
            s_nodes.add(int(conn[e_idx, 1]))
        s_nodes = [nd for nd in s_nodes if nd < n_strand_nodes]

        # Z中央に近いノードを選択
        z_dists = [(nd, abs(coords_final[nd, 2] - z_mid)) for nd in s_nodes]
        z_dists.sort(key=lambda x: x[1])
        if not z_dists:
            continue
        mid_nd = z_dists[0][0]

        color = _STRAND_COLORS[s % len(_STRAND_COLORS)]
        x_c = coords_final[mid_nd, 0]
        y_c = coords_final[mid_nd, 1]

        circle = plt.Circle(
            (x_c, y_c),
            wire_radius,
            fill=True,
            alpha=0.4,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
        )
        ax_cs.add_patch(circle)
        ax_cs.plot(x_c, y_c, "o", color=color, markersize=3)
        ax_cs.annotate(f"S{s}", (x_c, y_c), fontsize=8, ha="center", va="center")

    ax_cs.set_aspect("equal")
    ax_cs.set_xlabel("X (mm)")
    ax_cs.set_ylabel("Y (mm)")
    ax_cs.grid(True, alpha=0.3)
    ax_cs.autoscale()
    # 少し余白を追加
    x_lim = ax_cs.get_xlim()
    y_lim = ax_cs.get_ylim()
    margin = wire_radius * 2
    ax_cs.set_xlim(x_lim[0] - margin, x_lim[1] + margin)
    ax_cs.set_ylim(y_lim[0] - margin, y_lim[1] + margin)

    out_path2 = "docs/verification/strand_bending_cross_section.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.close()

    # --- 貫入定量検査 ---
    print("\n=== 貫入定量検査（最終フレーム）===")
    # 各ストランドの中央ノード座標をリスト化
    strand_centers = []
    for s in range(cfg.n_strands):
        s_mask = strand_ids == s if strand_ids is not None else np.ones(len(conn), dtype=bool)
        s_elems = np.where(s_mask)[0]
        s_nodes = set()
        for e_idx in s_elems:
            s_nodes.add(int(conn[e_idx, 0]))
            s_nodes.add(int(conn[e_idx, 1]))
        s_nodes = sorted([nd for nd in s_nodes if nd < n_strand_nodes])
        strand_centers.append(s_nodes)

    n_penetrations = 0
    max_penetration = 0.0
    for si in range(cfg.n_strands):
        for sj in range(si + 1, cfg.n_strands):
            for ni in strand_centers[si]:
                pi = coords_final[ni]
                for nj in strand_centers[sj]:
                    pj = coords_final[nj]
                    # 同じZ位置付近のノードペアのみチェック
                    if abs(pi[2] - pj[2]) > z_range / (cfg.n_elements_per_pitch * 2):
                        continue
                    dist_xy = np.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
                    gap = dist_xy - 2 * wire_radius
                    if gap < 0:
                        n_penetrations += 1
                        if -gap > max_penetration:
                            max_penetration = -gap
                        print(
                            f"  PENETRATION: S{si} node {ni} - S{sj} node {nj}, "
                            f"dist={dist_xy:.4f}mm, gap={gap:.4f}mm"
                        )

    if n_penetrations == 0:
        print("  No penetration detected (all inter-strand gaps >= 0)")
    else:
        print(f"\n  Total penetrations: {n_penetrations}, max depth: {max_penetration:.4f}mm")
        print(f"  max_penetration / wire_diameter = {max_penetration / (2 * wire_radius):.4f}")


if __name__ == "__main__":
    main()
