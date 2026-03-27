"""動的三点曲げの3D変形時系列可視化.

n_periods=1, E=25, push=30 の結果を複数時点で3D描画し、
貫入・非物理的変形がないか検証する。

Usage:
    python contracts/visualize_bending_3d.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)


def main():
    cfg = DynamicThreePointBendContactJigConfig(
        E=25.0,
        n_periods=1.0,
        jig_push=30.0,
        max_increments=300,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
    )

    print(f"E={cfg.E}, push={cfg.jig_push}, n_periods={cfg.n_periods}")
    print("solving...")

    r = DynamicThreePointBendContactJigProcess().process(cfg)
    sr = r.solver_result

    final_frac = sr.load_history[-1] if sr.load_history else 0.0
    fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0
    print(f"\nRESULT: conv={sr.converged} frac={final_frac:.4f} fc={fc_last:.2f}N")
    print(f"  incr={sr.n_increments} cutback={sr.n_cutbacks}")

    # 初期座標
    X_ref = r.mesh.node_coords  # (n_nodes, 3)
    n_wire = r.n_wire_nodes
    n_hex = r.n_hex_nodes
    conn = r.mesh.connectivity  # (n_elem, 2)
    n_wire_elems = cfg.n_elems_wire

    disp_hist = sr.displacement_history
    load_hist = sr.load_history
    n_snaps = len(disp_hist)

    if n_snaps == 0:
        print("ERROR: displacement_history is empty!")
        return

    # 時系列から等間隔に8スナップショット選択
    n_frames = min(8, n_snaps)
    indices = np.linspace(0, n_snaps - 1, n_frames, dtype=int)

    # --- 3D plot ---
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Dynamic 3-point bend: E={cfg.E}, push={cfg.jig_push}mm, "
        f"n_periods={cfg.n_periods}\n"
        f"frac={final_frac:.4f}, fc={fc_last:.2f}N, "
        f"incr={sr.n_increments}, cutback={sr.n_cutbacks}",
        fontsize=13,
    )

    for panel_idx, snap_idx in enumerate(indices):
        u_snap = disp_hist[snap_idx]
        frac = load_hist[snap_idx] if snap_idx < len(load_hist) else 0.0

        # 変位から現在座標を計算
        X_cur = X_ref.copy()
        for nd in range(len(X_ref)):
            X_cur[nd, 0] += u_snap[6 * nd + 0]
            X_cur[nd, 1] += u_snap[6 * nd + 1]
            X_cur[nd, 2] += u_snap[6 * nd + 2]

        ax = fig.add_subplot(2, 4, panel_idx + 1, projection="3d")

        # ワイヤ要素（青）
        for e in range(n_wire_elems):
            n0, n1 = conn[e]
            xs = [X_cur[n0, 0], X_cur[n1, 0]]
            ys = [X_cur[n0, 1], X_cur[n1, 1]]
            zs = [X_cur[n0, 2], X_cur[n1, 2]]
            ax.plot(xs, ys, zs, "b-", linewidth=2.0)

        # ジグ要素（赤）
        for e in range(n_wire_elems, len(conn)):
            n0, n1 = conn[e]
            xs = [X_cur[n0, 0], X_cur[n1, 0]]
            ys = [X_cur[n0, 1], X_cur[n1, 1]]
            zs = [X_cur[n0, 2], X_cur[n1, 2]]
            ax.plot(xs, ys, zs, "r-", linewidth=2.5)

        # ジグ節点（赤丸）
        jig_x = X_cur[n_wire:, 0]
        jig_y = X_cur[n_wire:, 1]
        jig_z = X_cur[n_wire:, 2]
        ax.scatter(jig_x, jig_y, jig_z, c="red", s=30, zorder=5)

        # ワイヤ節点（青丸）
        wire_x = X_cur[:n_wire, 0]
        wire_y = X_cur[:n_wire, 1]
        wire_z = X_cur[:n_wire, 2]
        ax.scatter(wire_x, wire_y, wire_z, c="blue", s=15, zorder=5)

        # 初期位置（灰色点線）
        for e in range(n_wire_elems):
            n0, n1 = conn[e]
            ax.plot(
                [X_ref[n0, 0], X_ref[n1, 0]],
                [X_ref[n0, 1], X_ref[n1, 1]],
                [X_ref[n0, 2], X_ref[n1, 2]],
                "k--",
                linewidth=0.5,
                alpha=0.3,
            )

        ax.set_title(f"frac={frac:.3f} (#{snap_idx})", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 視点固定（yz平面を見る角度）
        ax.view_init(elev=15, azim=-80)

    plt.tight_layout()
    out_path = "docs/verification/dynamic_bend_3d_np1.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # --- yz平面投影（2D側面図） ---
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 8))
    fig2.suptitle(
        f"YZ-plane projection: E={cfg.E}, push={cfg.jig_push}mm, n_periods={cfg.n_periods}",
        fontsize=13,
    )

    for panel_idx, snap_idx in enumerate(indices):
        u_snap = disp_hist[snap_idx]
        frac = load_hist[snap_idx] if snap_idx < len(load_hist) else 0.0

        X_cur = X_ref.copy()
        for nd in range(len(X_ref)):
            X_cur[nd, 0] += u_snap[6 * nd + 0]
            X_cur[nd, 1] += u_snap[6 * nd + 1]
            X_cur[nd, 2] += u_snap[6 * nd + 2]

        ax = axes2[panel_idx // 4, panel_idx % 4]

        # ワイヤ
        for e in range(n_wire_elems):
            n0, n1 = conn[e]
            ax.plot(
                [X_cur[n0, 0], X_cur[n1, 0]],
                [X_cur[n0, 1], X_cur[n1, 1]],
                "b-",
                linewidth=2.0,
            )

        # ジグ
        for e in range(n_wire_elems, len(conn)):
            n0, n1 = conn[e]
            ax.plot(
                [X_cur[n0, 0], X_cur[n1, 0]],
                [X_cur[n0, 1], X_cur[n1, 1]],
                "r-",
                linewidth=2.5,
            )

        # 初期位置
        for e in range(n_wire_elems):
            n0, n1 = conn[e]
            ax.plot(
                [X_ref[n0, 0], X_ref[n1, 0]],
                [X_ref[n0, 1], X_ref[n1, 1]],
                "k--",
                linewidth=0.5,
                alpha=0.3,
            )

        ax.set_title(f"frac={frac:.3f}", fontsize=10)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = "docs/verification/dynamic_bend_2d_np1.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.close()

    # --- 貫入検査 ---
    print("\n=== 貫入検査 ===")
    mid_node = r.wire_mid_node
    for snap_idx in indices:
        u_snap = disp_hist[snap_idx]
        frac = load_hist[snap_idx] if snap_idx < len(load_hist) else 0.0

        # ワイヤ中央のy座標
        wire_mid_y = X_ref[mid_node, 1] + u_snap[6 * mid_node + 1]
        # ジグの最低y座標
        jig_min_y = min(X_ref[n_wire + i, 1] + u_snap[6 * (n_wire + i) + 1] for i in range(n_hex))
        # gap = ワイヤ上面 - ジグ底面（正=隙間、負=貫入）
        gap = wire_mid_y + cfg.wire_diameter / 2.0 - jig_min_y
        print(
            f"  frac={frac:.3f}: wire_mid_y={wire_mid_y:+.3f}mm, "
            f"jig_min_y={jig_min_y:+.3f}mm, gap={gap:+.4f}mm"
        )

    # --- 変形量の物理的妥当性 ---
    print("\n=== 変形量の妥当性 ===")
    u_final = disp_hist[-1]
    max_disp_y = 0.0
    max_disp_node = -1
    for nd in range(n_wire):
        dy = abs(u_snap[6 * nd + 1])
        if dy > max_disp_y:
            max_disp_y = dy
            max_disp_node = nd
    print(f"  max |u_y| = {max_disp_y:.3f}mm at node {max_disp_node}")
    print(f"  push = {cfg.jig_push}mm, L = {cfg.wire_length}mm")
    print(f"  max_disp/push = {max_disp_y / cfg.jig_push:.3f}")
    print(f"  max_disp/L = {max_disp_y / cfg.wire_length:.4f}")

    # EB 解析解
    I = np.pi * (cfg.wire_diameter / 2) ** 4 / 4
    k_eb = 48.0 * cfg.E * I / cfg.wire_length**3
    push_reached = final_frac * cfg.jig_push
    P_eb = k_eb * push_reached
    print(f"\n  k_EB = {k_eb:.2f} N/mm")
    print(f"  push_reached = {push_reached:.2f}mm")
    print(f"  P_EB = {P_eb:.1f}N (解析解の曲げ荷重)")
    print(f"  fc_sim = {fc_last:.2f}N (シミュレーション接触力)")


if __name__ == "__main__":
    main()
