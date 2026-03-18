#!/usr/bin/env python3
"""単一ヘリカルワイヤ vs 直線ワイヤのCR梁収束比較.

7本撚線の収束失敗が:
A) ヘリカル形状に起因するか
B) 複数ワイヤの組み合わせに起因するか
を切り分ける。
"""

import sys

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, ".")

from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSectionInput

_E = 200e9
_NU = 0.3
_WIRE_D = 0.002
_NDOF = 6


def _G(E, nu):
    return E / (2.0 * (1.0 + nu))


def _kappa(nu):
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


def run_nr_force_control(node_coords, connectivity, bend_angle_deg, n_steps, max_iter=50, tol=1e-8):
    """純NR力制御曲げテスト."""
    section = BeamSectionInput.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)
    n_nodes = len(node_coords)
    ndof = n_nodes * _NDOF

    # 固定: 最初の節点（z最小）
    fixed_dofs = np.arange(_NDOF)
    free_dofs = np.array([d for d in range(ndof) if d not in fixed_dofs])

    # 外力: 最後の節点にモーメント Mx
    bend_rad = np.deg2rad(bend_angle_deg)
    L = np.linalg.norm(node_coords[-1] - node_coords[0])
    M = _E * section.Iy * bend_rad / L
    f_ext = np.zeros(ndof)
    f_ext[(n_nodes - 1) * _NDOF + 3] = M

    u = np.zeros(ndof)
    for step in range(n_steps):
        load_frac = (step + 1) / n_steps
        f_step = f_ext * load_frac

        for it in range(max_iter):
            K_T, _ = assemble_cr_beam3d(
                node_coords, connectivity, u,
                _E, G, section.A, section.Iy, section.Iz, section.J,
                kappa, kappa, stiffness=True, internal_force=False,
            )
            _, f_int = assemble_cr_beam3d(
                node_coords, connectivity, u,
                _E, G, section.A, section.Iy, section.Iz, section.J,
                kappa, kappa, stiffness=False, internal_force=True,
            )
            R = f_int - f_step
            R[fixed_dofs] = 0.0
            res = np.linalg.norm(R[free_dofs])
            f_ref = max(np.linalg.norm(f_step[free_dofs]), 1e-30)
            rel = res / f_ref

            if rel < tol:
                break

            K_ff = K_T[np.ix_(free_dofs, free_dofs)]
            du_f = spsolve(csc_matrix(K_ff), -R[free_dofs])
            du = np.zeros(ndof)
            du[free_dofs] = du_f
            u += du
        else:
            return False, step + 1, n_steps, it + 1, rel

    return True, n_steps, n_steps, it + 1, rel


def main():
    print("=" * 70)
    print("  単一ワイヤ収束比較: ヘリカル vs 直線")
    print("=" * 70)

    # 直線ワイヤ（8要素）
    n_elems = 8
    L = 0.020  # 20mm
    n_nodes = n_elems + 1
    coords_straight = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        coords_straight[i, 2] = i * L / n_elems
    conn_straight = np.array([[i, i + 1] for i in range(n_elems)])

    # ヘリカルワイヤ（7本メッシュから1本だけ取り出す）
    mesh7 = make_twisted_wire_mesh(
        7, _WIRE_D, 0.040, length=0.0,
        n_elems_per_strand=8, n_pitches=0.5, min_elems_per_pitch=16,
    )

    # strand 0 (中心) と strand 1 (外周) を比較
    for strand_id in [0, 1]:
        strand_nodes = mesh7.strand_nodes(strand_id)
        strand_elems = mesh7.strand_elems(strand_id)

        # ローカルメッシュを作成（節点を0から振り直す）
        node_map = {old: new for new, old in enumerate(strand_nodes)}
        coords_strand = mesh7.node_coords[strand_nodes].copy()
        conn_strand = np.array([
            [node_map[mesh7.connectivity[ei, 0]], node_map[mesh7.connectivity[ei, 1]]]
            for ei in strand_elems
        ])

        label = f"中心(strand 0)" if strand_id == 0 else f"外周(strand 1)"
        print(f"\n--- {label} ---")
        print(f"  節点数: {len(coords_strand)}, 要素数: {len(conn_strand)}")

        # 要素長と方向の確認
        for i in range(len(conn_strand)):
            n1, n2 = conn_strand[i]
            dx = coords_strand[n2] - coords_strand[n1]
            L_e = np.linalg.norm(dx)
            e_x = dx / L_e
            print(f"  elem {i}: L={L_e*1000:.4f}mm, "
                  f"dir=({e_x[0]:.4f}, {e_x[1]:.4f}, {e_x[2]:.4f})")

        # 角度ごとのテスト
        for angle in [10, 20, 30, 45, 60, 90]:
            n_steps = max(5, angle // 3)
            ok, done, total, iters, rel = run_nr_force_control(
                coords_strand, conn_strand, angle, n_steps, max_iter=50, tol=1e-8
            )
            status = "OK" if ok else "FAIL"
            print(f"  {angle:3d}°: {status} ({done}/{total} steps, "
                  f"last_iters={iters}, rel={rel:.3e})")
            if not ok:
                break

    # 直線ワイヤ
    print(f"\n--- 直線ワイヤ ---")
    for angle in [10, 20, 30, 45, 60, 90]:
        n_steps = max(5, angle // 3)
        ok, done, total, iters, rel = run_nr_force_control(
            coords_straight, conn_straight, angle, n_steps, max_iter=50, tol=1e-8
        )
        status = "OK" if ok else "FAIL"
        print(f"  {angle:3d}°: {status} ({done}/{total} steps, "
              f"last_iters={iters}, rel={rel:.3e})")

    # 要素数を増やしたヘリカルワイヤ
    print(f"\n--- 外周ヘリカルワイヤ 32要素/ピッチ ---")
    mesh7_fine = make_twisted_wire_mesh(
        7, _WIRE_D, 0.040, length=0.0,
        n_elems_per_strand=16, n_pitches=0.5, min_elems_per_pitch=16,
    )
    strand_nodes = mesh7_fine.strand_nodes(1)
    strand_elems = mesh7_fine.strand_elems(1)
    node_map = {old: new for new, old in enumerate(strand_nodes)}
    coords_fine = mesh7_fine.node_coords[strand_nodes].copy()
    conn_fine = np.array([
        [node_map[mesh7_fine.connectivity[ei, 0]], node_map[mesh7_fine.connectivity[ei, 1]]]
        for ei in strand_elems
    ])
    print(f"  節点数: {len(coords_fine)}, 要素数: {len(conn_fine)}")
    for angle in [10, 20, 30, 45, 60, 90]:
        n_steps = max(5, angle // 3)
        ok, done, total, iters, rel = run_nr_force_control(
            coords_fine, conn_fine, angle, n_steps, max_iter=50, tol=1e-8
        )
        status = "OK" if ok else "FAIL"
        print(f"  {angle:3d}°: {status} ({done}/{total} steps, "
              f"last_iters={iters}, rel={rel:.3e})")
        if not ok:
            break


if __name__ == "__main__":
    main()
