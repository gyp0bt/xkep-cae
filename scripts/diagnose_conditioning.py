#!/usr/bin/env python3
"""ヘリカル梁の線形系条件数と線形解精度の診断."""

import sys

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, ".")

from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSection

_E = 200e9
_NU = 0.3
_WIRE_D = 0.002
_NDOF = 6


def _G(E, nu):
    return E / (2.0 * (1.0 + nu))


def _kappa(nu):
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


def diagnose():
    print("=" * 70)
    print("  ヘリカル梁の線形系精度診断")
    print("=" * 70)

    mesh = make_twisted_wire_mesh(
        7, _WIRE_D, 0.040, length=0.0,
        n_elems_per_strand=8, n_pitches=0.5, min_elems_per_pitch=16,
    )
    section = BeamSection.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)
    ndof = mesh.n_nodes * _NDOF

    # 固定DOF
    fixed = set()
    for sid in range(7):
        nodes = mesh.strand_nodes(sid)
        for d in range(_NDOF):
            fixed.add(_NDOF * nodes[0] + d)
    fixed_dofs = np.array(sorted(fixed))
    free_dofs = np.array([d for d in range(ndof) if d not in fixed])

    # 外力
    M = _E * section.Iy * np.deg2rad(10) / mesh.length
    f_ext = np.zeros(ndof)
    for sid in range(7):
        nodes = mesh.strand_nodes(sid)
        n_end = nodes[-1]
        f_ext[_NDOF * n_end + 3] = M

    # Step 1: 初期状態の接線剛性
    u = np.zeros(ndof)
    f_step = f_ext * 0.2  # load frac = 0.2

    K_T, _ = assemble_cr_beam3d(
        mesh.node_coords, mesh.connectivity, u,
        _E, G, section.A, section.Iy, section.Iz, section.J,
        kappa, kappa, stiffness=True, internal_force=False,
    )
    _, f_int = assemble_cr_beam3d(
        mesh.node_coords, mesh.connectivity, u,
        _E, G, section.A, section.Iy, section.Iz, section.J,
        kappa, kappa, stiffness=False, internal_force=True,
    )

    R = f_int - f_step
    R[fixed_dofs] = 0.0

    K_ff = K_T[np.ix_(free_dofs, free_dofs)].toarray()

    print(f"\n  自由DOF数: {len(free_dofs)}")
    print(f"  ||R||: {np.linalg.norm(R[free_dofs]):.6e}")
    print(f"  ||f_ext||: {np.linalg.norm(f_step[free_dofs]):.6e}")

    # 条件数
    try:
        eigs = np.linalg.eigvalsh(K_ff)
        pos_eigs = eigs[eigs > 1e-20]
        neg_eigs = eigs[eigs < -1e-20]
        zero_eigs = eigs[np.abs(eigs) <= 1e-20]
        print(f"\n  固有値分布:")
        print(f"    正: {len(pos_eigs)} (min={pos_eigs.min():.3e}, max={pos_eigs.max():.3e})")
        if len(neg_eigs) > 0:
            print(f"    負: {len(neg_eigs)} (min={neg_eigs.min():.3e}, max={neg_eigs.max():.3e})")
        print(f"    ゼロ近傍: {len(zero_eigs)}")
        if len(pos_eigs) > 0:
            cond = pos_eigs.max() / pos_eigs.min()
            print(f"    条件数: {cond:.3e}")
    except Exception as e:
        print(f"  固有値計算エラー: {e}")

    # 線形解の精度
    du_f = np.linalg.solve(K_ff, -R[free_dofs])
    residual_check = K_ff @ du_f + R[free_dofs]
    print(f"\n  線形解の残差: ||K*du + R|| = {np.linalg.norm(residual_check):.6e}")
    print(f"  相対残差: {np.linalg.norm(residual_check) / np.linalg.norm(R[free_dofs]):.6e}")

    # 1ステップ後の残差
    du = np.zeros(ndof)
    du[free_dofs] = du_f
    u_new = u + du

    _, f_int_new = assemble_cr_beam3d(
        mesh.node_coords, mesh.connectivity, u_new,
        _E, G, section.A, section.Iy, section.Iz, section.J,
        kappa, kappa, stiffness=False, internal_force=True,
    )
    R_new = f_int_new - f_step
    R_new[fixed_dofs] = 0.0

    res_old = np.linalg.norm(R[free_dofs])
    res_new = np.linalg.norm(R_new[free_dofs])
    print(f"\n  NR更新前後の残差:")
    print(f"    更新前: {res_old:.6e}")
    print(f"    更新後: {res_new:.6e}")
    print(f"    低減率: {res_new / res_old:.6e}")

    # DOFスケールの確認
    print(f"\n  du成分の大きさ (代表的なDOF):")
    # 各ストランドの端点DOF
    for sid in [0, 1]:
        nodes = mesh.strand_nodes(sid)
        n_end = nodes[-1]
        for d in range(_NDOF):
            dof = _NDOF * n_end + d
            if dof in fixed:
                continue
            names = ["ux", "uy", "uz", "rx", "ry", "rz"]
            print(f"    strand {sid} end {names[d]}: du={du[dof]:.6e}")

    # ストランド別の残差
    print(f"\n  ストランド別の残差:")
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        strand_dofs = []
        for n in nodes:
            for d in range(_NDOF):
                dof = _NDOF * n + d
                if dof not in fixed:
                    strand_dofs.append(dof)
        strand_res = np.linalg.norm(R_new[strand_dofs])
        print(f"    strand {sid}: ||R|| = {strand_res:.6e}")

    # 外周ワイヤ単独テスト（ストランド1のみ）
    print(f"\n\n  --- ストランド1単独テスト ---")
    strand_nodes_list = mesh.strand_nodes(1)
    strand_elems = mesh.strand_elems(1)
    node_map = {old: new for new, old in enumerate(strand_nodes_list)}
    coords_s = mesh.node_coords[strand_nodes_list].copy()
    conn_s = np.array([
        [node_map[mesh.connectivity[ei, 0]], node_map[mesh.connectivity[ei, 1]]]
        for ei in strand_elems
    ])
    n_nodes_s = len(coords_s)
    ndof_s = n_nodes_s * _NDOF

    fixed_s = np.arange(_NDOF)
    free_s = np.array([d for d in range(ndof_s) if d not in fixed_s])

    u_s = np.zeros(ndof_s)
    M_s = _E * section.Iy * np.deg2rad(10) / np.linalg.norm(coords_s[-1] - coords_s[0])
    f_ext_s = np.zeros(ndof_s)
    f_ext_s[(n_nodes_s - 1) * _NDOF + 3] = M_s
    f_step_s = f_ext_s * 0.2

    K_T_s, _ = assemble_cr_beam3d(
        coords_s, conn_s, u_s,
        _E, G, section.A, section.Iy, section.Iz, section.J,
        kappa, kappa, stiffness=True, internal_force=False,
    )
    _, f_int_s = assemble_cr_beam3d(
        coords_s, conn_s, u_s,
        _E, G, section.A, section.Iy, section.Iz, section.J,
        kappa, kappa, stiffness=False, internal_force=True,
    )
    R_s = f_int_s - f_step_s
    R_s[fixed_s] = 0.0

    K_ff_s = K_T_s[np.ix_(free_s, free_s)].toarray()

    eigs_s = np.linalg.eigvalsh(K_ff_s)
    pos_eigs_s = eigs_s[eigs_s > 1e-20]
    print(f"  自由DOF数: {len(free_s)}")
    print(f"  条件数: {pos_eigs_s.max() / pos_eigs_s.min():.3e}")

    du_s = np.linalg.solve(K_ff_s, -R_s[free_s])
    u_s_new = np.zeros(ndof_s)
    u_s_new[free_s] = du_s

    _, f_int_s_new = assemble_cr_beam3d(
        coords_s, conn_s, u_s_new,
        _E, G, section.A, section.Iy, section.Iz, section.J,
        kappa, kappa, stiffness=False, internal_force=True,
    )
    R_s_new = f_int_s_new - f_step_s
    R_s_new[fixed_s] = 0.0

    res_s_old = np.linalg.norm(R_s[free_s])
    res_s_new = np.linalg.norm(R_s_new[free_s])
    print(f"  NR更新前: {res_s_old:.6e}")
    print(f"  NR更新後: {res_s_new:.6e}")
    print(f"  低減率: {res_s_new / res_s_old:.6e}")

    # 反復を複数回行って収束率を確認
    print(f"\n  --- 反復ごとの残差推移 ---")
    u_it = np.zeros(ndof_s)
    for it in range(20):
        K_T_it, _ = assemble_cr_beam3d(
            coords_s, conn_s, u_it,
            _E, G, section.A, section.Iy, section.Iz, section.J,
            kappa, kappa, stiffness=True, internal_force=False,
        )
        _, f_int_it = assemble_cr_beam3d(
            coords_s, conn_s, u_it,
            _E, G, section.A, section.Iy, section.Iz, section.J,
            kappa, kappa, stiffness=False, internal_force=True,
        )
        R_it = f_int_it - f_step_s
        R_it[fixed_s] = 0.0
        res_it = np.linalg.norm(R_it[free_s])
        f_ref_it = max(np.linalg.norm(f_step_s[free_s]), 1e-30)
        rel_it = res_it / f_ref_it

        print(f"    iter {it}: ||R||/||f|| = {rel_it:.6e}")

        if rel_it < 1e-10:
            break

        K_ff_it = K_T_it[np.ix_(free_s, free_s)].toarray()
        du_it = np.linalg.solve(K_ff_it, -R_it[free_s])
        u_it[free_s] += du_it


if __name__ == "__main__":
    diagnose()
