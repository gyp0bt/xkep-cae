#!/usr/bin/env python3
"""ヘリカル梁要素の初回Newtonステップ過大応答の診断.

条件数診断で判明した1475倍の残差増大の原因を特定する。
単一ヘリカル要素レベルで、f_int(alpha*du) vs alpha の非線形性を調べる。
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from xkep_cae.elements.beam_timo3d import (
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent,
    timo_beam3d_ke_local,
    _beam3d_length_and_direction,
    _build_local_axes,
    _rotvec_to_rotmat,
    _rotmat_to_rotvec,
    _rodrigues_rotation,
    _transformation_matrix_3d,
)
from xkep_cae.sections.beam import BeamSection

_E = 200e9
_NU = 0.3
_WIRE_D = 0.002


def _G(E, nu):
    return E / (2.0 * (1.0 + nu))


def _kappa(nu):
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


def diagnose_single_element():
    """単一要素レベルでの非線形性診断."""
    print("=" * 70)
    print("  単一要素レベル: f_int(alpha*du) の非線形性")
    print("=" * 70)

    section = BeamSection.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)

    # ヘリカル要素: 外周ワイヤの1要素を模擬
    angle = 2 * np.pi / 16  # 1/16回転 (半ピッチ16要素 → 1要素あたり)
    r = _WIRE_D  # 配置半径 = 2mm (実際の7本撚線と同等)
    dz = 0.040 / (2 * 16)  # ピッチ40mm, 半ピッチ, 16要素
    x0 = r * np.cos(0)
    y0 = r * np.sin(0)
    x1 = r * np.cos(angle)
    y1 = r * np.sin(angle)
    coords_hel = np.array([[x0, y0, 0.0], [x1, y1, dz]], dtype=float)

    # 直線要素 (同じ長さ)
    L_hel = np.linalg.norm(coords_hel[1] - coords_hel[0])
    coords_str = np.array([[0, 0, 0], [0, 0, L_hel]], dtype=float)

    print(f"\n  ヘリカル要素: {coords_hel[0]} → {coords_hel[1]}")
    print(f"  要素長: {L_hel*1000:.4f} mm")
    print(f"  直線要素: {coords_str[0]} → {coords_str[1]}")

    for name, coords in [("直線", coords_str), ("ヘリカル", coords_hel)]:
        print(f"\n--- {name}要素 ---")

        u0 = np.zeros(12)

        # 接線剛性
        K_T = timo_beam3d_cr_tangent(
            coords, u0, _E, G, section.A, section.Iy, section.Iz,
            section.J, kappa, kappa,
        )
        f0 = timo_beam3d_cr_internal_force(
            coords, u0, _E, G, section.A, section.Iy, section.Iz,
            section.J, kappa, kappa,
        )

        print(f"  ||f_int(0)||: {np.linalg.norm(f0):.6e}")
        print(f"  ||K_T||: {np.linalg.norm(K_T, 'fro'):.6e}")

        # 固有値
        eigvals = np.linalg.eigvalsh(K_T)
        pos = eigvals[eigvals > 1e-20]
        neg = eigvals[eigvals < -1e-20]
        zero = eigvals[np.abs(eigvals) <= 1e-20]
        print(f"  K_T 固有値: 正={len(pos)}, 負={len(neg)}, ゼロ={len(zero)}")
        if len(pos) > 0:
            print(f"    正: min={pos.min():.3e}, max={pos.max():.3e}, cond={pos.max()/pos.min():.3e}")
        if len(neg) > 0:
            print(f"    負: min={neg.min():.3e}, max={neg.max():.3e}")

        # 節点2のrx DOFに力を加えて、Newton方向を計算
        f_ext = np.zeros(12)
        f_ext[9] = _E * section.Iy * np.deg2rad(2) / np.linalg.norm(coords[1] - coords[0])

        # 固定: 節点1の全DOF
        free = np.arange(6, 12)
        K_ff = K_T[np.ix_(free, free)]
        R0 = f0[free] - f_ext[free]  # = -f_ext (since f0=0)
        du_f = np.linalg.solve(K_ff, -R0)
        du = np.zeros(12)
        du[free] = du_f

        print(f"\n  外力: f_ext[9] = {f_ext[9]:.6e} (2°相当)")
        print(f"  Newton方向 du:")
        names = ["ux", "uy", "uz", "rx", "ry", "rz"]
        for i in range(6):
            print(f"    node2 {names[i]}: {du[6+i]:.6e}")

        # f_int(alpha * du) の非線形性チェック
        print(f"\n  f_int(alpha*du) の非線形性:")
        print(f"  {'alpha':>8s} {'||f_int||':>12s} {'||K*du*a||':>12s} {'ratio':>10s} {'R_norm':>12s}")

        for alpha in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            u_test = alpha * du
            f_test = timo_beam3d_cr_internal_force(
                coords, u_test, _E, G, section.A, section.Iy, section.Iz,
                section.J, kappa, kappa,
            )
            f_linear = K_T @ u_test
            ratio = np.linalg.norm(f_test) / max(np.linalg.norm(f_linear), 1e-30)

            # 残差
            R_test = f_test - f_ext * alpha  # scale f_ext with alpha too? No
            # Actually the NR step uses f_step = f_ext (constant), du from K_T^{-1}*f_step
            R_test2 = f_test - f_ext  # for alpha=1, this is the actual residual
            print(f"  {alpha:8.3f} {np.linalg.norm(f_test):12.6e} "
                  f"{np.linalg.norm(f_linear):12.6e} {ratio:10.4f} "
                  f"{np.linalg.norm(R_test2[free]):12.6e}")

        # corotated フレームの回転をトレース
        print(f"\n  corotated フレーム追跡 (alpha=0→1):")
        L_0, e_x_0 = _beam3d_length_and_direction(coords)
        R_0 = _build_local_axes(e_x_0, None)
        print(f"  R_0 =")
        for row in R_0:
            print(f"    [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")

        for alpha in [0.0, 0.01, 0.1, 0.5, 1.0]:
            u_test = alpha * du
            x1_def = coords[0] + u_test[0:3]
            x2_def = coords[1] + u_test[6:9]
            coords_def = np.array([x1_def, x2_def])
            L_def, e_x_def = _beam3d_length_and_direction(coords_def)
            R_rod = _rodrigues_rotation(e_x_0, e_x_def)
            R_cr = R_0 @ R_rod.T

            # 自然変形
            R_node1 = _rotvec_to_rotmat(u_test[3:6])
            R_node2 = _rotvec_to_rotmat(u_test[9:12])
            R_def1 = R_cr @ R_node1 @ R_0.T
            R_def2 = R_cr @ R_node2 @ R_0.T
            theta_def1 = _rotmat_to_rotvec(R_def1)
            theta_def2 = _rotmat_to_rotvec(R_def2)

            d_cr = np.zeros(12)
            d_cr[3:6] = theta_def1
            d_cr[6] = L_def - L_0
            d_cr[9:12] = theta_def2

            print(f"\n  alpha={alpha:.2f}: L_def={L_def*1000:.6f}mm, "
                  f"dL={d_cr[6]*1000:.6e}mm")
            print(f"    e_x_def = ({e_x_def[0]:+.6f}, {e_x_def[1]:+.6f}, {e_x_def[2]:+.6f})")
            print(f"    theta_def1 = ({theta_def1[0]:+.6e}, {theta_def1[1]:+.6e}, {theta_def1[2]:+.6e})")
            print(f"    theta_def2 = ({theta_def2[0]:+.6e}, {theta_def2[1]:+.6e}, {theta_def2[2]:+.6e})")

            # f_cr = K_local @ d_cr
            Ke_local = timo_beam3d_ke_local(
                _E, G, section.A, section.Iy, section.Iz, section.J,
                L_0, kappa, kappa,
            )
            f_cr = Ke_local @ d_cr
            print(f"    ||d_cr||: {np.linalg.norm(d_cr):.6e}")
            print(f"    ||f_cr||: {np.linalg.norm(f_cr):.6e}")
            # DOF-wise
            for i in [3, 4, 5, 6, 9, 10, 11]:
                print(f"    d_cr[{i}]={d_cr[i]:+.6e}  f_cr[{i}]={f_cr[i]:+.6e}")


def diagnose_local_stiffness_check():
    """局所剛性行列の固有値確認."""
    print("\n" + "=" * 70)
    print("  局所剛性行列 K_local の確認")
    print("=" * 70)

    section = BeamSection.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)

    L = 0.00155  # typical element length
    Ke = timo_beam3d_ke_local(
        _E, G, section.A, section.Iy, section.Iz, section.J,
        L, kappa, kappa,
    )
    print(f"  L = {L*1000:.4f} mm")
    print(f"  E = {_E:.3e}, G = {G:.3e}")
    print(f"  A = {section.A:.6e}")
    print(f"  Iy = {section.Iy:.6e}, Iz = {section.Iz:.6e}")
    print(f"  J = {section.J:.6e}")

    eigvals = np.linalg.eigvalsh(Ke)
    print(f"\n  K_local 固有値:")
    for i, ev in enumerate(eigvals):
        print(f"    {i}: {ev:.6e}")

    # 対角要素
    print(f"\n  K_local 対角要素:")
    dof_names = ["ux1", "uy1", "uz1", "rx1", "ry1", "rz1",
                 "ux2", "uy2", "uz2", "rx2", "ry2", "rz2"]
    for i in range(12):
        print(f"    K[{dof_names[i]},{dof_names[i]}] = {Ke[i,i]:.6e}")


if __name__ == "__main__":
    diagnose_single_element()
    diagnose_local_stiffness_check()
