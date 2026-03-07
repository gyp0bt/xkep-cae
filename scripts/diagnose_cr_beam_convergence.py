#!/usr/bin/env python3
"""CR梁要素の大回転収束診断.

7本撚線ではなく単一梁でCR要素のNR収束を確認し、
接触を完全に排除した状態での収束限界を特定する。

また、Modified NRの閾値、線探索の有無の影響を調査。
"""

import sys
import time

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, ".")

from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
from xkep_cae.sections.beam import BeamSection

_E = 200e9
_NU = 0.3
_WIRE_D = 0.002
_NDOF = 6


def _G(E, nu):
    return E / (2.0 * (1.0 + nu))


def _kappa(nu):
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


def make_single_beam(n_elems=8, length=0.020):
    """単一直線梁メッシュを作成."""
    n_nodes = n_elems + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i, 2] = i * length / n_elems  # z軸方向
    connectivity = np.array([[i, i + 1] for i in range(n_elems)])
    return node_coords, connectivity, n_nodes


def single_beam_bending_test(
    n_elems=8,
    length=0.020,
    bend_angle_deg=90.0,
    n_steps=20,
    max_iter=50,
    tol=1e-8,
    modified_nr_threshold=0,
    use_line_search=False,
):
    """単一CR梁の片持ちモーメント荷重曲げテスト."""
    section = BeamSection.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)

    node_coords, connectivity, n_nodes = make_single_beam(n_elems, length)
    ndof = n_nodes * _NDOF

    # 固定: z=0端の全DOF
    fixed_dofs = np.arange(_NDOF)
    free_dofs = np.array([d for d in range(ndof) if d not in fixed_dofs])

    # 外力: z=L端にモーメント Mx
    bend_angle_rad = np.deg2rad(bend_angle_deg)
    M_target = _E * section.Iy * bend_angle_rad / length
    f_ext = np.zeros(ndof)
    f_ext[(n_nodes - 1) * _NDOF + 3] = M_target  # rx DOF

    u = np.zeros(ndof)
    converged_steps = 0

    for step in range(n_steps):
        load_frac = (step + 1) / n_steps
        f_ext_step = f_ext * load_frac

        K_T_frozen = None
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

            R = f_int - f_ext_step
            R[fixed_dofs] = 0.0

            res_norm = np.linalg.norm(R[free_dofs])
            f_norm = max(np.linalg.norm(f_ext_step[free_dofs]), 1e-30)
            rel_res = res_norm / f_norm

            if rel_res < tol:
                if step % 5 == 0 or step == n_steps - 1:
                    print(
                        f"  Step {step + 1}/{n_steps} (frac={load_frac:.3f}): "
                        f"converged iter={it}, ||R||/||f||={rel_res:.3e}"
                    )
                converged_steps += 1
                break

            if it % 10 == 0:
                print(
                    f"  Step {step + 1}/{n_steps}, iter {it}: "
                    f"||R||/||f||={rel_res:.3e}"
                )

            # Modified NR: 接線凍結
            if modified_nr_threshold > 0 and it >= modified_nr_threshold:
                if K_T_frozen is None or (it - modified_nr_threshold) % 5 == 0:
                    K_T_frozen = K_T.copy()
                K_solve = K_T_frozen
            else:
                K_solve = K_T

            # 線形求解
            K_ff = K_solve[np.ix_(free_dofs, free_dofs)]
            R_f = R[free_dofs]
            du_f = spsolve(csc_matrix(K_ff), -R_f)

            du = np.zeros(ndof)
            du[free_dofs] = du_f

            # Line search
            if use_line_search:
                alpha = 1.0
                for ls in range(6):
                    u_try = u + alpha * du
                    _, f_int_try = assemble_cr_beam3d(
                        node_coords, connectivity, u_try,
                        _E, G, section.A, section.Iy, section.Iz, section.J,
                        kappa, kappa, stiffness=False, internal_force=True,
                    )
                    R_try = f_int_try - f_ext_step
                    R_try[fixed_dofs] = 0.0
                    res_try = np.linalg.norm(R_try[free_dofs])
                    if res_try < res_norm * 1.5:
                        break
                    alpha *= 0.5
                u += alpha * du
            else:
                u += du
        else:
            print(
                f"  Step {step + 1}/{n_steps}: NOT CONVERGED "
                f"(last ||R||/||f||={rel_res:.3e})"
            )
            break

    all_converged = converged_steps == n_steps
    tip_disp = u[-_NDOF:-_NDOF + 3] if ndof >= _NDOF else np.zeros(3)
    return all_converged, converged_steps, n_steps, tip_disp


def main():
    print("=" * 70)
    print("  CR梁要素 大回転NR収束診断")
    print("=" * 70)

    # テスト1: 単一梁で角度ごとの収束テスト
    print("\n--- テスト1: 単一梁、角度ごとの収束（Full NR, tol=1e-8）---")
    for angle in [10, 20, 30, 45, 60, 90, 120, 180]:
        n_steps = max(5, angle // 5)
        ok, done, total, tip = single_beam_bending_test(
            n_elems=8, bend_angle_deg=angle, n_steps=n_steps,
            max_iter=50, tol=1e-8, modified_nr_threshold=0,
        )
        status = "OK" if ok else "FAIL"
        print(f"  {angle:4d}°: {status} ({done}/{total} steps) "
              f"tip=({tip[0]*1e3:.3f}, {tip[1]*1e3:.3f}, {tip[2]*1e3:.3f}) mm\n")

    # テスト2: Modified NR (threshold=5) vs Full NR for 45°
    print("\n--- テスト2: Modified NR比較 (45°) ---")
    for mnr in [0, 3, 5, 10]:
        label = f"MNR_threshold={mnr}" if mnr > 0 else "Full NR"
        ok, done, total, tip = single_beam_bending_test(
            n_elems=8, bend_angle_deg=45, n_steps=10,
            max_iter=50, tol=1e-8, modified_nr_threshold=mnr,
        )
        status = "OK" if ok else "FAIL"
        print(f"  {label}: {status} ({done}/{total})\n")

    # テスト3: Line search効果 (45°)
    print("\n--- テスト3: Line search比較 (45°) ---")
    for ls in [False, True]:
        label = f"LS={'ON' if ls else 'OFF'}"
        ok, done, total, tip = single_beam_bending_test(
            n_elems=8, bend_angle_deg=45, n_steps=10,
            max_iter=50, tol=1e-8, use_line_search=ls,
        )
        status = "OK" if ok else "FAIL"
        print(f"  {label}: {status} ({done}/{total})\n")

    # テスト4: 7本撚線（接触なし、NRのみ）で角度テスト
    print("\n--- テスト4: 7本撚線 接触なし 純NR ---")
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

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

    for angle in [10, 20, 30, 45, 60, 90]:
        bend_angle_rad = np.deg2rad(angle)
        M_per_strand = _E * section.Iy * bend_angle_rad / mesh.length
        f_ext = np.zeros(ndof)
        for sid in range(7):
            nodes = mesh.strand_nodes(sid)
            n_end = nodes[-1]
            f_ext[_NDOF * n_end + 3] = M_per_strand

        n_steps = max(5, angle // 5)
        u = np.zeros(ndof)
        converged_steps = 0

        for step in range(n_steps):
            load_frac = (step + 1) / n_steps
            f_step = f_ext * load_frac

            for it in range(50):
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
                res = np.linalg.norm(R[free_dofs])
                f_ref = max(np.linalg.norm(f_step[free_dofs]), 1e-30)
                if res / f_ref < 1e-8:
                    converged_steps += 1
                    break

                K_ff = K_T[np.ix_(free_dofs, free_dofs)]
                du_f = spsolve(csc_matrix(K_ff), -R[free_dofs])
                du = np.zeros(ndof)
                du[free_dofs] = du_f
                u += du
            else:
                print(f"  7-wire {angle}° step {step+1}: NOT CONVERGED "
                      f"(||R||/||f||={res/f_ref:.3e})")
                break

        status = "OK" if converged_steps == n_steps else "FAIL"
        print(f"  7-wire {angle:3d}°: {status} ({converged_steps}/{n_steps} steps)")

    # テスト5: 7本撚線 変位制御（NCP方式と同じ）で角度テスト
    print("\n--- テスト5: 7本撚線 変位制御 純NR ---")
    for angle in [10, 20, 30, 45, 60, 90]:
        bend_angle_rad = np.deg2rad(angle)
        n_steps = max(5, angle // 3)

        # 処方: rx DOF at tip
        rx_dofs_end = []
        for sid in range(7):
            nodes = mesh.strand_nodes(sid)
            n_end = nodes[-1]
            rx_dofs_end.append(_NDOF * n_end + 3)
        rx_arr = np.array(rx_dofs_end)

        fixed_all = np.unique(np.concatenate([fixed_dofs, rx_arr]))
        free_all = np.array([d for d in range(ndof) if d not in fixed_all])

        u = np.zeros(ndof)
        converged_steps = 0

        for step in range(n_steps):
            load_frac = (step + 1) / n_steps
            # 処方変位
            u[rx_arr] = bend_angle_rad * load_frac

            for it in range(50):
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
                R = f_int.copy()
                R[fixed_dofs] = 0.0
                R[rx_arr] = 0.0  # 処方DOFの残差はゼロ
                res = np.linalg.norm(R[free_all])

                # 基準ノルム: 初回の残差
                if it == 0:
                    f_ref = max(res, 1e-30)

                if res / f_ref < 1e-6:
                    converged_steps += 1
                    break

                if it % 10 == 0:
                    print(f"    {angle}° step {step+1}, iter {it}: "
                          f"||R||/||f||={res/f_ref:.3e}")

                K_ff = K_T[np.ix_(free_all, free_all)]
                du_f = spsolve(csc_matrix(K_ff), -R[free_all])
                du = np.zeros(ndof)
                du[free_all] = du_f
                u += du
            else:
                print(f"  7-wire disp-ctrl {angle}° step {step+1}: NOT CONVERGED "
                      f"(||R||/||f||={res/f_ref:.3e})")
                break

        status = "OK" if converged_steps == n_steps else "FAIL"
        print(f"  7-wire disp-ctrl {angle:3d}°: {status} ({converged_steps}/{n_steps} steps)")


if __name__ == "__main__":
    main()
