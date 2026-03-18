#!/usr/bin/env python3
"""変位制御でのヘリカルワイヤ収束テスト.

実際のNCP solverは変位制御(prescribed rotation)を使う。
力制御と変位制御での収束性の違いを確認する。
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


def test_disp_control(node_coords, connectivity, n_strands,
                      strand_nodes_fn, bend_angle_deg, n_steps,
                      max_iter=50, tol=1e-8, label=""):
    """変位制御（処方回転）でのNR収束テスト."""
    section = BeamSectionInput.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)
    n_nodes = len(node_coords)
    ndof = n_nodes * _NDOF

    # 固定: z=0端の全DOF
    fixed = set()
    for sid in range(n_strands):
        nodes = strand_nodes_fn(sid)
        for d in range(_NDOF):
            fixed.add(_NDOF * nodes[0] + d)
    fixed_dofs = np.array(sorted(fixed))

    # 処方: z=L端のrx DOF
    rx_dofs = []
    for sid in range(n_strands):
        nodes = strand_nodes_fn(sid)
        rx_dofs.append(_NDOF * nodes[-1] + 3)
    rx_dofs = np.array(rx_dofs)

    # 全拘束DOF
    all_fixed = np.unique(np.concatenate([fixed_dofs, rx_dofs]))
    free_dofs = np.array([d for d in range(ndof) if d not in all_fixed])

    bend_rad = np.deg2rad(bend_angle_deg)
    u = np.zeros(ndof)

    for step in range(n_steps):
        load_frac = (step + 1) / n_steps
        u[rx_dofs] = bend_rad * load_frac

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

            R = f_int.copy()
            R[fixed_dofs] = 0.0
            R[rx_dofs] = 0.0
            res = np.linalg.norm(R[free_dofs])

            if it == 0:
                f_ref = max(res, 1e-30)

            rel = res / f_ref
            if rel < tol:
                break

            K_ff = K_T[np.ix_(free_dofs, free_dofs)]
            du_f = spsolve(csc_matrix(K_ff), -R[free_dofs])
            du = np.zeros(ndof)
            du[free_dofs] = du_f
            u += du
        else:
            print(f"  {label} {bend_angle_deg}° step {step+1}/{n_steps}: "
                  f"NOT CONVERGED (iters={it+1}, rel={rel:.3e})")
            return False, step, n_steps, it + 1, rel

    return True, n_steps, n_steps, it + 1, rel


def main():
    print("=" * 70)
    print("  変位制御ヘリカルワイヤ収束テスト")
    print("=" * 70)

    mesh = make_twisted_wire_mesh(
        7, _WIRE_D, 0.040, length=0.0,
        n_elems_per_strand=8, n_pitches=0.5, min_elems_per_pitch=16,
    )

    # 7本撚線、変位制御
    print("\n--- 7本撚線 変位制御 ---")
    for angle in [10, 20, 30, 45, 60, 90]:
        n_steps = max(5, angle // 3)
        ok, done, total, iters, rel = test_disp_control(
            mesh.node_coords, mesh.connectivity, 7,
            mesh.strand_nodes, angle, n_steps, max_iter=50, tol=1e-8,
            label="7-wire",
        )
        status = "OK" if ok else "FAIL"
        print(f"  {angle:3d}°: {status} ({done}/{total} steps, "
              f"last_iters={iters}, rel={rel:.3e})")

    # 外周1本だけ、変位制御
    print("\n--- 外周strand 1のみ 変位制御 ---")
    strand_nodes_list = mesh.strand_nodes(1)
    strand_elems = mesh.strand_elems(1)
    node_map = {old: new for new, old in enumerate(strand_nodes_list)}
    coords_s = mesh.node_coords[strand_nodes_list].copy()
    conn_s = np.array([
        [node_map[mesh.connectivity[ei, 0]], node_map[mesh.connectivity[ei, 1]]]
        for ei in strand_elems
    ])

    for angle in [10, 20, 30, 45, 60, 90]:
        n_steps = max(5, angle // 3)
        ok, done, total, iters, rel = test_disp_control(
            coords_s, conn_s, 1,
            lambda sid: np.arange(len(coords_s)),
            angle, n_steps, max_iter=50, tol=1e-8,
            label="strand1",
        )
        status = "OK" if ok else "FAIL"
        print(f"  {angle:3d}°: {status} ({done}/{total} steps, "
              f"last_iters={iters}, rel={rel:.3e})")

    # 外周1本、変位制御、反復詳細
    print("\n--- 外周strand 1 変位制御 10° 反復詳細 ---")
    section = BeamSectionInput.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)
    n_nodes_s = len(coords_s)
    ndof_s = n_nodes_s * _NDOF
    fixed_s = np.arange(_NDOF)
    rx_s = np.array([(n_nodes_s - 1) * _NDOF + 3])
    all_fixed_s = np.unique(np.concatenate([fixed_s, rx_s]))
    free_s = np.array([d for d in range(ndof_s) if d not in all_fixed_s])

    u_s = np.zeros(ndof_s)
    bend_rad = np.deg2rad(10)
    n_steps = 5

    for step in range(n_steps):
        load_frac = (step + 1) / n_steps
        u_s[rx_s] = bend_rad * load_frac
        print(f"\n  Step {step+1}/{n_steps} (rx={np.rad2deg(bend_rad*load_frac):.1f}°)")

        for it in range(50):
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
            R_s = f_int_s.copy()
            R_s[fixed_s] = 0.0
            R_s[rx_s] = 0.0
            res_s = np.linalg.norm(R_s[free_s])

            if it == 0:
                f_ref_s = max(res_s, 1e-30)

            rel_s = res_s / f_ref_s
            if it < 15 or rel_s < 1e-6:
                print(f"    iter {it}: ||R||/||f||={rel_s:.6e}, ||R||={res_s:.6e}")

            if rel_s < 1e-8:
                break

            K_ff_s = K_T_s[np.ix_(free_s, free_s)]
            du_f_s = spsolve(csc_matrix(K_ff_s), -R_s[free_s])
            du_s = np.zeros(ndof_s)
            du_s[free_s] = du_f_s
            u_s += du_s


if __name__ == "__main__":
    main()
