#!/usr/bin/env python3
"""CR梁の数値微分接線剛性の品質診断.

数値微分のepsが大回転・ヘリカル形状で不適切になることを検証。
解析的接線と数値微分接線の比較、eps感度テストを実施。
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from xkep_cae.elements.beam_timo3d import (
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent,
    timo_beam3d_ke_local,
)
from xkep_cae.sections.beam import BeamSection

_E = 200e9
_NU = 0.3
_WIRE_D = 0.002


def _G(E, nu):
    return E / (2.0 * (1.0 + nu))


def _kappa(nu):
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


def numerical_tangent_with_eps(
    coords, u_elem, E, G, A, Iy, Iz, J, ky, kz, eps, v_ref=None
):
    """指定epsでの数値微分接線剛性."""
    K = np.zeros((12, 12))
    for j in range(12):
        u_p = u_elem.copy()
        u_m = u_elem.copy()
        u_p[j] += eps
        u_m[j] -= eps
        f_p = timo_beam3d_cr_internal_force(
            coords, u_p, E, G, A, Iy, Iz, J, ky, kz, v_ref=v_ref
        )
        f_m = timo_beam3d_cr_internal_force(
            coords, u_m, E, G, A, Iy, Iz, J, ky, kz, v_ref=v_ref
        )
        K[:, j] = (f_p - f_m) / (2 * eps)
    return 0.5 * (K + K.T)


def test_eps_sensitivity():
    """異なるepsでの接線剛性の変動を評価."""
    print("=" * 70)
    print("  診断: 数値微分eps感度")
    print("=" * 70)

    section = BeamSection.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)

    # ケース1: 直線梁、変形なし
    coords_straight = np.array([[0, 0, 0], [0, 0, 0.0025]], dtype=float)
    u_zero = np.zeros(12)

    # ケース2: 直線梁、大回転 (30°)
    u_rotated = np.zeros(12)
    u_rotated[9] = np.deg2rad(30)  # node2 θx = 30°

    # ケース3: ヘリカル梁（初期傾斜）
    # 1本のヘリカルワイヤの要素を模擬
    angle = 2 * np.pi / 16  # 1/16回転
    r = 0.002  # 配置半径
    z0, z1 = 0.0, 0.0025  # 要素長さ
    x0 = r * np.cos(0)
    y0 = r * np.sin(0)
    x1 = r * np.cos(angle)
    y1 = r * np.sin(angle)
    coords_helical = np.array([[x0, y0, z0], [x1, y1, z1]], dtype=float)

    # ケース4: ヘリカル梁 + 大回転
    u_hel_rotated = np.zeros(12)
    u_hel_rotated[9] = np.deg2rad(15)
    u_hel_rotated[10] = np.deg2rad(10)

    cases = [
        ("直線梁・変形なし", coords_straight, u_zero),
        ("直線梁・30°回転", coords_straight, u_rotated),
        ("ヘリカル梁・変形なし", coords_helical, u_zero),
        ("ヘリカル梁・15°回転", coords_helical, u_hel_rotated),
    ]

    eps_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    for name, coords, u_e in cases:
        print(f"\n--- {name} ---")
        # 基準: eps=1e-7（現在のデフォルト）
        K_ref = numerical_tangent_with_eps(
            coords, u_e, _E, G, section.A, section.Iy, section.Iz,
            section.J, kappa, kappa, eps=1e-7
        )
        K_ref_norm = np.linalg.norm(K_ref, "fro")

        print(f"  ||K_ref||_F = {K_ref_norm:.6e}")

        for eps in eps_values:
            K_eps = numerical_tangent_with_eps(
                coords, u_e, _E, G, section.A, section.Iy, section.Iz,
                section.J, kappa, kappa, eps=eps
            )
            diff = np.linalg.norm(K_eps - K_ref, "fro")
            rel_diff = diff / K_ref_norm if K_ref_norm > 0 else 0
            print(f"  eps={eps:.0e}: rel_diff={rel_diff:.6e}")

        # 対称性チェック
        K_asym = numerical_tangent_with_eps(
            coords, u_e, _E, G, section.A, section.Iy, section.Iz,
            section.J, kappa, kappa, eps=1e-7
        )
        sym_err = np.linalg.norm(K_asym - K_asym.T, "fro") / max(K_ref_norm, 1e-30)
        print(f"  対称性誤差 (対称化前): {sym_err:.6e}")

        # 条件数
        try:
            eigvals = np.linalg.eigvalsh(K_ref)
            pos = eigvals[eigvals > 1e-30]
            if len(pos) > 0:
                cond = pos[-1] / pos[0]
                print(f"  条件数: {cond:.3e} (正固有値数: {len(pos)}/{len(eigvals)})")
        except Exception:
            pass


def test_tangent_consistency():
    """接線剛性の整合性テスト: f(u+du) - f(u) ≈ K*du を評価."""
    print("\n" + "=" * 70)
    print("  診断: 接線剛性の整合性 (f(u+du) - f(u) vs K*du)")
    print("=" * 70)

    section = BeamSection.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)

    # ヘリカル梁要素
    angle = 2 * np.pi / 16
    r = 0.002
    coords = np.array([
        [r * np.cos(0), r * np.sin(0), 0.0],
        [r * np.cos(angle), r * np.sin(angle), 0.0025],
    ])

    # 異なる変形状態でのテスト
    rng = np.random.RandomState(42)
    deform_scales = [0.0, 0.001, 0.01, 0.1, 0.3, 0.5]

    for scale in deform_scales:
        u_base = rng.randn(12) * scale
        # 並進は小さく、回転はスケーリング
        u_base[:3] *= 1e-3
        u_base[6:9] *= 1e-3

        K_T = timo_beam3d_cr_tangent(
            coords, u_base, _E, G, section.A, section.Iy, section.Iz,
            section.J, kappa, kappa
        )
        f_base = timo_beam3d_cr_internal_force(
            coords, u_base, _E, G, section.A, section.Iy, section.Iz,
            section.J, kappa, kappa
        )

        # 小さい摂動
        for du_scale in [1e-4, 1e-5, 1e-6, 1e-7]:
            du = rng.randn(12) * du_scale
            du[:3] *= 1e-3
            du[6:9] *= 1e-3

            f_pert = timo_beam3d_cr_internal_force(
                coords, u_base + du, _E, G, section.A, section.Iy, section.Iz,
                section.J, kappa, kappa
            )

            df_actual = f_pert - f_base
            df_approx = K_T @ du
            err = np.linalg.norm(df_actual - df_approx)
            ref = max(np.linalg.norm(df_actual), 1e-30)

            if scale == 0.0 and du_scale == 1e-6:
                print(
                    f"  scale={scale:.3f}, du_scale={du_scale:.0e}: "
                    f"||df-K*du||/||df|| = {err/ref:.3e}"
                )
            elif du_scale == 1e-6:
                print(
                    f"  scale={scale:.3f}, du_scale={du_scale:.0e}: "
                    f"||df-K*du||/||df|| = {err/ref:.3e}"
                )


def test_7wire_tangent_convergence():
    """7本撚線での実際の接線剛性品質を反復ごとに追跡."""
    print("\n" + "=" * 70)
    print("  診断: 7本撚線 接線剛性品質（反復ごとの二次収束チェック）")
    print("=" * 70)

    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve

    from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

    mesh = make_twisted_wire_mesh(
        7, _WIRE_D, 0.040, length=0.0,
        n_elems_per_strand=8, n_pitches=0.5, min_elems_per_pitch=16,
    )
    section = BeamSection.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)
    ndof = mesh.n_nodes * 6

    fixed = set()
    for sid in range(7):
        nodes = mesh.strand_nodes(sid)
        for d in range(6):
            fixed.add(6 * nodes[0] + d)
    fixed_dofs = np.array(sorted(fixed))
    free_dofs = np.array([d for d in range(ndof) if d not in fixed])

    # 力制御: rx DOFにモーメント
    for angle in [10, 20, 30]:
        bend_rad = np.deg2rad(angle)
        M = _E * section.Iy * bend_rad / mesh.length
        f_ext = np.zeros(ndof)
        for sid in range(7):
            nodes = mesh.strand_nodes(sid)
            n_end = nodes[-1]
            f_ext[6 * n_end + 3] = M

        n_steps = max(3, angle // 5)
        u = np.zeros(ndof)

        print(f"\n  --- {angle}° 力制御, {n_steps} steps ---")
        for step in range(n_steps):
            load_frac = (step + 1) / n_steps
            f_step = f_ext * load_frac

            residuals = []
            for it in range(30):
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
                rel = res / f_ref
                residuals.append(rel)

                if rel < 1e-8:
                    break

                K_ff = K_T[np.ix_(free_dofs, free_dofs)]
                du_f = spsolve(csc_matrix(K_ff), -R[free_dofs])
                du = np.zeros(ndof)
                du[free_dofs] = du_f
                u += du

            # 収束率分析
            n_its = len(residuals)
            conv_rates = []
            for i in range(2, n_its):
                if residuals[i - 1] > 1e-30 and residuals[i - 2] > 1e-30:
                    # 二次収束: log(r_n)/log(r_{n-1}) ≈ 2
                    r = np.log10(max(residuals[i], 1e-30)) / np.log10(max(residuals[i - 1], 1e-30))
                    conv_rates.append(r)

            print(f"  Step {step+1}/{n_steps}: {n_its} iters, "
                  f"final={residuals[-1]:.3e}")
            if len(residuals) <= 8:
                for i, r in enumerate(residuals):
                    rate_str = f", rate={conv_rates[i-2]:.2f}" if i >= 2 and i - 2 < len(conv_rates) else ""
                    print(f"    iter {i}: {r:.6e}{rate_str}")

            if residuals[-1] > 1e-4:
                print(f"    *** 収束失敗 ***")
                break


def main():
    test_eps_sensitivity()
    test_tangent_consistency()
    test_7wire_tangent_convergence()


if __name__ == "__main__":
    main()
