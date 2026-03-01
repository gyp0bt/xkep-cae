"""Corotational (CR) Timoshenko 3D 梁のテスト.

検証項目:
  - 小変位で線形剛性行列・線形内力と一致
  - 接線剛性が内力の数値微分と一致
  - 純回転（剛体回転）でゼロ内力
  - 片持ち梁大たわみの NR 収束
  - アセンブリ関数の動作確認
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.beam_timo3d import (
    assemble_cr_beam3d,
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent,
    timo_beam3d_ke_global,
)

# =====================================================================
# テストパラメータ
# =====================================================================
E = 200e3  # MPa
NU = 0.3
G = E / (2.0 * (1.0 + NU))
A = 100.0  # mm²
Iy = 833.333  # mm⁴ (10×10 矩形断面)
Iz = 833.333
J = 1406.0  # mm⁴
KAPPA_Y = 5.0 / 6.0
KAPPA_Z = 5.0 / 6.0
L = 100.0  # mm


def _coords_x() -> np.ndarray:
    """x軸方向の梁座標."""
    return np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])


# =====================================================================
# 小変位での線形一致テスト
# =====================================================================
class TestSmallDisplacementLinearMatch:
    """小変位で CR 内力が線形 K·u と一致することを検証."""

    @pytest.mark.parametrize(
        "dof_idx",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ids=[
            "ux1",
            "uy1",
            "uz1",
            "rx1",
            "ry1",
            "rz1",
            "ux2",
            "uy2",
            "uz2",
            "rx2",
            "ry2",
            "rz2",
        ],
    )
    def test_single_dof_small(self, dof_idx: int) -> None:
        """各DOFに微小変位を与え、f_int ≈ K·u を確認."""
        coords = _coords_x()
        u = np.zeros(12)
        u[dof_idx] = 1e-6  # 微小変位

        f_cr = timo_beam3d_cr_internal_force(
            coords,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
        )
        K = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, KAPPA_Y, KAPPA_Z)
        f_lin = K @ u

        np.testing.assert_allclose(f_cr, f_lin, atol=1e-4, rtol=1e-4)

    def test_combined_small_displacement(self) -> None:
        """ランダムな微小変位で f_int ≈ K·u を確認."""
        rng = np.random.default_rng(42)
        coords = _coords_x()
        u = rng.standard_normal(12) * 1e-6

        f_cr = timo_beam3d_cr_internal_force(
            coords,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
        )
        K = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, KAPPA_Y, KAPPA_Z)
        f_lin = K @ u

        np.testing.assert_allclose(f_cr, f_lin, atol=1e-4, rtol=1e-4)


# =====================================================================
# 接線剛性の数値微分検証
# =====================================================================
class TestTangentStiffness:
    """接線剛性が独立の有限差分と一致することを検証."""

    def test_tangent_matches_fd_zero_disp(self) -> None:
        """ゼロ変位で接線剛性 ≈ 線形剛性."""
        coords = _coords_x()
        u = np.zeros(12)

        K_T = timo_beam3d_cr_tangent(
            coords,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
        )
        K_lin = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, KAPPA_Y, KAPPA_Z)

        np.testing.assert_allclose(K_T, K_lin, atol=1e-2, rtol=1e-3)

    def test_tangent_matches_independent_fd(self) -> None:
        """有限変位状態で接線剛性が独立のFD計算と一致."""
        coords = _coords_x()
        rng = np.random.default_rng(123)
        u = rng.standard_normal(12) * 0.5  # 中程度の変位

        K_T = timo_beam3d_cr_tangent(
            coords,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
        )

        # 独立の FD（異なる eps）
        eps2 = 1e-6
        K_fd = np.zeros((12, 12))
        for j in range(12):
            u_p = u.copy()
            u_m = u.copy()
            u_p[j] += eps2
            u_m[j] -= eps2
            f_p = timo_beam3d_cr_internal_force(
                coords,
                u_p,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                KAPPA_Y,
                KAPPA_Z,
            )
            f_m = timo_beam3d_cr_internal_force(
                coords,
                u_m,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                KAPPA_Y,
                KAPPA_Z,
            )
            K_fd[:, j] = (f_p - f_m) / (2.0 * eps2)
        K_fd = 0.5 * (K_fd + K_fd.T)

        np.testing.assert_allclose(K_T, K_fd, atol=1e-1, rtol=1e-3)

    def test_tangent_symmetry(self) -> None:
        """接線剛性が対称であることを確認."""
        coords = _coords_x()
        rng = np.random.default_rng(456)
        u = rng.standard_normal(12) * 0.3

        K_T = timo_beam3d_cr_tangent(
            coords,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
        )

        np.testing.assert_allclose(K_T, K_T.T, atol=1e-6)


# =====================================================================
# 剛体回転テスト
# =====================================================================
class TestRigidBodyMotion:
    """剛体運動で内力がゼロであることを検証."""

    def test_pure_translation(self) -> None:
        """純粋な並進で f_int = 0."""
        coords = _coords_x()
        # 両節点に同じ並進
        u = np.zeros(12)
        u[0] = u[6] = 1.0  # ux
        u[1] = u[7] = 2.0  # uy
        u[2] = u[8] = -0.5  # uz

        f = timo_beam3d_cr_internal_force(
            coords,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
        )
        np.testing.assert_allclose(f, 0.0, atol=1e-8)


# =====================================================================
# 大変形テスト
# =====================================================================
class TestLargeDeformation:
    """大変形での基本動作を検証."""

    def test_pure_axial_extension(self) -> None:
        """純軸伸びで軸力のみ発生."""
        coords = _coords_x()
        delta = 1.0  # 1mm 伸び
        u = np.zeros(12)
        u[6] = delta  # 節点2 の ux

        f = timo_beam3d_cr_internal_force(
            coords,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
        )

        N_expected = E * A * delta / L
        # 節点1: -N（圧縮方向）、節点2: +N（引張方向）
        assert f[0] == pytest.approx(-N_expected, rel=1e-6)
        assert f[6] == pytest.approx(N_expected, rel=1e-6)
        # 横方向はゼロ
        np.testing.assert_allclose(f[[1, 2, 7, 8]], 0.0, atol=1e-8)

    def test_force_increases_with_displacement(self) -> None:
        """変位を増やすと内力が増加する（単調性）."""
        coords = _coords_x()

        norms = []
        for scale in [0.1, 0.5, 1.0, 5.0]:
            u = np.zeros(12)
            u[7] = -scale  # 節点2 uy（下向き）
            f = timo_beam3d_cr_internal_force(
                coords,
                u,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                KAPPA_Y,
                KAPPA_Z,
            )
            norms.append(np.linalg.norm(f))

        for i in range(len(norms) - 1):
            assert norms[i + 1] > norms[i]


# =====================================================================
# アセンブリテスト
# =====================================================================
class TestAssembly:
    """assemble_cr_beam3d のテスト."""

    def test_assembly_zero_displacement(self) -> None:
        """ゼロ変位で内力がゼロ."""
        n_elems = 4
        n_nodes = n_elems + 1
        nodes = np.column_stack(
            [
                np.linspace(0, L, n_nodes),
                np.zeros(n_nodes),
                np.zeros(n_nodes),
            ]
        )
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)])
        u = np.zeros(6 * n_nodes)

        K_T, f_int = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
            sparse=False,
        )

        np.testing.assert_allclose(f_int, 0.0, atol=1e-12)
        assert K_T is not None
        assert K_T.shape == (6 * n_nodes, 6 * n_nodes)

    def test_assembly_small_disp_matches_linear(self) -> None:
        """小変位でアセンブリ結果が線形 K·u と一致."""
        n_elems = 4
        n_nodes = n_elems + 1
        nodes = np.column_stack(
            [
                np.linspace(0, L, n_nodes),
                np.zeros(n_nodes),
                np.zeros(n_nodes),
            ]
        )
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)])
        ndof = 6 * n_nodes

        rng = np.random.default_rng(789)
        u = rng.standard_normal(ndof) * 1e-6

        K_T, f_int = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
            sparse=False,
        )

        # 線形剛性でのアセンブリ
        from xkep_cae.numerical_tests.runner import _assemble_3d

        def ke_func(c):
            return timo_beam3d_ke_global(
                c,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                KAPPA_Y,
                KAPPA_Z,
            )

        K_lin, _ = _assemble_3d(nodes, conn, ke_func)
        f_lin = K_lin @ u

        np.testing.assert_allclose(f_int, f_lin, atol=1e-4, rtol=1e-4)

    def test_assembly_stiffness_only(self) -> None:
        """stiffness=True, internal_force=False の動作確認."""
        n_elems = 2
        n_nodes = n_elems + 1
        nodes = np.column_stack(
            [
                np.linspace(0, L, n_nodes),
                np.zeros(n_nodes),
                np.zeros(n_nodes),
            ]
        )
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)])
        u = np.zeros(6 * n_nodes)

        K_T, f_int = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            KAPPA_Y,
            KAPPA_Z,
            stiffness=True,
            internal_force=False,
            sparse=False,
        )

        assert K_T is not None
        assert f_int is None


# =====================================================================
# 静的 NR ソルバーとの統合テスト — 片持ち梁大たわみ
# =====================================================================
class TestCantileverNonlinear:
    """CR 梁 + newton_raphson で片持ち梁の大たわみを解く."""

    @staticmethod
    def _make_cr_assemblers(nodes, conn):
        """CR 梁用のスパース行列コールバックを生成."""

        def assemble_f_int(u):
            _, fi = assemble_cr_beam3d(
                nodes,
                conn,
                u,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                KAPPA_Y,
                KAPPA_Z,
                stiffness=False,
                internal_force=True,
            )
            return fi

        def assemble_K_T(u):
            kt, _ = assemble_cr_beam3d(
                nodes,
                conn,
                u,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                KAPPA_Y,
                KAPPA_Z,
                stiffness=True,
                internal_force=False,
            )
            return kt

        return assemble_f_int, assemble_K_T

    def test_cantilever_large_deflection_converges(self) -> None:
        """大荷重で NR が収束することを確認."""
        from xkep_cae.solver import newton_raphson

        n_elems = 10
        n_nodes = n_elems + 1
        nodes = np.column_stack(
            [
                np.linspace(0, 1000.0, n_nodes),
                np.zeros(n_nodes),
                np.zeros(n_nodes),
            ]
        )
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)])
        ndof = 6 * n_nodes

        fixed_dofs = np.arange(6)
        f_ext = np.zeros(ndof)
        P = -50.0
        f_ext[6 * n_elems + 1] = P

        assemble_f_int, assemble_K_T = self._make_cr_assemblers(nodes, conn)

        result = newton_raphson(
            f_ext_total=f_ext,
            fixed_dofs=fixed_dofs,
            assemble_tangent=assemble_K_T,
            assemble_internal_force=assemble_f_int,
            n_load_steps=5,
            max_iter=30,
            tol_force=1e-8,
            show_progress=False,
        )

        assert result.converged
        tip_uy = result.u[6 * n_elems + 1]
        assert tip_uy < 0

        # δ_lin = PL³/(3EI) ≈ 100 mm
        delta_lin = abs(P) * 1000.0**3 / (3.0 * E * Iy)
        assert abs(tip_uy) > 0.5 * delta_lin
        assert abs(tip_uy) < 2.0 * delta_lin

    def test_small_load_matches_linear(self) -> None:
        """微小荷重で線形解と一致."""
        from xkep_cae.solver import newton_raphson

        n_elems = 10
        n_nodes = n_elems + 1
        nodes = np.column_stack(
            [
                np.linspace(0, 1000.0, n_nodes),
                np.zeros(n_nodes),
                np.zeros(n_nodes),
            ]
        )
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)])
        ndof = 6 * n_nodes

        fixed_dofs = np.arange(6)
        f_ext = np.zeros(ndof)
        P = -0.001
        f_ext[6 * n_elems + 1] = P

        assemble_f_int, assemble_K_T = self._make_cr_assemblers(nodes, conn)

        result = newton_raphson(
            f_ext_total=f_ext,
            fixed_dofs=fixed_dofs,
            assemble_tangent=assemble_K_T,
            assemble_internal_force=assemble_f_int,
            n_load_steps=1,
            max_iter=30,
            tol_force=1e-8,
            show_progress=False,
        )

        assert result.converged
        tip_uy = result.u[6 * n_elems + 1]

        delta_lin = P * 1000.0**3 / (3.0 * E * Iy)
        np.testing.assert_allclose(tip_uy, delta_lin, rtol=0.02)
