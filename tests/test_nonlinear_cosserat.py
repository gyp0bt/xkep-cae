"""非線形 Cosserat rod 要素のテスト.

テスト方針:
  1. 非線形歪み計算の基本検証（ゼロ変位、小変位で線形一致）
  2. 非線形内力の基本検証（ゼロ変位→ゼロ、小変位で線形一致）
  3. 非線形接線剛性の有限差分検証（最重要）
  4. 接線剛性の対称性
"""

from __future__ import annotations

import numpy as np

from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    cosserat_internal_force_nonlinear,
    cosserat_nonlinear_strains,
    cosserat_tangent_stiffness_nonlinear,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection

# --- テスト用パラメータ ---
E = 200_000.0  # MPa
NU = 0.3
G = E / (2.0 * (1.0 + NU))


def _make_section() -> BeamSection:
    return BeamSection.rectangle(10.0, 20.0)


def _make_material() -> BeamElastic1D:
    return BeamElastic1D(E=E, nu=NU)


def _x_beam_coords(L: float) -> np.ndarray:
    """x軸方向の2節点梁座標."""
    return np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])


class TestNonlinearStrains:
    """非線形歪み計算の基本検証."""

    def test_zero_displacement_gives_zero_strain(self):
        """ゼロ変位ではゼロ歪み."""
        coords = _x_beam_coords(10.0)
        u = np.zeros(12)
        strains = cosserat_nonlinear_strains(coords, u)
        np.testing.assert_array_almost_equal(strains.gamma, [0, 0, 0])
        np.testing.assert_array_almost_equal(strains.kappa, [0, 0, 0])

    def test_pure_axial_small(self):
        """小さな軸伸び: Γ₁ ≈ δ/L."""
        L = 10.0
        coords = _x_beam_coords(L)
        delta = 0.001  # very small
        u = np.zeros(12)
        u[6] = delta  # node2 x-displacement
        strains = cosserat_nonlinear_strains(coords, u)
        np.testing.assert_almost_equal(strains.gamma[0], delta / L, decimal=6)
        np.testing.assert_almost_equal(strains.gamma[1], 0.0, decimal=10)
        np.testing.assert_almost_equal(strains.gamma[2], 0.0, decimal=10)

    def test_pure_rotation_90deg(self):
        """90度回転: 断面だけ回転、変位なし → Γ を検証.

        θz = π/2 で z 軸まわり 90 度回転。
        R(θz=π/2) = [[0,-1,0],[1,0,0],[0,0,1]]
        R^T = [[0,1,0],[-1,0,0],[0,0,1]]
        r' = [1,0,0]（変位なし、x軸方向梁）
        R0 = I（x軸方向梁）
        Γ = R^T [1,0,0] - [1,0,0] = [0,-1,0] - [1,0,0] = [-1,-1,0]
        """
        L = 10.0
        coords = _x_beam_coords(L)
        u = np.zeros(12)
        theta_z = np.pi / 2.0
        u[5] = theta_z  # node1 θz
        u[11] = theta_z  # node2 θz
        strains = cosserat_nonlinear_strains(coords, u)
        np.testing.assert_almost_equal(strains.gamma[0], -1.0, decimal=6)
        np.testing.assert_almost_equal(strains.gamma[1], -1.0, decimal=6)
        np.testing.assert_almost_equal(strains.gamma[2], 0.0, decimal=6)

    def test_pure_bending_curvature(self):
        """純曲げ: 一端固定、他端を θz だけ回転 → κ₃ = θz/L."""
        L = 10.0
        coords = _x_beam_coords(L)
        theta_z = 0.1
        u = np.zeros(12)
        u[11] = theta_z  # node2 θz のみ
        strains = cosserat_nonlinear_strains(coords, u)
        np.testing.assert_almost_equal(strains.kappa[2], theta_z / L, decimal=6)


class TestNonlinearInternalForce:
    """非線形内力の基本検証."""

    def test_zero_displacement_gives_zero_force(self):
        """ゼロ変位ではゼロ内力."""
        sec = _make_section()
        L = 10.0
        coords = _x_beam_coords(L)
        u = np.zeros(12)
        f_int = cosserat_internal_force_nonlinear(
            coords,
            u,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            5.0 / 6.0,
            5.0 / 6.0,
        )
        np.testing.assert_array_almost_equal(f_int, np.zeros(12))

    def test_small_displacement_matches_linear(self):
        """小変位では線形版と非線形版が一致."""
        sec = _make_section()
        mat = _make_material()
        L = 10.0
        coords = _x_beam_coords(L)

        # 小さな変位
        rng = np.random.default_rng(42)
        u = rng.standard_normal(12) * 1e-6

        rod_linear = CosseratRod(section=sec, nonlinear=False)
        rod_nonlinear = CosseratRod(section=sec, nonlinear=True)

        f_lin = rod_linear.internal_force(coords, u, mat)
        f_nl = rod_nonlinear.internal_force(coords, u, mat)

        np.testing.assert_array_almost_equal(f_nl, f_lin, decimal=3)


class TestNonlinearTangentStiffness:
    """非線形接線剛性の検証（最重要テスト）."""

    def test_symmetry(self):
        """接線剛性行列は対称."""
        sec = _make_section()
        L = 10.0
        coords = _x_beam_coords(L)

        rng = np.random.default_rng(42)
        u = rng.standard_normal(12) * 0.1

        K_T = cosserat_tangent_stiffness_nonlinear(
            coords,
            u,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            5.0 / 6.0,
            5.0 / 6.0,
        )
        np.testing.assert_array_almost_equal(K_T, K_T.T, decimal=6)

    def test_small_displacement_matches_linear(self):
        """小変位では線形版の K_T と一致."""
        sec = _make_section()
        mat = _make_material()
        L = 10.0
        coords = _x_beam_coords(L)

        u = np.zeros(12)

        rod_linear = CosseratRod(section=sec, nonlinear=False)
        rod_nonlinear = CosseratRod(section=sec, nonlinear=True)

        K_lin = rod_linear.tangent_stiffness(coords, u, mat)
        K_nl = rod_nonlinear.tangent_stiffness(coords, u, mat)

        # ゼロ変位では両方とも材料剛性のみ
        # 非線形版は数値微分なので相対誤差 1e-6 程度
        K_max = np.max(np.abs(K_lin))
        np.testing.assert_allclose(K_nl, K_lin, atol=K_max * 1e-6)

    def test_matches_independent_fd(self):
        """接線剛性が独立した有限差分の対称化版と一致.

        回転ベクトルパラメトリゼーションでは接線剛性は厳密には非対称。
        対称化した有限差分同士で比較する。
        """
        sec = _make_section()
        L = 10.0
        coords = _x_beam_coords(L)

        rng = np.random.default_rng(123)
        u = rng.standard_normal(12) * 0.05

        params = dict(
            E=E,
            G=G,
            A=sec.A,
            Iy=sec.Iy,
            Iz=sec.Iz,
            J=sec.J,
            kappa_y=5.0 / 6.0,
            kappa_z=5.0 / 6.0,
        )

        K_T = cosserat_tangent_stiffness_nonlinear(coords, u, **params)

        # 異なる eps で独立に有限差分 + 対称化
        eps = 1e-6
        K_fd = np.zeros((12, 12))
        for j in range(12):
            u_plus = u.copy()
            u_plus[j] += eps
            u_minus = u.copy()
            u_minus[j] -= eps
            f_plus = cosserat_internal_force_nonlinear(coords, u_plus, **params)
            f_minus = cosserat_internal_force_nonlinear(coords, u_minus, **params)
            K_fd[:, j] = (f_plus - f_minus) / (2.0 * eps)
        K_fd = 0.5 * (K_fd + K_fd.T)  # 対称化

        K_max = max(np.max(np.abs(K_T)), 1.0)
        np.testing.assert_allclose(K_T, K_fd, atol=K_max * 1e-4)

    def test_tangent_zero_displacement(self):
        """ゼロ変位での接線剛性が合理的."""
        sec = _make_section()
        mat = _make_material()
        L = 10.0
        coords = _x_beam_coords(L)
        u = np.zeros(12)

        rod_nl = CosseratRod(section=sec, nonlinear=True)
        K_T = rod_nl.tangent_stiffness(coords, u, mat)

        # 対称
        np.testing.assert_array_almost_equal(K_T, K_T.T, decimal=6)
        # 非ゼロ
        assert np.max(np.abs(K_T)) > 0
        # 線形版と相対誤差 1e-5 以内で一致
        rod_lin = CosseratRod(section=sec, nonlinear=False)
        K_lin = rod_lin.tangent_stiffness(coords, u, mat)
        K_max = np.max(np.abs(K_lin))
        np.testing.assert_allclose(K_T, K_lin, atol=K_max * 1e-5)


class TestCosseratRodNonlinearDispatch:
    """CosseratRod クラスの nonlinear ディスパッチ."""

    def test_default_is_linear(self):
        """デフォルトは nonlinear=False."""
        sec = _make_section()
        rod = CosseratRod(section=sec)
        assert not rod.nonlinear

    def test_nonlinear_flag(self):
        """nonlinear=True が設定される."""
        sec = _make_section()
        rod = CosseratRod(section=sec, nonlinear=True)
        assert rod.nonlinear

    def test_nonlinear_internal_force_dispatches(self):
        """nonlinear=True で非線形内力が使われる."""
        sec = _make_section()
        mat = _make_material()
        L = 10.0
        coords = _x_beam_coords(L)

        # 非ゼロ変位
        u = np.zeros(12)
        u[6] = 0.01  # small axial

        rod_lin = CosseratRod(section=sec, nonlinear=False)
        rod_nl = CosseratRod(section=sec, nonlinear=True)

        f_lin = rod_lin.internal_force(coords, u, mat)
        f_nl = rod_nl.internal_force(coords, u, mat)

        # 小変位なのでほぼ同じだがゼロではない
        assert np.linalg.norm(f_nl) > 1e-3
        # 内力の大きさが近い（小変位なので）
        np.testing.assert_array_almost_equal(f_nl, f_lin, decimal=2)
