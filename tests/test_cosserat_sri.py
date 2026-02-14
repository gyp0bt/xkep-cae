"""Cosserat rod SRI（選択的低減積分）のテスト.

テスト方針:
  1. SRI剛性行列の基本性質（対称性、正半定値性）
  2. 軸力・ねじり: 1要素で厳密（SRIでも完全積分と一致）
  3. 曲げ: SRIの方がuniform-1点より少ない要素数で精度良好
  4. 内力の線形等価性（f_int = Ke · u）
  5. CosseratRod クラスの integration_scheme="sri" 対応
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    cosserat_internal_force_local_sri,
    cosserat_ke_global_sri,
    cosserat_ke_local,
    cosserat_ke_local_sri,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection


# --- テスト用パラメータ ---
E = 200_000.0   # MPa
NU = 0.3
G = E / (2.0 * (1.0 + NU))
L = 100.0       # mm
B_WIDTH = 10.0  # mm
H_HEIGHT = 20.0 # mm


def _make_section() -> BeamSection:
    return BeamSection.rectangle(B_WIDTH, H_HEIGHT)


def _make_material() -> BeamElastic1D:
    return BeamElastic1D(E=E, nu=NU)


def _solve_cantilever_ke(n_elems, beam_length, load_dof, load_value, ke_func):
    """汎用片持ち梁ソルバー（ke_func で剛性行列関数を切替可能）."""
    n_nodes = n_elems + 1
    total_dof = n_nodes * 6
    elem_len = beam_length / n_elems

    K = np.zeros((total_dof, total_dof))
    for i in range(n_elems):
        coords = np.array([
            [i * elem_len, 0.0, 0.0],
            [(i + 1) * elem_len, 0.0, 0.0],
        ])
        Ke = ke_func(coords)
        K[6 * i:6 * (i + 2), 6 * i:6 * (i + 2)] += Ke

    fixed_dofs = list(range(6))
    free_dofs = [d for d in range(total_dof) if d not in fixed_dofs]

    f = np.zeros(total_dof)
    f[6 * n_elems + load_dof] = load_value

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    u_f = np.linalg.solve(K_ff, f[free_dofs])
    u = np.zeros(total_dof)
    for i, d in enumerate(free_dofs):
        u[d] = u_f[i]
    return u


class TestSRIStiffnessMatrix:
    """SRI 剛性行列の基本性質テスト."""

    def test_shape(self):
        sec = _make_section()
        Ke = cosserat_ke_local_sri(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        assert Ke.shape == (12, 12)

    def test_symmetric(self):
        sec = _make_section()
        Ke = cosserat_ke_local_sri(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=12)

    def test_positive_semidefinite(self):
        """6つのゼロ固有値（剛体モード）、6つの正固有値."""
        sec = _make_section()
        Ke = cosserat_ke_local_sri(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        eigvals = np.sort(np.linalg.eigvalsh(Ke))
        for i in range(6):
            assert abs(eigvals[i]) < 1e-6, f"eigval[{i}] = {eigvals[i]} should be zero"
        for i in range(6, 12):
            assert eigvals[i] > 1e-6, f"eigval[{i}] = {eigvals[i]} should be positive"

    def test_axial_stiffness_matches_uniform(self):
        """軸方向剛性は uniform と SRI で一致する."""
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        Ke_uni = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz, n_gauss=1)
        Ke_sri = cosserat_ke_local_sri(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz)
        # 軸方向: DOF 0, 6
        np.testing.assert_almost_equal(Ke_sri[0, 0], Ke_uni[0, 0], decimal=8)
        np.testing.assert_almost_equal(Ke_sri[0, 6], Ke_uni[0, 6], decimal=8)

    def test_torsional_stiffness_matches_uniform(self):
        """ねじり剛性は uniform と SRI で一致する."""
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        Ke_uni = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz, n_gauss=1)
        Ke_sri = cosserat_ke_local_sri(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz)
        np.testing.assert_almost_equal(Ke_sri[3, 3], Ke_uni[3, 3], decimal=8)

    def test_global_transformation(self):
        """全体座標系でも対称."""
        sec = _make_section()
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        Ke = cosserat_ke_global_sri(coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, 5.0 / 6.0, 5.0 / 6.0)
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=12)


class TestSRIAxialTorsion:
    """SRI での軸引張・ねじりが厳密であることの確認."""

    def test_axial_single_element(self):
        """軸引張: δ = PL/(EA)."""
        sec = _make_section()
        P = 1000.0
        ke_func = lambda coords: cosserat_ke_global_sri(
            coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, 5.0 / 6.0, 5.0 / 6.0,
        )
        u = _solve_cantilever_ke(1, L, 0, P, ke_func)
        delta_expected = P * L / (E * sec.A)
        np.testing.assert_almost_equal(u[6], delta_expected, decimal=8)

    def test_torsion_single_element(self):
        """ねじり: θ = TL/(GJ)."""
        sec = _make_section()
        T = 500.0
        ke_func = lambda coords: cosserat_ke_global_sri(
            coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, 5.0 / 6.0, 5.0 / 6.0,
        )
        u = _solve_cantilever_ke(1, L, 3, T, ke_func)
        theta_expected = T * L / (G * sec.J)
        np.testing.assert_almost_equal(u[6 + 3], theta_expected, decimal=8)


class TestSRIBendingConvergence:
    """SRI の曲げ収束特性テスト.

    SRI は uniform-1点 より少ない要素数で良好な精度を示す。
    """

    def test_bending_y_convergence(self):
        """y方向曲げ: SRI が uniform-1点 より高精度であることを確認."""
        sec = _make_section()
        P = 100.0
        kappa = 5.0 / 6.0

        delta_bending = P * L**3 / (3.0 * E * sec.Iz)
        delta_shear = P * L / (kappa * G * sec.A)
        delta_exact = delta_bending + delta_shear

        ke_func_sri = lambda coords: cosserat_ke_global_sri(
            coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, kappa, kappa,
        )
        from xkep_cae.elements.beam_cosserat import cosserat_ke_global
        ke_func_uni = lambda coords: cosserat_ke_global(
            coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, kappa, kappa, n_gauss=1,
        )

        n_elems_test = 8
        u_sri = _solve_cantilever_ke(n_elems_test, L, 1, P, ke_func_sri)
        u_uni = _solve_cantilever_ke(n_elems_test, L, 1, P, ke_func_uni)

        err_sri = abs(u_sri[6 * n_elems_test + 1] - delta_exact) / abs(delta_exact)
        err_uni = abs(u_uni[6 * n_elems_test + 1] - delta_exact) / abs(delta_exact)

        # SRI は uniform-1点 より精度が良い（またはほぼ同等）
        assert err_sri <= err_uni + 1e-10, (
            f"SRI error {err_sri:.6f} > uniform error {err_uni:.6f}"
        )

    def test_bending_sri_32elem_accuracy(self):
        """32要素 SRI で十分な精度が出ること."""
        sec = _make_section()
        P = 100.0
        kappa = 5.0 / 6.0

        delta_bending = P * L**3 / (3.0 * E * sec.Iz)
        delta_shear = P * L / (kappa * G * sec.A)
        delta_exact = delta_bending + delta_shear

        ke_func = lambda coords: cosserat_ke_global_sri(
            coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, kappa, kappa,
        )

        errors = []
        for n_elems in [4, 8, 16, 32]:
            u = _solve_cantilever_ke(n_elems, L, 1, P, ke_func)
            delta = u[6 * n_elems + 1]
            errors.append(abs(delta - delta_exact) / abs(delta_exact))

        # 32要素で十分な精度
        assert errors[-1] < 0.001, f"32要素で相対誤差 {errors[-1]:.6f} > 0.1%"
        # 収束性
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], "要素数増加で誤差が減少しない"


class TestSRIInternalForce:
    """SRI版 内力ベクトルのテスト."""

    def test_linear_equivalence(self):
        """f_int_sri = Ke_sri · u の確認."""
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        Ke = cosserat_ke_local_sri(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz)
        u_local = np.zeros(12)
        u_local[6] = 0.05
        u_local[7] = 0.01
        u_local[11] = 0.001
        f_ke = Ke @ u_local
        f_int = cosserat_internal_force_local_sri(
            E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz, u_local,
        )
        np.testing.assert_array_almost_equal(f_int, f_ke, decimal=8)

    def test_zero_displacement_gives_zero(self):
        sec = _make_section()
        u_local = np.zeros(12)
        f_int = cosserat_internal_force_local_sri(
            E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0, u_local,
        )
        np.testing.assert_array_almost_equal(f_int, np.zeros(12))


class TestCosseratRodSRIClass:
    """CosseratRod クラスの integration_scheme="sri" テスト."""

    def test_sri_scheme_creation(self):
        sec = _make_section()
        rod = CosseratRod(section=sec, integration_scheme="sri")
        assert rod.integration_scheme == "sri"

    def test_invalid_scheme_raises(self):
        sec = _make_section()
        with pytest.raises(ValueError, match="integration_scheme"):
            CosseratRod(section=sec, integration_scheme="invalid")

    def test_default_scheme_is_uniform(self):
        sec = _make_section()
        rod = CosseratRod(section=sec)
        assert rod.integration_scheme == "uniform"

    def test_sri_stiffness(self):
        """SRIクラスの剛性行列が対称であること."""
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec, integration_scheme="sri")
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        Ke = rod.local_stiffness(coords, mat)
        assert Ke.shape == (12, 12)
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=12)

    def test_sri_internal_force_consistency(self):
        """SRIクラスの f_int = Ke · u."""
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec, integration_scheme="sri")
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        u = np.zeros(12)
        u[6] = 0.05
        f_int = rod.internal_force(coords, u, mat)
        Ke = rod.local_stiffness(coords, mat)
        f_ke = Ke @ u
        np.testing.assert_array_almost_equal(f_int, f_ke, decimal=6)

    def test_sri_tangent_stiffness(self):
        """tangent_stiffness = Km + Kg が計算できること."""
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec, integration_scheme="sri")
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        u = np.zeros(12)
        u[6] = 0.05
        K_T = rod.tangent_stiffness(coords, u, mat)
        assert K_T.shape == (12, 12)
        np.testing.assert_array_almost_equal(K_T, K_T.T, decimal=10)
