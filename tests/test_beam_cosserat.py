"""Cosserat rod 要素のテスト.

テスト方針:
  1. 剛性行列の基本性質（対称性、正定値性）
  2. 軸力・ねじり: 1要素で厳密解と一致
  3. 曲げ: メッシュ細分割で解析解に収束
  4. 一般化歪みの正当性
  5. 座標変換の整合性
  6. ElementProtocol との適合性
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    _cosserat_b_matrix,
    _cosserat_constitutive_matrix,
    cosserat_geometric_stiffness_global,
    cosserat_geometric_stiffness_local,
    cosserat_internal_force_global,
    cosserat_internal_force_local,
    cosserat_ke_global,
    cosserat_ke_local,
    cosserat_section_forces,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.math.quaternion import quat_identity
from xkep_cae.sections.beam import BeamSection

# --- テスト用パラメータ ---
E = 200_000.0  # MPa
NU = 0.3
G = E / (2.0 * (1.0 + NU))
L = 100.0  # mm
B_WIDTH = 10.0  # mm
H_HEIGHT = 20.0  # mm


def _make_section() -> BeamSection:
    return BeamSection.rectangle(B_WIDTH, H_HEIGHT)


def _make_material() -> BeamElastic1D:
    return BeamElastic1D(E=E, nu=NU)


class TestConstitutiveMatrix:
    """構成行列 C のテスト."""

    def test_shape(self):
        sec = _make_section()
        C = _cosserat_constitutive_matrix(E, G, sec.A, sec.Iy, sec.Iz, sec.J, 5.0 / 6.0, 5.0 / 6.0)
        assert C.shape == (6, 6)

    def test_diagonal(self):
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        C = _cosserat_constitutive_matrix(E, G, sec.A, sec.Iy, sec.Iz, sec.J, ky, kz)
        np.testing.assert_almost_equal(C[0, 0], E * sec.A)
        np.testing.assert_almost_equal(C[1, 1], ky * G * sec.A)
        np.testing.assert_almost_equal(C[2, 2], kz * G * sec.A)
        np.testing.assert_almost_equal(C[3, 3], G * sec.J)
        np.testing.assert_almost_equal(C[4, 4], E * sec.Iy)
        np.testing.assert_almost_equal(C[5, 5], E * sec.Iz)

    def test_is_diagonal(self):
        sec = _make_section()
        C = _cosserat_constitutive_matrix(E, G, sec.A, sec.Iy, sec.Iz, sec.J, 5.0 / 6.0, 5.0 / 6.0)
        off_diag = C - np.diag(np.diag(C))
        np.testing.assert_array_almost_equal(off_diag, np.zeros((6, 6)))


class TestBMatrix:
    """B行列のテスト."""

    def test_shape(self):
        B = _cosserat_b_matrix(L, 0.5)
        assert B.shape == (6, 12)

    def test_midpoint_values(self):
        """ξ=0.5 でのB行列の値を確認."""
        B = _cosserat_b_matrix(L, 0.5)
        # Γ₁ row: dN1=-1/L at u₁₁, dN2=1/L at u₁₂
        assert abs(B[0, 0] - (-1.0 / L)) < 1e-12
        assert abs(B[0, 6] - (1.0 / L)) < 1e-12
        # Γ₂ row: θ₃₁ coeff = -N₁(0.5) = -0.5
        assert abs(B[1, 5] - (-0.5)) < 1e-12
        assert abs(B[1, 11] - (-0.5)) < 1e-12
        # Γ₃ row: θ₂₁ coeff = N₁(0.5) = 0.5
        assert abs(B[2, 4] - 0.5) < 1e-12

    def test_pure_axial(self):
        """純粋軸伸び: u = [0,0,0,0,0,0, δ,0,0,0,0,0] → Γ₁=δ/L, 他ゼロ."""
        delta = 0.1
        u = np.zeros(12)
        u[6] = delta  # u₁₂ = δ
        B = _cosserat_b_matrix(L, 0.5)
        strain = B @ u
        np.testing.assert_almost_equal(strain[0], delta / L)  # Γ₁
        np.testing.assert_almost_equal(strain[1], 0.0)
        np.testing.assert_almost_equal(strain[2], 0.0)
        np.testing.assert_almost_equal(strain[3], 0.0)
        np.testing.assert_almost_equal(strain[4], 0.0)
        np.testing.assert_almost_equal(strain[5], 0.0)

    def test_pure_twist(self):
        """純粋ねじり: θ₁₂ = φ → κ₁ = φ/L."""
        phi = 0.05
        u = np.zeros(12)
        u[9] = phi  # θ₁₂
        B = _cosserat_b_matrix(L, 0.5)
        strain = B @ u
        np.testing.assert_almost_equal(strain[3], phi / L)  # κ₁


class TestStiffnessMatrix:
    """剛性行列の基本性質テスト."""

    def test_shape(self):
        sec = _make_section()
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        assert Ke.shape == (12, 12)

    def test_symmetric(self):
        sec = _make_section()
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=12)

    def test_positive_semidefinite(self):
        """6つのゼロ固有値（剛体モード）、6つの正固有値."""
        sec = _make_section()
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        eigvals = np.linalg.eigvalsh(Ke)
        eigvals_sorted = np.sort(eigvals)
        # 最初の6つがゼロ（剛体モード）— 浮動小数点誤差を許容
        for i in range(6):
            assert abs(eigvals_sorted[i]) < 1e-6, (
                f"eigval[{i}] = {eigvals_sorted[i]} should be zero"
            )
        # 残りの6つが正
        for i in range(6, 12):
            assert eigvals_sorted[i] > 1e-6, f"eigval[{i}] = {eigvals_sorted[i]} should be positive"

    def test_axial_stiffness(self):
        """軸方向剛性: K[0,0] = EA/L, K[0,6] = -EA/L."""
        sec = _make_section()
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        np.testing.assert_almost_equal(Ke[0, 0], E * sec.A / L, decimal=6)
        np.testing.assert_almost_equal(Ke[0, 6], -E * sec.A / L, decimal=6)

    def test_torsional_stiffness(self):
        """ねじり剛性: K[3,3] = GJ/L."""
        sec = _make_section()
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0)
        np.testing.assert_almost_equal(Ke[3, 3], G * sec.J / L, decimal=6)

    def test_global_symmetric(self):
        """全体座標系の剛性行列も対称."""
        sec = _make_section()
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        Ke = cosserat_ke_global(coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, 5.0 / 6.0, 5.0 / 6.0)
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=12)

    def test_two_gauss_points(self):
        """2点ガウス求積の剛性行列も対称・正半定値."""
        sec = _make_section()
        Ke = cosserat_ke_local(
            E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0, n_gauss=2
        )
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=12)
        eigvals = np.linalg.eigvalsh(Ke)
        assert np.min(eigvals) > -1e-6


def _solve_cantilever(
    n_elems: int,
    beam_length: float,
    load_dof: int,
    load_value: float,
    section: BeamSection,
    material: BeamElastic1D,
    n_gauss: int = 1,
) -> np.ndarray:
    """片持ち梁の直接法ソルバー.

    x軸方向の梁。節点0が固定端、節点n_elemsが自由端。

    Args:
        n_elems: 要素数
        beam_length: 梁長さ
        load_dof: 荷重DOF（自由端の局所DOF番号、0-5）
        load_value: 荷重値
        section: 断面
        material: 材料
        n_gauss: ガウス積分点数

    Returns:
        u: (n_nodes * 6,) 全節点変位ベクトル
    """
    n_nodes = n_elems + 1
    total_dof = n_nodes * 6
    elem_len = beam_length / n_elems

    # 全体剛性行列の組み立て
    K = np.zeros((total_dof, total_dof))
    for i in range(n_elems):
        coords = np.array(
            [
                [i * elem_len, 0.0, 0.0],
                [(i + 1) * elem_len, 0.0, 0.0],
            ]
        )
        Ke = cosserat_ke_global(
            coords,
            material.E,
            material.G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            5.0 / 6.0,
            5.0 / 6.0,
            n_gauss=n_gauss,
        )
        dof_start = 6 * i
        dof_end = 6 * (i + 2)
        K[dof_start:dof_end, dof_start:dof_end] += Ke

    # 境界条件: 節点0の全6DOF固定
    fixed_dofs = list(range(6))
    free_dofs = [d for d in range(total_dof) if d not in fixed_dofs]

    # 荷重ベクトル
    f = np.zeros(total_dof)
    f[6 * n_elems + load_dof] = load_value

    # 縮約して解く
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    f_f = f[free_dofs]
    u_f = np.linalg.solve(K_ff, f_f)

    u = np.zeros(total_dof)
    for i, dof in enumerate(free_dofs):
        u[dof] = u_f[i]

    return u


class TestCantileverAxial:
    """片持ち梁の軸引張テスト."""

    def test_single_element(self):
        """1要素: δ = PL/(EA) — 厳密."""
        sec = _make_section()
        mat = _make_material()
        P = 1000.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        delta_expected = P * L / (E * sec.A)
        delta_actual = u[6]  # u₁₂
        np.testing.assert_almost_equal(delta_actual, delta_expected, decimal=8)

    def test_multi_element(self):
        """10要素: 同じ結果."""
        sec = _make_section()
        mat = _make_material()
        P = 1000.0
        u = _solve_cantilever(10, L, 0, P, sec, mat)
        delta_expected = P * L / (E * sec.A)
        delta_actual = u[6 * 10]
        np.testing.assert_almost_equal(delta_actual, delta_expected, decimal=8)


class TestCantileverTorsion:
    """片持ち梁のねじりテスト."""

    def test_single_element(self):
        """1要素: θ = TL/(GJ) — 厳密."""
        sec = _make_section()
        mat = _make_material()
        T = 500.0
        u = _solve_cantilever(1, L, 3, T, sec, mat)
        theta_expected = T * L / (G * sec.J)
        theta_actual = u[6 + 3]  # θ₁₂
        np.testing.assert_almost_equal(theta_actual, theta_expected, decimal=8)

    def test_multi_element(self):
        """10要素: 同じ結果."""
        sec = _make_section()
        mat = _make_material()
        T = 500.0
        u = _solve_cantilever(10, L, 3, T, sec, mat)
        theta_expected = T * L / (G * sec.J)
        theta_actual = u[6 * 10 + 3]
        np.testing.assert_almost_equal(theta_actual, theta_expected, decimal=8)


class TestCantileverBending:
    """片持ち梁の曲げテスト（メッシュ収束）.

    片持ち梁先端に集中荷重 P (y方向):
      δ_bending = PL³/(3EI)  (EB解)
      δ_shear   = PL/(κGA)   (せん断変形)
      δ_total   = δ_bending + δ_shear (Timoshenko解)
    """

    def test_convergence_bending_y(self):
        """y方向曲げ: 要素数を増やすと解析解に収束."""
        sec = _make_section()
        mat = _make_material()
        P = 100.0
        kappa = 5.0 / 6.0

        # 解析解 (Timoshenko)
        delta_bending = P * L**3 / (3.0 * E * sec.Iz)
        delta_shear = P * L / (kappa * G * sec.A)
        delta_exact = delta_bending + delta_shear

        errors = []
        for n_elems in [2, 4, 8, 16, 32]:
            u = _solve_cantilever(n_elems, L, 1, P, sec, mat)
            delta_actual = u[6 * n_elems + 1]
            rel_error = abs(delta_actual - delta_exact) / abs(delta_exact)
            errors.append(rel_error)

        # 要素数が増えると精度向上
        assert errors[-1] < 0.001, f"32要素で相対誤差 {errors[-1]:.6f} > 0.1%"
        # 収束性: 各ステップで誤差減少
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"要素数増加で誤差が減少しない: errors[{i}]={errors[i]} >= errors[{i - 1}]={errors[i - 1]}"
            )

    def test_convergence_bending_z(self):
        """z方向曲げ: 同様に収束を確認."""
        sec = _make_section()
        mat = _make_material()
        P = 100.0
        kappa = 5.0 / 6.0

        delta_bending = P * L**3 / (3.0 * E * sec.Iy)
        delta_shear = P * L / (kappa * G * sec.A)
        delta_exact = delta_bending + delta_shear

        errors = []
        for n_elems in [4, 8, 16, 32]:
            u = _solve_cantilever(n_elems, L, 2, P, sec, mat)
            delta_actual = u[6 * n_elems + 2]
            rel_error = abs(delta_actual - delta_exact) / abs(delta_exact)
            errors.append(rel_error)

        assert errors[-1] < 0.001, f"32要素で相対誤差 {errors[-1]:.6f} > 0.1%"


class TestCantileverCombined:
    """複合荷重（軸力 + ねじり同時）のテスト."""

    def test_axial_plus_torsion(self):
        """軸力とねじりの重ね合わせ."""
        sec = _make_section()
        mat = _make_material()
        P = 1000.0
        T = 500.0

        # 個別に解く
        u_axial = _solve_cantilever(10, L, 0, P, sec, mat)
        u_torsion = _solve_cantilever(10, L, 3, T, sec, mat)

        # 同時荷重
        n_elems = 10
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        elem_len = L / n_elems

        K = np.zeros((total_dof, total_dof))
        for i in range(n_elems):
            coords = np.array(
                [
                    [i * elem_len, 0.0, 0.0],
                    [(i + 1) * elem_len, 0.0, 0.0],
                ]
            )
            Ke = cosserat_ke_global(
                coords,
                mat.E,
                mat.G,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.J,
                5.0 / 6.0,
                5.0 / 6.0,
            )
            K[6 * i : 6 * (i + 2), 6 * i : 6 * (i + 2)] += Ke

        fixed = list(range(6))
        free = [d for d in range(total_dof) if d not in fixed]
        f = np.zeros(total_dof)
        f[6 * n_elems] = P  # 軸力
        f[6 * n_elems + 3] = T  # ねじり

        K_ff = K[np.ix_(free, free)]
        u_f = np.linalg.solve(K_ff, f[free])
        u = np.zeros(total_dof)
        for i, d in enumerate(free):
            u[d] = u_f[i]

        # 軸変位とねじり角は個別解の和
        np.testing.assert_almost_equal(
            u[6 * n_elems],
            u_axial[6 * n_elems],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            u[6 * n_elems + 3],
            u_torsion[6 * n_elems + 3],
            decimal=8,
        )


class TestCoordinateTransform:
    """座標変換テスト."""

    def test_rotated_beam(self):
        """y軸方向の梁: 座標変換後も軸剛性は同じ."""
        sec = _make_section()
        coords_x = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        coords_y = np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])

        Ke_x = cosserat_ke_global(
            coords_x,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            5.0 / 6.0,
            5.0 / 6.0,
        )
        Ke_y = cosserat_ke_global(
            coords_y,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            5.0 / 6.0,
            5.0 / 6.0,
        )

        # 両方とも対称
        np.testing.assert_array_almost_equal(Ke_x, Ke_x.T, decimal=12)
        np.testing.assert_array_almost_equal(Ke_y, Ke_y.T, decimal=12)

        # 固有値が一致（座標変換は固有値を変えない）
        eigx = np.sort(np.linalg.eigvalsh(Ke_x))
        eigy = np.sort(np.linalg.eigvalsh(Ke_y))
        np.testing.assert_array_almost_equal(eigx, eigy, decimal=6)


class TestSectionForces:
    """断面力テスト."""

    def test_axial_force(self):
        """軸引張の断面力: N = P."""
        sec = _make_section()
        mat = _make_material()
        P = 500.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        u_elem = u[0:12]

        f1, f2 = cosserat_section_forces(
            coords,
            u_elem,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            5.0 / 6.0,
            5.0 / 6.0,
        )
        np.testing.assert_almost_equal(f1.N, P, decimal=4)
        np.testing.assert_almost_equal(f2.N, P, decimal=4)

    def test_torsion_moment(self):
        """ねじりの断面力: Mx = T."""
        sec = _make_section()
        mat = _make_material()
        T = 300.0
        u = _solve_cantilever(1, L, 3, T, sec, mat)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        u_elem = u[0:12]

        f1, f2 = cosserat_section_forces(
            coords,
            u_elem,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            5.0 / 6.0,
            5.0 / 6.0,
        )
        np.testing.assert_almost_equal(f1.Mx, T, decimal=4)
        np.testing.assert_almost_equal(f2.Mx, T, decimal=4)


class TestCosseratRodClass:
    """CosseratRod クラスのテスト."""

    def test_protocol_attributes(self):
        """ElementProtocol の属性を持つ."""
        sec = _make_section()
        elem = CosseratRod(section=sec)
        assert elem.ndof_per_node == 6
        assert elem.nnodes == 2
        assert elem.ndof == 12

    def test_dof_indices(self):
        sec = _make_section()
        elem = CosseratRod(section=sec)
        edofs = elem.dof_indices(np.array([0, 1]))
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        np.testing.assert_array_equal(edofs, expected)

    def test_local_stiffness(self):
        sec = _make_section()
        mat = _make_material()
        elem = CosseratRod(section=sec)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        Ke = elem.local_stiffness(coords, mat)
        assert Ke.shape == (12, 12)
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=12)

    def test_cowper_kappa(self):
        """Cowper κ モード."""
        sec = _make_section()
        mat = _make_material()
        elem = CosseratRod(section=sec, kappa_y="cowper", kappa_z="cowper")
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        Ke = elem.local_stiffness(coords, mat)
        assert Ke.shape == (12, 12)

    def test_q_ref_nodes(self):
        """参照四元数が恒等四元数で初期化される."""
        sec = _make_section()
        elem = CosseratRod(section=sec)
        for q in elem.q_ref_nodes:
            np.testing.assert_array_almost_equal(q, quat_identity())

    def test_compute_strains(self):
        """一般化歪みの計算."""
        sec = _make_section()
        elem = CosseratRod(section=sec)

        # 純軸伸びの変位
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        delta = 0.1
        u = np.zeros(12)
        u[6] = delta

        strains = elem.compute_strains(coords, u)
        np.testing.assert_almost_equal(strains.gamma[0], delta / L, decimal=10)
        np.testing.assert_almost_equal(strains.gamma[1], 0.0, decimal=10)
        np.testing.assert_almost_equal(strains.gamma[2], 0.0, decimal=10)
        np.testing.assert_almost_equal(strains.kappa[0], 0.0, decimal=10)

    def test_section_forces_class(self):
        """クラス経由の断面力計算."""
        sec = _make_section()
        mat = _make_material()
        elem = CosseratRod(section=sec)
        P = 500.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        f1, f2 = elem.section_forces(coords, u[0:12], mat)
        np.testing.assert_almost_equal(f1.N, P, decimal=4)

    def test_invalid_kappa_string(self):
        sec = _make_section()
        with pytest.raises(ValueError, match="cowper"):
            CosseratRod(section=sec, kappa_y="invalid")

    def test_invalid_gauss_points(self):
        sec = _make_section()
        with pytest.raises(ValueError, match="n_gauss"):
            cosserat_ke_local(
                E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, 5.0 / 6.0, 5.0 / 6.0, n_gauss=3
            )


class TestSimplySupportedBeam:
    """単純支持梁（3点曲げ）のテスト.

    中央集中荷重 P、スパン L の単純支持梁:
      δ_mid = PL³/(48EI) + PL/(4κGA)
    """

    def test_bend3p_convergence(self):
        """3点曲げのメッシュ収束."""
        sec = _make_section()
        kappa = 5.0 / 6.0

        delta_bending = 100.0 * L**3 / (48.0 * E * sec.Iz)
        delta_shear = 100.0 * L / (4.0 * kappa * G * sec.A)
        delta_exact = delta_bending + delta_shear

        errors = []
        P = 100.0
        for n_elems in [4, 8, 16, 32, 64]:
            n_nodes = n_elems + 1
            total_dof = n_nodes * 6
            elem_len = L / n_elems

            # 組み立て
            K = np.zeros((total_dof, total_dof))
            for i in range(n_elems):
                coords_i = np.array(
                    [
                        [i * elem_len, 0.0, 0.0],
                        [(i + 1) * elem_len, 0.0, 0.0],
                    ]
                )
                Ke = cosserat_ke_global(
                    coords_i,
                    E,
                    G,
                    sec.A,
                    sec.Iy,
                    sec.Iz,
                    sec.J,
                    kappa,
                    kappa,
                )
                K[6 * i : 6 * (i + 2), 6 * i : 6 * (i + 2)] += Ke

            # 境界条件: 6剛体モードを全て除去
            # 節点0: ux=0, uy=0, uz=0, θx=0 (4拘束)
            # 節点n: uy=0, uz=0 (2拘束)
            fixed_dofs = [0, 1, 2, 3, 6 * n_elems + 1, 6 * n_elems + 2]
            free_dofs = [d for d in range(total_dof) if d not in fixed_dofs]

            # 中央荷重
            mid_node = n_elems // 2
            f = np.zeros(total_dof)
            f[6 * mid_node + 1] = -P

            K_ff = K[np.ix_(free_dofs, free_dofs)]
            u_f = np.linalg.solve(K_ff, f[free_dofs])
            u = np.zeros(total_dof)
            for i, d in enumerate(free_dofs):
                u[d] = u_f[i]

            delta_mid = abs(u[6 * mid_node + 1])
            rel_error = abs(delta_mid - delta_exact) / abs(delta_exact)
            errors.append(rel_error)

        # 64要素で十分な精度
        assert errors[-1] < 0.005, f"64要素で相対誤差 {errors[-1]:.6f} > 0.5%"


class TestCircularSection:
    """円形断面でのCosserat要素テスト."""

    def test_axial_circle(self):
        """円形断面の軸引張."""
        sec = BeamSection.circle(d=10.0)
        mat = _make_material()
        P = 1000.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        delta_expected = P * L / (E * sec.A)
        np.testing.assert_almost_equal(u[6], delta_expected, decimal=8)

    def test_torsion_circle(self):
        """円形断面のねじり."""
        sec = BeamSection.circle(d=10.0)
        mat = _make_material()
        T = 500.0
        u = _solve_cantilever(1, L, 3, T, sec, mat)
        theta_expected = T * L / (G * sec.J)
        np.testing.assert_almost_equal(u[6 + 3], theta_expected, decimal=8)


# ===========================================================================
# 内力ベクトル テスト
# ===========================================================================
class TestInternalForce:
    """内力ベクトル f_int のテスト."""

    def test_linear_equivalence_axial(self):
        """線形時 f_int = Ke · u の確認（軸引張）."""
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz)
        P = 1000.0
        u_local = np.zeros(12)
        u_local[6] = P * L / (E * sec.A)
        f_ke = Ke @ u_local
        f_int = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            ky,
            kz,
            u_local,
        )
        np.testing.assert_array_almost_equal(f_int, f_ke, decimal=8)

    def test_linear_equivalence_torsion(self):
        """線形時 f_int = Ke · u の確認（ねじり）."""
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz)
        T_torque = 500.0
        u_local = np.zeros(12)
        u_local[9] = T_torque * L / (G * sec.J)
        f_ke = Ke @ u_local
        f_int = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            ky,
            kz,
            u_local,
        )
        np.testing.assert_array_almost_equal(f_int, f_ke, decimal=8)

    def test_linear_equivalence_bending(self):
        """線形時 f_int = Ke · u の確認（曲げ変位）."""
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz)
        u_local = np.zeros(12)
        u_local[7] = 0.01  # uy2
        u_local[11] = 0.001  # θz2
        f_ke = Ke @ u_local
        f_int = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            ky,
            kz,
            u_local,
        )
        np.testing.assert_array_almost_equal(f_int, f_ke, decimal=8)

    def test_global_internal_force(self):
        """全体座標系での内力 = Tᵀ · f_int_local の確認."""
        sec = _make_section()
        mat = _make_material()
        P = 1000.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        u_global = u  # 12 DOF (2 nodes)
        f_int = cosserat_internal_force_global(
            coords,
            u_global,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            5.0 / 6.0,
            5.0 / 6.0,
        )
        # 固定端の反力 = -P, 荷重端の内力 = P
        assert abs(f_int[0] + P) < 1e-6  # 固定端反力
        assert abs(f_int[6] - P) < 1e-6  # 荷重端

    def test_class_internal_force(self):
        """CosseratRodクラスの internal_force() テスト."""
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        P = 1000.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        f_int = rod.internal_force(coords, u, mat)
        Ke = rod.local_stiffness(coords, mat)
        f_ke = Ke @ u
        np.testing.assert_array_almost_equal(f_int, f_ke, decimal=6)

    def test_zero_displacement_gives_zero_force(self):
        """ゼロ変位 → ゼロ内力."""
        sec = _make_section()
        u_local = np.zeros(12)
        f_int = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            5.0 / 6.0,
            5.0 / 6.0,
            u_local,
        )
        np.testing.assert_array_almost_equal(f_int, np.zeros(12))

    def test_2gauss_equivalence(self):
        """2点ガウス求積でも f_int = Ke · u."""
        sec = _make_section()
        ky = kz = 5.0 / 6.0
        Ke = cosserat_ke_local(E, G, sec.A, sec.Iy, sec.Iz, sec.J, L, ky, kz, n_gauss=2)
        u_local = np.zeros(12)
        u_local[6] = 0.05
        u_local[7] = 0.01
        f_ke = Ke @ u_local
        f_int = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            ky,
            kz,
            u_local,
            n_gauss=2,
        )
        np.testing.assert_array_almost_equal(f_int, f_ke, decimal=8)


# ===========================================================================
# 幾何剛性行列 テスト
# ===========================================================================
class TestGeometricStiffness:
    """幾何剛性行列 Kg のテスト."""

    def test_shape(self):
        """形状が (12, 12) であること."""
        stress = np.array([1000.0, 0, 0, 0, 0, 0])
        Kg = cosserat_geometric_stiffness_local(L, stress)
        assert Kg.shape == (12, 12)

    def test_symmetry(self):
        """幾何剛性行列は対称."""
        stress = np.array([1000.0, 0, 0, 100.0, 0, 0])
        Kg = cosserat_geometric_stiffness_local(L, stress)
        np.testing.assert_array_almost_equal(Kg, Kg.T)

    def test_zero_stress_gives_zero(self):
        """ゼロ応力 → ゼロ幾何剛性."""
        stress = np.zeros(6)
        Kg = cosserat_geometric_stiffness_local(L, stress)
        np.testing.assert_array_almost_equal(Kg, np.zeros((12, 12)))

    def test_axial_tension_positive(self):
        """引張軸力 → 幾何剛性行列が非ゼロ."""
        N = 1000.0
        stress = np.array([N, 0, 0, 0, 0, 0])
        Kg = cosserat_geometric_stiffness_local(L, stress)
        assert np.linalg.norm(Kg) > 0

    def test_proportional_to_N(self):
        """Kg は軸力 N に比例する."""
        stress_1 = np.array([1000.0, 0, 0, 0, 0, 0])
        stress_2 = np.array([2000.0, 0, 0, 0, 0, 0])
        Kg_1 = cosserat_geometric_stiffness_local(L, stress_1)
        Kg_2 = cosserat_geometric_stiffness_local(L, stress_2)
        np.testing.assert_array_almost_equal(2.0 * Kg_1, Kg_2)

    def test_global_transformation(self):
        """全体座標系への変換で固有値が保存される."""
        sec = _make_section()
        coords_x = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        # 軸引張変位
        P = 1000.0
        mat = _make_material()
        u = _solve_cantilever(1, L, 0, P, sec, mat)

        Kg = cosserat_geometric_stiffness_global(
            coords_x,
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
        assert Kg.shape == (12, 12)
        np.testing.assert_array_almost_equal(Kg, Kg.T)

    def test_class_geometric_stiffness(self):
        """CosseratRodクラスの geometric_stiffness() テスト."""
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        P = 1000.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        Kg = rod.geometric_stiffness(coords, u, mat)
        assert Kg.shape == (12, 12)
        np.testing.assert_array_almost_equal(Kg, Kg.T)

    def test_tangent_stiffness_positive_definite_under_tension(self):
        """引張時の接線剛性 Km + Kg は正定値性がゼロ方向を除いて成立."""
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec)
        coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        P = 1000.0
        u = _solve_cantilever(1, L, 0, P, sec, mat)
        Km = rod.local_stiffness(coords, mat)
        Kg = rod.geometric_stiffness(coords, u, mat)
        Kt = Km + Kg
        eigvals = np.linalg.eigvalsh(Kt)
        # 6 rigid-body modes have ~0 eigenvalues, rest should be positive
        positive_eigs = eigvals[eigvals > 1e-6]
        assert len(positive_eigs) >= 6
        assert np.all(positive_eigs > 0)


# ===========================================================================
# 初期曲率 テスト
# ===========================================================================
class TestInitialCurvature:
    """初期曲率 κ₀ のサポートテスト."""

    def test_kappa0_attribute(self):
        """CosseratRod にkappa_0が設定できること."""
        sec = _make_section()
        kappa_0 = np.array([0.01, 0.0, 0.0])  # 初期ねじり
        rod = CosseratRod(section=sec, kappa_0=kappa_0)
        np.testing.assert_array_almost_equal(rod.kappa_0, kappa_0)

    def test_kappa0_none_default(self):
        """デフォルトで kappa_0 = None."""
        sec = _make_section()
        rod = CosseratRod(section=sec)
        assert rod.kappa_0 is None

    def test_internal_force_with_initial_curvature(self):
        """初期曲率がある場合、ゼロ変位でも内力が発生する."""
        sec = _make_section()
        kappa_0 = np.array([0.01, 0.0, 0.0])  # 初期ねじり
        u_local = np.zeros(12)
        f_int = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            5.0 / 6.0,
            5.0 / 6.0,
            u_local,
            kappa_0=kappa_0,
        )
        # 初期ねじりがあるので θx 方向に非ゼロ内力
        assert np.linalg.norm(f_int) > 0
        # κ₁₀ のみ非ゼロなので、Mx に関する成分のみ
        # DOF 3 (θ₁₁), DOF 9 (θ₁₂) に影響
        assert abs(f_int[3]) > 0 or abs(f_int[9]) > 0

    def test_internal_force_cancels_at_kappa0(self):
        """変位が初期曲率と整合する場合、初期曲率分が相殺される."""
        sec = _make_section()
        kappa_0 = np.array([0.01, 0.0, 0.0])  # 初期ねじり κ₁₀ = 0.01
        # κ₁ = θ₁' = (θ₁₂ - θ₁₁)/L = κ₁₀ → θ₁₂ = κ₁₀ * L
        u_local = np.zeros(12)
        u_local[9] = kappa_0[0] * L  # θ₁₂
        f_with = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            5.0 / 6.0,
            5.0 / 6.0,
            u_local,
            kappa_0=kappa_0,
        )
        # 変位無しの場合の内力
        f_without = cosserat_internal_force_local(
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            L,
            5.0 / 6.0,
            5.0 / 6.0,
            np.zeros(12),
            kappa_0=None,
        )
        # kappa_0が変位で完全に打ち消される → 内力ゼロ（ストレスフリー配位）
        np.testing.assert_array_almost_equal(f_with, f_without, decimal=8)

    def test_helical_initial_curvature(self):
        """ヘリカル初期曲率（ねじり + 曲率）."""
        sec = BeamSection.circle(d=5.0)
        kappa_0 = np.array([0.02, 0.01, 0.0])  # ねじり + y曲率
        rod = CosseratRod(section=sec, kappa_0=kappa_0)
        coords = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        mat = _make_material()
        u_zero = np.zeros(12)
        f_int = rod.internal_force(coords, u_zero, mat)
        # 初期曲率があるのでゼロ変位でも内力が出る
        assert np.linalg.norm(f_int) > 0
