"""幾何学的非線形連続体要素（Q4 TL定式化）のテスト.

テスト方針:
  1. 基本検証: ゼロ変位 → ゼロ内力、GL ひずみがゼロ
  2. 小変形極限: 線形剛性行列・内力との一致
  3. 接線剛性の有限差分検証（最重要）
  4. 接線剛性の対称性
  5. 一様ひずみ（パッチテスト）: 正しい応力の再現
  6. Newton-Raphson 統合テスト: 単純圧縮/引張の収束
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.continuum_nl import (
    b_matrix_nl_2d,
    deformation_gradient_2d,
    green_lagrange_strain_2d,
    make_nl_assembler_q4,
    quad4_nl_force_and_stiffness,
    quad4_nl_internal_force,
    quad4_nl_tangent_stiffness,
)
from xkep_cae.elements.quad4 import quad4_ke_plane_strain
from xkep_cae.materials.elastic import constitutive_plane_strain

# ---------------------------------------------------------------------------
# テスト用パラメータ
# ---------------------------------------------------------------------------
E = 200_000.0  # MPa
NU = 0.3
D = constitutive_plane_strain(E, NU)

# 単位正方形要素
UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)

# 矩形要素 (2×1)
RECT_21 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)


# ===================================================================
# 基本検証
# ===================================================================


class TestBasicNonlinear:
    """ゼロ変位・基本的な整合性の検証."""

    def test_zero_disp_zero_strain(self):
        """ゼロ変位では GL ひずみがゼロ."""
        from xkep_cae.elements.continuum_nl import _jacobian_2d, _q4_shape_deriv

        u = np.zeros(8)
        for xi, eta in [(-0.5, -0.5), (0.0, 0.0), (0.5, 0.5)]:
            dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
            dN_dX, dN_dY, _ = _jacobian_2d(dN_dxi, dN_deta, UNIT_SQUARE)
            F, H = deformation_gradient_2d(dN_dX, dN_dY, u)
            E_v = green_lagrange_strain_2d(F)
            np.testing.assert_allclose(F, np.eye(2), atol=1e-15)
            np.testing.assert_allclose(H, np.zeros((2, 2)), atol=1e-15)
            np.testing.assert_allclose(E_v, np.zeros(3), atol=1e-15)

    def test_zero_disp_zero_internal_force(self):
        """ゼロ変位では内力がゼロ."""
        u = np.zeros(8)
        f_int = quad4_nl_internal_force(UNIT_SQUARE, u, D)
        np.testing.assert_allclose(f_int, np.zeros(8), atol=1e-12)

    def test_zero_disp_stiffness_equals_linear(self):
        """ゼロ変位では接線剛性が線形剛性と一致する."""
        u = np.zeros(8)
        K_nl = quad4_nl_tangent_stiffness(UNIT_SQUARE, u, D)
        K_lin = quad4_ke_plane_strain(UNIT_SQUARE, D)
        np.testing.assert_allclose(K_nl, K_lin, atol=1e-10)

    def test_zero_disp_stiffness_equals_linear_rect(self):
        """矩形要素でも同様にゼロ変位で線形剛性と一致."""
        u = np.zeros(8)
        K_nl = quad4_nl_tangent_stiffness(RECT_21, u, D)
        K_lin = quad4_ke_plane_strain(RECT_21, D)
        np.testing.assert_allclose(K_nl, K_lin, atol=1e-10)

    def test_combined_matches_individual(self):
        """統合関数の結果が個別関数と一致する."""
        u = np.random.default_rng(42).normal(0, 0.01, 8)
        f1 = quad4_nl_internal_force(UNIT_SQUARE, u, D)
        K1 = quad4_nl_tangent_stiffness(UNIT_SQUARE, u, D)
        f2, K2 = quad4_nl_force_and_stiffness(UNIT_SQUARE, u, D)
        np.testing.assert_allclose(f1, f2, atol=1e-14)
        np.testing.assert_allclose(K1, K2, atol=1e-14)


# ===================================================================
# 小変形極限
# ===================================================================


class TestSmallDisplacementLimit:
    """小変形での線形解との比較."""

    def test_small_disp_internal_force_linear(self):
        """小さな変位では f_int ≈ K_lin @ u."""
        rng = np.random.default_rng(123)
        u = rng.normal(0, 1e-8, 8)
        K_lin = quad4_ke_plane_strain(UNIT_SQUARE, D)
        f_int = quad4_nl_internal_force(UNIT_SQUARE, u, D)
        f_lin = K_lin @ u
        np.testing.assert_allclose(f_int, f_lin, rtol=1e-5, atol=1e-15)

    def test_small_disp_tangent_close_to_linear(self):
        """小さな変位では K_T ≈ K_lin."""
        rng = np.random.default_rng(456)
        u = rng.normal(0, 1e-8, 8)
        K_lin = quad4_ke_plane_strain(UNIT_SQUARE, D)
        K_T = quad4_nl_tangent_stiffness(UNIT_SQUARE, u, D)
        np.testing.assert_allclose(K_T, K_lin, rtol=1e-5, atol=1e-10)

    def test_b_matrix_linear_part_matches_standard(self):
        """変位ゼロでの B_L が標準 B 行列と一致する."""
        from xkep_cae.elements.continuum_nl import _jacobian_2d, _q4_shape_deriv

        xi, eta = 0.0, 0.0
        dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
        dN_dX, dN_dY, _ = _jacobian_2d(dN_dxi, dN_deta, UNIT_SQUARE)

        H = np.zeros((2, 2))
        B_L = b_matrix_nl_2d(dN_dX, dN_dY, H)

        # 手動で構築した標準 B 行列
        B_std = np.zeros((3, 8))
        for ni in range(4):
            B_std[0, 2 * ni] = dN_dX[ni]
            B_std[1, 2 * ni + 1] = dN_dY[ni]
            B_std[2, 2 * ni] = dN_dY[ni]
            B_std[2, 2 * ni + 1] = dN_dX[ni]

        np.testing.assert_allclose(B_L, B_std, atol=1e-15)


# ===================================================================
# 接線剛性の有限差分検証
# ===================================================================


class TestTangentFiniteDifference:
    """接線剛性行列の有限差分による検証（最重要）."""

    @pytest.fixture(params=["unit_square", "rect_21"])
    def geom(self, request):
        if request.param == "unit_square":
            return UNIT_SQUARE
        return RECT_21

    @pytest.fixture(params=["zero", "small", "moderate"])
    def disp_level(self, request, geom):
        rng = np.random.default_rng(789)
        if request.param == "zero":
            return np.zeros(8)
        elif request.param == "small":
            return rng.normal(0, 1e-4, 8)
        else:
            return rng.normal(0, 0.05, 8)

    def test_tangent_fd(self, geom, disp_level):
        """K_T の各列を f_int の有限差分で検証する.

        K_T[:, j] ≈ (f_int(u + δe_j) - f_int(u - δe_j)) / (2δ)
        """
        u = disp_level
        eps = 1e-7

        K_T = quad4_nl_tangent_stiffness(geom, u, D)
        K_fd = np.zeros_like(K_T)

        for j in range(8):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[j] += eps
            u_minus[j] -= eps
            f_plus = quad4_nl_internal_force(geom, u_plus, D)
            f_minus = quad4_nl_internal_force(geom, u_minus, D)
            K_fd[:, j] = (f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(K_T, K_fd, rtol=1e-4, atol=1e-6)


# ===================================================================
# 対称性
# ===================================================================


class TestSymmetry:
    """接線剛性行列の対称性検証."""

    def test_tangent_symmetric_zero_disp(self):
        K_T = quad4_nl_tangent_stiffness(UNIT_SQUARE, np.zeros(8), D)
        np.testing.assert_allclose(K_T, K_T.T, atol=1e-12)

    def test_tangent_symmetric_nonzero_disp(self):
        rng = np.random.default_rng(101)
        u = rng.normal(0, 0.02, 8)
        K_T = quad4_nl_tangent_stiffness(UNIT_SQUARE, u, D)
        np.testing.assert_allclose(K_T, K_T.T, atol=1e-10)

    def test_tangent_symmetric_rect(self):
        rng = np.random.default_rng(202)
        u = rng.normal(0, 0.02, 8)
        K_T = quad4_nl_tangent_stiffness(RECT_21, u, D)
        np.testing.assert_allclose(K_T, K_T.T, atol=1e-10)


# ===================================================================
# パッチテスト（一様ひずみ）
# ===================================================================


class TestPatchTest:
    """一様ひずみ状態の再現を検証."""

    def test_uniform_tension_x(self):
        """x 方向一様引張: ε_xx = 0.001.

        変位場: u_x = 0.001 * x, u_y = 0
        一様ひずみ場では全 Gauss 点で同一の GL ひずみ・応力になる。
        """
        from xkep_cae.elements.continuum_nl import (
            _GAUSS_POINTS_2x2,
            _jacobian_2d,
            _q4_shape_deriv,
        )

        eps_xx = 0.001
        u = np.zeros(8)
        for ni in range(4):
            u[2 * ni] = eps_xx * UNIT_SQUARE[ni, 0]

        # 全 Gauss 点での GL ひずみが一様であることを検証
        E_expected = np.array([eps_xx + 0.5 * eps_xx**2, 0.0, 0.0])
        for xi, eta in _GAUSS_POINTS_2x2:
            dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
            dN_dX, dN_dY, _ = _jacobian_2d(dN_dxi, dN_deta, UNIT_SQUARE)
            F, _ = deformation_gradient_2d(dN_dX, dN_dY, u)
            E_v = green_lagrange_strain_2d(F)
            np.testing.assert_allclose(E_v, E_expected, atol=1e-14)

        # S2PK 応力の検証
        S_expected = D @ E_expected
        # ひずみエネルギー U = ∫ 0.5*E:D:E dV₀ = 0.5*S:E*V
        # (一様場なので解析積分可能)
        V = 1.0
        U_exact = 0.5 * S_expected @ E_expected * V

        # 数値積分のエネルギーと比較
        U_num = 0.0
        for xi, eta in _GAUSS_POINTS_2x2:
            dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
            dN_dX, dN_dY, detJ = _jacobian_2d(dN_dxi, dN_deta, UNIT_SQUARE)
            F, _ = deformation_gradient_2d(dN_dX, dN_dY, u)
            E_v = green_lagrange_strain_2d(F)
            S_v = D @ E_v
            U_num += 0.5 * E_v @ S_v * detJ * 1.0
        np.testing.assert_allclose(U_num, U_exact, rtol=1e-12)

    def test_uniform_shear(self):
        """一様せん断: γ_xy = 0.001.

        変位場: u_x = 0.0005 * y, u_y = 0.0005 * x
        """
        gamma = 0.001
        u = np.zeros(8)
        for ni in range(4):
            u[2 * ni] = 0.5 * gamma * UNIT_SQUARE[ni, 1]  # u_x = γ/2 * y
            u[2 * ni + 1] = 0.5 * gamma * UNIT_SQUARE[ni, 0]  # u_y = γ/2 * x

        f_int = quad4_nl_internal_force(UNIT_SQUARE, u, D)
        energy_elem = 0.5 * u @ f_int

        # GL shear strain 2E_12 ≈ γ for small strain
        E_expected = np.array([0.5 * (0.5 * gamma) ** 2, 0.5 * (0.5 * gamma) ** 2, gamma])
        # Add higher-order terms from GL strain
        S_expected = D @ E_expected
        energy_exact = 0.5 * S_expected @ E_expected * 1.0
        np.testing.assert_allclose(energy_elem, energy_exact, rtol=1e-5)


# ===================================================================
# GL ひずみの検証
# ===================================================================


class TestGreenLagrangeStrain:
    """GL ひずみの基本特性を検証."""

    def test_pure_stretch_x(self):
        """x 方向純伸長: F = [[λ,0],[0,1]], E₁₁ = 0.5*(λ²-1)."""
        lam = 1.1
        F = np.array([[lam, 0.0], [0.0, 1.0]])
        E_v = green_lagrange_strain_2d(F)
        np.testing.assert_allclose(E_v, [0.5 * (lam**2 - 1), 0.0, 0.0], atol=1e-14)

    def test_pure_rotation_zero_strain(self):
        """純回転では GL ひずみがゼロ（剛体回転不変）."""
        theta = np.pi / 6
        c, s = np.cos(theta), np.sin(theta)
        F = np.array([[c, -s], [s, c]])
        E_v = green_lagrange_strain_2d(F)
        np.testing.assert_allclose(E_v, np.zeros(3), atol=1e-14)

    def test_biaxial_stretch(self):
        """二軸伸長: F = diag(λ₁, λ₂)."""
        lam1, lam2 = 1.2, 0.9
        F = np.diag([lam1, lam2])
        E_v = green_lagrange_strain_2d(F)
        np.testing.assert_allclose(
            E_v,
            [0.5 * (lam1**2 - 1), 0.5 * (lam2**2 - 1), 0.0],
            atol=1e-14,
        )

    def test_simple_shear(self):
        """単純せん断: F = [[1,γ],[0,1]].

        C = F^T F = [[1, γ],[γ, 1+γ²]]
        E = 0.5*(C-I) → E₁₁=0, E₂₂=γ²/2, 2E₁₂=γ
        """
        gamma = 0.3
        F = np.array([[1.0, gamma], [0.0, 1.0]])
        E_v = green_lagrange_strain_2d(F)
        np.testing.assert_allclose(
            E_v,
            [0.0, 0.5 * gamma**2, gamma],
            atol=1e-14,
        )


# ===================================================================
# Newton-Raphson 統合テスト
# ===================================================================


class TestNewtonRaphsonIntegration:
    """NR ソルバーとの統合テスト."""

    def _make_single_elem_problem(self, n_x=2, n_y=2, Lx=2.0, Ly=1.0):
        """n_x × n_y 要素のメッシュを生成する."""
        nodes = []
        for j in range(n_y + 1):
            for i in range(n_x + 1):
                nodes.append([Lx * i / n_x, Ly * j / n_y])
        nodes = np.array(nodes, dtype=float)

        conn = []
        for j in range(n_y):
            for i in range(n_x):
                n0 = j * (n_x + 1) + i
                n1 = n0 + 1
                n2 = n1 + (n_x + 1)
                n3 = n0 + (n_x + 1)
                conn.append([n0, n1, n2, n3])
        conn = np.array(conn, dtype=int)

        return nodes, conn

    def test_uniaxial_tension_small(self):
        """小さな一様引張でNRが1ステップ1反復で収束する.

        左端固定、右端に引張荷重。
        """
        from xkep_cae.solver import newton_raphson

        nodes, conn = self._make_single_elem_problem(n_x=2, n_y=1)
        n_nodes = len(nodes)
        ndof = 2 * n_nodes

        f_int_fn, K_T_fn = make_nl_assembler_q4(nodes, conn, D, t=1.0)

        # 左端固定（x=0 の節点）
        fixed_dofs = []
        for i in range(n_nodes):
            if abs(nodes[i, 0]) < 1e-12:
                fixed_dofs.extend([2 * i, 2 * i + 1])
        fixed_dofs = np.array(fixed_dofs, dtype=int)

        # 右端に小さな引張荷重
        f_ext = np.zeros(ndof)
        for i in range(n_nodes):
            if abs(nodes[i, 0] - 2.0) < 1e-12:
                f_ext[2 * i] = 10.0  # small tensile load in x

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            K_T_fn,
            f_int_fn,
            n_load_steps=1,
            max_iter=20,
            tol_force=1e-10,
            show_progress=False,
        )
        assert result.converged
        # 全ての x 変位が正
        for i in range(n_nodes):
            if abs(nodes[i, 0]) > 1e-12:
                assert result.u[2 * i] > 0.0

    def test_compression_multi_step(self):
        """圧縮荷重の多ステップNR解析の収束を検証.

        4×2 メッシュ、左端固定、右端に圧縮荷重。
        """
        from xkep_cae.solver import newton_raphson

        nodes, conn = self._make_single_elem_problem(n_x=4, n_y=2)
        n_nodes = len(nodes)
        ndof = 2 * n_nodes

        f_int_fn, K_T_fn = make_nl_assembler_q4(nodes, conn, D, t=1.0)

        # 左端固定
        fixed_dofs = []
        for i in range(n_nodes):
            if abs(nodes[i, 0]) < 1e-12:
                fixed_dofs.extend([2 * i, 2 * i + 1])
        fixed_dofs = np.array(fixed_dofs, dtype=int)

        # 右端に圧縮荷重（中程度）
        f_ext = np.zeros(ndof)
        for i in range(n_nodes):
            if abs(nodes[i, 0] - 2.0) < 1e-12:
                f_ext[2 * i] = -100.0

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            K_T_fn,
            f_int_fn,
            n_load_steps=5,
            max_iter=30,
            tol_force=1e-8,
            show_progress=False,
        )
        assert result.converged

    def test_nonlinear_vs_linear_small_load(self):
        """小荷重での非線形解が線形解に近いことを検証."""
        from xkep_cae.assembly import assemble_global_stiffness
        from xkep_cae.elements.quad4 import Quad4PlaneStrain
        from xkep_cae.materials.elastic import PlaneStrainElastic
        from xkep_cae.solver import newton_raphson

        nodes, conn = self._make_single_elem_problem(n_x=3, n_y=2)
        n_nodes = len(nodes)
        ndof = 2 * n_nodes

        # 非線形アセンブラ
        f_int_fn, K_T_fn = make_nl_assembler_q4(nodes, conn, D, t=1.0)

        # 線形アセンブリ
        elem = Quad4PlaneStrain()
        mat = PlaneStrainElastic(E=E, nu=NU)
        K_lin = assemble_global_stiffness(
            nodes, [(elem, conn)], mat, thickness=1.0, show_progress=False
        )

        # 左端固定
        fixed_dofs = []
        for i in range(n_nodes):
            if abs(nodes[i, 0]) < 1e-12:
                fixed_dofs.extend([2 * i, 2 * i + 1])
        fixed_dofs = np.array(fixed_dofs, dtype=int)

        # 非常に小さな荷重
        f_ext = np.zeros(ndof)
        for i in range(n_nodes):
            if abs(nodes[i, 0] - 2.0) < 1e-12:
                f_ext[2 * i] = 1.0  # very small

        # 非線形解
        result_nl = newton_raphson(
            f_ext,
            fixed_dofs,
            K_T_fn,
            f_int_fn,
            n_load_steps=1,
            max_iter=10,
            tol_force=1e-12,
            show_progress=False,
        )
        assert result_nl.converged

        # 線形解（直接法）
        import scipy.sparse.linalg as spla

        K_bc = K_lin.tolil()
        f_bc = f_ext.copy()
        for dof in fixed_dofs:
            K_bc[dof, :] = 0.0
            K_bc[:, dof] = 0.0
            K_bc[dof, dof] = 1.0
            f_bc[dof] = 0.0
        u_lin = spla.spsolve(K_bc.tocsr(), f_bc)

        np.testing.assert_allclose(result_nl.u, u_lin, rtol=1e-4, atol=1e-12)


# ===================================================================
# エネルギー整合性
# ===================================================================


class TestEnergyConsistency:
    """内力とひずみエネルギーの整合性を検証."""

    def test_energy_gradient(self):
        """f_int ≈ dU/du を有限差分で検証する.

        ひずみエネルギー U = ∫ 0.5 * E:D:E dV₀
        f_int = ∂U/∂u
        """
        rng = np.random.default_rng(303)
        u = rng.normal(0, 0.01, 8)

        def strain_energy(u_e):
            from xkep_cae.elements.continuum_nl import (
                _GAUSS_POINTS_2x2,
                _jacobian_2d,
                _q4_shape_deriv,
            )

            W = 0.0
            for xi, eta in _GAUSS_POINTS_2x2:
                dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
                dN_dX, dN_dY, detJ = _jacobian_2d(dN_dxi, dN_deta, UNIT_SQUARE)
                F, _ = deformation_gradient_2d(dN_dX, dN_dY, u_e)
                E_v = green_lagrange_strain_2d(F)
                S_v = D @ E_v
                W += 0.5 * E_v @ S_v * detJ * 1.0
            return W

        f_int = quad4_nl_internal_force(UNIT_SQUARE, u, D)

        # 有限差分
        eps = 1e-7
        f_fd = np.zeros(8)
        for j in range(8):
            u_p = u.copy()
            u_m = u.copy()
            u_p[j] += eps
            u_m[j] -= eps
            f_fd[j] = (strain_energy(u_p) - strain_energy(u_m)) / (2 * eps)

        np.testing.assert_allclose(f_int, f_fd, rtol=1e-5, atol=1e-10)


# ===================================================================
# Updated Lagrangian (UL) 定式化テスト
# ===================================================================


class TestUpdatedLagrangian:
    """Updated Lagrangian 定式化の検証."""

    def _make_mesh(self, n_x=2, n_y=2, Lx=2.0, Ly=1.0):
        """n_x × n_y 要素のメッシュを生成する."""
        nodes = []
        for j in range(n_y + 1):
            for i in range(n_x + 1):
                nodes.append([Lx * i / n_x, Ly * j / n_y])
        nodes = np.array(nodes, dtype=float)

        conn = []
        for j in range(n_y):
            for i in range(n_x):
                n0 = j * (n_x + 1) + i
                n1 = n0 + 1
                n2 = n1 + (n_x + 1)
                n3 = n0 + (n_x + 1)
                conn.append([n0, n1, n2, n3])
        conn = np.array(conn, dtype=int)
        return nodes, conn

    def _get_fixed_and_load(self, nodes, load_x=10.0, Lx=2.0):
        """左端固定、右端荷重の境界条件."""
        n_nodes = len(nodes)
        ndof = 2 * n_nodes
        fixed = []
        for i in range(n_nodes):
            if abs(nodes[i, 0]) < 1e-12:
                fixed.extend([2 * i, 2 * i + 1])
        fixed = np.array(fixed, dtype=int)

        f_ext = np.zeros(ndof)
        for i in range(n_nodes):
            if abs(nodes[i, 0] - Lx) < 1e-12:
                f_ext[2 * i] = load_x
        return fixed, f_ext

    def test_ul_assembler_zero_disp(self):
        """ULアセンブラ: ゼロ変位でゼロ内力."""
        from xkep_cae.elements.continuum_nl import ULAssemblerQ4

        nodes, conn = self._make_mesh()
        ul = ULAssemblerQ4(nodes, conn, D)

        f_int = ul.assemble_internal_force(np.zeros(ul.ndof))
        np.testing.assert_allclose(f_int, 0.0, atol=1e-14)

    def test_ul_assembler_matches_tl_single_step(self):
        """ULアセンブラ: 単一ステップでTLと同じ結果."""
        from xkep_cae.elements.continuum_nl import ULAssemblerQ4

        nodes, conn = self._make_mesh(n_x=2, n_y=1)
        ndof = 2 * len(nodes)

        rng = np.random.default_rng(42)
        u = rng.normal(0, 0.01, ndof)

        # TL
        f_int_tl, K_T_tl = make_nl_assembler_q4(nodes, conn, D)
        f_tl = f_int_tl(u)
        K_tl = K_T_tl(u).toarray()

        # UL（参照配置未更新なので同じ）
        ul = ULAssemblerQ4(nodes, conn, D)
        f_ul = ul.assemble_internal_force(u)
        K_ul = ul.assemble_tangent(u).toarray()

        np.testing.assert_allclose(f_ul, f_tl, atol=1e-12)
        np.testing.assert_allclose(K_ul, K_tl, atol=1e-12)

    def test_ul_reference_update(self):
        """参照配置更新後、ゼロ変位でも蓄積応力による内力が非ゼロ."""
        from xkep_cae.elements.continuum_nl import ULAssemblerQ4

        nodes, conn = self._make_mesh(n_x=2, n_y=1)
        ndof = 2 * len(nodes)

        ul = ULAssemblerQ4(nodes, conn, D)

        # 増分変位を適用して更新
        u_inc = np.zeros(ndof)
        for i in range(len(nodes)):
            u_inc[2 * i] = 0.01 * nodes[i, 0] / 2.0  # 線形分布

        ul.update_reference(u_inc)

        # 累積変位が更新されている
        np.testing.assert_allclose(ul.u_total, u_inc)

        # 更新後のゼロ変位でも蓄積応力があるため内力は非ゼロ
        f_int = ul.assemble_internal_force(np.zeros(ndof))
        assert np.linalg.norm(f_int) > 0, "蓄積応力による内力が存在するはず"

    def test_ul_reset(self):
        """reset() で初期状態に戻ることを確認."""
        from xkep_cae.elements.continuum_nl import ULAssemblerQ4

        nodes, conn = self._make_mesh()
        ul = ULAssemblerQ4(nodes, conn, D)

        # 変位を適用して更新
        u_inc = np.ones(ul.ndof) * 0.001
        ul.update_reference(u_inc)
        assert np.linalg.norm(ul.u_total) > 0

        # リセット
        ul.reset()
        np.testing.assert_allclose(ul.u_total, 0.0)
        np.testing.assert_allclose(ul.nodes_ref, nodes)

    def test_ul_nr_small_tension(self):
        """UL NRソルバー: 小荷重の引張で収束."""
        from xkep_cae.elements.continuum_nl import (
            ULAssemblerQ4,
            newton_raphson_ul,
        )

        nodes, conn = self._make_mesh(n_x=2, n_y=1)
        fixed, f_ext = self._get_fixed_and_load(nodes, load_x=10.0)

        ul = ULAssemblerQ4(nodes, conn, D)
        result = newton_raphson_ul(
            f_ext,
            fixed,
            ul,
            n_load_steps=1,
            max_iter=20,
            tol_force=1e-10,
            show_progress=False,
        )

        assert result.converged
        # 右端のx変位は正
        for i in range(len(nodes)):
            if abs(nodes[i, 0] - 2.0) < 1e-12:
                assert result.u[2 * i] > 0.0

    def test_ul_nr_matches_tl_small_load(self):
        """UL と TL が小荷重で同じ結果を返す."""
        from xkep_cae.elements.continuum_nl import (
            ULAssemblerQ4,
            newton_raphson_ul,
        )
        from xkep_cae.solver import newton_raphson

        nodes, conn = self._make_mesh(n_x=3, n_y=2)
        fixed, f_ext = self._get_fixed_and_load(nodes, load_x=1.0)

        # TL
        f_int_fn, K_T_fn = make_nl_assembler_q4(nodes, conn, D)
        result_tl = newton_raphson(
            f_ext,
            fixed,
            K_T_fn,
            f_int_fn,
            n_load_steps=1,
            max_iter=10,
            tol_force=1e-12,
            show_progress=False,
        )
        assert result_tl.converged

        # UL
        ul = ULAssemblerQ4(nodes, conn, D)
        result_ul = newton_raphson_ul(
            f_ext,
            fixed,
            ul,
            n_load_steps=1,
            max_iter=10,
            tol_force=1e-12,
            show_progress=False,
        )
        assert result_ul.converged

        np.testing.assert_allclose(result_ul.u, result_tl.u, rtol=1e-6, atol=1e-12)

    def test_ul_nr_multi_step_convergence(self):
        """UL NR多ステップ: 中程度の荷重で収束."""
        from xkep_cae.elements.continuum_nl import (
            ULAssemblerQ4,
            newton_raphson_ul,
        )

        nodes, conn = self._make_mesh(n_x=4, n_y=2)
        fixed, f_ext = self._get_fixed_and_load(nodes, load_x=-100.0)

        ul = ULAssemblerQ4(nodes, conn, D)
        result = newton_raphson_ul(
            f_ext,
            fixed,
            ul,
            n_load_steps=5,
            max_iter=30,
            tol_force=1e-8,
            show_progress=False,
        )

        assert result.converged
        assert len(result.load_history) == 5
        assert len(result.displacement_history) == 5

    def test_ul_nr_matches_tl_moderate_load(self):
        """UL と TL が中程度荷重で近い結果を返す.

        大変形では定式化の違いで微小な差が出るが、
        十分なステップ数であればほぼ一致する。
        """
        from xkep_cae.elements.continuum_nl import (
            ULAssemblerQ4,
            newton_raphson_ul,
        )
        from xkep_cae.solver import newton_raphson

        nodes, conn = self._make_mesh(n_x=4, n_y=2)
        fixed, f_ext = self._get_fixed_and_load(nodes, load_x=50.0)
        n_steps = 10

        # TL
        f_int_fn, K_T_fn = make_nl_assembler_q4(nodes, conn, D)
        result_tl = newton_raphson(
            f_ext,
            fixed,
            K_T_fn,
            f_int_fn,
            n_load_steps=n_steps,
            max_iter=30,
            tol_force=1e-10,
            show_progress=False,
        )
        assert result_tl.converged

        # UL
        ul = ULAssemblerQ4(nodes, conn, D)
        result_ul = newton_raphson_ul(
            f_ext,
            fixed,
            ul,
            n_load_steps=n_steps,
            max_iter=30,
            tol_force=1e-10,
            show_progress=False,
        )
        assert result_ul.converged

        # TLとULの結果は近い（完全に一致はしない）
        np.testing.assert_allclose(result_ul.u, result_tl.u, rtol=0.05, atol=1e-8)

    def test_ul_load_history_monotonic(self):
        """荷重履歴が単調増加."""
        from xkep_cae.elements.continuum_nl import (
            ULAssemblerQ4,
            newton_raphson_ul,
        )

        nodes, conn = self._make_mesh(n_x=2, n_y=1)
        fixed, f_ext = self._get_fixed_and_load(nodes, load_x=20.0)

        ul = ULAssemblerQ4(nodes, conn, D)
        result = newton_raphson_ul(
            f_ext,
            fixed,
            ul,
            n_load_steps=5,
            show_progress=False,
        )

        assert result.converged
        for i in range(1, len(result.load_history)):
            assert result.load_history[i] > result.load_history[i - 1]

    def test_ul_cumulative_displacement(self):
        """累積変位が各ステップの増分の合計と一致."""
        from xkep_cae.elements.continuum_nl import (
            ULAssemblerQ4,
            newton_raphson_ul,
        )

        nodes, conn = self._make_mesh(n_x=2, n_y=1)
        fixed, f_ext = self._get_fixed_and_load(nodes, load_x=10.0)

        ul = ULAssemblerQ4(nodes, conn, D)
        result = newton_raphson_ul(
            f_ext,
            fixed,
            ul,
            n_load_steps=3,
            show_progress=False,
        )

        assert result.converged
        # displacement_history の最後が u と一致
        np.testing.assert_allclose(result.displacement_history[-1], result.u, atol=1e-14)
