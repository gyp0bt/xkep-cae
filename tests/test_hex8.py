"""HEX8 B-bar 要素のテスト.

テスト構成:
  1. 形状関数の基本性質
  2. 剛性行列の基本性質（対称性、ランク、剛体モード）
  3. パッチテスト（一様ひずみ再現性）
  4. 一軸圧縮テスト（解析解との比較）
  5. 3D弾性テンソルの検証
  6. ElementProtocol 適合性
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.hex8 import (
    Hex8BBar,
    _hex8_dNdxi,
    _hex8_shape,
    hex8_ke_bbar,
)
from xkep_cae.materials.elastic import IsotropicElastic3D, constitutive_3d

# ============================================================
# テスト用パラメータ
# ============================================================
E_STEEL = 200e9  # [Pa]
NU_STEEL = 0.3


def _unit_cube() -> np.ndarray:
    """単位立方体の節点座標 (8, 3)."""
    return np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )


def _brick(Lx: float, Ly: float, Lz: float) -> np.ndarray:
    """任意サイズの直方体要素座標."""
    return np.array(
        [
            [0, 0, 0],
            [Lx, 0, 0],
            [Lx, Ly, 0],
            [0, Ly, 0],
            [0, 0, Lz],
            [Lx, 0, Lz],
            [Lx, Ly, Lz],
            [0, Ly, Lz],
        ],
        dtype=float,
    )


# ============================================================
# 1. 形状関数テスト
# ============================================================
class TestShapeFunctions:
    """HEX8 形状関数の基本性質."""

    def test_partition_of_unity(self):
        """形状関数の和 = 1（任意点で）."""
        for _ in range(10):
            xi, eta, zeta = np.random.uniform(-1, 1, 3)
            N = _hex8_shape(xi, eta, zeta)
            assert N.sum() == pytest.approx(1.0, abs=1e-14)

    def test_nodal_values(self):
        """節点位置で N_i = δ_ij."""
        corners = [
            (-1, -1, -1),
            (+1, -1, -1),
            (+1, +1, -1),
            (-1, +1, -1),
            (-1, -1, +1),
            (+1, -1, +1),
            (+1, +1, +1),
            (-1, +1, +1),
        ]
        for i, (xi, eta, zeta) in enumerate(corners):
            N = _hex8_shape(xi, eta, zeta)
            for j in range(8):
                expected = 1.0 if i == j else 0.0
                assert N[j] == pytest.approx(expected, abs=1e-14)

    def test_dN_shape(self):
        """dN/dξ は (3, 8) 行列."""
        dN = _hex8_dNdxi(0.0, 0.0, 0.0)
        assert dN.shape == (3, 8)

    def test_dN_sum_zero(self):
        """∂N_i/∂ξ の和 = 0（分配性の微分）."""
        for _ in range(10):
            xi, eta, zeta = np.random.uniform(-1, 1, 3)
            dN = _hex8_dNdxi(xi, eta, zeta)
            np.testing.assert_allclose(dN.sum(axis=1), 0.0, atol=1e-14)


# ============================================================
# 2. 剛性行列の基本性質
# ============================================================
class TestStiffnessBasic:
    """剛性行列の基本性質テスト."""

    def test_symmetry(self):
        """対称性: Ke = Ke^T."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar(_unit_cube(), D)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-6)

    def test_rank(self):
        """ランク = 18（24 DOF - 6 剛体モード）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 18

    def test_rigid_body_modes(self):
        """6 つの剛体モード（3 並進 + 3 回転）がゼロ固有値."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar(_unit_cube(), D)
        eigvals = np.sort(np.linalg.eigvalsh(Ke))
        # 最小6つの固有値がほぼゼロ
        threshold = 1e-6 * eigvals[-1]
        assert np.all(np.abs(eigvals[:6]) < threshold)

    def test_positive_semidefinite(self):
        """正半定値（全固有値 >= -ε）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        threshold = -1e-6 * np.max(np.abs(eigvals))
        assert np.all(eigvals >= threshold)

    def test_nonzero_stiffness(self):
        """非ゼロ剛性（最大固有値 > 0）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar(_unit_cube(), D)
        assert np.max(np.linalg.eigvalsh(Ke)) > 0

    def test_scaled_geometry(self):
        """スケーリング: 2倍拡大で剛性行列が変化するが有限."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke_scaled = hex8_ke_bbar(2.0 * _unit_cube(), D)
        assert np.linalg.norm(Ke_scaled) > 0

    def test_distorted_element(self):
        """歪んだ要素でも正常に動作."""
        coords = _unit_cube().copy()
        coords[6, :] += [0.1, 0.1, 0.1]  # 1つの節点を少し動かす
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar(coords, D)
        assert Ke.shape == (24, 24)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-4)


# ============================================================
# 3. パッチテスト（一様ひずみ再現性）
# ============================================================
class TestPatchTest:
    """パッチテスト — 一様ひずみを正確に再現."""

    @staticmethod
    def _assemble_multi_elem(nodes: np.ndarray, elements: np.ndarray, D: np.ndarray) -> np.ndarray:
        """複数 HEX8 要素のアセンブリ."""
        n_nodes = len(nodes)
        ndof = 3 * n_nodes
        K = np.zeros((ndof, ndof))
        for elem in elements:
            coords = nodes[elem]
            ke = hex8_ke_bbar(coords, D)
            dofs = []
            for nid in elem:
                dofs.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])
            dofs = np.array(dofs)
            for ii in range(24):
                for jj in range(24):
                    K[dofs[ii], dofs[jj]] += ke[ii, jj]
        return K

    def test_uniaxial_strain_patch(self):
        """一軸ひずみパッチテスト: 2×2×2 要素."""
        # 2×2×2 = 8要素、3×3×3 = 27節点
        nx, ny, nz = 2, 2, 2
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

        # 節点生成
        nodes = []
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    nodes.append([i * dx, j * dy, k * dz])
        nodes = np.array(nodes)
        n_nodes = len(nodes)

        # 要素生成
        def nid(i, j, k):
            return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

        elements = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    elements.append(
                        [
                            nid(i, j, k),
                            nid(i + 1, j, k),
                            nid(i + 1, j + 1, k),
                            nid(i, j + 1, k),
                            nid(i, j, k + 1),
                            nid(i + 1, j, k + 1),
                            nid(i + 1, j + 1, k + 1),
                            nid(i, j + 1, k + 1),
                        ]
                    )
        elements = np.array(elements)

        D = constitutive_3d(E_STEEL, NU_STEEL)
        K = self._assemble_multi_elem(nodes, elements, D)
        ndof = 3 * n_nodes

        # 一軸ひずみ εxx = ε₀ を課す
        # 変位場: u = ε₀ * x, v = 0, w = 0
        eps0 = 1e-4
        u_exact = np.zeros(ndof)
        for i in range(n_nodes):
            u_exact[3 * i] = eps0 * nodes[i, 0]  # u = ε₀ * x
            # v = 0, w = 0

        # 内力 = K @ u → 等価節点力
        F_int = K @ u_exact

        # 面上の節点に対する力のバランスを確認
        # x=0 面: 反力
        # x=Lx 面: 引張力
        # 側面: ポアソン効果による拘束力

        # 重要: パッチテストの判定基準は
        # K @ u_exact が正しい節点力を与えること
        # → 一様ひずみ εxx = ε₀ → σxx = (λ + 2μ) ε₀
        lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        sigma_xx = (lam + 2 * mu) * eps0

        # x=Lx 面の全ノードの x 方向力の合計 = σxx * A
        A_face = Ly * Lz
        x_max_nodes = [i for i in range(n_nodes) if abs(nodes[i, 0] - Lx) < 1e-10]
        Fx_right = sum(F_int[3 * i] for i in x_max_nodes)
        np.testing.assert_allclose(Fx_right, sigma_xx * A_face, rtol=1e-10)

    def test_constant_strain_single_element(self):
        """単一要素での定ひずみ → 正確な応力."""
        coords = _brick(2.0, 1.0, 1.0)
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar(coords, D)

        # 一軸ひずみ: u = ε₀ * x
        eps0 = 1e-4
        u = np.zeros(24)
        for i in range(8):
            u[3 * i] = eps0 * coords[i, 0]

        # F = Ke @ u
        F = Ke @ u

        # x=Lx 面の節点: 4,5,6,7 → DOF 3*i
        # ...ではなく、x=2 の節点
        Lx, Ly, Lz = 2.0, 1.0, 1.0
        lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        sigma_xx = (lam + 2 * mu) * eps0

        # x=Lx の節点: 1, 2, 5, 6
        right_nodes = [i for i in range(8) if abs(coords[i, 0] - Lx) < 1e-10]
        Fx_right = sum(F[3 * i] for i in right_nodes)
        np.testing.assert_allclose(Fx_right, sigma_xx * Ly * Lz, rtol=1e-10)


# ============================================================
# 4. 一軸圧縮テスト（解析解比較）
# ============================================================
class TestUniaxialCompression:
    """一軸圧縮の解析解との比較テスト."""

    @staticmethod
    def _solve_bar(
        L: float,
        A_cs: float,
        n_elem: int,
        E: float,
        nu: float,
        F_applied: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """z 方向一軸引張棒を HEX8 で解く.

        断面: A_cs = a × a の正方形
        """
        a = np.sqrt(A_cs)
        dz = L / n_elem

        # 節点生成 (2×2×(n_elem+1))
        nodes = []
        for k in range(n_elem + 1):
            for j in range(2):
                for i in range(2):
                    nodes.append([i * a, j * a, k * dz])
        nodes = np.array(nodes)
        n_nodes = len(nodes)

        def nid(i, j, k):
            return k * 4 + j * 2 + i

        elements = []
        for k in range(n_elem):
            elements.append(
                [
                    nid(0, 0, k),
                    nid(1, 0, k),
                    nid(1, 1, k),
                    nid(0, 1, k),
                    nid(0, 0, k + 1),
                    nid(1, 0, k + 1),
                    nid(1, 1, k + 1),
                    nid(0, 1, k + 1),
                ]
            )
        elements = np.array(elements)

        D = constitutive_3d(E, nu)
        ndof = 3 * n_nodes
        K = np.zeros((ndof, ndof))
        for elem in elements:
            coords_e = nodes[elem]
            ke = hex8_ke_bbar(coords_e, D)
            dofs = []
            for nid_val in elem:
                dofs.extend([3 * nid_val, 3 * nid_val + 1, 3 * nid_val + 2])
            dofs = np.array(dofs)
            for ii in range(24):
                for jj in range(24):
                    K[dofs[ii], dofs[jj]] += ke[ii, jj]

        # 境界条件: z=0 面を固定（uz=0）、
        # x方向: x=0でux=0
        # y方向: y=0でuy=0
        # → ポアソン収縮を許容
        F = np.zeros(ndof)

        # z=L 面に引張力
        top_nodes = [i for i in range(n_nodes) if abs(nodes[i, 2] - L) < 1e-10]
        f_per_node = F_applied / len(top_nodes)
        for nid_val in top_nodes:
            F[3 * nid_val + 2] = f_per_node

        # 固定 DOF
        fixed = []
        for i in range(n_nodes):
            if abs(nodes[i, 2]) < 1e-10:
                fixed.append(3 * i + 2)  # uz = 0
            if abs(nodes[i, 0]) < 1e-10:
                fixed.append(3 * i)  # ux = 0
            if abs(nodes[i, 1]) < 1e-10:
                fixed.append(3 * i + 1)  # uy = 0

        free = [d for d in range(ndof) if d not in fixed]
        K_ff = K[np.ix_(free, free)]
        F_f = F[free]
        u_f = np.linalg.solve(K_ff, F_f)

        u = np.zeros(ndof)
        u[free] = u_f

        return nodes, u

    def test_axial_displacement(self):
        """z 方向引張: 先端変位 = FL/(EA)."""
        L = 10.0
        a = 1.0  # 断面 1×1
        A = a**2
        n_elem = 4
        F = 1000.0  # [N]

        nodes, u = self._solve_bar(L, A, n_elem, E_STEEL, NU_STEEL, F)

        # 解析解: δ = FL/(EA)
        delta_exact = F * L / (E_STEEL * A)

        # z=L のノードの uz を取得
        top_nodes = [i for i in range(len(nodes)) if abs(nodes[i, 2] - L) < 1e-10]
        uz_top = np.mean([u[3 * i + 2] for i in top_nodes])

        np.testing.assert_allclose(uz_top, delta_exact, rtol=1e-6)

    def test_poisson_contraction(self):
        """ポアソン収縮: Δa/a = -ν * ε_zz."""
        L = 10.0
        a = 1.0
        A = a**2
        n_elem = 4
        F = 1000.0

        nodes, u = self._solve_bar(L, A, n_elem, E_STEEL, NU_STEEL, F)

        eps_zz = F / (E_STEEL * A)
        # x=a の先端ノードの ux
        top_right = [
            i
            for i in range(len(nodes))
            if abs(nodes[i, 0] - a) < 1e-10 and abs(nodes[i, 2] - L) < 1e-10
        ]
        ux_mean = np.mean([u[3 * i] for i in top_right])
        lateral_strain = ux_mean / a
        np.testing.assert_allclose(lateral_strain, -NU_STEEL * eps_zz, rtol=1e-6)


# ============================================================
# 5. 3D 弾性テンソル テスト
# ============================================================
class TestElastic3D:
    """3D 等方弾性テンソルのテスト."""

    def test_shape(self):
        """(6, 6) 行列."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        assert D.shape == (6, 6)

    def test_symmetric(self):
        """対称性: D = D^T."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_positive_definite(self):
        """正定値."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        eigvals = np.linalg.eigvalsh(D)
        assert np.all(eigvals > 0)

    def test_uniaxial_stress(self):
        """一軸応力: σ = E * ε for uniaxial (拘束なし) ではなく、
        一軸ひずみ: σxx = (λ + 2μ) εxx."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        eps = np.array([1e-4, 0, 0, 0, 0, 0])
        sigma = D @ eps
        lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        assert sigma[0] == pytest.approx((lam + 2 * mu) * 1e-4, rel=1e-12)
        assert sigma[1] == pytest.approx(lam * 1e-4, rel=1e-12)
        assert sigma[2] == pytest.approx(lam * 1e-4, rel=1e-12)

    def test_shear(self):
        """純せん断: τ = μ * γ."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        eps = np.array([0, 0, 0, 1e-4, 0, 0])  # γyz
        sigma = D @ eps
        assert sigma[3] == pytest.approx(mu * 1e-4, rel=1e-12)

    def test_class_interface(self):
        """IsotropicElastic3D がConstitutiveProtocol的に tangent() を持つ."""
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        D = mat.tangent()
        assert D.shape == (6, 6)
        np.testing.assert_allclose(D, constitutive_3d(E_STEEL, NU_STEEL), atol=1e-10)


# ============================================================
# 6. ElementProtocol 適合テスト
# ============================================================
class TestHex8BBarProtocol:
    """Hex8BBar の ElementProtocol 適合性."""

    def test_attributes(self):
        """基本属性."""
        elem = Hex8BBar()
        assert elem.ndof_per_node == 3
        assert elem.nnodes == 8
        assert elem.ndof == 24

    def test_local_stiffness(self):
        """local_stiffness が (24,24) を返す."""
        elem = Hex8BBar()
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        assert Ke.shape == (24, 24)

    def test_dof_indices(self):
        """dof_indices が正しいインデックスを返す."""
        elem = Hex8BBar()
        node_ids = np.array([10, 11, 12, 13, 20, 21, 22, 23])
        dofs = elem.dof_indices(node_ids)
        assert dofs.shape == (24,)
        # node 10 → DOF 30, 31, 32
        assert dofs[0] == 30
        assert dofs[1] == 31
        assert dofs[2] == 32

    def test_invalid_coords(self):
        """不正な座標形状でエラー."""
        elem = Hex8BBar()
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        with pytest.raises(ValueError, match="8,3"):
            elem.local_stiffness(np.zeros((4, 3)), mat)
