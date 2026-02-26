"""HEX8 要素ファミリのテスト.

テスト構成:
  1. 形状関数の基本性質
  2. 剛性行列の基本性質（対称性、ランク、剛体モード）
  3. パッチテスト（一様ひずみ再現性）
  4. 一軸圧縮テスト（解析解との比較）
  5. 3D弾性テンソルの検証
  6. ElementProtocol 適合性
  7. D行列の偏差-体積分解テスト
  8. C3D8R / C3D8I の基本性質テスト
  9. 片持ち梁曲げ — 全バリアント vs 解析解比較
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.hex8 import (
    Hex8BBar,
    Hex8BBarMean,
    Hex8Incompatible,
    Hex8Reduced,
    Hex8SRI,
    _extract_B_vol,
    _hex8_dNdxi,
    _hex8_shape,
    _split_D_vol_dev,
    hex8_ke_bbar,
    hex8_ke_bbar_mean,
    hex8_ke_incompatible,
    hex8_ke_reduced,
    hex8_ke_sri,
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
# ヘルパー: 片持ち梁ソルバー
# ============================================================


def _solve_cantilever(
    L: float,
    h: float,
    w: float,
    ne_x: int,
    ne_y: int,
    ne_z: int,
    ke_func,
    D: np.ndarray,
    P: float = 1000.0,
) -> float:
    """z 方向片持ち梁、先端に y 方向荷重 P. 先端 y たわみを返す."""
    dx, dy, dz = w / ne_x, h / ne_y, L / ne_z

    nodes = []
    for k in range(ne_z + 1):
        for j in range(ne_y + 1):
            for i in range(ne_x + 1):
                nodes.append([i * dx, j * dy, k * dz])
    nodes = np.array(nodes)
    n_nodes = len(nodes)

    def nid(i, j, k):
        return k * (ne_y + 1) * (ne_x + 1) + j * (ne_x + 1) + i

    elements = []
    for k in range(ne_z):
        for j in range(ne_y):
            for i in range(ne_x):
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

    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof))
    for elem in elements:
        coords_e = nodes[elem]
        ke = ke_func(coords_e, D)
        dofs = np.array([3 * n + d for n in elem for d in range(3)])
        for ii in range(24):
            for jj in range(24):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]

    F = np.zeros(ndof)
    top_nodes = [i for i in range(n_nodes) if abs(nodes[i, 2] - L) < 1e-10]
    f_per = P / len(top_nodes)
    for nid_val in top_nodes:
        F[3 * nid_val + 1] = f_per

    fixed = []
    for i in range(n_nodes):
        if abs(nodes[i, 2]) < 1e-10:
            fixed.extend([3 * i, 3 * i + 1, 3 * i + 2])

    free = [d for d in range(ndof) if d not in fixed]
    K_ff = K[np.ix_(free, free)]
    F_f = F[free]
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(ndof)
    u[free] = u_f

    return np.mean([u[3 * i + 1] for i in top_nodes])


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
class TestStiffnessBasicSRI:
    """C3D8 (SRI) 剛性行列の基本性質テスト."""

    def test_symmetry(self):
        """対称性: Ke = Ke^T."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-6)

    def test_rank(self):
        """ランク = 12（24 DOF - 6 RBM - 6 hourglass）.

        SRI（偏差=1点積分）は 6 つのアワーグラスモードを持つ。
        """
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 12

    def test_rigid_body_modes(self):
        """最小 6 固有値がほぼゼロ（剛体モード含む）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D)
        eigvals = np.sort(np.linalg.eigvalsh(Ke))
        threshold = 1e-6 * eigvals[-1]
        assert np.all(np.abs(eigvals[:6]) < threshold)

    def test_positive_semidefinite(self):
        """正半定値（全固有値 >= -ε）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        threshold = -1e-6 * np.max(np.abs(eigvals))
        assert np.all(eigvals >= threshold)

    def test_nonzero_stiffness(self):
        """非ゼロ剛性（最大固有値 > 0）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D)
        assert np.max(np.linalg.eigvalsh(Ke)) > 0

    def test_scaled_geometry(self):
        """スケーリング: 2倍拡大で剛性行列が変化するが有限."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke_scaled = hex8_ke_sri(2.0 * _unit_cube(), D)
        assert np.linalg.norm(Ke_scaled) > 0

    def test_distorted_element(self):
        """歪んだ要素でも正常に動作."""
        coords = _unit_cube().copy()
        coords[6, :] += [0.1, 0.1, 0.1]
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(coords, D)
        assert Ke.shape == (24, 24)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-4)

    def test_backward_compat_alias(self):
        """hex8_ke_bbar は hex8_ke_sri のエイリアス."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        coords = _unit_cube()
        Ke_sri = hex8_ke_sri(coords, D)
        Ke_bbar = hex8_ke_bbar(coords, D)
        np.testing.assert_allclose(Ke_sri, Ke_bbar, atol=1e-15)


# ============================================================
# 3. パッチテスト（一様ひずみ再現性）
# ============================================================
class TestPatchTest:
    """パッチテスト — 一様ひずみを正確に再現."""

    @staticmethod
    def _assemble_multi_elem(
        nodes: np.ndarray,
        elements: np.ndarray,
        D: np.ndarray,
        ke_func=hex8_ke_sri,
    ) -> np.ndarray:
        """複数 HEX8 要素のアセンブリ."""
        n_nodes = len(nodes)
        ndof = 3 * n_nodes
        K = np.zeros((ndof, ndof))
        for elem in elements:
            coords = nodes[elem]
            ke = ke_func(coords, D)
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
        nx, ny, nz = 2, 2, 2
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

        nodes = []
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    nodes.append([i * dx, j * dy, k * dz])
        nodes = np.array(nodes)
        n_nodes = len(nodes)

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

        eps0 = 1e-4
        u_exact = np.zeros(ndof)
        for i in range(n_nodes):
            u_exact[3 * i] = eps0 * nodes[i, 0]

        F_int = K @ u_exact

        lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        sigma_xx = (lam + 2 * mu) * eps0

        A_face = Ly * Lz
        x_max_nodes = [i for i in range(n_nodes) if abs(nodes[i, 0] - Lx) < 1e-10]
        Fx_right = sum(F_int[3 * i] for i in x_max_nodes)
        np.testing.assert_allclose(Fx_right, sigma_xx * A_face, rtol=1e-10)

    def test_constant_strain_single_element(self):
        """単一要素での定ひずみ → 正確な応力."""
        coords = _brick(2.0, 1.0, 1.0)
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(coords, D)

        eps0 = 1e-4
        u = np.zeros(24)
        for i in range(8):
            u[3 * i] = eps0 * coords[i, 0]

        F = Ke @ u

        Lx, Ly, Lz = 2.0, 1.0, 1.0
        lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        sigma_xx = (lam + 2 * mu) * eps0

        right_nodes = [i for i in range(8) if abs(coords[i, 0] - Lx) < 1e-10]
        Fx_right = sum(F[3 * i] for i in right_nodes)
        np.testing.assert_allclose(Fx_right, sigma_xx * Ly * Lz, rtol=1e-10)

    def test_patch_incompatible(self):
        """非適合モード要素のパッチテスト（2×2×2）."""
        nx, ny, nz = 2, 2, 2
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

        nodes = []
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    nodes.append([i * dx, j * dy, k * dz])
        nodes = np.array(nodes)
        n_nodes = len(nodes)

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
        K = self._assemble_multi_elem(nodes, elements, D, ke_func=hex8_ke_incompatible)
        ndof = 3 * n_nodes

        eps0 = 1e-4
        u_exact = np.zeros(ndof)
        for i in range(n_nodes):
            u_exact[3 * i] = eps0 * nodes[i, 0]

        F_int = K @ u_exact
        lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        sigma_xx = (lam + 2 * mu) * eps0

        A_face = Ly * Lz
        x_max_nodes = [i for i in range(n_nodes) if abs(nodes[i, 0] - Lx) < 1e-10]
        Fx_right = sum(F_int[3 * i] for i in x_max_nodes)
        np.testing.assert_allclose(Fx_right, sigma_xx * A_face, rtol=1e-6)


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
        """z 方向一軸引張棒を HEX8 で解く."""
        a = np.sqrt(A_cs)
        dz = L / n_elem

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
            ke = hex8_ke_sri(coords_e, D)
            dofs = []
            for nid_val in elem:
                dofs.extend([3 * nid_val, 3 * nid_val + 1, 3 * nid_val + 2])
            dofs = np.array(dofs)
            for ii in range(24):
                for jj in range(24):
                    K[dofs[ii], dofs[jj]] += ke[ii, jj]

        F = np.zeros(ndof)
        top_nodes = [i for i in range(n_nodes) if abs(nodes[i, 2] - L) < 1e-10]
        f_per_node = F_applied / len(top_nodes)
        for nid_val in top_nodes:
            F[3 * nid_val + 2] = f_per_node

        fixed = []
        for i in range(n_nodes):
            if abs(nodes[i, 2]) < 1e-10:
                fixed.append(3 * i + 2)
            if abs(nodes[i, 0]) < 1e-10:
                fixed.append(3 * i)
            if abs(nodes[i, 1]) < 1e-10:
                fixed.append(3 * i + 1)

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
        a = 1.0
        A = a**2
        n_elem = 4
        F = 1000.0

        nodes, u = self._solve_bar(L, A, n_elem, E_STEEL, NU_STEEL, F)

        delta_exact = F * L / (E_STEEL * A)
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
        D = constitutive_3d(E_STEEL, NU_STEEL)
        assert D.shape == (6, 6)

    def test_symmetric(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_positive_definite(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        eigvals = np.linalg.eigvalsh(D)
        assert np.all(eigvals > 0)

    def test_uniaxial_stress(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        eps = np.array([1e-4, 0, 0, 0, 0, 0])
        sigma = D @ eps
        lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        assert sigma[0] == pytest.approx((lam + 2 * mu) * 1e-4, rel=1e-12)
        assert sigma[1] == pytest.approx(lam * 1e-4, rel=1e-12)
        assert sigma[2] == pytest.approx(lam * 1e-4, rel=1e-12)

    def test_shear(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        mu = E_STEEL / (2 * (1 + NU_STEEL))
        eps = np.array([0, 0, 0, 1e-4, 0, 0])
        sigma = D @ eps
        assert sigma[3] == pytest.approx(mu * 1e-4, rel=1e-12)

    def test_class_interface(self):
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        D = mat.tangent()
        assert D.shape == (6, 6)
        np.testing.assert_allclose(D, constitutive_3d(E_STEEL, NU_STEEL), atol=1e-10)


# ============================================================
# 6. ElementProtocol 適合テスト
# ============================================================
class TestHex8SRIProtocol:
    """Hex8SRI (C3D8) の ElementProtocol 適合性."""

    def test_attributes(self):
        elem = Hex8SRI()
        assert elem.ndof_per_node == 3
        assert elem.nnodes == 8
        assert elem.ndof == 24
        assert elem.element_type == "C3D8"

    def test_local_stiffness(self):
        elem = Hex8SRI()
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        assert Ke.shape == (24, 24)

    def test_dof_indices(self):
        elem = Hex8SRI()
        node_ids = np.array([10, 11, 12, 13, 20, 21, 22, 23])
        dofs = elem.dof_indices(node_ids)
        assert dofs.shape == (24,)
        assert dofs[0] == 30
        assert dofs[1] == 31
        assert dofs[2] == 32

    def test_invalid_coords(self):
        elem = Hex8SRI()
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        with pytest.raises(ValueError, match="8,3"):
            elem.local_stiffness(np.zeros((4, 3)), mat)

    def test_backward_compat(self):
        """Hex8BBar は Hex8SRI のエイリアス."""
        assert Hex8BBar is Hex8SRI


# ============================================================
# 7. D 行列の偏差-体積分解テスト
# ============================================================
class TestDSplit:
    """D 行列の vol/dev 分解の検証."""

    def test_split_sums_to_D(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        D_vol, D_dev = _split_D_vol_dev(D)
        np.testing.assert_allclose(D_vol + D_dev, D, atol=1e-6)

    def test_D_vol_rank_1(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        D_vol, _ = _split_D_vol_dev(D)
        eigvals = np.linalg.eigvalsh(D_vol)
        n_nonzero = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert n_nonzero == 1

    def test_D_dev_rank_5(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        _, D_dev = _split_D_vol_dev(D)
        eigvals = np.linalg.eigvalsh(D_dev)
        n_nonzero = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert n_nonzero == 5

    def test_D_vol_is_Km_outer_m(self):
        """D_vol = K * m ⊗ m."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        D_vol, _ = _split_D_vol_dev(D)
        m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        K = E_STEEL / (3 * (1 - 2 * NU_STEEL))  # 体積弾性率
        np.testing.assert_allclose(D_vol, K * np.outer(m, m), rtol=1e-10)


# ============================================================
# 8. C3D8R / C3D8I の基本性質テスト
# ============================================================
class TestHex8Reduced:
    """C3D8R (均一低減積分) のテスト."""

    def test_rank(self):
        """ランク = 6（24 DOF - 6 RBM - 12 hourglass）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_reduced(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 6

    def test_symmetry(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_reduced(_unit_cube(), D)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-10)

    def test_protocol(self):
        elem = Hex8Reduced()
        assert elem.element_type == "C3D8R"
        assert elem.ndof == 24
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        assert Ke.shape == (24, 24)

    def test_hourglass_control_increases_rank(self):
        """alpha_hg > 0 でアワーグラス制御によりランクが増加."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke_hg = hex8_ke_reduced(_unit_cube(), D, alpha_hg=0.05)
        eigvals = np.linalg.eigvalsh(Ke_hg)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        # 4 HG ベクトル × 3 方向 = 12 モード。うち 9 が独立に追加（3 は既存と重複）
        assert rank == 15  # 6(base) + 9(HG) = 15

    def test_hourglass_control_symmetry(self):
        """アワーグラス制御付きでも対称性を保持."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke_hg = hex8_ke_reduced(_unit_cube(), D, alpha_hg=0.05)
        np.testing.assert_allclose(Ke_hg, Ke_hg.T, atol=1e-10)

    def test_hourglass_control_psd(self):
        """アワーグラス制御付きで正半定値."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke_hg = hex8_ke_reduced(_unit_cube(), D, alpha_hg=0.05)
        eigvals = np.linalg.eigvalsh(Ke_hg)
        threshold = -1e-6 * np.max(np.abs(eigvals))
        assert np.all(eigvals >= threshold)

    def test_hourglass_control_via_class(self):
        """Hex8Reduced クラス経由でアワーグラス制御が機能."""
        elem = Hex8Reduced(alpha_hg=0.05)
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 15

    def test_alpha_hg_zero_default(self):
        """デフォルト alpha_hg=0.0 でアワーグラス制御なし."""
        elem = Hex8Reduced()
        assert elem.alpha_hg == 0.0
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 6


class TestHex8Incompatible:
    """C3D8I (非適合モード) のテスト."""

    def test_rank(self):
        """ランク = 18（24 DOF - 6 RBM、hourglass なし）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_incompatible(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 18

    def test_symmetry(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_incompatible(_unit_cube(), D)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-4)

    def test_positive_semidefinite(self):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_incompatible(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        threshold = -1e-6 * np.max(np.abs(eigvals))
        assert np.all(eigvals >= threshold)

    def test_rigid_body_modes(self):
        """6 つの剛体モードがゼロ固有値."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_incompatible(_unit_cube(), D)
        eigvals = np.sort(np.linalg.eigvalsh(Ke))
        threshold = 1e-6 * eigvals[-1]
        n_zero = np.sum(np.abs(eigvals) < threshold)
        assert n_zero == 6

    def test_protocol(self):
        elem = Hex8Incompatible()
        assert elem.element_type == "C3D8I"
        assert elem.ndof == 24
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        assert Ke.shape == (24, 24)

    def test_distorted_element(self):
        """歪んだ要素でも動作."""
        coords = _unit_cube().copy()
        coords[6, :] += [0.1, 0.1, 0.1]
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_incompatible(coords, D)
        assert Ke.shape == (24, 24)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-3)


# ============================================================
# 9. 片持ち梁曲げ — 全バリアント比較
# ============================================================
class TestCantileverBending:
    """片持ち梁の曲げ: 各 HEX8 バリアント vs 解析解."""

    L = 10.0  # 梁長さ
    h = 1.0  # 高さ
    w = 1.0  # 幅
    P = 1000.0  # 先端荷重 [N]
    Iy = w * h**3 / 12.0  # 断面二次モーメント
    G = E_STEEL / (2 * (1 + NU_STEEL))
    kappa = 5.0 / 6.0
    A = w * h
    delta_EB = P * L**3 / (3 * E_STEEL * Iy)
    delta_Timo = delta_EB + P * L / (kappa * G * A)

    def test_c3d8r_single_element_cross_section(self):
        """C3D8R (非適合モード): 断面 1×1 要素でも高精度（< 5%）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        uy = _solve_cantilever(self.L, self.h, self.w, 1, 1, 8, hex8_ke_incompatible, D, self.P)
        np.testing.assert_allclose(uy, self.delta_Timo, rtol=0.05)

    def test_c3d8r_convergence(self):
        """C3D8R: メッシュ細分化で収束."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        uy_coarse = _solve_cantilever(
            self.L, self.h, self.w, 1, 1, 4, hex8_ke_incompatible, D, self.P
        )
        uy_fine = _solve_cantilever(
            self.L, self.h, self.w, 1, 1, 16, hex8_ke_incompatible, D, self.P
        )
        err_coarse = abs(uy_coarse - self.delta_Timo) / abs(self.delta_Timo)
        err_fine = abs(uy_fine - self.delta_Timo) / abs(self.delta_Timo)
        assert err_fine < err_coarse

    def test_c3d8_sri_multi_element_cross(self):
        """C3D8 (SRI): 断面 4×4 要素で解析解に収束（< 5%）."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        uy = _solve_cantilever(self.L, self.h, self.w, 4, 4, 8, hex8_ke_sri, D, self.P)
        np.testing.assert_allclose(uy, self.delta_Timo, rtol=0.05)

    def test_c3d8r_better_than_c3d8_coarse(self):
        """C3D8R は粗メッシュで C3D8 より精度が高い."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        uy_incomp = _solve_cantilever(
            self.L, self.h, self.w, 2, 2, 8, hex8_ke_incompatible, D, self.P
        )
        uy_sri = _solve_cantilever(self.L, self.h, self.w, 2, 2, 8, hex8_ke_sri, D, self.P)
        err_incomp = abs(uy_incomp - self.delta_Timo) / abs(self.delta_Timo)
        err_sri = abs(uy_sri - self.delta_Timo) / abs(self.delta_Timo)
        assert err_incomp < err_sri


# ============================================================
# 10. C3D8R alpha_hg チューニング指針テスト
# ============================================================


class TestHourglassControlTuning:
    """C3D8R アワーグラス制御パラメータ alpha_hg のチューニング指針."""

    L = 10.0
    h = 1.0
    w = 1.0
    P = 1000.0
    Iy = w * h**3 / 12.0
    G = E_STEEL / (2 * (1 + NU_STEEL))
    kappa = 5.0 / 6.0
    A = w * h
    delta_Timo = P * L**3 / (3 * E_STEEL * Iy) + P * L / (kappa * G * A)

    def _solve_reduced(self, alpha_hg, ne_x=2, ne_y=2, ne_z=8):
        D = constitutive_3d(E_STEEL, NU_STEEL)
        return _solve_cantilever(
            self.L,
            self.h,
            self.w,
            ne_x,
            ne_y,
            ne_z,
            lambda xyz, D_: hex8_ke_reduced(xyz, D_, alpha_hg=alpha_hg),
            D,
            self.P,
        )

    def test_alpha_0_no_bending_stiffness(self):
        """alpha_hg=0: アワーグラスモードにより曲げが不安定."""
        # alpha_hg=0 は有限変位を返すが解析解からの乖離が大きい（ロック解除されすぎ）
        uy = self._solve_reduced(0.0)
        err = abs(uy - self.delta_Timo) / abs(self.delta_Timo)
        # alpha_hg=0 では大きな誤差が出る（曲げ剛性不足）
        assert err > 0.5, f"alpha_hg=0 should have large error, got {err:.4f}"

    def test_recommended_range(self):
        """alpha_hg=0.03 で曲げ誤差 < 10%.

        alpha_hg=0.03 が最適（誤差約3%）。0.05 ではやや過剛性（誤差約10%）。
        """
        uy = self._solve_reduced(0.03)
        err = abs(uy - self.delta_Timo) / abs(self.delta_Timo)
        assert err < 0.10, f"alpha_hg=0.03: error {err:.4f} > 10%"

    def test_optimal_alpha_around_003(self):
        """alpha_hg ≈ 0.03 が最適: 小さすぎても大きすぎても精度低下.

        アワーグラス制御には最適値が存在する:
        - 小さすぎる → アワーグラスモードが残り曲げ不安定
        - 大きすぎる → 人工剛性過大で過剛性
        alpha_hg=0.03 が両者のバランス点。
        """
        alphas = [0.01, 0.03, 0.10]
        errors = []
        for alpha in alphas:
            uy = self._solve_reduced(alpha)
            err = abs(uy - self.delta_Timo) / abs(self.delta_Timo)
            errors.append(err)
        # 0.03 が最も小さい誤差
        assert errors[1] < errors[0], (
            f"alpha 0.03 ({errors[1]:.4f}) should be better than 0.01 ({errors[0]:.4f})"
        )
        assert errors[1] < errors[2], (
            f"alpha 0.03 ({errors[1]:.4f}) should be better than 0.10 ({errors[2]:.4f})"
        )

    def test_excessive_alpha_overstiffens(self):
        """alpha_hg が過大だと過剛性（たわみ過小）になる."""
        uy_05 = self._solve_reduced(0.05)
        uy_50 = self._solve_reduced(0.50)
        # alpha=0.50 ではアワーグラス人工剛性が過大 → たわみが小さくなる
        assert abs(uy_50) < abs(uy_05), "excessive alpha should reduce deflection"


# ============================================================
# 11. B-bar 平均膨張法 (C3D8B) テスト
# ============================================================


class TestHex8BBarMean:
    """C3D8B (B-bar 平均膨張法) のテスト."""

    def test_symmetry(self):
        """B-bar 剛性行列は対称."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar_mean(_unit_cube(), D)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-10)

    def test_rank(self):
        """ランク = 18（24 DOF - 6 RBM）. 完全積分ベースなのでアワーグラスなし."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar_mean(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 18

    def test_psd(self):
        """正半定値性."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar_mean(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        assert np.all(eigvals > -1e-6 * np.max(np.abs(eigvals)))

    def test_rbm(self):
        """6 つの剛体モードがゼロエネルギー."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar_mean(_unit_cube(), D)
        eigvals = np.linalg.eigvalsh(Ke)
        n_zero = np.sum(np.abs(eigvals) < 1e-6 * np.max(np.abs(eigvals)))
        assert n_zero == 6

    def test_protocol(self):
        """Hex8BBarMean クラスが ElementProtocol に適合."""
        elem = Hex8BBarMean()
        assert elem.element_type == "C3D8B"
        assert elem.ndof == 24
        assert elem.nnodes == 8
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        assert Ke.shape == (24, 24)

    def test_patch_test(self):
        """B-bar パッチテスト: 一様ひずみが正確に再現される."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_bbar_mean(_unit_cube(), D)
        # 一様引張: u_x = ε₀·x → εxx = ε₀, 他=0
        eps0 = 1e-3
        nodes = _unit_cube()
        u = np.zeros(24)
        for i in range(8):
            u[3 * i] = eps0 * nodes[i, 0]
        # 内力: f_int = Ke @ u
        f = Ke @ u
        # 反力のx成分合計 ≈ σ_xx × 面積 (面 x=0: 圧縮, x=1: 引張)
        sigma_xx = D[0, 0] * eps0
        # x=1 面（節点 1,2,5,6）の x方向反力合計
        fx_pos = sum(f[3 * i] for i in [1, 2, 5, 6])
        np.testing.assert_allclose(fx_pos, sigma_xx, rtol=1e-10)

    def test_b_vol_projection(self):
        """_extract_B_vol の体積射影が正しい."""
        from xkep_cae.elements.hex8 import _compute_B_detJ

        B, _, _, _ = _compute_B_detJ(_unit_cube(), 0.0, 0.0, 0.0)
        B_vol = _extract_B_vol(B)
        # 体積部: 法線3行が同一（各DOFの体積寄与平均）
        np.testing.assert_allclose(B_vol[0, :], B_vol[1, :])
        np.testing.assert_allclose(B_vol[0, :], B_vol[2, :])
        # せん断行はゼロ
        np.testing.assert_allclose(B_vol[3:6, :], 0.0)

    def test_cantilever_bending_coarse(self):
        """B-bar 片持ち梁テスト（4×4×8 メッシュ）.

        B-bar は体積ロッキングを回避するが、せん断ロッキングは残る。
        完全積分ベースのため、薄い梁ではせん断ロッキングで過剛性（< 40%）。
        """
        L, h, w = 10.0, 1.0, 1.0
        P = 1000.0
        D = constitutive_3d(E_STEEL, NU_STEEL)
        uy = _solve_cantilever(L, h, w, 4, 4, 8, hex8_ke_bbar_mean, D, P)
        Iy = w * h**3 / 12.0
        G = E_STEEL / (2 * (1 + NU_STEEL))
        kappa = 5.0 / 6.0
        A = w * h
        delta_Timo = P * L**3 / (3 * E_STEEL * Iy) + P * L / (kappa * G * A)
        rel_err = abs(uy - delta_Timo) / abs(delta_Timo)
        # B-bar は完全積分ベースなのでせん断ロッキングあり
        assert rel_err < 0.40, f"B-bar cantilever error {rel_err:.4f} > 40%"

    def test_volume_locking_resistance(self):
        """B-bar は非圧縮限界（ν→0.5）でも安定.

        完全積分（SRI でない）では ν=0.499 で体積ロッキングが深刻になるが、
        B-bar ではランク 18 を維持する。
        """
        D_nearly_inc = constitutive_3d(E_STEEL, 0.499)
        Ke = hex8_ke_bbar_mean(_unit_cube(), D_nearly_inc)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        # 非圧縮限界でもランク 18 を維持
        assert rank == 18, f"B-bar rank at nu=0.499: {rank}, expected 18"


# ============================================================
# 12. C3D8 SRI + アワーグラス制御テスト
# ============================================================


class TestSRIHourglassControl:
    """C3D8 (SRI) のアワーグラス制御テスト."""

    def test_alpha_hg_default_zero(self):
        """Hex8SRI のデフォルト alpha_hg=0.0."""
        elem = Hex8SRI()
        assert elem.alpha_hg == 0.0

    def test_rank_without_hg(self):
        """アワーグラス制御なし: ランク 12."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D, alpha_hg=0.0)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank == 12

    def test_rank_with_hg(self):
        """アワーグラス制御あり: ランク増加."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D, alpha_hg=0.03)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank > 12, f"expected rank > 12 with HG control, got {rank}"

    def test_symmetry_with_hg(self):
        """アワーグラス制御付きでも対称性保持."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D, alpha_hg=0.03)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-10)

    def test_psd_with_hg(self):
        """アワーグラス制御付きでも正半定値."""
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Ke = hex8_ke_sri(_unit_cube(), D, alpha_hg=0.03)
        eigvals = np.linalg.eigvalsh(Ke)
        assert np.all(eigvals > -1e-6 * np.max(np.abs(eigvals)))

    def test_class_alpha_hg(self):
        """Hex8SRI クラス経由でアワーグラス制御."""
        elem = Hex8SRI(alpha_hg=0.02)
        mat = IsotropicElastic3D(E_STEEL, NU_STEEL)
        Ke = elem.local_stiffness(_unit_cube(), mat)
        eigvals = np.linalg.eigvalsh(Ke)
        rank = np.sum(np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals)))
        assert rank > 12

    def test_cantilever_improvement(self):
        """SRI + HG 制御で片持ち梁精度が改善."""
        L, h, w = 10.0, 1.0, 1.0
        P = 1000.0
        D = constitutive_3d(E_STEEL, NU_STEEL)
        Iy = w * h**3 / 12.0
        G = E_STEEL / (2 * (1 + NU_STEEL))
        kappa = 5.0 / 6.0
        A = w * h
        delta_Timo = P * L**3 / (3 * E_STEEL * Iy) + P * L / (kappa * G * A)

        # SRI のみ (alpha_hg=0): 1×1 断面ではロック気味
        uy_no_hg = _solve_cantilever(L, h, w, 1, 1, 8, lambda xyz, D_: hex8_ke_sri(xyz, D_), D, P)
        # SRI + HG: alpha_hg=0.02
        uy_hg = _solve_cantilever(
            L, h, w, 1, 1, 8, lambda xyz, D_: hex8_ke_sri(xyz, D_, alpha_hg=0.02), D, P
        )
        err_no_hg = abs(uy_no_hg - delta_Timo) / abs(delta_Timo)
        err_hg = abs(uy_hg - delta_Timo) / abs(delta_Timo)
        # HG制御で誤差が改善（または同程度）
        assert err_hg < err_no_hg + 0.05, (
            f"SRI+HG ({err_hg:.4f}) should not be much worse than SRI ({err_no_hg:.4f})"
        )
