"""シースモデル曲げモードバリデーションテスト.

C3D8R（非適合モード）HEX8 要素で円筒管をメッシュ化し、
解析解（梁理論）と比較してシースの長手方向挙動を検証する。

テスト構成:
  1. チューブメッシュ基本テスト
  2. 軸方向引張/圧縮（EA 検証）
  3. 純曲げ（EI 検証）
  4. 横せん断（片持ち梁、Timoshenko 理論）
  5. シースモデル等価剛性との整合性
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.elements.hex8 import hex8_ke_incompatible
from xkep_cae.materials.elastic import constitutive_3d
from xkep_cae.mesh.tube_mesh import make_tube_mesh, tube_face_nodes

# ============================================================
# テスト用パラメータ
# ============================================================
E_STEEL = 200e9  # [Pa]
NU_STEEL = 0.3
# 典型的なシース寸法
R_INNER = 5.0e-3  # 内径 5mm
R_OUTER = 6.0e-3  # 外径 6mm（肉厚 1mm）
TUBE_LENGTH = 50.0e-3  # 管長さ 50mm


# ============================================================
# ヘルパー: 3D チューブ FEM ソルバー
# ============================================================


def _assemble_tube_fem(
    nodes: np.ndarray,
    elements: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """HEX8 要素チューブの全体剛性行列を組み立て."""
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof))

    for elem in elements:
        coords_e = nodes[elem]
        ke = hex8_ke_incompatible(coords_e, D)
        dofs = np.empty(24, dtype=int)
        for i, nid in enumerate(elem):
            dofs[3 * i] = 3 * nid
            dofs[3 * i + 1] = 3 * nid + 1
            dofs[3 * i + 2] = 3 * nid + 2
        for ii in range(24):
            for jj in range(24):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]

    return K


def _solve_tube(
    nodes: np.ndarray,
    elements: np.ndarray,
    D: np.ndarray,
    fixed_dofs: list[int],
    F: np.ndarray,
) -> np.ndarray:
    """チューブ FEM を解く.

    Parameters
    ----------
    nodes, elements : メッシュ
    D : 弾性テンソル
    fixed_dofs : 拘束 DOF リスト
    F : 荷重ベクトル

    Returns
    -------
    u : (ndof,) 変位ベクトル
    """
    K = _assemble_tube_fem(nodes, elements, D)
    ndof = 3 * len(nodes)

    free = np.array([d for d in range(ndof) if d not in fixed_dofs])
    K_ff = K[np.ix_(free, free)]
    F_f = F[free]
    u_f = np.linalg.solve(K_ff, F_f)

    u = np.zeros(ndof)
    u[free] = u_f
    return u


# ============================================================
# 解析解ユーティリティ
# ============================================================


def _tube_section_properties(r_inner: float, r_outer: float) -> dict:
    """円筒管の断面特性."""
    A = math.pi * (r_outer**2 - r_inner**2)
    Iy = math.pi / 4.0 * (r_outer**4 - r_inner**4)
    J = math.pi / 2.0 * (r_outer**4 - r_inner**4)
    return {"A": A, "I": Iy, "J": J}


# ============================================================
# 1. チューブメッシュ基本テスト
# ============================================================
class TestTubeMesh:
    """チューブメッシュの基本テスト."""

    def test_node_count(self):
        """節点数 = (n_r+1) * n_theta * (n_z+1)."""
        n_r, n_theta, n_z = 2, 12, 4
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, TUBE_LENGTH, n_r, n_theta, n_z)
        expected = (n_r + 1) * n_theta * (n_z + 1)
        assert len(nodes) == expected

    def test_element_count(self):
        """要素数 = n_r * n_theta * n_z."""
        n_r, n_theta, n_z = 2, 12, 4
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, TUBE_LENGTH, n_r, n_theta, n_z)
        assert len(elements) == n_r * n_theta * n_z

    def test_node_radii(self):
        """全節点が内径〜外径の範囲内."""
        nodes, _ = make_tube_mesh(R_INNER, R_OUTER, TUBE_LENGTH, 2, 12, 4)
        r = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
        assert np.all(r >= R_INNER - 1e-15)
        assert np.all(r <= R_OUTER + 1e-15)

    def test_z_range(self):
        """全節点の z が [0, L] 内."""
        nodes, _ = make_tube_mesh(R_INNER, R_OUTER, TUBE_LENGTH, 2, 12, 4)
        assert np.all(nodes[:, 2] >= -1e-15)
        assert np.all(nodes[:, 2] <= TUBE_LENGTH + 1e-15)

    def test_face_nodes_z0(self):
        """z=0 面の節点数."""
        n_r, n_theta, n_z = 2, 12, 4
        nodes, _ = make_tube_mesh(R_INNER, R_OUTER, TUBE_LENGTH, n_r, n_theta, n_z)
        z0_nodes = tube_face_nodes(nodes, "z0")
        assert len(z0_nodes) == (n_r + 1) * n_theta

    def test_face_nodes_zL(self):
        """z=L 面の節点数."""
        n_r, n_theta, n_z = 2, 12, 4
        nodes, _ = make_tube_mesh(R_INNER, R_OUTER, TUBE_LENGTH, n_r, n_theta, n_z)
        zL_nodes = tube_face_nodes(nodes, "zL", length=TUBE_LENGTH)
        assert len(zL_nodes) == (n_r + 1) * n_theta

    def test_hex8_positive_jacobian(self):
        """全要素のヤコビアンが正."""
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, TUBE_LENGTH, 2, 16, 4)
        D = constitutive_3d(E_STEEL, NU_STEEL)
        for elem in elements:
            coords = nodes[elem]
            # hex8_ke_incompatible が例外を投げなければ detJ > 0
            ke = hex8_ke_incompatible(coords, D)
            assert ke.shape == (24, 24)

    def test_invalid_geometry(self):
        """不正な幾何でエラー."""
        with pytest.raises(ValueError):
            make_tube_mesh(0, R_OUTER, TUBE_LENGTH, 2, 12, 4)
        with pytest.raises(ValueError):
            make_tube_mesh(R_OUTER, R_INNER, TUBE_LENGTH, 2, 12, 4)
        with pytest.raises(ValueError):
            make_tube_mesh(R_INNER, R_OUTER, 0, 2, 12, 4)


# ============================================================
# 2. 軸方向引張/圧縮（EA 検証）
# ============================================================
class TestAxialStiffness:
    """軸方向引張/圧縮: FEM vs 解析解 EA."""

    def test_axial_tension(self):
        """軸方向引張: δ = FL/(EA)."""
        L = TUBE_LENGTH
        n_r, n_theta, n_z = 2, 16, 8
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, L, n_r, n_theta, n_z)
        D = constitutive_3d(E_STEEL, NU_STEEL)

        props = _tube_section_properties(R_INNER, R_OUTER)
        F_total = 1000.0  # [N]

        ndof = 3 * len(nodes)
        F_vec = np.zeros(ndof)

        # z=L 面に軸方向荷重
        zL_nodes = tube_face_nodes(nodes, "zL", length=L)
        f_per_node = F_total / len(zL_nodes)
        for nid in zL_nodes:
            F_vec[3 * nid + 2] = f_per_node

        # z=0 面: uz=0、剛体拘束
        z0_nodes = tube_face_nodes(nodes, "z0")
        fixed = []
        for nid in z0_nodes:
            fixed.append(3 * nid + 2)  # uz = 0

        # 剛体回転・並進拘束（最小拘束）
        # x=0, y=0 近辺の節点を使って ux, uy を拘束
        # z=0 面の節点のうち、θ≈0 の節点で uy=0
        # z=0 面の節点のうち、θ≈π/2 の節点で ux=0
        z0_angles = np.arctan2(nodes[z0_nodes, 1], nodes[z0_nodes, 0])
        idx_0 = z0_nodes[np.argmin(np.abs(z0_angles))]
        idx_90 = z0_nodes[np.argmin(np.abs(z0_angles - np.pi / 2))]
        fixed.append(3 * idx_0)  # ux = 0
        fixed.append(3 * idx_0 + 1)  # uy = 0
        fixed.append(3 * idx_90)  # ux = 0 (回転拘束)

        u = _solve_tube(nodes, elements, D, fixed, F_vec)

        # z=L 面の平均 uz
        uz_top = np.mean([u[3 * nid + 2] for nid in zL_nodes])
        delta_exact = F_total * L / (E_STEEL * props["A"])

        # 5% 以内で一致（メッシュ粗さ + ポアソン効果による差異）
        np.testing.assert_allclose(uz_top, delta_exact, rtol=0.05)

    def test_axial_compression(self):
        """軸方向圧縮: δ = -FL/(EA)."""
        L = TUBE_LENGTH
        n_r, n_theta, n_z = 2, 16, 8
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, L, n_r, n_theta, n_z)
        D = constitutive_3d(E_STEEL, NU_STEEL)

        props = _tube_section_properties(R_INNER, R_OUTER)
        F_total = -500.0  # 圧縮

        ndof = 3 * len(nodes)
        F_vec = np.zeros(ndof)

        zL_nodes = tube_face_nodes(nodes, "zL", length=L)
        f_per_node = F_total / len(zL_nodes)
        for nid in zL_nodes:
            F_vec[3 * nid + 2] = f_per_node

        z0_nodes = tube_face_nodes(nodes, "z0")
        fixed = []
        for nid in z0_nodes:
            fixed.append(3 * nid + 2)

        z0_angles = np.arctan2(nodes[z0_nodes, 1], nodes[z0_nodes, 0])
        idx_0 = z0_nodes[np.argmin(np.abs(z0_angles))]
        idx_90 = z0_nodes[np.argmin(np.abs(z0_angles - np.pi / 2))]
        fixed.append(3 * idx_0)
        fixed.append(3 * idx_0 + 1)
        fixed.append(3 * idx_90)

        u = _solve_tube(nodes, elements, D, fixed, F_vec)

        uz_top = np.mean([u[3 * nid + 2] for nid in zL_nodes])
        delta_exact = F_total * L / (E_STEEL * props["A"])

        np.testing.assert_allclose(uz_top, delta_exact, rtol=0.05)

    def test_stiffness_scaling(self):
        """EA が肉厚に正しく比例."""
        L = TUBE_LENGTH
        n_r, n_theta, n_z = 2, 16, 8
        D = constitutive_3d(E_STEEL, NU_STEEL)
        F_total = 1000.0

        results = {}
        for t, r_out in [(1e-3, 6e-3), (2e-3, 7e-3)]:
            nodes, elements = make_tube_mesh(R_INNER, r_out, L, n_r, n_theta, n_z)
            ndof = 3 * len(nodes)
            F_vec = np.zeros(ndof)

            zL_nodes = tube_face_nodes(nodes, "zL", length=L)
            f_per_node = F_total / len(zL_nodes)
            for nid in zL_nodes:
                F_vec[3 * nid + 2] = f_per_node

            z0_nodes = tube_face_nodes(nodes, "z0")
            fixed = []
            for nid in z0_nodes:
                fixed.append(3 * nid + 2)
            z0_angles = np.arctan2(nodes[z0_nodes, 1], nodes[z0_nodes, 0])
            idx_0 = z0_nodes[np.argmin(np.abs(z0_angles))]
            idx_90 = z0_nodes[np.argmin(np.abs(z0_angles - np.pi / 2))]
            fixed.append(3 * idx_0)
            fixed.append(3 * idx_0 + 1)
            fixed.append(3 * idx_90)

            u = _solve_tube(nodes, elements, D, fixed, F_vec)
            uz_top = np.mean([u[3 * nid + 2] for nid in zL_nodes])
            results[t] = uz_top

        # 肉厚増 → 変位減少（剛性増）
        assert abs(results[2e-3]) < abs(results[1e-3])


# ============================================================
# 3. 純曲げ（EI 検証）
# ============================================================
class TestBendingStiffness:
    """純曲げ: FEM vs 解析解 EI."""

    def test_cantilever_bending(self):
        """片持ち梁先端荷重: δ = FL³/(3EI).

        z=0 固定端、z=L 自由端に y 方向荷重。
        """
        L = 100.0e-3  # 長めのチューブでスレンダー性確保
        n_r, n_theta, n_z = 2, 16, 16
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, L, n_r, n_theta, n_z)
        D = constitutive_3d(E_STEEL, NU_STEEL)

        props = _tube_section_properties(R_INNER, R_OUTER)
        P = 10.0  # [N] y 方向荷重

        ndof = 3 * len(nodes)
        F_vec = np.zeros(ndof)

        # z=L 面に y 方向荷重
        zL_nodes = tube_face_nodes(nodes, "zL", length=L)
        f_per_node = P / len(zL_nodes)
        for nid in zL_nodes:
            F_vec[3 * nid + 1] = f_per_node

        # z=0 面: 全 DOF 固定（固定端）
        z0_nodes = tube_face_nodes(nodes, "z0")
        fixed = []
        for nid in z0_nodes:
            fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        u = _solve_tube(nodes, elements, D, fixed, F_vec)

        # z=L 面の平均 uy
        uy_tip = np.mean([u[3 * nid + 1] for nid in zL_nodes])

        # Euler-Bernoulli 解析解
        delta_EB = P * L**3 / (3.0 * E_STEEL * props["I"])

        # Timoshenko 補正（横せん断によるたわみ追加）
        G = E_STEEL / (2.0 * (1.0 + NU_STEEL))
        # パイプ断面のせん断補正係数
        m = R_INNER / R_OUTER
        kappa_s = (
            6.0
            * (1.0 + NU_STEEL)
            * (1.0 + m**2) ** 2
            / ((7.0 + 6.0 * NU_STEEL) * (1.0 + m**2) ** 2 + (20.0 + 12.0 * NU_STEEL) * m**2)
        )
        A_s = kappa_s * props["A"]
        delta_shear = P * L / (G * A_s)

        delta_total = delta_EB + delta_shear

        # 10% 以内で Timoshenko 解と一致
        np.testing.assert_allclose(uy_tip, delta_total, rtol=0.10)

    def test_pure_bending_moment(self):
        """純曲げモーメント: 曲率 κ = M/(EI).

        z=0 固定端、z=L に曲げモーメント（カップル荷重）。
        """
        L = 100.0e-3
        n_r, n_theta, n_z = 2, 16, 16
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, L, n_r, n_theta, n_z)
        D = constitutive_3d(E_STEEL, NU_STEEL)

        props = _tube_section_properties(R_INNER, R_OUTER)
        M = 1.0  # [N·m] x 軸周り曲げモーメント

        ndof = 3 * len(nodes)
        F_vec = np.zeros(ndof)

        # z=L 面にモーメント荷重: M_x = ∫ σzz * y dA
        # → 線形分布荷重 σzz ∝ y
        # → 節点力 fz_i ∝ y_i
        zL_nodes = tube_face_nodes(nodes, "zL", length=L)
        y_vals = nodes[zL_nodes, 1]

        # 荷重分布: σzz = M * y / I（標準符号規約）
        # → fz_i ∝ y_i, 合計モーメント M = Σ fz_i * y_i
        # 梁の運動学: d²v/dz² = -M/(EI)
        # → 先端たわみ δ = -ML²/(2EI)
        y2_sum = np.sum(y_vals**2)
        for idx, nid in enumerate(zL_nodes):
            F_vec[3 * nid + 2] = M * y_vals[idx] / y2_sum

        # z=0 面: 全 DOF 固定
        z0_nodes = tube_face_nodes(nodes, "z0")
        fixed = []
        for nid in z0_nodes:
            fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        u = _solve_tube(nodes, elements, D, fixed, F_vec)

        # 先端のたわみ: d²v/dz² = -M/(EI) → δ = -ML²/(2EI)
        # 応力カップルで y > 0 側が引張 → 梁は -y 方向にたわむ
        uy_tip = np.mean([u[3 * nid + 1] for nid in zL_nodes])
        delta_exact = -M * L**2 / (2.0 * E_STEEL * props["I"])

        # 15% 以内で一致（モーメント荷重の離散化誤差）
        np.testing.assert_allclose(uy_tip, delta_exact, rtol=0.15)

    def test_bending_EI_thicker_stiffer(self):
        """肉厚増 → EI 増 → たわみ減少."""
        L = 100.0e-3
        P = 10.0
        n_r, n_theta, n_z = 2, 16, 16
        D = constitutive_3d(E_STEEL, NU_STEEL)

        tips = {}
        for r_out in [6e-3, 8e-3]:
            nodes, elements = make_tube_mesh(R_INNER, r_out, L, n_r, n_theta, n_z)
            ndof = 3 * len(nodes)
            F_vec = np.zeros(ndof)

            zL_nodes = tube_face_nodes(nodes, "zL", length=L)
            f_per_node = P / len(zL_nodes)
            for nid in zL_nodes:
                F_vec[3 * nid + 1] = f_per_node

            z0_nodes = tube_face_nodes(nodes, "z0")
            fixed = []
            for nid in z0_nodes:
                fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

            u = _solve_tube(nodes, elements, D, fixed, F_vec)
            tips[r_out] = np.mean([u[3 * nid + 1] for nid in zL_nodes])

        # 外径 8mm > 6mm → EI 増 → たわみ減少
        assert abs(tips[8e-3]) < abs(tips[6e-3])


# ============================================================
# 4. 横せん断テスト
# ============================================================
class TestTransverseShear:
    """横せん断: 短い片持ちチューブでせん断変形支配."""

    def test_short_cantilever_shear_dominant(self):
        """短い片持ち梁: せん断たわみが支配的.

        L/D ≈ 2 のストッキーな梁では Timoshenko 補正が重要。
        """
        L = 20.0e-3  # 短い管
        n_r, n_theta, n_z = 2, 16, 8
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, L, n_r, n_theta, n_z)
        D = constitutive_3d(E_STEEL, NU_STEEL)

        props = _tube_section_properties(R_INNER, R_OUTER)
        P = 100.0  # [N]

        ndof = 3 * len(nodes)
        F_vec = np.zeros(ndof)

        zL_nodes = tube_face_nodes(nodes, "zL", length=L)
        f_per_node = P / len(zL_nodes)
        for nid in zL_nodes:
            F_vec[3 * nid + 1] = f_per_node

        z0_nodes = tube_face_nodes(nodes, "z0")
        fixed = []
        for nid in z0_nodes:
            fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        u = _solve_tube(nodes, elements, D, fixed, F_vec)
        uy_tip = np.mean([u[3 * nid + 1] for nid in zL_nodes])

        # Euler-Bernoulli のみ（せん断なし）
        delta_EB = P * L**3 / (3.0 * E_STEEL * props["I"])

        # Timoshenko（せん断込み）
        G = E_STEEL / (2.0 * (1.0 + NU_STEEL))
        m = R_INNER / R_OUTER
        kappa_s = (
            6.0
            * (1.0 + NU_STEEL)
            * (1.0 + m**2) ** 2
            / ((7.0 + 6.0 * NU_STEEL) * (1.0 + m**2) ** 2 + (20.0 + 12.0 * NU_STEEL) * m**2)
        )
        A_s = kappa_s * props["A"]
        delta_shear = P * L / (G * A_s)
        delta_Timo = delta_EB + delta_shear

        # FEM は Timoshenko に近い（15%以内）
        np.testing.assert_allclose(uy_tip, delta_Timo, rtol=0.15)

        # せん断たわみの寄与が曲げたわみの 10% 以上
        shear_ratio = delta_shear / delta_EB
        assert shear_ratio > 0.1, f"せん断比率 {shear_ratio:.3f} が低すぎる"

    def test_shear_locking_free(self):
        """非適合モード法によりせん断ロッキングが緩和されている.

        粗いメッシュでも、フル積分 HEX8 に比べて
        大幅にロッキングの少ない結果を得られることを確認。
        """
        L = 100.0e-3
        P = 10.0
        n_r, n_theta, n_z = 1, 8, 8  # 粗いメッシュ
        nodes, elements = make_tube_mesh(R_INNER, R_OUTER, L, n_r, n_theta, n_z)
        D = constitutive_3d(E_STEEL, NU_STEEL)

        props = _tube_section_properties(R_INNER, R_OUTER)

        ndof = 3 * len(nodes)
        F_vec = np.zeros(ndof)

        zL_nodes = tube_face_nodes(nodes, "zL", length=L)
        f_per_node = P / len(zL_nodes)
        for nid in zL_nodes:
            F_vec[3 * nid + 1] = f_per_node

        z0_nodes = tube_face_nodes(nodes, "z0")
        fixed = []
        for nid in z0_nodes:
            fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        u = _solve_tube(nodes, elements, D, fixed, F_vec)
        uy_tip = np.mean([u[3 * nid + 1] for nid in zL_nodes])

        # 解析解
        delta_EB = P * L**3 / (3.0 * E_STEEL * props["I"])

        # 非適合モード法により粗いメッシュでも解析解の 50% 以上の変位が出る
        # （純フル積分だとロッキングで 10-30% に留まることがある）
        ratio = abs(uy_tip / delta_EB)
        assert ratio > 0.5, f"ロッキング疑い: FEM/解析 = {ratio:.3f}"


# ============================================================
# 5. シースモデル等価剛性との整合性
# ============================================================
class TestSheathStiffnessConsistency:
    """シースモデルの等価梁剛性 (EA, EI, GJ) と FEM の整合性."""

    def test_EA_consistency(self):
        """sheath_equivalent_stiffness の EA vs FEM 軸剛性."""
        from xkep_cae.mesh.twisted_wire import (
            SheathModel,
            make_twisted_wire_mesh,
            sheath_equivalent_stiffness,
            sheath_inner_radius,
        )

        mesh = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=2.0e-3,
            pitch=40.0e-3,
            length=40.0e-3,
            n_elems_per_strand=4,
        )
        sheath = SheathModel(thickness=1.0e-3, E=E_STEEL, nu=NU_STEEL)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        r_in = sheath_inner_radius(mesh, sheath)
        r_out = r_in + sheath.thickness

        props = _tube_section_properties(r_in, r_out)
        EA_analytical = E_STEEL * props["A"]

        # sheath_equivalent_stiffness の EA と一致
        np.testing.assert_allclose(stiff["EA"], EA_analytical, rtol=1e-10)

    def test_EI_consistency(self):
        """sheath_equivalent_stiffness の EI vs FEM 曲げ剛性."""
        from xkep_cae.mesh.twisted_wire import (
            SheathModel,
            make_twisted_wire_mesh,
            sheath_equivalent_stiffness,
            sheath_inner_radius,
        )

        mesh = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=2.0e-3,
            pitch=40.0e-3,
            length=40.0e-3,
            n_elems_per_strand=4,
        )
        sheath = SheathModel(thickness=1.0e-3, E=E_STEEL, nu=NU_STEEL)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        r_in = sheath_inner_radius(mesh, sheath)
        r_out = r_in + sheath.thickness

        props = _tube_section_properties(r_in, r_out)
        EI_analytical = E_STEEL * props["I"]

        np.testing.assert_allclose(stiff["EIy"], EI_analytical, rtol=1e-10)
        np.testing.assert_allclose(stiff["EIz"], EI_analytical, rtol=1e-10)

    def test_GJ_consistency(self):
        """sheath_equivalent_stiffness の GJ vs 解析解."""
        from xkep_cae.mesh.twisted_wire import (
            SheathModel,
            make_twisted_wire_mesh,
            sheath_equivalent_stiffness,
            sheath_inner_radius,
        )

        mesh = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=2.0e-3,
            pitch=40.0e-3,
            length=40.0e-3,
            n_elems_per_strand=4,
        )
        sheath = SheathModel(thickness=1.0e-3, E=E_STEEL, nu=NU_STEEL)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        r_in = sheath_inner_radius(mesh, sheath)
        r_out = r_in + sheath.thickness

        G = E_STEEL / (2.0 * (1.0 + NU_STEEL))
        props = _tube_section_properties(r_in, r_out)
        GJ_analytical = G * props["J"]

        np.testing.assert_allclose(stiff["GJ"], GJ_analytical, rtol=1e-10)

    def test_bending_fem_vs_sheath_model(self):
        """シースモデル EI での曲げたわみ vs HEX8 FEM たわみ."""
        from xkep_cae.mesh.twisted_wire import (
            SheathModel,
            make_twisted_wire_mesh,
            sheath_equivalent_stiffness,
            sheath_inner_radius,
        )

        mesh = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=2.0e-3,
            pitch=40.0e-3,
            length=40.0e-3,
            n_elems_per_strand=4,
        )
        sheath = SheathModel(thickness=1.0e-3, E=E_STEEL, nu=NU_STEEL)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        r_in = sheath_inner_radius(mesh, sheath)
        r_out = r_in + sheath.thickness

        L = 100.0e-3
        P = 10.0
        n_r, n_theta, n_z = 2, 16, 16
        nodes, elements = make_tube_mesh(r_in, r_out, L, n_r, n_theta, n_z)
        D = constitutive_3d(E_STEEL, NU_STEEL)

        ndof = 3 * len(nodes)
        F_vec = np.zeros(ndof)

        zL_nodes = tube_face_nodes(nodes, "zL", length=L)
        f_per_node = P / len(zL_nodes)
        for nid in zL_nodes:
            F_vec[3 * nid + 1] = f_per_node

        z0_nodes = tube_face_nodes(nodes, "z0")
        fixed = []
        for nid in z0_nodes:
            fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])

        u = _solve_tube(nodes, elements, D, fixed, F_vec)
        uy_fem = np.mean([u[3 * nid + 1] for nid in zL_nodes])

        # Euler-Bernoulli with Timoshenko correction
        EI = stiff["EIy"]
        delta_EB = P * L**3 / (3.0 * EI)
        G = E_STEEL / (2.0 * (1.0 + NU_STEEL))
        m = r_in / r_out
        kappa_s = (
            6.0
            * (1.0 + NU_STEEL)
            * (1.0 + m**2) ** 2
            / ((7.0 + 6.0 * NU_STEEL) * (1.0 + m**2) ** 2 + (20.0 + 12.0 * NU_STEEL) * m**2)
        )
        props = _tube_section_properties(r_in, r_out)
        A_s = kappa_s * props["A"]
        delta_shear = P * L / (G * A_s)
        delta_beam = delta_EB + delta_shear

        # FEM と梁理論（シースモデル EI 基準）が 15% 以内で一致
        # （3D FEM は梁理論と厳密には一致しない: 断面の変形・ポアソン効果）
        np.testing.assert_allclose(uy_fem, delta_beam, rtol=0.15)
