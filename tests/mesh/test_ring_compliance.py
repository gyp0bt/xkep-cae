"""厚肉弾性リング コンプライアンス行列テスト (Stage S1).

テスト構成:
  1. モード 0 (Lamé 解) — 解析解との一致
  2. モード n≥2 (Michell 解) — 基本性質
  3. コンプライアンス行列 — 対称性・正定値性・循環行列
  4. 薄肉リング極限 — Euler 曲がり梁理論との一致
  5. FEM リング解との比較
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.mesh.ring_compliance import (
    _build_michell_system,
    build_ring_compliance_matrix,
    ring_compliance_summary,
    ring_mode0_compliance,
    ring_mode_n_compliance,
)

# ============================================================
# テスト用パラメータ
# ============================================================
# 鋼製シース想定
E_STEEL = 200e9  # [Pa]
NU_STEEL = 0.3
# 典型的なシース寸法
A_INNER = 5.0e-3  # 内径 5mm
B_OUTER = 6.0e-3  # 外径 6mm（肉厚 1mm）


# ============================================================
# 1. モード 0 (Lamé 解) テスト
# ============================================================
class TestMode0Compliance:
    """モード 0 コンプライアンス (Lamé 解) のテスト."""

    def test_positive_compliance(self):
        """内圧で膨張 → c₀ > 0."""
        c0 = ring_mode0_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        assert c0 > 0

    def test_plane_stress_positive(self):
        """平面応力でも c₀ > 0."""
        c0 = ring_mode0_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, plane="stress")
        assert c0 > 0

    def test_lame_analytical(self):
        """Lamé 解との厳密一致（平面応力）."""
        a, b = 5e-3, 8e-3
        E, nu = 100e9, 0.25
        c0 = ring_mode0_compliance(a, b, E, nu, plane="stress")
        # 解析解: c₀ = a/[E(b²-a²)] · [(1-ν)a² + (1+ν)b²]
        expected = a / (E * (b**2 - a**2)) * ((1.0 - nu) * a**2 + (1.0 + nu) * b**2)
        np.testing.assert_allclose(c0, expected, rtol=1e-14)

    def test_lame_plane_strain(self):
        """Lamé 解との厳密一致（平面ひずみ）."""
        a, b = 5e-3, 8e-3
        E, nu = 100e9, 0.25
        c0 = ring_mode0_compliance(a, b, E, nu, plane="strain")
        # 平面ひずみ: c₀ = a(1+ν)/[E(b²-a²)] · [(1-2ν)a² + b²]
        expected = a * (1.0 + nu) / (E * (b**2 - a**2)) * ((1.0 - 2.0 * nu) * a**2 + b**2)
        np.testing.assert_allclose(c0, expected, rtol=1e-14)

    def test_thicker_ring_stiffer(self):
        """肉厚が増すと剛性上昇 → c₀ 減少."""
        c0_thin = ring_mode0_compliance(5e-3, 5.5e-3, E_STEEL, NU_STEEL)
        c0_thick = ring_mode0_compliance(5e-3, 8e-3, E_STEEL, NU_STEEL)
        assert c0_thick < c0_thin

    def test_stiffer_material(self):
        """ヤング率上昇 → c₀ 減少."""
        c0_soft = ring_mode0_compliance(A_INNER, B_OUTER, 100e9, NU_STEEL)
        c0_hard = ring_mode0_compliance(A_INNER, B_OUTER, 400e9, NU_STEEL)
        assert c0_hard < c0_soft

    def test_invalid_geometry(self):
        """不正な幾何でエラー."""
        with pytest.raises(ValueError):
            ring_mode0_compliance(0, B_OUTER, E_STEEL, NU_STEEL)
        with pytest.raises(ValueError):
            ring_mode0_compliance(B_OUTER, A_INNER, E_STEEL, NU_STEEL)

    def test_invalid_plane(self):
        """不正な plane 指定でエラー."""
        with pytest.raises(ValueError):
            ring_mode0_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, plane="foo")


# ============================================================
# 2. モード n≥2 (Michell 解) テスト
# ============================================================
class TestModeNCompliance:
    """モード n コンプライアンス (Michell 解) のテスト."""

    def test_positive_compliance_n2(self):
        """モード 2 のコンプライアンスは正."""
        cn = ring_mode_n_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, n=2)
        assert cn > 0

    def test_positive_compliance_n6(self):
        """モード 6 のコンプライアンスは正."""
        cn = ring_mode_n_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, n=6)
        assert cn > 0

    def test_compliance_decreases_with_n(self):
        """高次モードほどコンプライアンスが小さい（剛性が高い）."""
        cn = []
        for n in [2, 4, 6, 8, 12]:
            cn.append(ring_mode_n_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, n=n))
        # 単調減少
        for i in range(len(cn) - 1):
            assert cn[i] > cn[i + 1], f"c_{2 + 2 * i} > c_{4 + 2 * i} を期待"

    def test_thicker_ring_stiffer_mode_n(self):
        """肉厚増 → モード n コンプライアンス減少."""
        cn_thin = ring_mode_n_compliance(5e-3, 5.5e-3, E_STEEL, NU_STEEL, n=6)
        cn_thick = ring_mode_n_compliance(5e-3, 8e-3, E_STEEL, NU_STEEL, n=6)
        assert cn_thick < cn_thin

    def test_plane_stress_vs_strain(self):
        """平面応力と平面ひずみで値が異なる."""
        cn_stress = ring_mode_n_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, n=6, plane="stress")
        cn_strain = ring_mode_n_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, n=6, plane="strain")
        assert cn_stress != cn_strain

    def test_invalid_n(self):
        """n < 2 でエラー."""
        with pytest.raises(ValueError):
            ring_mode_n_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, n=1)
        with pytest.raises(ValueError):
            ring_mode_n_compliance(A_INNER, B_OUTER, E_STEEL, NU_STEEL, n=0)

    def test_michell_system_rank(self):
        """Michell 4×4 行列が正則であること."""
        for n in [2, 3, 6, 12]:
            M, rhs = _build_michell_system(beta=1.2, n=n)
            det = np.linalg.det(M)
            assert abs(det) > 1e-20, f"n={n} で行列が特異"

    def test_stress_bc_satisfied(self):
        """解いた係数が応力境界条件を満たすことを検証."""
        a, b = 5e-3, 7e-3
        n = 6
        beta = b / a
        M, rhs = _build_michell_system(beta, n)
        alpha = np.linalg.solve(M, rhs)

        # σ_r(a) = -p_n = -1 → M[0,:] @ alpha = -1
        np.testing.assert_allclose(M[0, :] @ alpha, -1.0, atol=1e-12)
        # σ_r(b) = 0
        np.testing.assert_allclose(M[1, :] @ alpha, 0.0, atol=1e-12)
        # τ_rθ(a) = 0
        np.testing.assert_allclose(M[2, :] @ alpha, 0.0, atol=1e-12)
        # τ_rθ(b) = 0
        np.testing.assert_allclose(M[3, :] @ alpha, 0.0, atol=1e-12)

    def test_high_mode_stable(self):
        """高次モード (n=50) でも数値的に安定."""
        cn = ring_mode_n_compliance(5e-3, 6e-3, E_STEEL, NU_STEEL, n=50)
        assert np.isfinite(cn)
        assert cn > 0


# ============================================================
# 3. コンプライアンス行列テスト
# ============================================================
class TestComplianceMatrix:
    """N×N コンプライアンス行列の性質テスト."""

    def test_symmetric(self):
        """対称行列."""
        C = build_ring_compliance_matrix(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        np.testing.assert_allclose(C, C.T, atol=1e-20)

    def test_positive_definite(self):
        """正定値."""
        C = build_ring_compliance_matrix(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"負の固有値: {eigvals}"

    def test_circulant(self):
        """循環行列（各行は前行の巡回シフト）."""
        N = 6
        C = build_ring_compliance_matrix(N, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        # C[i,j] = C[0, (j-i) % N]
        for i in range(N):
            for j in range(N):
                idx = (j - i) % N
                np.testing.assert_allclose(
                    C[i, j], C[0, idx], atol=1e-20, err_msg=f"C[{i},{j}] != C[0,{idx}]"
                )

    def test_diagonal_dominant(self):
        """自己コンプライアンス（対角）が最大."""
        C = build_ring_compliance_matrix(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        diag = np.diag(C)
        off_diag_max = np.max(np.abs(C - np.diag(diag)))
        assert np.all(diag > off_diag_max)

    def test_size_6(self):
        """7本撚り（外層6本）のサイズ."""
        C = build_ring_compliance_matrix(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        assert C.shape == (6, 6)

    def test_size_12(self):
        """19本撚り（外層12本）のサイズ."""
        C = build_ring_compliance_matrix(12, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        assert C.shape == (12, 12)

    def test_uniform_load_mode0(self):
        """全点に等荷重 → モード 0 のみ励起、変位が均一."""
        N = 6
        C = build_ring_compliance_matrix(N, A_INNER, B_OUTER, E_STEEL, NU_STEEL, n_modes=60)
        F = np.ones(N)  # 均等荷重
        delta = C @ F
        # 全変位が同一
        np.testing.assert_allclose(delta, delta[0] * np.ones(N), rtol=1e-10)

    def test_single_force_response(self):
        """1点のみに力 → 最大変位はその点."""
        N = 6
        C = build_ring_compliance_matrix(N, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        F = np.zeros(N)
        F[0] = 1.0
        delta = C @ F
        assert np.argmax(delta) == 0

    def test_n_modes_convergence(self):
        """モード数増加で行列が収束."""
        C_low = build_ring_compliance_matrix(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL, n_modes=12)
        C_high = build_ring_compliance_matrix(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL, n_modes=60)
        # 高モードはほぼ同じ値に収束
        np.testing.assert_allclose(C_low, C_high, rtol=0.05)

    def test_invalid_N(self):
        """N < 2 でエラー."""
        with pytest.raises(ValueError):
            build_ring_compliance_matrix(1, A_INNER, B_OUTER, E_STEEL, NU_STEEL)


# ============================================================
# 4. 薄肉リング極限テスト
# ============================================================
class TestThinRingLimit:
    """薄肉リング (t << R) では曲がり梁理論と一致.

    薄肉極限 (t/R → 0) での非伸張性リング公式:
      モード 0: c₀ = R²/(E·t)  （膜変形, Lamé 極限）
      モード n≥2: c_n ≈ R⁴ / [EI(n²-1)²]  （非伸張性リング）

    ここで I = t³/12（単位奥行あたり）。
    モード n≥2 では膜剛性が高いため非伸張性仮定がよく成り立つ。

    注意: t/R = 0.1 では 5-30% の薄肉近似誤差がある。
    Michell 厚肉解が基準解（エネルギー法で平面ひずみ一致を検証済み）。
    """

    def test_mode0_thin_ring(self):
        """モード 0: 薄肉極限で c₀ → R²/(E·t) (平面応力, ν=0)."""
        R = 10e-3
        t = 1.0e-3  # t/R = 0.1
        a = R - t / 2
        b = R + t / 2
        E, nu = 200e9, 0.0

        c0 = ring_mode0_compliance(a, b, E, nu, plane="stress")
        c0_thin = R**2 / (E * t)
        np.testing.assert_allclose(c0, c0_thin, rtol=0.05)

    def test_mode_n_inextensible_thin_ring(self):
        """モード n: 非伸張性薄肉リングでは c_n ≈ R⁴/(EI(n²-1)²).

        低次モードでは膜剛性 >> 曲げ剛性のため非伸張性がよく成り立つ。
        t/R=0.1 では近似誤差あり。
        """
        R = 10e-3
        t = 1.0e-3  # t/R = 0.1
        a = R - t / 2
        b = R + t / 2
        E, nu = 200e9, 0.0
        I_ring = t**3 / 12.0

        for n in [2, 3, 6]:
            cn = ring_mode_n_compliance(a, b, E, nu, n=n, plane="stress")
            cn_inext = R**4 / (E * I_ring * (n**2 - 1) ** 2)
            np.testing.assert_allclose(
                cn,
                cn_inext,
                rtol=0.5,
                err_msg=f"薄肉非伸張極限と大幅に乖離 (n={n})",
            )

    def test_mode_n_convergence_to_thin(self):
        """t/R → 0 で Michell 解が薄肉公式に収束."""
        R = 10e-3
        E, nu = 200e9, 0.0
        n = 6

        errors = []
        for t_ratio in [0.2, 0.1, 0.05]:
            t = R * t_ratio
            a, b = R - t / 2, R + t / 2
            I_ring = t**3 / 12.0
            cn = ring_mode_n_compliance(a, b, E, nu, n=n, plane="stress")
            cn_thin = R**4 / (E * I_ring * (n**2 - 1) ** 2)
            errors.append(abs(cn - cn_thin) / cn_thin)

        # 肉厚が薄くなるほど薄肉解に収束（誤差減少）
        assert errors[1] < errors[0], "t/R=0.1 は t/R=0.2 より薄肉解に近いべき"
        assert errors[2] < errors[1], "t/R=0.05 は t/R=0.1 より薄肉解に近いべき"


# ============================================================
# 5. FEM リング解との比較テスト
# ============================================================
class TestFEMComparison:
    """Q4 要素で構成したリングメッシュの FEM 解との比較."""

    @staticmethod
    def _make_ring_mesh(
        a: float, b: float, n_r: int, n_theta: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """構造格子リングメッシュ生成.

        Parameters
        ----------
        a : float
            内径
        b : float
            外径
        n_r : int
            径方向要素数
        n_theta : int
            周方向要素数

        Returns
        -------
        nodes : (n_nodes, 2) ndarray
        elements : (n_elems, 4) ndarray — Q4 接続（反時計回り）
        """
        # 構造格子点
        r_vals = np.linspace(a, b, n_r + 1)
        theta_vals = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]  # 周期

        n_nodes = (n_r + 1) * n_theta
        nodes = np.zeros((n_nodes, 2))

        for i, r in enumerate(r_vals):
            for j, th in enumerate(theta_vals):
                nid = i * n_theta + j
                nodes[nid, 0] = r * np.cos(th)
                nodes[nid, 1] = r * np.sin(th)

        # Q4 要素（反時計回り: ξ→径方向, η→周方向で det(J)>0）
        elements = []
        for i in range(n_r):
            for j in range(n_theta):
                n0 = i * n_theta + j  # 内, θ_j
                n1 = (i + 1) * n_theta + j  # 外, θ_j
                n2 = (i + 1) * n_theta + (j + 1) % n_theta  # 外, θ_{j+1}
                n3 = i * n_theta + (j + 1) % n_theta  # 内, θ_{j+1}
                elements.append([n0, n1, n2, n3])

        return nodes, np.array(elements)

    @staticmethod
    def _assemble_ring_fem(
        nodes: np.ndarray,
        elements: np.ndarray,
        E: float,
        nu: float,
        plane: str = "strain",
    ) -> np.ndarray:
        """2D Q4 要素で全体剛性行列を組み立て.

        平面ひずみ or 平面応力の D 行列を使用。
        """
        n_nodes = len(nodes)
        ndof = 2 * n_nodes
        # 疎行列の代わりに密行列（小規模なので）
        K = np.zeros((ndof, ndof))

        if plane == "strain":
            factor = E / ((1 + nu) * (1 - 2 * nu))
            D = factor * np.array(
                [
                    [1 - nu, nu, 0],
                    [nu, 1 - nu, 0],
                    [0, 0, (1 - 2 * nu) / 2],
                ]
            )
        else:  # plane stress
            factor = E / (1 - nu**2)
            D = factor * np.array(
                [
                    [1, nu, 0],
                    [nu, 1, 0],
                    [0, 0, (1 - nu) / 2],
                ]
            )

        # 2×2 ガウス積分点
        gp = 1.0 / np.sqrt(3)
        gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
        gauss_wts = [1.0, 1.0, 1.0, 1.0]

        for elem in elements:
            x_e = nodes[elem, 0]  # (4,)
            y_e = nodes[elem, 1]

            ke = np.zeros((8, 8))
            for (xi, eta), w in zip(gauss_pts, gauss_wts, strict=True):
                # 形状関数の自然座標微分
                dN_dxi = (
                    np.array(
                        [
                            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
                        ]
                    )
                    / 4.0
                )

                J = dN_dxi @ np.column_stack([x_e, y_e])  # (2,2)
                detJ = np.linalg.det(J)
                dN_dx = np.linalg.solve(J, dN_dxi)  # (2,4)

                # B 行列 (3×8)
                B = np.zeros((3, 8))
                for k in range(4):
                    B[0, 2 * k] = dN_dx[0, k]
                    B[1, 2 * k + 1] = dN_dx[1, k]
                    B[2, 2 * k] = dN_dx[1, k]
                    B[2, 2 * k + 1] = dN_dx[0, k]

                ke += B.T @ D @ B * detJ * w

            # 全体剛性行列に組み込み
            dofs = []
            for nid in elem:
                dofs.extend([2 * nid, 2 * nid + 1])
            dofs = np.array(dofs)

            for ii in range(8):
                for jj in range(8):
                    K[dofs[ii], dofs[jj]] += ke[ii, jj]

        return K

    @staticmethod
    def _find_node_at_angle(nodes, target_angle, node_set):
        """指定角度に最も近いノードを返す."""
        best_nid = None
        best_diff = float("inf")
        for nid in node_set:
            x, y = nodes[nid]
            angle = np.arctan2(y, x)
            diff = abs(np.arctan2(np.sin(angle - target_angle), np.cos(angle - target_angle)))
            if diff < best_diff:
                best_diff = diff
                best_nid = nid
        return best_nid

    def test_fem_ring_uniform_pressure(self):
        """FEM リング解 vs Lamé 解（均等内圧）."""
        a, b = 5e-3, 7e-3
        E, nu = 200e9, 0.3
        plane = "strain"

        n_r, n_theta = 8, 48
        nodes, elements = self._make_ring_mesh(a, b, n_r, n_theta)
        n_nodes = len(nodes)
        ndof = 2 * n_nodes

        K = self._assemble_ring_fem(nodes, elements, E, nu, plane)

        # 内面に均等圧力 p=1 を荷重ベクトルとして適用
        p = 1.0  # [Pa]
        F = np.zeros(ndof)
        inner_nodes = list(range(n_theta))
        for idx in range(len(inner_nodes)):
            nid = inner_nodes[idx]
            nid_next = inner_nodes[(idx + 1) % len(inner_nodes)]
            x0, y0 = nodes[nid]
            x1, y1 = nodes[nid_next]
            edge_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            r0 = np.sqrt(x0**2 + y0**2)
            r1 = np.sqrt(x1**2 + y1**2)
            nx0, ny0 = x0 / r0, y0 / r0
            nx1, ny1 = x1 / r1, y1 / r1
            force_mag = p * edge_len / 2.0
            F[2 * nid] += force_mag * nx0
            F[2 * nid + 1] += force_mag * ny0
            F[2 * nid_next] += force_mag * nx1
            F[2 * nid_next + 1] += force_mag * ny1

        # 剛体拘束: 接線方向のみ拘束（径方向膨張を妨げない）
        # θ≈0 のノード: u_y = 0 (y方向 = 接線方向)
        # θ≈π/2 のノード: u_x = 0 (x方向 = 接線方向)
        node_0 = self._find_node_at_angle(nodes, 0.0, inner_nodes)
        node_90 = self._find_node_at_angle(nodes, np.pi / 2, inner_nodes)
        fixed_dofs = [
            2 * node_0 + 1,  # u_y = 0 at θ≈0
            2 * node_90,  # u_x = 0 at θ≈π/2
        ]

        free_dofs = [d for d in range(ndof) if d not in fixed_dofs]
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        u_f = np.linalg.solve(K_ff, F_f)

        u = np.zeros(ndof)
        u[free_dofs] = u_f

        # 内面の径方向変位を計算
        u_r_fem = []
        for nid in inner_nodes:
            ux, uy = u[2 * nid], u[2 * nid + 1]
            x, y = nodes[nid]
            r = np.sqrt(x**2 + y**2)
            u_r_fem.append((ux * x + uy * y) / r)
        u_r_fem = np.array(u_r_fem)

        # Lamé 解析解
        c0 = ring_mode0_compliance(a, b, E, nu, plane)
        u_r_analytical = c0 * p

        # FEM 解は解析解と 2% 以内で一致
        np.testing.assert_allclose(
            np.mean(u_r_fem), u_r_analytical, rtol=0.02, err_msg="FEM リング解が Lamé 解と不一致"
        )

    def test_fem_ring_point_loads(self):
        """FEM リング解 vs コンプライアンス行列（6点集中荷重）."""
        a, b = 5e-3, 7e-3
        E, nu = 200e9, 0.3
        plane = "strain"
        N = 6  # 6点荷重

        n_r, n_theta = 8, 60  # n_theta は N の倍数
        nodes, elements = self._make_ring_mesh(a, b, n_r, n_theta)
        n_nodes = len(nodes)
        ndof = 2 * n_nodes

        K = self._assemble_ring_fem(nodes, elements, E, nu, plane)

        F_total = np.zeros(ndof)
        load_magnitude = 1.0  # [N/m]

        load_angles = [2 * np.pi * k / N for k in range(N)]
        inner_nodes = list(range(n_theta))

        load_node_ids = []
        for angle in load_angles:
            nid = self._find_node_at_angle(nodes, angle, inner_nodes)
            load_node_ids.append(nid)
            x, y = nodes[nid]
            r = np.sqrt(x**2 + y**2)
            F_total[2 * nid] += load_magnitude * x / r
            F_total[2 * nid + 1] += load_magnitude * y / r

        # 剛体拘束: 接線方向のみ
        node_0 = self._find_node_at_angle(nodes, 0.0, inner_nodes)
        node_90 = self._find_node_at_angle(nodes, np.pi / 2, inner_nodes)
        fixed_dofs = [
            2 * node_0 + 1,  # u_y = 0 at θ≈0
            2 * node_90,  # u_x = 0 at θ≈π/2
        ]

        free_dofs = [d for d in range(ndof) if d not in fixed_dofs]
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F_total[free_dofs]
        u_f = np.linalg.solve(K_ff, F_f)

        u = np.zeros(ndof)
        u[free_dofs] = u_f

        # 荷重点での径方向変位
        u_r_fem = []
        for nid in load_node_ids:
            ux, uy = u[2 * nid], u[2 * nid + 1]
            x, y = nodes[nid]
            r = np.sqrt(x**2 + y**2)
            u_r_fem.append((ux * x + uy * y) / r)
        u_r_fem = np.array(u_r_fem)

        # コンプライアンス行列による予測
        C = build_ring_compliance_matrix(N, a, b, E, nu, n_modes=60, plane=plane)
        F_vec = np.ones(N) * load_magnitude
        u_r_analytical = C @ F_vec

        # 均等荷重なので全変位は同一（対称性）
        # FEM でも荷重点の変位はほぼ均一
        np.testing.assert_allclose(
            np.mean(u_r_fem),
            np.mean(u_r_analytical),
            rtol=0.05,
            err_msg="FEM 平均径方向変位が不一致",
        )

        # 変位のばらつき: 均等荷重なのでコンプライアンス行列は均一応答を予測
        # FEM は BC 非対称性と離散化誤差でばらつくが、
        # mean に対する変動が 50% 以内であれば許容
        u_r_fem_var = u_r_fem - np.mean(u_r_fem)
        assert np.max(np.abs(u_r_fem_var)) < 0.5 * np.mean(np.abs(u_r_fem))


# ============================================================
# 6. summary 関数テスト
# ============================================================
class TestComplianceSummary:
    """ring_compliance_summary のテスト."""

    def test_returns_dict(self):
        """辞書を返す."""
        result = ring_compliance_summary(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        assert isinstance(result, dict)
        assert "C" in result
        assert "eigenvalues" in result
        assert "condition_number" in result
        assert "c0" in result
        assert "mode_compliances" in result

    def test_eigenvalues_positive(self):
        """固有値が全て正."""
        result = ring_compliance_summary(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        assert np.all(result["eigenvalues"] > 0)

    def test_condition_number_finite(self):
        """条件数が有限."""
        result = ring_compliance_summary(6, A_INNER, B_OUTER, E_STEEL, NU_STEEL)
        assert np.isfinite(result["condition_number"])
        assert result["condition_number"] >= 1.0
