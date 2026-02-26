"""2D定常熱伝導FEMの検証テスト.

解析解との比較による検証:
1. 1D定常熱伝導（温度固定境界）
2. 一様発熱 + 対流冷却（フィンモデル）
3. メッシュ生成の検証
4. 熱流束計算の検証
"""

from __future__ import annotations

import numpy as np

from xkep_cae.thermal.fem import (
    assemble_thermal_system,
    compute_heat_flux,
    make_rect_mesh,
    quad4_conductivity,
    quad4_convection_surface,
    quad4_heat_load,
    solve_steady_thermal,
)


class TestMeshGeneration:
    """メッシュ生成の基本検証."""

    def test_rect_mesh_node_count(self):
        nodes, conn, edges = make_rect_mesh(1.0, 0.5, 4, 2)
        assert nodes.shape == (5 * 3, 2)
        assert conn.shape == (4 * 2, 4)

    def test_rect_mesh_boundary_edges(self):
        nodes, conn, edges = make_rect_mesh(1.0, 1.0, 3, 3)
        assert len(edges["bottom"]) == 3
        assert len(edges["top"]) == 3
        assert len(edges["left"]) == 3
        assert len(edges["right"]) == 3

    def test_rect_mesh_coordinates(self):
        nodes, _, _ = make_rect_mesh(2.0, 1.0, 2, 1)
        assert np.isclose(nodes[:, 0].min(), 0.0)
        assert np.isclose(nodes[:, 0].max(), 2.0)
        assert np.isclose(nodes[:, 1].min(), 0.0)
        assert np.isclose(nodes[:, 1].max(), 1.0)


class TestElementMatrices:
    """要素レベル行列の検証."""

    def test_conductivity_symmetry(self):
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        Ke = quad4_conductivity(xy, k=1.0, t=1.0)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-14)

    def test_conductivity_positive_semidefinite(self):
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        Ke = quad4_conductivity(xy, k=1.0, t=1.0)
        eigvals = np.linalg.eigvalsh(Ke)
        assert np.all(eigvals >= -1e-14)

    def test_convection_surface_symmetry(self):
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        He = quad4_convection_surface(xy, h=10.0)
        np.testing.assert_allclose(He, He.T, atol=1e-14)

    def test_heat_load_uniform(self):
        """一様発熱で荷重が要素面積×q×tの1/4ずつ分配."""
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        q = np.ones(4) * 100.0
        fe = quad4_heat_load(xy, q, t=0.01)
        # 面積1.0, q=100, t=0.01 → 合計 = 1.0
        np.testing.assert_allclose(fe.sum(), 1.0, rtol=1e-10)
        # 正方形で一様 → 等配分
        np.testing.assert_allclose(fe, 0.25, rtol=1e-10)

    def test_conductivity_scaling(self):
        """伝導率のスケーリング検証."""
        xy = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
        Ke1 = quad4_conductivity(xy, k=1.0, t=1.0)
        Ke2 = quad4_conductivity(xy, k=2.0, t=1.0)
        np.testing.assert_allclose(Ke2, 2.0 * Ke1, rtol=1e-14)


class TestSteadyConduction1D:
    """1D定常熱伝導の解析解比較.

    問題設定:
    - 長さ L=1.0m, 幅 W=0.1m の矩形
    - k=50 W/(m·K), 対流なし (h=0)
    - 左端 T=100, 右端 T=0（Dirichlet条件で模擬）
    - 解析解: T(x) = 100(1 - x/L)
    """

    def test_1d_conduction_no_convection(self):
        L, W = 1.0, 0.1
        nx, ny = 20, 2
        k = 50.0
        t = 0.01
        nodes, conn, edges = make_rect_mesh(L, W, nx, ny)
        N = len(nodes)
        q_nodal = np.zeros(N)

        K, f = assemble_thermal_system(
            nodes,
            conn,
            edges,
            k=k,
            h_conv=0.0,
            t=t,
            T_inf=0.0,
            q_nodal=q_nodal,
        )

        # Dirichlet BC: 左端 T=100, 右端 T=0
        # h_conv=0 なので対流項なし → 純粋伝導
        # ただし表裏面対流もゼロ → K は半正定値（剛体モード:一様温度）
        # Dirichlet で解く

        tol_nodes = 1e-10
        left_nodes = np.where(nodes[:, 0] < tol_nodes)[0]
        right_nodes = np.where(nodes[:, 0] > L - tol_nodes)[0]

        # ペナルティ法で Dirichlet 条件を適用
        big = 1e20
        for n in left_nodes:
            K[n, n] += big
            f[n] += big * 100.0
        for n in right_nodes:
            K[n, n] += big
            f[n] += big * 0.0

        T = solve_steady_thermal(K, f)

        # 解析解との比較
        T_exact = 100.0 * (1.0 - nodes[:, 0] / L)
        np.testing.assert_allclose(T, T_exact, atol=0.5)


class TestFinModel:
    """フィンモデル（一様発熱 + 対流冷却）.

    問題設定:
    - 正方形 L=0.1m
    - k=200 W/(m·K), h=50 W/(m²·K), t=0.002m
    - T_∞=25°C
    - 一様発熱 q=1e6 W/m³
    - 解析解（薄板フィンの0次近似: 一様温度）:
      熱収支: q * t * A = 2h * A * (T - T_∞) + h * t * perimeter * (T - T_∞)
      T - T_∞ = q * t / (2h + h * t * perimeter / A)
    """

    def test_uniform_heat_generation(self):
        L = 0.1
        nx, ny = 10, 10
        k = 200.0
        h_conv = 50.0
        t = 0.002
        T_inf = 25.0
        q = 1.0e6

        nodes, conn, edges = make_rect_mesh(L, L, nx, ny)
        N = len(nodes)
        q_nodal = np.full(N, q)

        K, f = assemble_thermal_system(
            nodes,
            conn,
            edges,
            k=k,
            h_conv=h_conv,
            t=t,
            T_inf=T_inf,
            q_nodal=q_nodal,
        )
        T = solve_steady_thermal(K, f)

        # 0次近似解析解（一様温度仮定）
        A = L * L
        perimeter = 4 * L
        dT_approx = q * t / (2 * h_conv + h_conv * t * perimeter / A)
        T_approx = T_inf + dT_approx

        # 板中心の温度は解析近似と概ね一致（5%以内）
        center_idx = np.argmin((nodes[:, 0] - L / 2) ** 2 + (nodes[:, 1] - L / 2) ** 2)
        assert abs(T[center_idx] - T_approx) / dT_approx < 0.05

    def test_temperature_positive(self):
        """温度が周囲温度以上になることを確認（発熱がある場合）."""
        L = 0.1
        nodes, conn, edges = make_rect_mesh(L, L, 5, 5)
        N = len(nodes)
        T_inf = 20.0
        q_nodal = np.full(N, 5.0e5)

        K, f = assemble_thermal_system(
            nodes,
            conn,
            edges,
            k=100.0,
            h_conv=25.0,
            t=0.005,
            T_inf=T_inf,
            q_nodal=q_nodal,
        )
        T = solve_steady_thermal(K, f)
        assert np.all(T >= T_inf - 1e-10)


class TestHeatFlux:
    """熱流束計算の検証."""

    def test_flux_uniform_gradient(self):
        """一様温度勾配で正しい熱流束."""
        nodes, conn, edges = make_rect_mesh(1.0, 1.0, 4, 4)
        # 線形温度場: T = 100 - 50*x → qx = -k*(-50) = 50k, qy = 0
        T = 100.0 - 50.0 * nodes[:, 0]
        k = 10.0
        flux = compute_heat_flux(nodes, conn, T, k, t=1.0)
        np.testing.assert_allclose(flux[:, 0], 500.0, atol=1e-10)
        np.testing.assert_allclose(flux[:, 1], 0.0, atol=1e-10)
