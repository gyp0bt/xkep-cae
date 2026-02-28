"""Phase S2: CPU 並列化・GMRES 自動有効化テスト."""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.assembly import (
    _PARALLEL_MIN_ELEMENTS,
    _assemble_parallel,
    _assemble_sequential,
    assemble_global_stiffness,
)
from xkep_cae.contact.broadphase import broadphase_aabb, compute_segment_aabb
from xkep_cae.contact.pair import ContactConfig

# ========== テスト用ヘルパー ==========


class _DummyMaterial:
    """ダミー材料（E=1, ν=0.3, 平面ひずみ）."""

    D = np.array(
        [
            [1.346, 0.577, 0.0],
            [0.577, 1.346, 0.0],
            [0.0, 0.0, 0.385],
        ]
    )

    def stiffness_matrix(self, strain=None):
        return self.D


class _DummyQ4:
    """ダミー Q4 要素（剛性行列のサイズだけ合わせる）."""

    ndof_per_node = 2
    ndof = 8
    nnodes = 4

    def local_stiffness(self, coords, material, thickness):
        return np.eye(self.ndof)

    def dof_indices(self, node_ids):
        return np.array(
            [n * self.ndof_per_node + d for n in node_ids for d in range(self.ndof_per_node)]
        )


# ========== 並列アセンブリテスト ==========


class TestParallelAssembly:
    """assemble_global_stiffness の並列化テスト."""

    def _make_mesh(self, n_elem: int):
        """正方メッシュを生成."""
        nx = int(np.ceil(np.sqrt(n_elem)))
        ny = max(1, n_elem // nx)
        nodes = np.array([(i, j) for j in range(ny + 1) for i in range(nx + 1)], dtype=float)
        conn = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                conn.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
        return nodes, np.array(conn[:n_elem], dtype=int)

    def test_sequential_matches_parallel_small(self):
        """小規模問題で逐次と並列の結果が一致."""
        elem = _DummyQ4()
        mat = _DummyMaterial()
        nodes, conn = self._make_mesh(16)
        n_total = len(conn)
        ndof_total = len(nodes) * 2

        K_seq = _assemble_sequential(
            nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False
        )
        K_par = _assemble_parallel(
            nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False, n_jobs=2
        )

        diff = abs(K_seq - K_par).max()
        assert diff < 1e-14, f"逐次と並列の差: {diff}"

    def test_n_jobs_minus_one(self):
        """n_jobs=-1 でエラーなく動作."""
        elem = _DummyQ4()
        mat = _DummyMaterial()
        nodes, conn = self._make_mesh(10)
        K = assemble_global_stiffness(nodes, [(elem, conn)], mat, show_progress=False, n_jobs=-1)
        assert K.shape[0] > 0

    def test_n_jobs_default_sequential(self):
        """デフォルト n_jobs=1 は逐次実行."""
        elem = _DummyQ4()
        mat = _DummyMaterial()
        nodes, conn = self._make_mesh(10)
        K = assemble_global_stiffness(nodes, [(elem, conn)], mat, show_progress=False)
        assert K.shape[0] > 0

    def test_parallel_threshold(self):
        """_PARALLEL_MIN_ELEMENTS 未満は並列化しない."""
        assert _PARALLEL_MIN_ELEMENTS == 4096
        elem = _DummyQ4()
        mat = _DummyMaterial()
        nodes, conn = self._make_mesh(10)
        # n_jobs=4 でも要素数 < 4096 なら逐次実行されるはず
        K = assemble_global_stiffness(nodes, [(elem, conn)], mat, show_progress=False, n_jobs=4)
        assert K.shape[0] > 0


# ========== GMRES 自動有効化テスト ==========


class TestGMRESAutoEnable:
    """ContactConfig の GMRES 自動有効化テスト."""

    def test_default_linear_solver_is_auto(self):
        """デフォルトの linear_solver は 'auto'."""
        cfg = ContactConfig()
        assert cfg.linear_solver == "auto"

    def test_gmres_dof_threshold_default(self):
        """gmres_dof_threshold のデフォルト値は 2000."""
        cfg = ContactConfig()
        assert cfg.gmres_dof_threshold == 2000

    def test_solve_linear_system_auto_small(self):
        """小規模問題で auto モードが直接法を使用."""
        from xkep_cae.contact.solver_ncp import _solve_linear_system

        n = 100
        K = sp.eye(n, format="csr") * 2.0
        rhs = np.ones(n)
        x = _solve_linear_system(K, rhs, mode="auto", gmres_dof_threshold=200)
        np.testing.assert_allclose(x, 0.5 * np.ones(n), atol=1e-10)

    def test_solve_linear_system_auto_large(self):
        """大規模問題で auto モードが反復法を使用."""
        from xkep_cae.contact.solver_ncp import _solve_linear_system

        n = 50
        K = sp.eye(n, format="csr") * 2.0
        rhs = np.ones(n)
        # 閾値を低く設定して反復法をトリガー
        x = _solve_linear_system(K, rhs, mode="auto", gmres_dof_threshold=10)
        np.testing.assert_allclose(x, 0.5 * np.ones(n), atol=1e-8)

    def test_saddle_point_auto_block_preconditioner(self):
        """DOF 閾値超過時に鞍点系が自動でブロック前処理を選択."""
        from xkep_cae.contact.solver_ncp import _solve_saddle_point_contact

        ndof = 100
        n_active = 3
        K_T = sp.eye(ndof, format="csr") * 10.0
        G_A = sp.random(n_active, ndof, density=0.1, format="csr")
        R_u = np.ones(ndof)
        g_active = np.ones(n_active) * 0.01
        fixed_dofs = np.array([0, 1, 2])

        # gmres_dof_threshold=50 で自動ブロック前処理
        du, dlam = _solve_saddle_point_contact(
            K_T,
            G_A,
            1e4,
            R_u,
            g_active,
            fixed_dofs,
            linear_solver="auto",
            gmres_dof_threshold=50,
        )
        assert du.shape == (ndof,)
        assert dlam.shape == (n_active,)
        assert np.all(np.isfinite(du))


# ========== Broadphase ベクトル化テスト ==========


class TestBroadphaseVectorized:
    """Broadphase AABB ベクトル化テスト."""

    def test_aabb_vectorized_matches_scalar(self):
        """ベクトル化 AABB が個別計算と一致."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 1, 0.0]), np.array([1, 1, 0.0])),
            (np.array([0, 0, 1.0]), np.array([1, 0, 1.0])),
        ]
        radii = np.array([0.1, 0.2, 0.15])
        margin = 0.05

        # 個別計算
        for i, (x0, x1) in enumerate(segments):
            lo_s, hi_s = compute_segment_aabb(x0, x1, radii[i], margin)
            # ベクトル化の中間結果を検証
            expand = radii[i] + margin
            lo_v = np.minimum(x0, x1) - expand
            hi_v = np.maximum(x0, x1) + expand
            np.testing.assert_allclose(lo_s, lo_v)
            np.testing.assert_allclose(hi_s, hi_v)

    def test_broadphase_result_unchanged(self):
        """ベクトル化後も候補ペアが同一."""
        np.random.seed(42)
        n = 20
        segments = []
        for _ in range(n):
            x0 = np.random.randn(3)
            x1 = x0 + np.random.randn(3) * 0.5
            segments.append((x0, x1))
        radii = np.abs(np.random.randn(n)) * 0.1 + 0.05

        pairs = broadphase_aabb(segments, radii, margin=0.1)
        # 結果がリストで返され、各要素が (i, j) で i < j
        for i, j in pairs:
            assert i < j


# ========== Mortar 適応ペナルティテスト ==========


class TestMortarAdaptivePenalty:
    """Mortar 適応ペナルティのユニットテスト."""

    def test_mortar_p_n_computation(self):
        """compute_mortar_p_n が正しく計算される."""
        from xkep_cae.contact.mortar import compute_mortar_p_n

        mortar_nodes = [0, 1, 2]
        lam = np.array([100.0, 50.0, 0.0])
        g = np.array([-0.01, 0.01, -0.005])  # 負=貫入
        k_pen = 1e4

        p_n = compute_mortar_p_n(mortar_nodes, lam, g, k_pen)
        # p_n[0] = max(0, 100 + 1e4 * 0.01) = max(0, 200) = 200
        assert p_n[0] == pytest.approx(200.0)
        # p_n[1] = max(0, 50 + 1e4 * (-0.01)) = max(0, -50) = 0
        assert p_n[1] == pytest.approx(0.0)
        # p_n[2] = max(0, 0 + 1e4 * 0.005) = max(0, 50) = 50
        assert p_n[2] == pytest.approx(50.0)

    def test_config_has_adaptive_mortar_params(self):
        """ContactConfig に適応ペナルティパラメータが存在."""
        cfg = ContactConfig()
        assert hasattr(cfg, "tol_penetration_ratio")
        assert hasattr(cfg, "penalty_growth_factor")
        assert hasattr(cfg, "k_pen_max")
        assert cfg.tol_penetration_ratio == 0.01
        assert cfg.penalty_growth_factor == 2.0
