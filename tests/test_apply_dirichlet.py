"""apply_dirichlet のテスト — 特に非ゼロ規定変位のスパース行列処理."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.bc import apply_dirichlet


def _make_tridiag(n: int) -> sp.csr_matrix:
    """n×n の三重対角行列（バネ系）を生成."""
    diag_main = 2.0 * np.ones(n)
    diag_off = -1.0 * np.ones(n - 1)
    K = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csr")
    return K


def _solve_dense_reference(K_dense, f, fixed_dofs, values):
    """密行列で Dirichlet BC を適用して基準解を求める."""
    K_bc = K_dense.copy()
    f_bc = f.copy()
    for d, v in zip(fixed_dofs, values, strict=True):
        f_bc -= K_bc[:, d] * v
        K_bc[:, d] = 0.0
        K_bc[d, :] = 0.0
        K_bc[d, d] = 1.0
        f_bc[d] = v
    u = np.linalg.solve(K_bc, f_bc)
    return u


class TestApplyDirichletZero:
    """ゼロ規定変位（従来通り動作するはず）."""

    def test_zero_bc_simple(self):
        n = 5
        K = _make_tridiag(n)
        f = np.ones(n)
        fixed_dofs = np.array([0, n - 1])
        result = apply_dirichlet(K, f, fixed_dofs, values=0.0)
        u = sp.linalg.spsolve(result.K, result.f)
        assert u[0] == pytest.approx(0.0)
        assert u[n - 1] == pytest.approx(0.0)

    def test_zero_bc_matches_dense(self):
        n = 8
        K = _make_tridiag(n)
        f = np.random.default_rng(42).standard_normal(n)
        fixed_dofs = np.array([0, 3, 7])
        values = np.zeros(3)
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u_sparse = sp.linalg.spsolve(result.K, result.f)
        u_dense = _solve_dense_reference(K.toarray(), f, fixed_dofs, values)
        np.testing.assert_allclose(u_sparse, u_dense, atol=1e-12)


class TestApplyDirichletNonzero:
    """非ゼロ規定変位 — 旧実装でバグが発生していたケース."""

    def test_single_nonzero_prescribed(self):
        """単一DOFに非ゼロ規定変位を適用."""
        n = 5
        K = _make_tridiag(n)
        f = np.zeros(n)
        fixed_dofs = np.array([0])
        values = np.array([1.0])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u = sp.linalg.spsolve(result.K, result.f)
        assert u[0] == pytest.approx(1.0)

    def test_multiple_nonzero_prescribed(self):
        """複数DOFに非ゼロ規定変位を適用 — バグの主要トリガー."""
        n = 6
        K = _make_tridiag(n)
        f = np.zeros(n)
        fixed_dofs = np.array([0, 5])
        values = np.array([1.0, -2.0])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u = sp.linalg.spsolve(result.K, result.f)
        assert u[0] == pytest.approx(1.0)
        assert u[5] == pytest.approx(-2.0)

    def test_nonzero_matches_dense(self):
        """スパース版と密行列版の結果が一致することを検証."""
        n = 10
        K = _make_tridiag(n)
        f = np.ones(n) * 5.0
        fixed_dofs = np.array([0, 4, 9])
        values = np.array([0.5, -1.0, 2.0])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u_sparse = sp.linalg.spsolve(result.K, result.f)
        u_dense = _solve_dense_reference(K.toarray(), f, fixed_dofs, values)
        np.testing.assert_allclose(u_sparse, u_dense, atol=1e-12)

    def test_nonzero_with_coupled_dofs(self):
        """隣接する（カップリングのある）DOF同士に非ゼロ規定変位."""
        n = 6
        K = _make_tridiag(n)
        f = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        # DOF 2 と DOF 3 は隣接（K[2,3] != 0, K[3,2] != 0）
        fixed_dofs = np.array([2, 3])
        values = np.array([0.5, -0.5])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u_sparse = sp.linalg.spsolve(result.K, result.f)
        u_dense = _solve_dense_reference(K.toarray(), f, fixed_dofs, values)
        np.testing.assert_allclose(u_sparse, u_dense, atol=1e-12)
        assert u_sparse[2] == pytest.approx(0.5)
        assert u_sparse[3] == pytest.approx(-0.5)

    def test_mixed_zero_and_nonzero(self):
        """ゼロと非ゼロの混在する規定変位."""
        n = 8
        K = _make_tridiag(n)
        f = np.ones(n)
        fixed_dofs = np.array([0, 3, 7])
        values = np.array([0.0, 1.5, 0.0])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u_sparse = sp.linalg.spsolve(result.K, result.f)
        u_dense = _solve_dense_reference(K.toarray(), f, fixed_dofs, values)
        np.testing.assert_allclose(u_sparse, u_dense, atol=1e-12)
        assert u_sparse[0] == pytest.approx(0.0)
        assert u_sparse[3] == pytest.approx(1.5)
        assert u_sparse[7] == pytest.approx(0.0)


class TestApplyDirichletFEM:
    """FEM的な剛性行列での非ゼロ規定変位テスト."""

    def test_beam_displacement_control(self):
        """梁の変位制御問題: 片端固定、他端に非ゼロ変位を指定."""
        # 3要素 2DOF/node (ux, uy) の簡易モデル
        n = 8  # 4 nodes × 2 DOF
        K_dense = np.zeros((n, n))
        # 各要素に簡易的な剛性を設定
        ke = np.array(
            [[12, 6, -12, 6], [6, 4, -6, 2], [-12, -6, 12, -6], [6, 2, -6, 4]],
            dtype=float,
        )
        for e in range(3):
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    K_dense[di, dj] += ke[i, j]

        K = sp.csr_matrix(K_dense)
        f = np.zeros(n)

        # node 0: 全拘束 (ux=0, uy=0), node 3: uy=1.0 (変位制御)
        fixed_dofs = np.array([0, 1, 7])
        values = np.array([0.0, 0.0, 1.0])

        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u_sparse = sp.linalg.spsolve(result.K, result.f)
        u_dense = _solve_dense_reference(K_dense, f, fixed_dofs, values)
        np.testing.assert_allclose(u_sparse, u_dense, atol=1e-12)
        assert u_sparse[0] == pytest.approx(0.0)
        assert u_sparse[1] == pytest.approx(0.0)
        assert u_sparse[7] == pytest.approx(1.0)

    def test_large_system_nonzero(self):
        """大きめの系で非ゼロ規定変位が正しいことを検証."""
        n = 100
        rng = np.random.default_rng(123)
        # ランダムな対称正定値行列
        A = rng.standard_normal((n, n))
        K_dense = A @ A.T + 10.0 * np.eye(n)
        K = sp.csr_matrix(K_dense)
        f = rng.standard_normal(n)

        fixed_dofs = np.array([0, 25, 50, 75, 99])
        values = np.array([1.0, -0.5, 2.0, 0.0, -1.0])

        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        u_sparse = sp.linalg.spsolve(result.K, result.f)
        u_dense = _solve_dense_reference(K_dense, f, fixed_dofs, values)
        np.testing.assert_allclose(u_sparse, u_dense, atol=1e-10)
        for dof, val in zip(fixed_dofs, values, strict=True):
            assert u_sparse[dof] == pytest.approx(val, abs=1e-12)


class TestApplyDirichletMatrixProperties:
    """適用後の行列の性質を確認."""

    def test_diagonal_is_one(self):
        """固定DOFの対角要素が1であること."""
        n = 6
        K = _make_tridiag(n)
        f = np.ones(n)
        fixed_dofs = np.array([0, 3, 5])
        values = np.array([1.0, -1.0, 0.5])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        K_out = result.K.toarray()
        for dof in fixed_dofs:
            assert K_out[dof, dof] == pytest.approx(1.0)

    def test_row_col_zero_except_diagonal(self):
        """固定DOFの行・列が対角以外ゼロであること."""
        n = 6
        K = _make_tridiag(n)
        f = np.ones(n)
        fixed_dofs = np.array([1, 4])
        values = np.array([2.0, -3.0])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        K_out = result.K.toarray()
        for dof in fixed_dofs:
            row = K_out[dof, :].copy()
            col = K_out[:, dof].copy()
            row[dof] = 0.0
            col[dof] = 0.0
            np.testing.assert_allclose(row, 0.0, atol=1e-15)
            np.testing.assert_allclose(col, 0.0, atol=1e-15)

    def test_rhs_equals_prescribed(self):
        """固定DOFの右辺が規定値と一致すること."""
        n = 6
        K = _make_tridiag(n)
        f = np.ones(n)
        fixed_dofs = np.array([0, 2, 5])
        values = np.array([1.5, -0.5, 3.0])
        result = apply_dirichlet(K, f, fixed_dofs, values=values)
        for dof, val in zip(fixed_dofs, values, strict=True):
            assert result.f[dof] == pytest.approx(val)
