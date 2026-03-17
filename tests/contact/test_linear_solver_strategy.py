"""LinearSolverStrategy のテスト.

DirectLinearSolver / IterativeLinearSolver / AutoLinearSolver の
Protocol 適合性と基本的な求解精度を検証する。
"""

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.skip(reason="__xkep_cae_deprecated 参照のため無効化 (status-193)")

from __xkep_cae_deprecated.process.strategies.linear_solver import (  # noqa: E402
    AutoLinearSolver,
    DirectLinearSolver,
    IterativeLinearSolver,
    create_linear_solver,
)
from xkep_cae.core.strategies.protocols import LinearSolverStrategy  # noqa: E402


def _make_spd_system(n: int = 50, seed: int = 42):
    """テスト用の正定値対称疎行列と右辺ベクトルを生成."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.3, format="csr", random_state=rng)
    K = (A + A.T) + n * sp.eye(n, format="csr")
    x_true = rng.standard_normal(n)
    rhs = K @ x_true
    return K, rhs, x_true


class TestLinearSolverProtocol:
    """Protocol 適合性のテスト."""

    def test_direct_is_strategy(self):
        assert isinstance(DirectLinearSolver(), LinearSolverStrategy)

    def test_iterative_is_strategy(self):
        assert isinstance(IterativeLinearSolver(), LinearSolverStrategy)

    def test_auto_is_strategy(self):
        assert isinstance(AutoLinearSolver(), LinearSolverStrategy)


class TestDirectLinearSolver:
    """DirectLinearSolver の求解精度テスト."""

    def test_solve_small_system(self):
        K, rhs, x_true = _make_spd_system(20)
        solver = DirectLinearSolver()
        x = solver.solve(K, rhs)
        np.testing.assert_allclose(x, x_true, atol=1e-10)

    def test_solve_medium_system(self):
        K, rhs, x_true = _make_spd_system(200)
        solver = DirectLinearSolver()
        x = solver.solve(K, rhs)
        np.testing.assert_allclose(x, x_true, atol=1e-8)


class TestIterativeLinearSolver:
    """IterativeLinearSolver の求解精度テスト."""

    def test_solve_small_system(self):
        K, rhs, x_true = _make_spd_system(20)
        solver = IterativeLinearSolver()
        x = solver.solve(K, rhs)
        np.testing.assert_allclose(x, x_true, atol=1e-6)

    def test_solve_medium_system(self):
        K, rhs, x_true = _make_spd_system(200)
        solver = IterativeLinearSolver()
        x = solver.solve(K, rhs)
        np.testing.assert_allclose(x, x_true, atol=1e-6)


class TestAutoLinearSolver:
    """AutoLinearSolver の自動選択テスト."""

    def test_small_uses_direct(self):
        """小規模問題で直接法が使われる."""
        K, rhs, x_true = _make_spd_system(20)
        solver = AutoLinearSolver(gmres_dof_threshold=100)
        x = solver.solve(K, rhs)
        np.testing.assert_allclose(x, x_true, atol=1e-10)

    def test_large_uses_iterative(self):
        """閾値以上で反復法に切り替わる."""
        K, rhs, x_true = _make_spd_system(200)
        solver = AutoLinearSolver(gmres_dof_threshold=100)
        x = solver.solve(K, rhs)
        np.testing.assert_allclose(x, x_true, atol=1e-6)


class TestCreateLinearSolver:
    """ファクトリ関数のテスト."""

    def test_create_direct(self):
        solver = create_linear_solver(mode="direct")
        assert isinstance(solver, DirectLinearSolver)

    def test_create_iterative(self):
        solver = create_linear_solver(mode="iterative")
        assert isinstance(solver, IterativeLinearSolver)

    def test_create_auto(self):
        solver = create_linear_solver(mode="auto")
        assert isinstance(solver, AutoLinearSolver)

    def test_create_auto_default(self):
        solver = create_linear_solver()
        assert isinstance(solver, AutoLinearSolver)

    def test_iterative_params_passed(self):
        solver = create_linear_solver(mode="iterative", iterative_tol=1e-8, ilu_drop_tol=1e-3)
        assert solver.iterative_tol == 1e-8
        assert solver.ilu_drop_tol == 1e-3

    def test_auto_params_passed(self):
        solver = create_linear_solver(mode="auto", gmres_dof_threshold=5000)
        assert solver.gmres_dof_threshold == 5000
