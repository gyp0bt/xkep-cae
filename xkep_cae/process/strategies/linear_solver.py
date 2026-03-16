"""LinearSolverStrategy 具象実装.

線形連立方程式 K x = rhs の解法を Strategy パターンで分離する。
solver_ncp.py の _solve_linear_system() if 分岐を Strategy 化したもの。

- DirectLinearSolver: scipy.sparse.linalg.spsolve 直接法
- IterativeLinearSolver: GMRES + ILU 前処理（大規模向け）
- AutoLinearSolver: DOF 閾値ベースで自動選択（デフォルト）
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class DirectLinearSolver:
    """直接法 (spsolve) による線形ソルバー."""

    def solve(self, K: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """spsolve で直接解く."""
        return spla.spsolve(K, rhs)


class IterativeLinearSolver:
    """GMRES + ILU 前処理による反復法線形ソルバー.

    Args:
        iterative_tol: GMRES 収束判定トレランス
        ilu_drop_tol: ILU 前処理の drop tolerance
    """

    def __init__(
        self,
        iterative_tol: float = 1e-10,
        ilu_drop_tol: float = 1e-4,
    ) -> None:
        self.iterative_tol = iterative_tol
        self.ilu_drop_tol = ilu_drop_tol

    def solve(self, K: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """GMRES + ILU で解く。ILU 構築失敗時は前処理なし、GMRES 非収束時は直接法に fallback."""
        K_csc = K.tocsc()
        try:
            ilu = spla.spilu(K_csc, drop_tol=self.ilu_drop_tol)
            M = spla.LinearOperator(K.shape, ilu.solve)
        except RuntimeError:
            M = None
        _restart_k = min(max(30, K.shape[0] // 10), 200)
        x, info = spla.gmres(
            K,
            rhs,
            M=M,
            atol=self.iterative_tol,
            restart=_restart_k,
            maxiter=max(500, K.shape[0]),
        )
        if info != 0:
            x = spla.spsolve(K, rhs)
        return x


class AutoLinearSolver:
    """DOF 閾値ベースの自動選択線形ソルバー.

    - DOF < gmres_dof_threshold: 直接法（特異行列検出時は反復法 fallback）
    - DOF >= gmres_dof_threshold: 反復法

    Args:
        iterative_tol: GMRES 収束判定トレランス
        ilu_drop_tol: ILU 前処理の drop tolerance
        gmres_dof_threshold: 反復法に切り替える DOF 閾値
    """

    def __init__(
        self,
        iterative_tol: float = 1e-10,
        ilu_drop_tol: float = 1e-4,
        gmres_dof_threshold: int = 2000,
    ) -> None:
        self.iterative_tol = iterative_tol
        self.ilu_drop_tol = ilu_drop_tol
        self.gmres_dof_threshold = gmres_dof_threshold
        self._iterative = IterativeLinearSolver(
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
        )

    def solve(self, K: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """DOF 閾値で直接法/反復法を自動選択して解く."""
        n = K.shape[0]
        if n >= self.gmres_dof_threshold:
            return self._iterative.solve(K, rhs)

        # 小規模: direct → iterative fallback
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x = spla.spsolve(K, rhs)
        for w in caught:
            if (
                "MatrixRankWarning" in str(w.category.__name__)
                or "singular" in str(w.message).lower()
            ):
                return self._iterative.solve(K, rhs)
        if not np.all(np.isfinite(x)):
            return self._iterative.solve(K, rhs)
        return x


def create_linear_solver(
    mode: str = "auto",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
    gmres_dof_threshold: int = 2000,
) -> DirectLinearSolver | IterativeLinearSolver | AutoLinearSolver:
    """ContactConfig のパラメータから LinearSolverStrategy を生成する.

    Args:
        mode: "direct" | "iterative" | "auto"
        iterative_tol: GMRES 収束判定トレランス
        ilu_drop_tol: ILU 前処理の drop tolerance
        gmres_dof_threshold: auto モードでの DOF 閾値

    Returns:
        LinearSolverStrategy を満たすインスタンス
    """
    if mode == "direct":
        return DirectLinearSolver()
    if mode == "iterative":
        return IterativeLinearSolver(
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
        )
    return AutoLinearSolver(
        iterative_tol=iterative_tol,
        ilu_drop_tol=ilu_drop_tol,
        gmres_dof_threshold=gmres_dof_threshold,
    )
