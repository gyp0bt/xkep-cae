"""線形・非線形ソルバーモジュール.

線形:
  - solve_displacement(): pyamg / spsolve 適応選択

非線形:
  - newton_raphson(): 荷重増分 + Newton-Raphson 法
  - Newton-Raphson の収束判定: 力ノルム / 変位ノルム / エネルギーノルム
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_displacement(
    K: sp.csr_matrix,
    f: np.ndarray,
    *,
    rtol: float = 1e-8,
    maxiter: int = 100000,
    size_threshold: int = 2000,
    show_progress: bool = True,
    use_pyamg: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """pyamgを主体としたソルバ。小規模はspsolveにフォールバック。

    Args:
        K: CSR剛性行列（SPD前提）
        f: 右辺ベクトル
        rtol: pyamgの収束tol
        maxiter: pyamg反復上限（V-cycle回数）
        size_threshold: これ未満の規模はspsolveで直接解く
        show_progress: セットアップ/ソルブ時間を表示

    Returns:
        u: 解ベクトル
        info: method, nit, residual_norm, setup_time, solve_time など
    """
    n = K.shape[0]
    info: dict[str, Any] = {
        "method": None,
        "nit": None,
        "success": True,
        "residual_norm": None,
        "setup_time": None,
        "solve_time": None,
    }

    # 小さい問題は素直に直達
    if n < size_threshold:
        t0 = time.time()
        u = spla.spsolve(K, f)
        elapsed = time.time() - t0
        res = K @ u - f
        info["method"] = "spsolve"
        info["nit"] = 1
        info["residual_norm"] = float(np.linalg.norm(res))
        info["setup_time"] = 0.0
        info["solve_time"] = elapsed
        if show_progress:
            print(f"[spsolve] n={n}, nnz={K.nnz}, elapsed={elapsed:.3f} s")
        return u, info

    # ここからpyamg主体
    try:
        import pyamg  # type: ignore
    except Exception:
        use_pyamg = False

    if not use_pyamg:
        # pyamgが使えない環境ではspsolveで落とす
        t0 = time.time()
        u = spla.spsolve(K, f)
        elapsed = time.time() - t0
        res = K @ u - f
        info["method"] = "spsolve(no-pyamg)"
        info["nit"] = 1
        info["residual_norm"] = float(np.linalg.norm(res))
        info["setup_time"] = 0.0
        info["solve_time"] = elapsed
        if show_progress:
            print(f"[spsolve(no-pyamg)] n={n}, nnz={K.nnz}, elapsed={elapsed:.3f} s")
        return u, info

    # pyamg setup
    t0 = time.time()
    ml = pyamg.smoothed_aggregation_solver(
        K,
        symmetry="symmetric",
        presmoother=("gauss_seidel", {"sweep": "symmetric"}),
        postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
    )
    setup_time = time.time() - t0

    # solve (V-cycle) + 残差ログ
    residuals: list[float] = []
    t1 = time.time()
    u = ml.solve(
        b=f,
        tol=rtol,
        maxiter=maxiter,
        cycle="V",
        residuals=residuals,
    )
    solve_time = time.time() - t1

    res_norm = float(residuals[-1]) if residuals else float(np.linalg.norm(K @ u - f))

    info["method"] = "pyamg-V"
    info["nit"] = len(residuals)
    info["success"] = res_norm <= rtol
    info["residual_norm"] = res_norm
    info["setup_time"] = setup_time
    info["solve_time"] = solve_time

    if show_progress:
        print(
            f"[pyamg-V] n={n}, nnz={K.nnz}, it={info['nit']}, "
            f"res={res_norm:.3e}, setup={setup_time:.3f}s, solve={solve_time:.3f}s"
        )

    return u, info


# ====================================================================
# 非線形ソルバー: Newton-Raphson 法
# ====================================================================

@dataclass
class NonlinearResult:
    """非線形解析の結果.

    Attributes:
        u: (ndof,) 最終変位ベクトル
        converged: 収束したかどうか
        n_load_steps: 荷重増分ステップ数
        total_iterations: 全ステップの合計Newton反復回数
        load_history: 各ステップの荷重係数の履歴
        displacement_history: 各ステップの変位ベクトルの履歴
        residual_history: 各ステップの最終残差ノルム
    """

    u: np.ndarray
    converged: bool
    n_load_steps: int
    total_iterations: int
    load_history: list[float] = field(default_factory=list)
    displacement_history: list[np.ndarray] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)


def newton_raphson(
    f_ext_total: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    *,
    n_load_steps: int = 10,
    max_iter: int = 30,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    tol_energy: float = 1e-10,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    fixed_values: float | np.ndarray = 0.0,
) -> NonlinearResult:
    """荷重増分 + Newton-Raphson 法による非線形静解析.

    荷重を n_load_steps に等分割し、各ステップで Newton-Raphson 反復を行う。

    残差: R(u) = λ·f_ext - f_int(u)
    接線: K_T(u) · Δu = R(u)
    更新: u ← u + Δu

    収束判定（いずれかを満たせば収束）:
      - ||R|| / ||f_ext|| < tol_force   （力ノルム）
      - ||Δu|| / ||u|| < tol_disp      （変位ノルム）
      - |Δu · R| / |Δu₀ · R₀| < tol_energy  （エネルギーノルム）

    Args:
        f_ext_total: (ndof,) 最終荷重ベクトル（全荷重）
        fixed_dofs: 拘束DOFの配列
        assemble_tangent: u → K_T(u) を返すコールバック（CSR行列）
        assemble_internal_force: u → f_int(u) を返すコールバック
        n_load_steps: 荷重増分ステップ数
        max_iter: 各ステップの最大Newton反復回数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_energy: エネルギーノルム収束判定
        show_progress: 進捗表示
        u0: 初期変位（None = ゼロ）
        fixed_values: 拘束変位値

    Returns:
        NonlinearResult: 解析結果
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    fixed_set = set(fixed_dofs.tolist())
    free_dofs = np.array([i for i in range(ndof) if i not in fixed_set])

    if u0 is not None:
        u = u0.copy()
    else:
        u = np.zeros(ndof, dtype=float)

    # 荷重増分
    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    res_history: list[float] = []
    total_iter = 0

    f_ext_ref_norm = np.linalg.norm(f_ext_total)
    if f_ext_ref_norm < 1e-30:
        f_ext_ref_norm = 1.0

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        f_ext = lam * f_ext_total

        converged_step = False
        energy_ref = None
        res_norm = 0.0

        for it in range(max_iter):
            total_iter += 1

            # 内力と残差
            f_int = assemble_internal_force(u)
            residual = f_ext - f_int

            # 拘束DOFの残差をゼロに
            residual[fixed_dofs] = 0.0

            res_norm = float(np.linalg.norm(residual))

            # 力ノルム判定
            if res_norm / f_ext_ref_norm < tol_force:
                converged_step = True
                if show_progress:
                    print(
                        f"  Step {step}/{n_load_steps}, "
                        f"iter {it}, ||R||/||f|| = {res_norm / f_ext_ref_norm:.3e} (force converged)"
                    )
                break

            # 接線剛性の組み立てと求解
            K_T = assemble_tangent(u)

            # 境界条件適用（行列消去法）
            K_bc = K_T.tolil()
            r_bc = residual.copy()
            for dof in fixed_dofs:
                K_bc[dof, :] = 0.0
                K_bc[:, dof] = 0.0
                K_bc[dof, dof] = 1.0
                r_bc[dof] = 0.0

            # 線形求解
            du = spla.spsolve(K_bc.tocsr(), r_bc)

            # エネルギーノルム判定
            energy = abs(float(np.dot(du, residual)))
            if energy_ref is None:
                energy_ref = energy if energy > 1e-30 else 1.0

            # 変位ノルム
            u_norm = float(np.linalg.norm(u))
            du_norm = float(np.linalg.norm(du))

            if show_progress and it % 5 == 0:
                print(
                    f"  Step {step}/{n_load_steps}, iter {it}, "
                    f"||R||/||f|| = {res_norm / f_ext_ref_norm:.3e}, "
                    f"||du||/||u|| = {du_norm / max(u_norm, 1e-30):.3e}"
                )

            # 更新
            u += du

            # 収束判定（更新後）
            if u_norm > 1e-30 and du_norm / u_norm < tol_disp:
                converged_step = True
                if show_progress:
                    print(
                        f"  Step {step}/{n_load_steps}, "
                        f"iter {it}, ||du||/||u|| = {du_norm / u_norm:.3e} (disp converged)"
                    )
                break

            if energy / energy_ref < tol_energy:
                converged_step = True
                if show_progress:
                    print(
                        f"  Step {step}/{n_load_steps}, "
                        f"iter {it}, energy ratio = {energy / energy_ref:.3e} (energy converged)"
                    )
                break

        if not converged_step:
            if show_progress:
                print(
                    f"  WARNING: Step {step} did not converge in {max_iter} iterations. "
                    f"||R||/||f|| = {res_norm / f_ext_ref_norm:.3e}"
                )
            return NonlinearResult(
                u=u,
                converged=False,
                n_load_steps=step,
                total_iterations=total_iter,
                load_history=load_history,
                displacement_history=disp_history,
                residual_history=res_history,
            )

        load_history.append(lam)
        disp_history.append(u.copy())
        res_history.append(res_norm)

    return NonlinearResult(
        u=u,
        converged=True,
        n_load_steps=n_load_steps,
        total_iterations=total_iter,
        load_history=load_history,
        displacement_history=disp_history,
        residual_history=res_history,
    )
