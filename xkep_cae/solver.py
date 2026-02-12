from __future__ import annotations

import time
from typing import Any

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
