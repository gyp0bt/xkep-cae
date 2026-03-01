"""線形・非線形ソルバーモジュール.

線形:
  - solve_displacement(): pyamg / spsolve 適応選択

非線形:
  - newton_raphson(): 荷重増分 + Newton-Raphson 法
  - arc_length(): 円筒弧長法 (Crisfield, 1981)
  - Newton-Raphson の収束判定: 力ノルム / 変位ノルム / エネルギーノルム
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from xkep_cae.core.results import LinearSolveResult


def solve_displacement(
    K: sp.csr_matrix,
    f: np.ndarray,
    *,
    rtol: float = 1e-8,
    maxiter: int = 100000,
    size_threshold: int = 2000,
    show_progress: bool = True,
    use_pyamg: bool = True,
) -> LinearSolveResult:
    """pyamgを主体としたソルバ。小規模はspsolveにフォールバック。

    Args:
        K: CSR剛性行列（SPD前提）
        f: 右辺ベクトル
        rtol: pyamgの収束tol
        maxiter: pyamg反復上限（V-cycle回数）
        size_threshold: これ未満の規模はspsolveで直接解く
        show_progress: セットアップ/ソルブ時間を表示

    Returns:
        LinearSolveResult: (u, info) の NamedTuple
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
        return LinearSolveResult(u=u, info=info)

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
        return LinearSolveResult(u=u, info=info)

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

    return LinearSolveResult(u=u, info=info)


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
    tangent_update_interval: int = 1,
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

    修正NR法:
      tangent_update_interval > 1 の場合、接線剛性を毎反復ではなく
      指定間隔でのみ再計算する。反復数は増えるが各反復のコストが削減される。

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
        tangent_update_interval: 接線剛性更新間隔。1=完全NR（毎反復更新）、
            2以上=修正NR（N反復ごとに更新）。大規模問題で有効。

    Returns:
        NonlinearResult: 解析結果
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
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
        K_bc_cached = None  # 修正NR用: キャッシュされたBC適用済み剛性行列

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

            # 接線剛性の組み立て（修正NR: interval に応じて再計算）
            if K_bc_cached is None or it % tangent_update_interval == 0:
                K_T = assemble_tangent(u)
                K_bc_cached, _ = _apply_bc(K_T, residual, fixed_dofs)

            # 右辺のみBC適用
            r_bc = residual.copy()
            r_bc[fixed_dofs] = 0.0

            # 線形求解
            du = spla.spsolve(K_bc_cached, r_bc)

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


# ====================================================================
# 非線形ソルバー: 円筒弧長法 (Crisfield, 1981)
# ====================================================================


@dataclass
class ArcLengthResult:
    """弧長法解析の結果.

    Attributes:
        u: (ndof,) 最終変位ベクトル
        lam: 最終荷重係数 λ
        converged: 全ステップが収束したかどうか
        n_steps: 実行ステップ数
        total_iterations: 全ステップの合計 Newton 反復回数
        load_history: 各ステップの荷重係数 λ の履歴
        displacement_history: 各ステップの変位ベクトルの履歴
    """

    u: np.ndarray
    lam: float
    converged: bool
    n_steps: int
    total_iterations: int
    load_history: list[float] = field(default_factory=list)
    displacement_history: list[np.ndarray] = field(default_factory=list)


def _apply_bc(
    K: sp.spmatrix,
    r: np.ndarray,
    fixed_dofs: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """境界条件適用（ベクトル化行列消去法）.

    tolil() + Python ループを CSR 直接操作に置換し高速化。
    """
    K_bc = K.tocsr().copy()
    r_bc = r.copy()

    if len(fixed_dofs) == 0:
        return K_bc, r_bc

    # 行の消去（CSR 行操作は効率的）
    for dof in fixed_dofs:
        K_bc[dof, :] = 0.0
    # 列の消去（CSC に変換して効率的に列操作）
    K_csc = K_bc.tocsc()
    for dof in fixed_dofs:
        K_csc[:, dof] = 0.0
    K_bc = K_csc.tocsr()
    # 対角に 1.0 を設定
    K_bc[fixed_dofs, fixed_dofs] = 1.0
    K_bc.eliminate_zeros()

    r_bc[fixed_dofs] = 0.0
    return K_bc, r_bc


def arc_length(
    f_ext_ref: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    *,
    n_steps: int = 50,
    delta_l: float | None = None,
    max_iter: int = 30,
    tol_force: float = 1e-8,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    lambda0: float = 0.0,
    lambda_max: float | None = None,
    max_cutbacks: int = 5,
) -> ArcLengthResult:
    """円筒弧長法 (Crisfield cylindrical arc-length) による非線形静解析.

    荷重係数 λ をも未知数として扱い、弧長拘束により
    リミットポイント（座屈・スナップスルー）を追跡する。

    弧長拘束（円筒型）:
      ||Δu||² = Δl²
    （荷重パラメータは拘束に含めない）

    予測子:
      K_T · δu_t = f_ext_ref
      Δλ₁ = ±Δl / ||δu_t||_free
      Δu₁ = Δλ₁ · δu_t

    修正子（各 NR 反復）:
      R = (λ + Δλ) · f_ext_ref - f_int(u + Δu)
      K_T · δu_R = R
      K_T · δu_t = f_ext_ref
      二次方程式から δλ を求め Δu, Δλ を更新

    Args:
        f_ext_ref: (ndof,) 参照荷重ベクトル（λ=1 での全荷重）
        fixed_dofs: 拘束DOFの配列
        assemble_tangent: u → K_T(u) を返すコールバック（CSR行列）
        assemble_internal_force: u → f_int(u) を返すコールバック
        n_steps: 最大ステップ数
        delta_l: 弧長パラメータ（None = 自動推定）
        max_iter: 各ステップの最大修正反復回数
        tol_force: 力ノルム収束判定
        show_progress: 進捗表示
        u0: 初期変位（None = ゼロ）
        lambda0: 初期荷重係数
        lambda_max: 荷重係数の上限（None = 制限なし）
        max_cutbacks: 各ステップの最大弧長カットバック回数

    Returns:
        ArcLengthResult: 解析結果
    """
    ndof = f_ext_ref.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)

    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)
    lam = lambda0

    f_ref_norm = float(np.linalg.norm(f_ext_ref))
    if f_ref_norm < 1e-30:
        raise ValueError("参照荷重ベクトルがゼロです。")

    # 弧長の自動推定: 初期接線から
    if delta_l is None:
        K0 = assemble_tangent(u)
        K0_bc, f_bc = _apply_bc(K0, f_ext_ref, fixed_dofs)
        du_t0 = spla.spsolve(K0_bc, f_bc)
        du_t0[fixed_dofs] = 0.0
        # 10ステップで λ=1 に到達する程度の弧長
        delta_l = float(np.linalg.norm(du_t0)) * 0.1

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    total_iter = 0
    sign = 1.0  # 荷重進行方向
    dl = delta_l  # 現在の弧長（適応的に変更）

    for step in range(1, n_steps + 1):
        converged_step = False

        for cutback in range(max_cutbacks + 1):
            # --- 予測子 ---
            K_T = assemble_tangent(u)
            K_bc, f_bc = _apply_bc(K_T, f_ext_ref, fixed_dofs)
            du_t = spla.spsolve(K_bc, f_bc)
            du_t[fixed_dofs] = 0.0

            du_t_norm = float(np.linalg.norm(du_t))
            if du_t_norm < 1e-30:
                if show_progress:
                    print(f"  Step {step}: 接線方向がゼロ。中断。")
                break

            d_lam_pred = sign * dl / du_t_norm
            Delta_u = d_lam_pred * du_t
            Delta_lam = d_lam_pred

            # --- 修正子 (NR 反復) ---
            need_cutback = False
            for it in range(max_iter):
                total_iter += 1

                f_int = assemble_internal_force(u + Delta_u)
                residual = (lam + Delta_lam) * f_ext_ref - f_int
                residual[fixed_dofs] = 0.0
                res_norm = float(np.linalg.norm(residual))

                if res_norm / f_ref_norm < tol_force:
                    converged_step = True
                    if show_progress:
                        print(
                            f"  Step {step}, λ={lam + Delta_lam:.6f}, "
                            f"iter {it}, ||R||/||f|| = {res_norm / f_ref_norm:.3e}"
                        )
                    break

                # 接線方向と残差修正（1回の factorize で2つの RHS を解く）
                K_T = assemble_tangent(u + Delta_u)
                K_bc, r_bc = _apply_bc(K_T, residual, fixed_dofs)
                du_R = spla.spsolve(K_bc, r_bc)
                du_R[fixed_dofs] = 0.0

                K_bc2, f_bc2 = _apply_bc(K_T, f_ext_ref, fixed_dofs)
                du_t = spla.spsolve(K_bc2, f_bc2)
                du_t[fixed_dofs] = 0.0

                # 円筒弧長拘束の二次方程式
                # ||(Δu + δu_R + δλ · δu_t)||² = Δl²
                v = Delta_u + du_R
                a1 = float(np.dot(du_t, du_t))
                a2 = 2.0 * float(np.dot(v, du_t))
                a3 = float(np.dot(v, v)) - dl**2

                disc = a2**2 - 4.0 * a1 * a3

                if disc < 0:
                    # 弧長が大きすぎて解が存在しない → カットバック
                    need_cutback = True
                    break

                sqrt_disc = np.sqrt(disc)
                d_lam_1 = (-a2 + sqrt_disc) / (2.0 * a1)
                d_lam_2 = (-a2 - sqrt_disc) / (2.0 * a1)

                # Δu の進行方向と一致する解を選択
                v1 = v + d_lam_1 * du_t
                v2 = v + d_lam_2 * du_t
                if float(np.dot(Delta_u, v1)) >= float(np.dot(Delta_u, v2)):
                    d_lam = d_lam_1
                else:
                    d_lam = d_lam_2

                Delta_u = v + d_lam * du_t
                Delta_lam += d_lam

            if converged_step:
                break

            if need_cutback and cutback < max_cutbacks:
                dl *= 0.5
                if show_progress:
                    print(
                        f"  Step {step}: 判別式 < 0。弧長を {dl:.4e} に縮小 (cutback {cutback + 1})"
                    )
                continue

            # max_iter reached without convergence
            break

        if not converged_step:
            if show_progress:
                print(f"  Step {step}: 収束せず。")
            return ArcLengthResult(
                u=u,
                lam=lam,
                converged=False,
                n_steps=step - 1,
                total_iterations=total_iter,
                load_history=load_history,
                displacement_history=disp_history,
            )

        # 更新
        u = u + Delta_u
        lam = lam + Delta_lam

        # 次ステップの荷重進行方向を更新
        if step > 1 and len(disp_history) > 0:
            du_step = u - disp_history[-1]
            dot_sign = float(np.dot(du_step, du_t))
            if dot_sign < 0:
                sign = -sign

        load_history.append(lam)
        disp_history.append(u.copy())

        # 弧長適応: 少ない反復で収束したら弧長を増加
        if it <= 3:
            dl = min(dl * 1.5, delta_l * 4.0)
        elif it > max_iter // 2:
            dl = max(dl * 0.5, delta_l * 0.01)

        if lambda_max is not None and lam >= lambda_max:
            if show_progress:
                print(f"  λ = {lam:.6f} >= lambda_max = {lambda_max}. 終了。")
            break

    return ArcLengthResult(
        u=u,
        lam=lam,
        converged=True,
        n_steps=len(load_history),
        total_iterations=total_iter,
        load_history=load_history,
        displacement_history=disp_history,
    )
