"""時間領域の過渡応答解析モジュール.

暗黙的時間積分スキーマ:
  - Newmark-β法（平均加速度法がデフォルト: β=1/4, γ=1/2）
  - HHT-α法（数値減衰付き: α ∈ [-1/3, 0]）

線形: M·ä + C·u̇ + K·u = f(t)          → solve_transient()
非線形: M·ä + C·u̇ + f_int(u) = f(t)   → solve_nonlinear_transient()
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

# ====================================================================
# コンフィグ・結果データクラス
# ====================================================================


@dataclass
class TransientConfig:
    """過渡応答解析の設定.

    Attributes:
        dt: 時間刻み [s]
        n_steps: 時間ステップ数
        beta: Newmark β パラメータ (デフォルト 0.25 = 平均加速度法)
        gamma: Newmark γ パラメータ (デフォルト 0.5)
        alpha_hht: HHT-α パラメータ (-1/3 ≤ α ≤ 0, デフォルト 0.0 = 標準 Newmark)
    """

    dt: float
    n_steps: int
    beta: float = 0.25
    gamma: float = 0.5
    alpha_hht: float = 0.0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt は正値: {self.dt}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps は1以上: {self.n_steps}")
        if not (-1.0 / 3.0 - 1e-12 <= self.alpha_hht <= 1e-12):
            raise ValueError(f"alpha_hht は [-1/3, 0]: {self.alpha_hht}")
        if self.beta <= 0:
            raise ValueError(f"beta は正値: {self.beta}")
        if self.gamma < 0.5 - 1e-12:
            raise ValueError(f"gamma は 0.5 以上: {self.gamma}")


@dataclass
class TransientResult:
    """過渡応答解析の結果.

    Attributes:
        time: (n_steps+1,) 時刻配列
        displacement: (n_steps+1, ndof) 変位履歴
        velocity: (n_steps+1, ndof) 速度履歴
        acceleration: (n_steps+1, ndof) 加速度履歴
        config: 解析設定
    """

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    config: TransientConfig = field(default_factory=lambda: TransientConfig(dt=1.0, n_steps=1))


# ====================================================================
# Newmark-β / HHT-α 時間積分ソルバー
# ====================================================================


def solve_transient(
    M: np.ndarray | sp.spmatrix,
    C: np.ndarray | sp.spmatrix,
    K: np.ndarray | sp.spmatrix,
    f_ext: Callable[[float], np.ndarray] | np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    config: TransientConfig,
    *,
    fixed_dofs: np.ndarray | None = None,
) -> TransientResult:
    """Newmark-β / HHT-α 法による線形過渡応答解析.

    運動方程式: M·ä + C·u̇ + K·u = f(t)

    Newmark 近似:
        u_{n+1} = u_n + Δt·v_n + Δt²·[(0.5-β)·a_n + β·a_{n+1}]
        v_{n+1} = v_n + Δt·[(1-γ)·a_n + γ·a_{n+1}]

    HHT-α 修正 (α ≠ 0):
        M·a_{n+1} + (1+α)·C·v_{n+1} - α·C·v_n
        + (1+α)·K·u_{n+1} - α·K·u_n = (1+α)·f_{n+1} - α·f_n

    Args:
        M: (ndof, ndof) 質量行列（密 or 疎）
        C: (ndof, ndof) 減衰行列（密 or 疎）
        K: (ndof, ndof) 剛性行列（密 or 疎）
        f_ext: 外力関数 f(t) → (ndof,) ベクトル、または定数力ベクトル (ndof,)
        u0: (ndof,) 初期変位
        v0: (ndof,) 初期速度
        config: TransientConfig
        fixed_dofs: 拘束 DOF のインデックス配列（None = 拘束なし）

    Returns:
        TransientResult: 時刻歴結果
    """
    dt = config.dt
    beta = config.beta
    gamma = config.gamma
    alpha = config.alpha_hht
    n_steps = config.n_steps

    ndof = len(u0)

    # 外力関数の準備
    if callable(f_ext):
        get_force = f_ext
    else:
        f_const = np.asarray(f_ext, dtype=float)

        def get_force(t: float) -> np.ndarray:
            return f_const

    # 疎行列 → 密行列に変換（小〜中規模を想定）
    M_d = _to_dense(M)
    C_d = _to_dense(C)
    K_d = _to_dense(K)

    # 拘束 DOF の処理
    if fixed_dofs is not None and len(fixed_dofs) > 0:
        fixed = np.asarray(fixed_dofs, dtype=int)
        all_dofs = np.arange(ndof)
        free_mask = np.ones(ndof, dtype=bool)
        free_mask[fixed] = False
        free = all_dofs[free_mask]
    else:
        fixed = np.array([], dtype=int)
        free = np.arange(ndof)

    # 自由 DOF に縮約
    M_ff = M_d[np.ix_(free, free)]
    C_ff = C_d[np.ix_(free, free)]
    K_ff = K_d[np.ix_(free, free)]

    u_f = u0[free].copy()
    v_f = v0[free].copy()

    # 初期加速度: M·a0 = f(0) - C·v0 - K·u0
    f0 = get_force(0.0)[free]
    rhs0 = f0 - C_ff @ v_f - K_ff @ u_f
    a_f = np.linalg.solve(M_ff, rhs0)

    # Newmark 定数
    c0 = 1.0 / (beta * dt**2)
    c1 = gamma / (beta * dt)
    c2 = 1.0 / (beta * dt)
    c3 = 1.0 / (2.0 * beta) - 1.0
    c4 = gamma / beta - 1.0
    c5 = dt * (gamma / (2.0 * beta) - 1.0)

    # HHT-α 修正の有効剛性行列
    # K_eff = c0·M + (1+α)·c1·C + (1+α)·K
    K_eff = c0 * M_ff + (1.0 + alpha) * c1 * C_ff + (1.0 + alpha) * K_ff

    # LU 分解を事前計算（定数係数系なので1回で済む）
    lu_piv = _lu_factor(K_eff)

    # 結果配列
    time_arr = np.linspace(0.0, dt * n_steps, n_steps + 1)
    u_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    v_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    a_hist = np.zeros((n_steps + 1, ndof), dtype=float)

    # 初期値を格納
    u_hist[0, free] = u_f
    v_hist[0, free] = v_f
    a_hist[0, free] = a_f

    # 前ステップの外力（HHT用）
    f_prev = f0.copy()

    # 時間ステップループ
    for n in range(n_steps):
        t_next = time_arr[n + 1]
        f_next = get_force(t_next)[free]

        # HHT-α 修正外力: (1+α)·f_{n+1} - α·f_n
        f_hht = (1.0 + alpha) * f_next - alpha * f_prev

        # 有効荷重ベクトル
        # f_eff = f_hht + M·(c0·u + c2·v + c3·a) + (1+α)·C·(c1·u + c4·v + c5·a)
        #         - α·K·u_n - α·C·v_n
        #       = f_hht + M·(...) + (1+α)·C·(...) + α·K·u - α·C·v
        #       ※ α·K·u_n と α·C·v_n は HHT 修正項。
        #         標準 Newmark (α=0) では消える。
        M_contrib = M_ff @ (c0 * u_f + c2 * v_f + c3 * a_f)
        C_contrib = (1.0 + alpha) * C_ff @ (c1 * u_f + c4 * v_f + c5 * a_f)

        # HHT補正: -α·K·u_n は (1+α)·K·u_{n+1} - α·K·u_n の形から来る。
        # 有効剛性に (1+α)·K が入っているので、右辺に α·K·u_n を足す。
        hht_K_corr = alpha * K_ff @ u_f
        hht_C_corr = alpha * C_ff @ v_f

        f_eff = f_hht + M_contrib + C_contrib + hht_K_corr + hht_C_corr

        # 求解: K_eff · u_{n+1} = f_eff
        u_new = _lu_solve(lu_piv, f_eff)

        # 加速度・速度の更新
        a_new = c0 * (u_new - u_f) - c2 * v_f - c3 * a_f
        v_new = v_f + dt * (1.0 - gamma) * a_f + dt * gamma * a_new

        # 状態更新
        u_f = u_new
        v_f = v_new
        a_f = a_new
        f_prev = f_next

        # 結果格納
        u_hist[n + 1, free] = u_f
        v_hist[n + 1, free] = v_f
        a_hist[n + 1, free] = a_f

    return TransientResult(
        time=time_arr,
        displacement=u_hist,
        velocity=v_hist,
        acceleration=a_hist,
        config=config,
    )


# ====================================================================
# 非線形過渡応答: Newton-Raphson + Newmark-β / HHT-α
# ====================================================================


@dataclass
class NonlinearTransientConfig:
    """非線形過渡応答解析の設定.

    Attributes:
        dt: 時間刻み [s]
        n_steps: 時間ステップ数
        beta: Newmark β パラメータ (デフォルト 0.25 = 平均加速度法)
        gamma: Newmark γ パラメータ (デフォルト 0.5)
        alpha_hht: HHT-α パラメータ (-1/3 ≤ α ≤ 0, デフォルト 0.0)
        max_iter: 各ステップの最大 Newton 反復回数
        tol_force: 力残差ノルムの収束判定値
    """

    dt: float
    n_steps: int
    beta: float = 0.25
    gamma: float = 0.5
    alpha_hht: float = 0.0
    max_iter: int = 30
    tol_force: float = 1e-8

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt は正値: {self.dt}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps は1以上: {self.n_steps}")
        if not (-1.0 / 3.0 - 1e-12 <= self.alpha_hht <= 1e-12):
            raise ValueError(f"alpha_hht は [-1/3, 0]: {self.alpha_hht}")
        if self.beta <= 0:
            raise ValueError(f"beta は正値: {self.beta}")
        if self.gamma < 0.5 - 1e-12:
            raise ValueError(f"gamma は 0.5 以上: {self.gamma}")


@dataclass
class NonlinearTransientResult:
    """非線形過渡応答解析の結果.

    Attributes:
        time: (n_steps+1,) 時刻配列
        displacement: (n_steps+1, ndof) 変位履歴
        velocity: (n_steps+1, ndof) 速度履歴
        acceleration: (n_steps+1, ndof) 加速度履歴
        config: 解析設定
        converged: 全ステップ収束したか
        iterations_per_step: 各ステップの NR 反復回数
        failed_step: 非収束時の失敗ステップ番号（1-indexed, None = 全収束）
    """

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    config: NonlinearTransientConfig
    converged: bool
    iterations_per_step: list[int] = field(default_factory=list)
    failed_step: int | None = None


def solve_nonlinear_transient(
    M: np.ndarray | sp.spmatrix,
    f_ext: Callable[[float], np.ndarray] | np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    config: NonlinearTransientConfig,
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    assemble_tangent: Callable[[np.ndarray], np.ndarray | sp.spmatrix],
    *,
    C: np.ndarray | sp.spmatrix | None = None,
    fixed_dofs: np.ndarray | None = None,
    show_progress: bool = True,
) -> NonlinearTransientResult:
    """非線形 Newmark-β / HHT-α 法による過渡応答解析.

    運動方程式: M·ä + C·u̇ + f_int(u) = f_ext(t)

    各時間ステップで Newmark 予測子を計算し、Newton-Raphson 反復で
    非線形平衡を解く。

    Newmark 予測子:
        ũ = u_n + Δt·v_n + 0.5·Δt²·(1-2β)·a_n
        ṽ = v_n + Δt·(1-γ)·a_n

    加速度・速度の Newmark 更新:
        a_{n+1} = (u_{n+1} - ũ) / (β·Δt²)
        v_{n+1} = ṽ + γ·Δt·a_{n+1}

    NR 残差 (HHT-α 一般形):
        R = (1+α)·f_{n+1} − α·f_n
          − M·a − (1+α)·[f_int(u) + C·v] + α·[f_int(u_n) + C·v_n]

    有効接線剛性:
        K_eff = c0·M + (1+α)·c1·C + (1+α)·K_T(u)

    Args:
        M: (ndof, ndof) 質量行列（定数、密 or 疎）
        f_ext: 外力関数 f(t) → (ndof,) または定数力ベクトル
        u0: (ndof,) 初期変位
        v0: (ndof,) 初期速度
        config: NonlinearTransientConfig
        assemble_internal_force: u → f_int(u) コールバック
        assemble_tangent: u → K_T(u) コールバック（密 or CSR）
        C: (ndof, ndof) 減衰行列（None = 減衰なし）
        fixed_dofs: 拘束 DOF インデックス（None = 拘束なし）
        show_progress: 進捗表示

    Returns:
        NonlinearTransientResult
    """
    dt = config.dt
    beta = config.beta
    gamma = config.gamma
    alpha = config.alpha_hht
    n_steps = config.n_steps
    max_iter = config.max_iter
    tol = config.tol_force

    ndof = len(u0)

    # 外力関数の準備
    if callable(f_ext):
        get_force = f_ext
    else:
        f_const = np.asarray(f_ext, dtype=float)

        def get_force(t: float) -> np.ndarray:
            return f_const

    # 密行列化
    M_d = _to_dense(M)
    C_d = _to_dense(C) if C is not None else np.zeros((ndof, ndof), dtype=float)

    # 拘束 DOF
    if fixed_dofs is not None and len(fixed_dofs) > 0:
        fixed = np.asarray(fixed_dofs, dtype=int)
    else:
        fixed = np.array([], dtype=int)

    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed] = False
    free = np.where(free_mask)[0]

    # Newmark 定数
    c0 = 1.0 / (beta * dt**2)
    c1 = gamma / (beta * dt)

    # 初期加速度: M·a₀ = f(0) - f_int(u₀) - C·v₀
    f0 = get_force(0.0)
    f_int0 = assemble_internal_force(u0)
    rhs0 = f0 - f_int0 - C_d @ v0

    a0 = np.zeros(ndof, dtype=float)
    if len(free) > 0:
        M_ff = M_d[np.ix_(free, free)]
        a0[free] = np.linalg.solve(M_ff, rhs0[free])

    # 状態初期化
    u = u0.copy()
    v = v0.copy()
    a = a0.copy()

    # 結果配列
    time_arr = np.linspace(0.0, dt * n_steps, n_steps + 1)
    u_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    v_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    a_hist = np.zeros((n_steps + 1, ndof), dtype=float)
    u_hist[0] = u.copy()
    v_hist[0] = v.copy()
    a_hist[0] = a.copy()

    iter_per_step: list[int] = []

    # 前ステップの保存値（HHT 用）
    f_int_prev = f_int0.copy()
    f_prev = f0.copy()

    # 進捗表示の間隔
    progress_interval = max(1, n_steps // 20)

    for n in range(n_steps):
        t_next = time_arr[n + 1]
        f_next = get_force(t_next)

        # Newmark 予測子
        u_pred = u + dt * v + 0.5 * dt**2 * (1.0 - 2.0 * beta) * a
        v_pred = v + dt * (1.0 - gamma) * a

        # 初期推定: 予測子
        u_new = u_pred.copy()
        u_new[fixed] = 0.0

        converged_step = False
        res_norm = 0.0
        ref_norm = 1.0

        for k in range(max_iter):
            # Newmark 更新による加速度・速度
            a_new = c0 * (u_new - u_pred)
            v_new = v_pred + dt * gamma * a_new

            # 内力
            f_int_new = assemble_internal_force(u_new)

            # 残差 (HHT-α 一般形)
            if abs(alpha) < 1e-15:
                R = f_next - f_int_new - M_d @ a_new - C_d @ v_new
            else:
                R = (
                    (1.0 + alpha) * f_next
                    - alpha * f_prev
                    - M_d @ a_new
                    - (1.0 + alpha) * (f_int_new + C_d @ v_new)
                    + alpha * (f_int_prev + C_d @ v)
                )

            R[fixed] = 0.0
            res_norm = float(np.linalg.norm(R))

            # 参照ノルム: 外力 or 慣性力
            f_ext_norm = float(np.linalg.norm(f_next))
            inertia_norm = float(np.linalg.norm(M_d @ a_new))
            ref_norm = max(f_ext_norm, inertia_norm, float(np.linalg.norm(f_int_new)), 1.0)

            if res_norm / ref_norm < tol:
                converged_step = True
                iter_per_step.append(k + 1)
                if show_progress and (n + 1) % progress_interval == 0:
                    print(
                        f"  Step {n + 1}/{n_steps}, t={t_next:.4e}, "
                        f"iter={k + 1}, ||R||/ref={res_norm / ref_norm:.3e}"
                    )
                break

            # 接線剛性
            K_T = assemble_tangent(u_new)
            K_T_d = _to_dense(K_T)

            # 有効接線剛性
            K_eff = c0 * M_d + (1.0 + alpha) * c1 * C_d + (1.0 + alpha) * K_T_d

            # 境界条件（行列消去法）
            for dof in fixed:
                K_eff[dof, :] = 0.0
                K_eff[:, dof] = 0.0
                K_eff[dof, dof] = 1.0

            # 線形求解
            du = np.linalg.solve(K_eff, R)
            u_new = u_new + du
            u_new[fixed] = 0.0

        if not converged_step:
            if show_progress:
                print(
                    f"  Step {n + 1}: 非収束 (iter={max_iter}, ||R||/ref={res_norm / ref_norm:.3e})"
                )
            iter_per_step.append(max_iter)
            return NonlinearTransientResult(
                time=time_arr[: n + 2],
                displacement=u_hist[: n + 2],
                velocity=v_hist[: n + 2],
                acceleration=a_hist[: n + 2],
                config=config,
                converged=False,
                iterations_per_step=iter_per_step,
                failed_step=n + 1,
            )

        # 最終更新
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + dt * gamma * a_new

        # 保存（HHT 用）
        f_int_prev = assemble_internal_force(u_new)
        f_prev = f_next.copy()

        # 状態更新
        u = u_new.copy()
        v = v_new.copy()
        a = a_new.copy()

        u_hist[n + 1] = u
        v_hist[n + 1] = v
        a_hist[n + 1] = a

    return NonlinearTransientResult(
        time=time_arr,
        displacement=u_hist,
        velocity=v_hist,
        acceleration=a_hist,
        config=config,
        converged=True,
        iterations_per_step=iter_per_step,
    )


# ====================================================================
# 内部ヘルパー
# ====================================================================


def _to_dense(A: np.ndarray | sp.spmatrix) -> np.ndarray:
    """疎行列を密行列に変換する（密行列はそのまま返す）."""
    if sp.issparse(A):
        return A.toarray()  # type: ignore[union-attr]
    return np.asarray(A, dtype=float)


def _lu_factor(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LU 分解（scipy.linalg.lu_factor のラッパー）."""
    import scipy.linalg as la

    return la.lu_factor(A)


def _lu_solve(
    lu_piv: tuple[np.ndarray, np.ndarray],
    b: np.ndarray,
) -> np.ndarray:
    """LU 分解を使った求解."""
    import scipy.linalg as la

    return la.lu_solve(lu_piv, b)
