"""Semi-smooth Newton ソルバー（NCP ベース接触） — Phase C6-L3.

Outer loop を廃止し、変位 u とラグランジュ乗数 λ を同時に更新する。

AL-NCP ハイブリッド + Alart-Curnier 鞍点定式化:
  接触力: p_n_i = max(0, λ_i + k_pen * (-g_i))  （AL 形式）
  NCP 条件 (Alart-Curnier):
    - Active  (λ + k_pen*(-g) > 0): C_i = k_pen * g_i  → g = 0 を強制
    - Inactive:                      C_i = λ_i          → λ = 0 を強制

鞍点系 (Saddle-point):
  [K_eff   -G_A^T] [Δu  ] = [-R_u    ]
  [G_A      0    ] [Δλ_A]   [-g_active]

  K_eff = K_T + k_pen * G_A^T * G_A

Schur complement on constraint space (n_active × n_active) で解く。
k_pen が正則化項として機能し、K_eff を確実に正定値に保つ。

参考文献:
- Alart, Curnier (1991): CMAME
- Hüeber, Stadler, Wohlmuth (2008): "A primal-dual active set algorithm
  for three-dimensional contact problems with Coulomb friction"

設計仕様: docs/contact/contact-algorithm-overhaul-c6.md §5
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.assembly import (
    _contact_dofs,
    _contact_shape_vector,
    _contact_tangent_shape_vector,
    compute_contact_force,
    compute_contact_stiffness,
)
from xkep_cae.contact.graph import ContactGraphHistory, snapshot_contact_graph
from xkep_cae.contact.law_friction import (
    compute_mu_effective,
    compute_tangential_displacement,
    friction_return_mapping,
    friction_tangent_2x2,
)
from xkep_cae.contact.pair import ContactManager, ContactStatus


@dataclass
class NCPSolveResult:
    """NCP ベース接触解析の結果.

    Attributes:
        u: (ndof,) 最終変位ベクトル
        lambdas: (n_pairs,) ラグランジュ乗数（全ペア、非活性は 0）
        converged: 収束したかどうか
        n_load_steps: 荷重増分ステップ数
        total_newton_iterations: 全ステップの合計 Newton 反復回数
        n_active_final: 最終的な ACTIVE ペア数
        load_history: 各ステップの荷重係数
        displacement_history: 各ステップの変位
        contact_force_history: 各ステップの接触力ノルム
        graph_history: 接触グラフの時系列
    """

    u: np.ndarray
    lambdas: np.ndarray
    converged: bool
    n_load_steps: int
    total_newton_iterations: int
    n_active_final: int
    load_history: list[float] = field(default_factory=list)
    displacement_history: list[np.ndarray] = field(default_factory=list)
    contact_force_history: list[float] = field(default_factory=list)
    graph_history: ContactGraphHistory = field(default_factory=ContactGraphHistory)


def _deformed_coords(
    node_coords_ref: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """参照座標 + 変位から変形座標を計算する."""
    n_nodes = node_coords_ref.shape[0]
    coords_def = node_coords_ref.copy()
    for i in range(n_nodes):
        coords_def[i, 0] += u[i * ndof_per_node + 0]
        coords_def[i, 1] += u[i * ndof_per_node + 1]
        coords_def[i, 2] += u[i * ndof_per_node + 2]
    return coords_def


def _compute_contact_force_from_lambdas(
    manager: ContactManager,
    lambdas: np.ndarray,
    ndof_total: int,
    ndof_per_node: int = 6,
    k_pen: float = 0.0,
) -> np.ndarray:
    """AL-NCP 接触力ベクトルを計算する.

    p_n_i = max(0, λ_i + k_pen * (-g_i))
    f_c = Σ p_n_i * g_shape_i

    Args:
        manager: 接触マネージャ
        lambdas: (n_pairs,) ラグランジュ乗数
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        k_pen: ペナルティ正則化パラメータ

    Returns:
        f_c: (ndof_total,) 接触内力ベクトル
    """
    f_c = np.zeros(ndof_total)

    for i, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        lam_i = lambdas[i] if i < len(lambdas) else 0.0
        g_i = pair.state.gap
        p_n = max(0.0, lam_i + k_pen * (-g_i))

        if p_n <= 0.0:
            continue

        g_shape = _contact_shape_vector(pair)
        dofs = _contact_dofs(pair, ndof_per_node)
        for k in range(4):
            for d in range(3):
                local_idx = k * 3 + d
                global_idx = dofs[k * ndof_per_node + d]
                f_c[global_idx] += p_n * g_shape[local_idx]

    return f_c


def _compute_friction_forces_ncp(
    manager: ContactManager,
    lambdas: np.ndarray,
    u: np.ndarray,
    u_ref: np.ndarray,
    node_coords_ref: np.ndarray,
    ndof_total: int,
    ndof_per_node: int = 6,
    k_pen: float = 0.0,
    mu: float = 0.3,
    mu_ramp_counter: int = 0,
    mu_ramp_steps: int = 0,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """NCP ソルバー用の摩擦力計算.

    各 ACTIVE ペアで:
    1. NCP 法線力 p_n = max(0, λ + k_pen*(-g)) を設定
    2. 接線変位を計算
    3. Coulomb return mapping を実行
    4. 摩擦力ベクトルをアセンブリ

    Args:
        manager: 接触マネージャ
        lambdas: (n_pairs,) ラグランジュ乗数
        u: (ndof,) 現在の変位
        u_ref: (ndof,) 参照変位（前ステップ収束解）
        node_coords_ref: (n_nodes, 3) 参照座標
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        k_pen: ペナルティ正則化パラメータ
        mu: 摩擦係数
        mu_ramp_counter: μランプカウンタ
        mu_ramp_steps: μランプ総ステップ数

    Returns:
        f_friction: (ndof_total,) 摩擦内力ベクトル
        friction_tangents: {pair_idx: D_t (2,2)} 摩擦接線剛性マップ
    """
    mu_eff = compute_mu_effective(mu, mu_ramp_counter, mu_ramp_steps)
    f_friction = np.zeros(ndof_total)
    friction_tangents: dict[int, np.ndarray] = {}

    for i, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        lam_i = lambdas[i] if i < len(lambdas) else 0.0
        g_i = pair.state.gap
        p_n = max(0.0, lam_i + k_pen * (-g_i))
        pair.state.p_n = p_n

        if p_n <= 0.0 or mu_eff <= 0.0:
            continue

        # ペナルティ剛性の初期化（未設定時）
        if pair.state.k_pen <= 0.0:
            pair.state.k_pen = k_pen
            pair.state.k_t = k_pen * manager.config.k_t_ratio

        # 接線変位
        delta_ut = compute_tangential_displacement(pair, u, u_ref, node_coords_ref, ndof_per_node)

        # Coulomb return mapping
        q = friction_return_mapping(pair, delta_ut, mu_eff)

        if float(np.linalg.norm(q)) < 1e-30:
            continue

        # 摩擦力アセンブリ
        dofs = _contact_dofs(pair, ndof_per_node)
        for axis in range(2):
            if abs(q[axis]) < 1e-30:
                continue
            g_t = _contact_tangent_shape_vector(pair, axis)
            for k in range(4):
                for d in range(3):
                    f_friction[dofs[k * ndof_per_node + d]] += q[axis] * g_t[k * 3 + d]

        # 摩擦接線剛性
        D_t = friction_tangent_2x2(pair, mu_eff)
        friction_tangents[i] = D_t

    return f_friction, friction_tangents


def _build_friction_stiffness(
    manager: ContactManager,
    friction_tangents: dict[int, np.ndarray],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """摩擦接線剛性行列のみを組み立てる（法線剛性を含まない）.

    NCP ソルバーでは法線剛性は鞍点系 (k_pen * G_A^T G_A) で処理されるため、
    compute_contact_stiffness を使うと法線剛性が二重カウントされる。
    この関数は摩擦接線剛性のみを構築する。

    K_f = Σ D_t[a1,a2] * g_t[a1] * g_t[a2]^T
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pair_idx, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        if pair_idx not in friction_tangents:
            continue

        D_t = friction_tangents[pair_idx]
        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        gdofs = np.empty(12, dtype=int)
        for k, node in enumerate(nodes):
            for d in range(3):
                gdofs[k * 3 + d] = node * ndof_per_node + d

        g_t = [
            _contact_tangent_shape_vector(pair, 0),
            _contact_tangent_shape_vector(pair, 1),
        ]
        for a1 in range(2):
            for a2 in range(2):
                d_val = D_t[a1, a2]
                if abs(d_val) < 1e-30:
                    continue
                for i in range(12):
                    for j in range(12):
                        val = d_val * g_t[a1][i] * g_t[a2][j]
                        if abs(val) > 1e-30:
                            rows.append(gdofs[i])
                            cols.append(gdofs[j])
                            data.append(val)

    if len(data) == 0:
        return sp.csr_matrix((ndof_total, ndof_total))

    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(ndof_total, ndof_total),
    ).tocsr()


def _build_constraint_jacobian(
    manager: ContactManager,
    ndof_total: int,
    ndof_per_node: int = 6,
) -> tuple[sp.csr_matrix, list[int]]:
    """制約ヤコビアン G = ∂g_n/∂u を構築する."""
    active_indices = []
    rows = []
    cols = []
    vals = []

    row_idx = 0
    for i, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        active_indices.append(i)
        s = pair.state.s
        t = pair.state.t
        normal = pair.state.normal

        dofs = _contact_dofs(pair, ndof_per_node)
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]
        for k in range(4):
            for d in range(3):
                global_dof = dofs[k * ndof_per_node + d]
                val = coeffs[k] * normal[d]
                if abs(val) > 1e-30:
                    rows.append(row_idx)
                    cols.append(global_dof)
                    vals.append(val)

        row_idx += 1

    n_active = len(active_indices)
    if n_active == 0:
        return sp.csr_matrix((0, ndof_total)), active_indices

    G = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_active, ndof_total),
    ).tocsr()
    return G, active_indices


def _build_tangential_constraint_jacobian(
    manager: ContactManager,
    active_indices: list[int],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """接線方向の制約ヤコビアン G_t を構築する.

    G_t は (2*n_active, ndof) の行列で、各アクティブペアに対して
    2行（t1方向, t2方向）を持つ。

    G_t[2*j,   :] = 接線変位の t1 成分の u 微分
    G_t[2*j+1, :] = 接線変位の t2 成分の u 微分

    接線相対変位: Δu_t = (u_B - u_A) · [t1, t2]
    → coeffs = [-(1-s), -s, (1-t), t]  (B-A方向)

    Args:
        manager: 接触マネージャ
        active_indices: アクティブペアインデックスリスト
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        G_t: (2*n_active, ndof_total) 接線制約ヤコビアン
    """
    n_active = len(active_indices)
    if n_active == 0:
        return sp.csr_matrix((0, ndof_total))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for j, pair_idx in enumerate(active_indices):
        pair = manager.pairs[pair_idx]
        s = pair.state.s
        t = pair.state.t
        t1 = pair.state.tangent1
        t2 = pair.state.tangent2

        dofs = _contact_dofs(pair, ndof_per_node)
        # B - A 方向: coeffs = [-(1-s), -s, (1-t), t]
        coeffs = [-(1.0 - s), -s, (1.0 - t), t]

        for axis, ti in enumerate([t1, t2]):
            row = 2 * j + axis
            for k in range(4):
                for d in range(3):
                    global_dof = dofs[k * ndof_per_node + d]
                    val = coeffs[k] * ti[d]
                    if abs(val) > 1e-30:
                        rows.append(row)
                        cols.append(global_dof)
                        vals.append(val)

    G_t = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(2 * n_active, ndof_total),
    ).tocsr()
    return G_t


def _solve_linear_system(
    K: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    mode: str = "auto",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
) -> np.ndarray:
    """線形連立方程式を解く."""
    import warnings

    import scipy.sparse.linalg as spla

    if mode == "direct":
        return spla.spsolve(K, rhs)

    if mode == "iterative":
        K_csc = K.tocsc()
        try:
            ilu = spla.spilu(K_csc, drop_tol=ilu_drop_tol)
            M = spla.LinearOperator(K.shape, ilu.solve)
        except RuntimeError:
            M = None
        x, info = spla.gmres(K, rhs, M=M, atol=iterative_tol, maxiter=max(500, K.shape[0]))
        if info != 0:
            x = spla.spsolve(K, rhs)
        return x

    # auto: direct → iterative fallback
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = spla.spsolve(K, rhs)
    for w in caught:
        if "MatrixRankWarning" in str(w.category.__name__) or "singular" in str(w.message).lower():
            return _solve_linear_system(K, rhs, mode="iterative", iterative_tol=iterative_tol)
    if not np.all(np.isfinite(x)):
        return _solve_linear_system(K, rhs, mode="iterative", iterative_tol=iterative_tol)
    return x


def _apply_bc(K_lil, rhs, fixed_dofs):
    """境界条件を適用する（in-place 変更）."""
    for dof in fixed_dofs:
        K_lil[dof, :] = 0.0
        K_lil[:, dof] = 0.0
        K_lil[dof, dof] = 1.0
        rhs[dof] = 0.0


def _solve_saddle_point_gmres(
    K_T: sp.csr_matrix,
    G_A: sp.csr_matrix,
    k_pen: float,
    R_u: np.ndarray,
    g_active: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Alart-Curnier NCP 鞍点系をブロック前処理付き GMRES で解く.

    拡大系:
      [K_eff   -G_A^T] [Δu  ] = [-R_u    ]
      [G_A      0    ] [Δλ_A]   [-g_active]

    ブロック対角前処理:
      P = [K_eff^{-1}   0      ]    K_eff^{-1} ≈ ILU
          [0            S_d^{-1}]    S_d ≈ diag(G_A * K_eff^{-1} * G_A^T)

    n_active が大きい場合に直接 Schur complement より効率的。
    K_eff^{-1} の (1 + n_active) 回の求解を GMRES 反復に置き換える。

    設計仕様: docs/contact/contact-algorithm-overhaul-c6.md §6

    Args:
        K_T: 構造接線剛性
        G_A: (n_active, ndof) NCP アクティブ制約ヤコビアン
        k_pen: ペナルティ正則化パラメータ
        R_u: (ndof,) 力残差
        g_active: (n_active,) アクティブペアのギャップ
        fixed_dofs: 拘束 DOF
        iterative_tol: GMRES 収束判定
        ilu_drop_tol: ILU drop tolerance

    Returns:
        (du, dlam_A): 変位増分, アクティブ乗数増分
    """
    import scipy.sparse.linalg as spla

    ndof = K_T.shape[0]
    n_active = G_A.shape[0]
    n_total = ndof + n_active

    # K_eff = K_T + k_pen * G_A^T * G_A
    K_eff = K_T + k_pen * (G_A.T @ G_A)

    # BC 適用
    K_bc = K_eff.tolil()
    rhs_u = -R_u.copy()
    _apply_bc(K_bc, rhs_u, fixed_dofs)
    K_bc_csr = K_bc.tocsr()

    # G_A の BC 処理（拘束 DOF 列をゼロに）
    G_A_bc = G_A.tolil()
    for dof in fixed_dofs:
        G_A_bc[:, dof] = 0.0
    G_A_bc_csr = G_A_bc.tocsr()
    G_A_bc_T = G_A_bc_csr.T.tocsr()

    # --- ILU 前処理構築 ---
    try:
        ilu = spla.spilu(K_bc_csr.tocsc(), drop_tol=ilu_drop_tol)
    except RuntimeError:
        ilu = None

    # --- Schur 補集合の対角近似: S_ii = G_A[i,:] * K_eff^{-1} * G_A[i,:]^T ---
    s_diag = np.ones(n_active) * 1e-12
    if ilu is not None:
        G_A_bc_dense = G_A_bc_csr.toarray()
        for i in range(n_active):
            g_row = G_A_bc_dense[i, :]
            v = ilu.solve(g_row)
            s_diag[i] = g_row @ v + 1e-12
    else:
        # ILU が失敗した場合の粗い近似
        for i in range(n_active):
            s_diag[i] = 1.0 / max(k_pen, 1e-12)

    # --- 拡大系行列-ベクトル積 ---
    def matvec(x):
        x_u = x[:ndof]
        x_lam = x[ndof:]
        y = np.zeros(n_total)
        y[:ndof] = K_bc_csr @ x_u - G_A_bc_T @ x_lam
        y[ndof:] = G_A_bc_csr @ x_u
        # BC 行: 拘束 DOF は identity
        for dof in fixed_dofs:
            y[dof] = x_u[dof]
        return y

    A_op = spla.LinearOperator((n_total, n_total), matvec=matvec)

    # --- ブロック対角前処理演算子 ---
    def precond(x):
        y = np.zeros(n_total)
        if ilu is not None:
            y[:ndof] = ilu.solve(x[:ndof])
        else:
            y[:ndof] = x[:ndof]
        y[ndof:] = x[ndof:] / s_diag
        return y

    M = spla.LinearOperator((n_total, n_total), matvec=precond)

    # --- RHS ---
    rhs = np.zeros(n_total)
    rhs[:ndof] = rhs_u
    rhs[ndof:] = -g_active

    # --- GMRES ---
    x, info = spla.gmres(
        A_op,
        rhs,
        M=M,
        atol=iterative_tol,
        maxiter=max(500, n_total),
    )

    if info != 0:
        # フォールバック: 直接 Schur complement
        return _solve_saddle_point_direct(
            K_T,
            G_A,
            k_pen,
            R_u,
            g_active,
            fixed_dofs,
            linear_solver="auto",
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
        )

    du = x[:ndof]
    dlam_A = x[ndof:]
    return du, dlam_A


def _solve_saddle_point_direct(
    K_T: sp.csr_matrix,
    G_A: sp.csr_matrix,
    k_pen: float,
    R_u: np.ndarray,
    g_active: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    linear_solver: str = "auto",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Alart-Curnier NCP 鞍点系を直接 Schur complement で解く.

    鞍点系:
      [K_eff   -G_A^T] [Δu  ] = [-R_u    ]
      [G_A      0    ] [Δλ_A]   [-g_active]

    K_eff = K_T + k_pen * G_A^T * G_A

    制約空間 Schur complement (n_active × n_active) で解く:
      1. v0 = K_eff^{-1} * (-R_u)        (非制約解)
      2. V  = K_eff^{-1} * G_A^T         (制約感度)
      3. S  = G_A * V                     (制約 Schur 補集合)
      4. Δλ = S^{-1} * (-g - G_A * v0)
      5. Δu = v0 + V * Δλ

    Args:
        K_T: 構造接線剛性
        G_A: (n_active, ndof) NCP アクティブ制約ヤコビアン
        k_pen: ペナルティ正則化パラメータ
        R_u: (ndof,) 力残差
        g_active: (n_active,) アクティブペアのギャップ
        fixed_dofs: 拘束 DOF
        linear_solver: 線形ソルバーモード
        iterative_tol: 反復ソルバー許容値
        ilu_drop_tol: ILU drop tolerance

    Returns:
        (du, dlam_A): 変位増分, アクティブ乗数増分
    """
    ndof = K_T.shape[0]
    n_active = G_A.shape[0]
    solve_kw = dict(mode=linear_solver, iterative_tol=iterative_tol, ilu_drop_tol=ilu_drop_tol)

    # K_eff = K_T + k_pen * G_A^T * G_A （ペナルティ正則化）
    K_eff = K_T + k_pen * (G_A.T @ G_A)

    # BC 適用
    K_bc = K_eff.tolil()
    rhs_u = -R_u.copy()
    _apply_bc(K_bc, rhs_u, fixed_dofs)
    K_bc_csr = K_bc.tocsr()

    # Step 1: v0 = K_eff^{-1} * (-R_u)
    v0 = _solve_linear_system(K_bc_csr, rhs_u, **solve_kw)

    # Step 2: V = K_eff^{-1} * G_A^T  (ndof × n_active)
    G_A_T_dense = G_A.T.toarray()  # ndof × n_active
    G_A_T_bc = G_A_T_dense.copy()
    for dof in fixed_dofs:
        G_A_T_bc[dof, :] = 0.0

    V = np.zeros((ndof, n_active))
    for j in range(n_active):
        V[:, j] = _solve_linear_system(K_bc_csr, G_A_T_bc[:, j], **solve_kw)

    # Step 3: S = G_A * V  (n_active × n_active — 制約 Schur 補集合)
    G_A_dense = G_A.toarray()
    S = G_A_dense @ V

    # 正則化（S が特異にならないよう保証）
    S += 1e-12 * np.eye(n_active)

    # Step 4: rhs_S = -g_active - G_A * v0
    rhs_S = -g_active - G_A_dense @ v0

    # Δλ_A を解く（n_active × n_active 密行列、通常は極めて小さい）
    dlam_A = np.linalg.solve(S, rhs_S)

    # Step 5: Δu = v0 + V * Δλ_A
    du = v0 + V @ dlam_A

    return du, dlam_A


def _solve_augmented_friction_system(
    K_T: sp.csr_matrix,
    G_n: sp.csr_matrix,
    G_t: sp.csr_matrix,
    k_pen: float,
    k_t: float,
    R_u: np.ndarray,
    C_n: np.ndarray,
    C_t: np.ndarray,
    J_blocks: dict,
    fixed_dofs: np.ndarray,
    n_ncp_active: int,
    ncp_active_mask: np.ndarray,
    active_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alart-Curnier 摩擦拡大鞍点系を解く.

    拡大系:
      [K_eff    -G_n_A^T  -G_t_A^T] [Δu    ]   [-R_u]
      [J_n_u    J_n_n      0      ] [Δλ_n_A] = [-C_n]
      [J_t_u    J_t_n      J_t_t  ] [Δλ_t_A]   [-C_t]

    K_eff = K_T + k_pen * G_n_A^T G_n_A

    各ペアの J ブロックは状態（active/inactive, stick/slip）に応じて異なる。

    Args:
        K_T: 構造接線剛性
        G_n: (n_geom_active, ndof) 法線制約ヤコビアン（全幾何アクティブ）
        G_t: (2*n_geom_active, ndof) 接線制約ヤコビアン（全幾何アクティブ）
        k_pen: 法線ペナルティ剛性
        k_t: 接線ペナルティ剛性
        R_u: (ndof,) 力残差
        C_n: (n_geom_active,) 法線 NCP 残差
        C_t: (2*n_geom_active,) 接線 NCP 残差
        J_blocks: 各ペアの Jacobian ブロック辞書
        fixed_dofs: 拘束 DOF
        n_ncp_active: NCP アクティブペア数
        ncp_active_mask: NCP アクティブマスク
        active_indices: アクティブペアインデックス

    Returns:
        (du, dlam_n, dlam_t): 変位増分, 法線乗数増分, 接線乗数増分
    """
    import scipy.sparse.linalg as spla

    ndof = K_T.shape[0]

    # NCP アクティブ行のみ抽出
    if n_ncp_active > 0:
        active_rows_n = np.where(ncp_active_mask)[0]
        G_n_A = G_n[active_rows_n, :]
    else:
        G_n_A = sp.csr_matrix((0, ndof))
        active_rows_n = np.array([], dtype=int)

    # 接線行: NCP アクティブに対応する 2*n 行
    if n_ncp_active > 0:
        t_rows = []
        for r in active_rows_n:
            t_rows.extend([2 * r, 2 * r + 1])
        t_rows = np.array(t_rows, dtype=int)
        G_t_A = G_t[t_rows, :]
    else:
        G_t_A = sp.csr_matrix((0, ndof))
        t_rows = np.array([], dtype=int)

    n_n = n_ncp_active
    n_t = 2 * n_ncp_active
    n_total = ndof + n_n + n_t

    # K_eff = K_T + k_pen * G_n_A^T G_n_A
    if n_n > 0:
        K_eff = K_T + k_pen * (G_n_A.T @ G_n_A)
    else:
        K_eff = K_T.copy()

    # --- 拡大系をブロック行列で構築 ---
    # J_n_u, J_n_n, J_t_u, J_t_n, J_t_t を密行列で構築
    J_n_u = np.zeros((n_n, ndof))
    J_n_n = np.zeros((n_n, n_n))
    J_t_u = np.zeros((n_t, ndof))
    J_t_n = np.zeros((n_t, n_n))
    J_t_t = np.zeros((n_t, n_t))

    G_n_A_dense = G_n_A.toarray() if n_n > 0 else np.zeros((0, ndof))
    G_t_A_dense = G_t_A.toarray() if n_t > 0 else np.zeros((0, ndof))

    for j_local in range(n_ncp_active):
        j_geom = int(active_rows_n[j_local])
        pair_idx = active_indices[j_geom]
        jb = J_blocks.get(pair_idx)
        if jb is None:
            # デフォルト: 法線 active, stick
            # C_n = k_pen * g → J_n_u = k_pen * G_n
            J_n_u[j_local, :] = k_pen * G_n_A_dense[j_local, :]
            # C_t = -k_t * Δu_t → J_t_u = -k_t * G_t, J_t_t = 0
            J_t_u[2 * j_local, :] = -k_t * G_t_A_dense[2 * j_local, :]
            J_t_u[2 * j_local + 1, :] = -k_t * G_t_A_dense[2 * j_local + 1, :]
            continue

        mode = jb["mode"]  # "active_stick", "active_slip", "inactive"

        if mode == "inactive":
            # C_n = λ_n → J_n_n = 1, C_t = λ_t → J_t_t = I
            J_n_n[j_local, j_local] = 1.0
            J_t_t[2 * j_local, 2 * j_local] = 1.0
            J_t_t[2 * j_local + 1, 2 * j_local + 1] = 1.0

        elif mode == "active_stick":
            # C_n = k_pen * g → J_n_u = k_pen * G_n
            J_n_u[j_local, :] = k_pen * G_n_A_dense[j_local, :]
            # C_t = -k_t * Δu_t → J_t_u = -k_t * G_t, J_t_t = 0
            J_t_u[2 * j_local, :] = -k_t * G_t_A_dense[2 * j_local, :]
            J_t_u[2 * j_local + 1, :] = -k_t * G_t_A_dense[2 * j_local + 1, :]

        elif mode == "active_slip":
            # 法線: C_n = k_pen * g → J_n_u = k_pen * G_n
            J_n_u[j_local, :] = k_pen * G_n_A_dense[j_local, :]

            # 接線 slip Jacobians
            # J_t_t = I - (μ*p_n/||λ̂_t||) * (I - q̂⊗q̂)
            J_t_t_local = jb["J_t_t"]  # (2,2)
            J_t_t[2 * j_local : 2 * j_local + 2, 2 * j_local : 2 * j_local + 2] = J_t_t_local

            # J_t_n = -μ * q̂  (2,1) — critical coupling term
            J_t_n_local = jb["J_t_n"]  # (2,)
            J_t_n[2 * j_local, j_local] = J_t_n_local[0]
            J_t_n[2 * j_local + 1, j_local] = J_t_n_local[1]

            # J_t_u: slip の変位微分
            # ∂C_t/∂u = μ*k_pen * outer(q̂, G_n) - μ*p_n*k_t/||λ̂_t|| * (I-q̂⊗q̂) @ G_t
            if "J_t_u" in jb:
                J_t_u[2 * j_local : 2 * j_local + 2, :] = jb["J_t_u"]  # (2, ndof)

    # C_n, C_t を NCP アクティブ行のみに絞る
    C_n_A = C_n[active_rows_n] if n_n > 0 else np.array([])
    C_t_A = C_t[t_rows] if n_t > 0 else np.array([])

    # 拡大系組立: sp.bmat で構築
    # [K_eff, -G_n_A^T, -G_t_A^T]   [-R_u ]
    # [J_n_u,  J_n_n,    0       ] = [-C_n_A]
    # [J_t_u,  J_t_n,    J_t_t   ]   [-C_t_A]
    K_eff_lil = K_eff.tolil()

    # BC を K_eff に適用
    rhs_u = -R_u.copy()
    _apply_bc(K_eff_lil, rhs_u, fixed_dofs)
    K_eff_bc = K_eff_lil.tocsr()

    # G の BC 処理（拘束 DOF 列をゼロに）
    if n_n > 0:
        G_n_A_bc = G_n_A_dense.copy()
        for dof in fixed_dofs:
            G_n_A_bc[:, dof] = 0.0
            J_n_u[:, dof] = 0.0
    if n_t > 0:
        G_t_A_bc = G_t_A_dense.copy()
        for dof in fixed_dofs:
            G_t_A_bc[:, dof] = 0.0
            J_t_u[:, dof] = 0.0

    # 拡大行列を構築
    if n_n == 0 and n_t == 0:
        du = spla.spsolve(K_eff_bc, rhs_u)
        return du, np.array([]), np.array([])

    blocks = [[K_eff_bc, None, None], [None, None, None], [None, None, None]]

    if n_n > 0:
        blocks[0][1] = sp.csr_matrix(-G_n_A_bc.T)
        blocks[1][0] = sp.csr_matrix(J_n_u)
        blocks[1][1] = sp.csr_matrix(J_n_n)
    if n_t > 0:
        blocks[0][2] = sp.csr_matrix(-G_t_A_bc.T) if n_t > 0 else None
        blocks[2][0] = sp.csr_matrix(J_t_u)
        blocks[2][2] = sp.csr_matrix(J_t_t)
    if n_n > 0 and n_t > 0:
        blocks[1][2] = sp.csr_matrix((n_n, n_t))
        blocks[2][1] = sp.csr_matrix(J_t_n)

    # None ブロックをゼロ行列で埋める
    sizes = [ndof, n_n, n_t]
    for i_b in range(3):
        for j_b in range(3):
            if blocks[i_b][j_b] is None:
                if sizes[i_b] > 0 and sizes[j_b] > 0:
                    blocks[i_b][j_b] = sp.csr_matrix((sizes[i_b], sizes[j_b]))

    # サイズ 0 のブロック行/列を除外
    active_block_rows = [i_b for i_b in range(3) if sizes[i_b] > 0]
    filtered_blocks = []
    for i_b in active_block_rows:
        row_blocks = []
        for j_b in active_block_rows:
            row_blocks.append(blocks[i_b][j_b])
        filtered_blocks.append(row_blocks)

    A = sp.bmat(filtered_blocks, format="csc")

    # RHS
    rhs_parts = []
    if ndof > 0:
        rhs_parts.append(rhs_u)
    if n_n > 0:
        rhs_parts.append(-C_n_A)
    if n_t > 0:
        rhs_parts.append(-C_t_A)
    rhs = np.concatenate(rhs_parts)

    # 解く
    x = spla.spsolve(A, rhs)
    if not np.all(np.isfinite(x)):
        # フォールバック: GMRES
        x_gm, info = spla.gmres(A.tocsr(), rhs, atol=1e-10, maxiter=max(500, n_total))
        if info == 0 and np.all(np.isfinite(x_gm)):
            x = x_gm

    du = x[:ndof]
    offset = ndof
    dlam_n = x[offset : offset + n_n] if n_n > 0 else np.array([])
    offset += n_n
    dlam_t = x[offset : offset + n_t] if n_t > 0 else np.array([])

    return du, dlam_n, dlam_t


def _solve_saddle_point_contact(
    K_T: sp.csr_matrix,
    G_A: sp.csr_matrix,
    k_pen: float,
    R_u: np.ndarray,
    g_active: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    linear_solver: str = "auto",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
    use_block_preconditioner: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Alart-Curnier NCP 鞍点系を解く（直接法 or ブロック前処理 GMRES）.

    use_block_preconditioner=False (デフォルト):
      制約空間 Schur complement で直接解法。n_active が小さい場合に高速。

    use_block_preconditioner=True:
      ブロック対角前処理付き GMRES。n_active が大きい場合に効率的。
      K_eff^{-1} を ILU で近似し、Schur 補集合の対角近似を前処理に使用。

    Args:
        K_T: 構造接線剛性
        G_A: (n_active, ndof) NCP アクティブ制約ヤコビアン
        k_pen: ペナルティ正則化パラメータ
        R_u: (ndof,) 力残差
        g_active: (n_active,) アクティブペアのギャップ
        fixed_dofs: 拘束 DOF
        linear_solver: 線形ソルバーモード（直接法のみ）
        iterative_tol: 反復ソルバー許容値
        ilu_drop_tol: ILU drop tolerance
        use_block_preconditioner: ブロック前処理 GMRES の使用（Phase C6-L4）

    Returns:
        (du, dlam_A): 変位増分, アクティブ乗数増分
    """
    n_active = G_A.shape[0]

    if n_active == 0:
        K_bc = K_T.tolil()
        rhs = -R_u.copy()
        _apply_bc(K_bc, rhs, fixed_dofs)
        solve_kw = dict(mode=linear_solver, iterative_tol=iterative_tol, ilu_drop_tol=ilu_drop_tol)
        du = _solve_linear_system(K_bc.tocsr(), rhs, **solve_kw)
        return du, np.array([])

    if use_block_preconditioner:
        return _solve_saddle_point_gmres(
            K_T,
            G_A,
            k_pen,
            R_u,
            g_active,
            fixed_dofs,
            iterative_tol=iterative_tol,
            ilu_drop_tol=ilu_drop_tol,
        )

    return _solve_saddle_point_direct(
        K_T,
        G_A,
        k_pen,
        R_u,
        g_active,
        fixed_dofs,
        linear_solver=linear_solver,
        iterative_tol=iterative_tol,
        ilu_drop_tol=ilu_drop_tol,
    )


def newton_raphson_contact_ncp(
    f_ext_total: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray | float,
    *,
    n_load_steps: int = 10,
    max_iter: int = 50,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    tol_ncp: float = 1e-8,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.0,
    broadphase_cell_size: float | None = None,
    ncp_type: str = "fb",
    ncp_reg: float = 1e-12,
    k_pen: float = 0.0,
    f_ext_base: np.ndarray | None = None,
    use_friction: bool = False,
    mu: float | None = None,
    mu_ramp_steps: int | None = None,
    line_contact: bool = False,
    n_gauss: int | None = None,
) -> NCPSolveResult:
    """Semi-smooth Newton 法による接触解析（AL-NCP + 鞍点定式化）.

    Outer loop 不要。u と λ を同時に更新。
    摩擦有効時は法線 NCP + 摩擦 return mapping のハイブリッド方式。
    line_contact 有効時は Gauss 積分で接触力・剛性を評価。

    Alart-Curnier NCP + 鞍点系:
      Active set: A = {i : λ_i + k_pen*(-g_i) > 0}
      Active pairs → g = 0 を制約として鞍点系で解く
      Inactive pairs → λ = 0 に設定

    摩擦（use_friction=True 時）:
      各 Newton 反復で Coulomb return mapping を実行し、
      摩擦力を残差に、摩擦接線剛性を K_T に加算する。
      法線方向は NCP で Outer loop 不要、摩擦は return mapping で処理。

    line contact（line_contact=True 時）:
      法線力と法線剛性を Gauss 積分で評価する。
      NCP 活性セット判定は代表点ギャップ（PtP）を使用し、
      力・剛性評価のみ line-to-line を使用するハイブリッド方式。

    Args:
        f_ext_total: (ndof,) 最終外荷重
        fixed_dofs: 拘束DOF
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径
        n_load_steps: 荷重増分数
        max_iter: Newton 最大反復数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_ncp: NCP 残差の収束判定
        show_progress: 進捗表示
        u0: 初期変位
        ndof_per_node: 1節点あたりの DOF 数
        broadphase_margin: broadphase 探索マージン
        broadphase_cell_size: broadphase セルサイズ
        ncp_type: NCP 関数の種類（"fb" or "min"）— 収束モニタリング用
        ncp_reg: FB 関数の正則化パラメータ
        k_pen: ペナルティ正則化パラメータ。0 の場合は
            manager.config.k_pen_scale を使用。
        f_ext_base: (ndof,) ベース外荷重
        use_friction: Coulomb 摩擦の有効化（NCP + return mapping ハイブリッド）
        mu: 摩擦係数（None なら config.mu）
        mu_ramp_steps: μランプ（None なら config.mu_ramp_steps）
        line_contact: Line-to-line Gauss 積分の有効化（None なら config.line_contact）
        n_gauss: Gauss 積分点数（None なら config.n_gauss）

    Returns:
        NCPSolveResult
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)

    # k_pen の決定
    if k_pen <= 0.0:
        k_pen = manager.config.k_pen_scale

    # 摩擦設定の解決
    _use_friction = use_friction or manager.config.use_friction
    _mu = mu if mu is not None else manager.config.mu
    _mu_ramp_steps = mu_ramp_steps if mu_ramp_steps is not None else manager.config.mu_ramp_steps

    # line contact 設定の解決
    _line_contact = line_contact or manager.config.line_contact
    _n_gauss = n_gauss if n_gauss is not None else manager.config.n_gauss

    # λ の初期化
    n_pairs = manager.n_pairs
    lam_all = np.zeros(n_pairs)
    # 接線乗数 λ_t（Alart-Curnier 摩擦用: 各ペア 2 成分）
    lam_t_all = np.zeros((n_pairs, 2))

    # 線形ソルバー設定
    linear_solver_mode = manager.config.linear_solver
    iterative_tol_cfg = manager.config.iterative_tol
    ilu_drop_tol_cfg = manager.config.ilu_drop_tol
    use_block_preconditioner = manager.config.ncp_block_preconditioner

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
    graph_history = ContactGraphHistory()
    total_newton = 0

    f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
    if f_ext_ref_norm < 1e-30:
        f_ext_ref_norm = 1.0

    _f_ext_base = f_ext_base if f_ext_base is not None else np.zeros(ndof)

    # 摩擦用: 参照変位（前ステップ収束解）
    u_ref = u.copy()

    for step in range(1, n_load_steps + 1):
        load_frac = step / n_load_steps
        f_ext = _f_ext_base + load_frac * f_ext_total

        step_converged = False

        # --- ステップ開始時: 候補検出 ---
        coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)
        manager.detect_candidates(
            coords_def,
            connectivity,
            radii,
            margin=broadphase_margin,
            cell_size=broadphase_cell_size,
        )
        manager.update_geometry(coords_def)

        # ペア数拡張
        if len(lam_all) < manager.n_pairs:
            lam_new = np.zeros(manager.n_pairs)
            lam_new[: len(lam_all)] = lam_all
            lam_all = lam_new
            lam_t_new = np.zeros((manager.n_pairs, 2))
            lam_t_new[: len(lam_t_all)] = lam_t_all
            lam_t_all = lam_t_new

        for it in range(max_iter):
            total_newton += 1

            # 1. 幾何更新（s,t,normal を毎反復更新）
            coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)
            manager.update_geometry(coords_def, freeze_active_set=True)

            # 2. 制約ヤコビアン（幾何的 ACTIVE ペア）
            G_mat, active_indices = _build_constraint_jacobian(manager, ndof, ndof_per_node)
            n_geom_active = len(active_indices)

            gaps = np.array([manager.pairs[i].state.gap for i in active_indices])
            lams = np.array([lam_all[i] for i in active_indices])

            # 3. NCP アクティブセット判定
            #    Active: λ + k_pen*(-g) > 0
            p_n_arr = np.maximum(0.0, lams + k_pen * (-gaps))
            ncp_active_mask = p_n_arr > 0.0  # boolean

            # 4. NCP 法線力を pair.state に同期（assembly 関数が参照するため）
            for idx_p, pair in enumerate(manager.pairs):
                if pair.state.status == ContactStatus.INACTIVE:
                    continue
                lam_i = lam_all[idx_p] if idx_p < len(lam_all) else 0.0
                p_n = max(0.0, lam_i + k_pen * (-pair.state.gap))
                pair.state.lambda_n = lam_i
                pair.state.p_n = p_n
                if pair.state.k_pen <= 0.0:
                    pair.state.k_pen = k_pen
                    pair.state.k_t = k_pen * manager.config.k_t_ratio

            # 4a. 接触力ベクトル（法線）
            if _line_contact:
                # Line-to-line Gauss 積分（assembly の既存インフラを利用）
                # line_contact フラグを一時的に有効化
                _orig_lc = manager.config.line_contact
                _orig_ng = manager.config.n_gauss
                manager.config.line_contact = True
                manager.config.n_gauss = _n_gauss
                f_c = compute_contact_force(
                    manager, ndof, ndof_per_node=ndof_per_node, node_coords=coords_def
                )
                manager.config.line_contact = _orig_lc
                manager.config.n_gauss = _orig_ng
            else:
                f_c = _compute_contact_force_from_lambdas(
                    manager, lam_all, ndof, ndof_per_node, k_pen=k_pen
                )

            # 4b. 摩擦力ベクトル（Alart-Curnier 接線乗数方式）
            if _use_friction:
                mu_eff = compute_mu_effective(_mu, step, _mu_ramp_steps)
                # 接線乗数から摩擦力を計算: f_fric = -G_t^T * λ_t
                f_friction = np.zeros(ndof)
                for _j, pair_idx in enumerate(active_indices):
                    pair = manager.pairs[pair_idx]
                    if pair.state.status == ContactStatus.INACTIVE:
                        continue
                    lam_t_j = lam_t_all[pair_idx]
                    if float(np.linalg.norm(lam_t_j)) < 1e-30:
                        continue
                    dofs = _contact_dofs(pair, ndof_per_node)
                    for axis in range(2):
                        if abs(lam_t_j[axis]) < 1e-30:
                            continue
                        g_t = _contact_tangent_shape_vector(pair, axis)
                        for kk in range(4):
                            for d in range(3):
                                # g_t_shape の符号は -(G_t^T) に対応
                                # f = -G_t^T * λ_t, g_t_shape は assembly 規約に従う
                                f_friction[dofs[kk * ndof_per_node + d]] += (
                                    lam_t_j[axis] * g_t[kk * 3 + d]
                                )
                f_c = f_c + f_friction

            # 5. 力残差
            f_int = assemble_internal_force(u)
            R_u = f_int + f_c - f_ext
            R_u[fixed_dofs] = 0.0

            # 6. NCP 残差（Alart-Curnier 方式）
            #    Active: C_i = k_pen * g_i  (g → 0 を目標)
            #    Inactive: C_i = λ_i        (λ → 0 を目標)
            C_ac = np.empty(n_geom_active) if n_geom_active > 0 else np.array([])
            for j in range(n_geom_active):
                if ncp_active_mask[j]:
                    C_ac[j] = k_pen * gaps[j]
                else:
                    C_ac[j] = lams[j]

            # 6b. 接線 NCP 残差 + J ブロック構築（Alart-Curnier 摩擦）
            C_t_ac = np.zeros(2 * n_geom_active)
            J_blocks: dict = {}
            if _use_friction and n_geom_active > 0:
                _k_t = k_pen * manager.config.k_t_ratio
                for j in range(n_geom_active):
                    pair_idx = active_indices[j]
                    pair = manager.pairs[pair_idx]
                    lam_t_j = lam_t_all[pair_idx]  # (2,)
                    p_n_j = p_n_arr[j] if ncp_active_mask[j] else 0.0

                    if not ncp_active_mask[j]:
                        # 法線非アクティブ → 接線も非アクティブ
                        C_t_ac[2 * j] = lam_t_j[0]
                        C_t_ac[2 * j + 1] = lam_t_j[1]
                        J_blocks[pair_idx] = {"mode": "inactive"}
                        continue

                    # 接線変位増分 Δu_t
                    delta_ut = compute_tangential_displacement(
                        pair, u, u_ref, node_coords_ref, ndof_per_node
                    )

                    # 増強接線乗数: λ̂_t = λ_t + k_t * Δu_t
                    lam_t_hat = lam_t_j + _k_t * delta_ut
                    lam_t_hat_norm = float(np.linalg.norm(lam_t_hat))

                    # Stick/slip 判定
                    if lam_t_hat_norm <= mu_eff * p_n_j or lam_t_hat_norm < 1e-30:
                        # Stick: C_t = λ_t - λ̂_t = -k_t * Δu_t
                        C_t_ac[2 * j : 2 * j + 2] = lam_t_j - lam_t_hat
                        J_blocks[pair_idx] = {"mode": "active_stick"}
                    else:
                        # Slip: C_t = λ_t - μ*p_n * q̂
                        q_hat = lam_t_hat / lam_t_hat_norm
                        C_t_ac[2 * j : 2 * j + 2] = lam_t_j - mu_eff * p_n_j * q_hat

                        # J_t_t = I - (μ*p_n/||λ̂_t||) * (I - q̂⊗q̂)
                        ratio = mu_eff * p_n_j / lam_t_hat_norm
                        I2 = np.eye(2)
                        J_t_t_local = I2 - ratio * (I2 - np.outer(q_hat, q_hat))

                        # J_t_n = -μ * q̂  (∂f_fric/∂λ_n coupling!)
                        J_t_n_local = -mu_eff * q_hat

                        # J_t_u: slip の変位微分
                        # ∂C_t/∂u = μ*k_pen * outer(q̂, G_n_row) - μ*p_n*k_t/||λ̂_t|| * (I-q̂⊗q̂) @ G_t_rows
                        G_n_row_j = G_mat[j, :].toarray().ravel()  # (ndof,)
                        G_t_rows_j = np.zeros((2, ndof))
                        dofs_j = _contact_dofs(pair, ndof_per_node)
                        for ax in range(2):
                            g_t_sv = _contact_tangent_shape_vector(pair, ax)
                            # G_t の符号: coeffs = [-(1-s), -s, (1-t), t]
                            # g_t_shape の符号: [(1-s), s, -(1-t), -t]
                            # G_t = -g_t_shape を DOF 配置
                            for kk in range(4):
                                for d in range(3):
                                    G_t_rows_j[ax, dofs_j[kk * ndof_per_node + d]] = -g_t_sv[
                                        kk * 3 + d
                                    ]

                        proj = np.eye(2) - np.outer(q_hat, q_hat)
                        J_t_u_local = (
                            mu_eff * k_pen * np.outer(q_hat, G_n_row_j)
                            - mu_eff * p_n_j * _k_t / lam_t_hat_norm * proj @ G_t_rows_j
                        )

                        J_blocks[pair_idx] = {
                            "mode": "active_slip",
                            "J_t_t": J_t_t_local,
                            "J_t_n": J_t_n_local,
                            "J_t_u": J_t_u_local,
                        }

            # 7. 収束判定
            res_u_norm = float(np.linalg.norm(R_u))
            ncp_norm = float(np.linalg.norm(C_ac)) if n_geom_active > 0 else 0.0
            ncp_t_norm = (
                float(np.linalg.norm(C_t_ac)) if (_use_friction and n_geom_active > 0) else 0.0
            )

            force_conv = res_u_norm / f_ext_ref_norm < tol_force
            ncp_conv = ncp_norm < tol_ncp
            ncp_t_conv = ncp_t_norm < tol_ncp if _use_friction else True
            all_conv = force_conv and ncp_conv and ncp_t_conv

            n_ncp_active = int(np.sum(ncp_active_mask))

            if all_conv:
                step_converged = True
                if show_progress:
                    msg = (
                        f"  Step {step}/{n_load_steps}, iter {it}, "
                        f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e}, "
                        f"||C_n|| = {ncp_norm:.3e}"
                    )
                    if _use_friction:
                        msg += f", ||C_t|| = {ncp_t_norm:.3e}"
                    msg += f" (converged, {n_ncp_active} active)"
                    print(msg)
                break

            if show_progress and it % 5 == 0:
                msg = (
                    f"  Step {step}/{n_load_steps}, iter {it}, "
                    f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e}, "
                    f"||C_n|| = {ncp_norm:.3e}"
                )
                if _use_friction:
                    msg += f", ||C_t|| = {ncp_t_norm:.3e}"
                msg += f", active={n_ncp_active}/{n_geom_active}"
                print(msg)

            # 8. 構造接線剛性
            K_T = assemble_tangent(u)

            # 8b. line contact 法線剛性を加算（Gauss 積分）
            if _line_contact:
                _orig_lc = manager.config.line_contact
                _orig_ng = manager.config.n_gauss
                manager.config.line_contact = True
                manager.config.n_gauss = _n_gauss
                K_line = compute_contact_stiffness(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                    use_geometric_stiffness=manager.config.use_geometric_stiffness,
                    node_coords=coords_def,
                )
                manager.config.line_contact = _orig_lc
                manager.config.n_gauss = _orig_ng
                K_T = K_T + K_line

            # 8c. 摩擦は拡大鞍点系で処理（K_T への加算不要）

            # 9-10. 鞍点系で解く（摩擦有無で分岐）
            if _use_friction and n_geom_active > 0:
                # Alart-Curnier 摩擦拡大鞍点系
                G_t_mat = _build_tangential_constraint_jacobian(
                    manager, active_indices, ndof, ndof_per_node
                )
                _k_t = k_pen * manager.config.k_t_ratio

                du, dlam_n, dlam_t = _solve_augmented_friction_system(
                    K_T,
                    G_mat,
                    G_t_mat,
                    k_pen,
                    _k_t,
                    R_u,
                    C_ac,
                    C_t_ac,
                    J_blocks,
                    fixed_dofs,
                    n_ncp_active,
                    ncp_active_mask,
                    active_indices,
                )

                # 11. 更新
                u += du

                # 法線乗数の更新
                if n_ncp_active > 0:
                    active_rows_n = np.where(ncp_active_mask)[0]
                    for j_local in range(n_ncp_active):
                        j_geom = int(active_rows_n[j_local])
                        pair_idx = active_indices[j_geom]
                        lam_all[pair_idx] += dlam_n[j_local]

                    # 接線乗数の更新
                    for j_local in range(n_ncp_active):
                        j_geom = int(active_rows_n[j_local])
                        pair_idx = active_indices[j_geom]
                        lam_t_all[pair_idx] += dlam_t[2 * j_local : 2 * j_local + 2]

            else:
                # 従来の法線のみ鞍点系
                if n_ncp_active > 0:
                    active_row_indices = np.where(ncp_active_mask)[0]
                    G_A = G_mat[active_row_indices, :]
                    g_A = gaps[active_row_indices]
                else:
                    G_A = sp.csr_matrix((0, ndof))
                    g_A = np.array([])

                du, dlam_A = _solve_saddle_point_contact(
                    K_T,
                    G_A,
                    k_pen,
                    R_u,
                    g_A,
                    fixed_dofs,
                    linear_solver=linear_solver_mode,
                    iterative_tol=iterative_tol_cfg,
                    ilu_drop_tol=ilu_drop_tol_cfg,
                    use_block_preconditioner=use_block_preconditioner,
                )

                # 11. 更新
                u += du

                # NCP アクティブ乗数の更新
                if n_ncp_active > 0:
                    active_pair_indices = [
                        active_indices[j] for j in range(n_geom_active) if ncp_active_mask[j]
                    ]
                    for j_local, pair_idx in enumerate(active_pair_indices):
                        lam_all[pair_idx] += dlam_A[j_local]

            # NCP 非アクティブ乗数をゼロに
            for j in range(n_geom_active):
                if not ncp_active_mask[j]:
                    lam_all[active_indices[j]] = 0.0
                    lam_t_all[active_indices[j]] = 0.0

            # λ_n ≥ 0 射影
            lam_all = np.maximum(lam_all, 0.0)

            # 変位ノルム判定
            u_norm = float(np.linalg.norm(u))
            du_norm = float(np.linalg.norm(du))
            if u_norm > 1e-30 and du_norm / u_norm < tol_disp and ncp_conv:
                step_converged = True
                if show_progress:
                    print(
                        f"  Step {step}/{n_load_steps}, iter {it}, "
                        f"||du||/||u|| = {du_norm / u_norm:.3e} "
                        f"(disp converged, {n_ncp_active} active)"
                    )
                break

        if not step_converged:
            if show_progress:
                print(f"  WARNING: Step {step} did not converge in {max_iter} iterations.")
            return NCPSolveResult(
                u=u,
                lambdas=lam_all,
                converged=False,
                n_load_steps=step,
                total_newton_iterations=total_newton,
                n_active_final=manager.n_active,
                load_history=load_history,
                displacement_history=disp_history,
                contact_force_history=contact_force_history,
                graph_history=graph_history,
            )

        # ステップ完了
        for i, pair in enumerate(manager.pairs):
            if i < len(lam_all):
                pair.state.lambda_n = lam_all[i]
                pair.state.p_n = max(0.0, lam_all[i] + k_pen * (-pair.state.gap))

        # 摩擦用: 参照変位を更新（次ステップ用）
        u_ref = u.copy()

        load_history.append(load_frac)
        disp_history.append(u.copy())
        f_c_norm = float(np.linalg.norm(f_c))
        contact_force_history.append(f_c_norm)

        try:
            graph = snapshot_contact_graph(manager, step_index=step - 1)
            graph_history.add_snapshot(graph)
        except Exception:
            pass

    return NCPSolveResult(
        u=u,
        lambdas=lam_all,
        converged=True,
        n_load_steps=n_load_steps,
        total_newton_iterations=total_newton,
        n_active_final=manager.n_active,
        load_history=load_history,
        displacement_history=disp_history,
        contact_force_history=contact_force_history,
        graph_history=graph_history,
    )
