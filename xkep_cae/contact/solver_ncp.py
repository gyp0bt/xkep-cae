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

from xkep_cae.contact.assembly import _contact_dofs, _contact_shape_vector
from xkep_cae.contact.graph import ContactGraphHistory, snapshot_contact_graph
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
) -> tuple[np.ndarray, np.ndarray]:
    """Alart-Curnier NCP 鞍点系を Schur complement で解く.

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

    if n_active == 0:
        K_bc = K_T.tolil()
        rhs = -R_u.copy()
        _apply_bc(K_bc, rhs, fixed_dofs)
        du = _solve_linear_system(K_bc.tocsr(), rhs, **solve_kw)
        return du, np.array([])

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
) -> NCPSolveResult:
    """Semi-smooth Newton 法による接触解析（AL-NCP + 鞍点定式化）.

    Outer loop 不要。u と λ を同時に更新。

    Alart-Curnier NCP + 鞍点系:
      Active set: A = {i : λ_i + k_pen*(-g_i) > 0}
      Active pairs → g = 0 を制約として鞍点系で解く
      Inactive pairs → λ = 0 に設定

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

    Returns:
        NCPSolveResult
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)

    # k_pen の決定
    if k_pen <= 0.0:
        k_pen = manager.config.k_pen_scale

    # λ の初期化
    n_pairs = manager.n_pairs
    lam_all = np.zeros(n_pairs)

    # 線形ソルバー設定
    linear_solver_mode = manager.config.linear_solver
    iterative_tol_cfg = manager.config.iterative_tol
    ilu_drop_tol_cfg = manager.config.ilu_drop_tol

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
    graph_history = ContactGraphHistory()
    total_newton = 0

    f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
    if f_ext_ref_norm < 1e-30:
        f_ext_ref_norm = 1.0

    _f_ext_base = f_ext_base if f_ext_base is not None else np.zeros(ndof)

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

            # 4. 接触力ベクトル
            f_c = _compute_contact_force_from_lambdas(
                manager, lam_all, ndof, ndof_per_node, k_pen=k_pen
            )

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

            # 7. 収束判定
            res_u_norm = float(np.linalg.norm(R_u))
            ncp_norm = float(np.linalg.norm(C_ac)) if n_geom_active > 0 else 0.0

            force_conv = res_u_norm / f_ext_ref_norm < tol_force
            ncp_conv = ncp_norm < tol_ncp

            n_ncp_active = int(np.sum(ncp_active_mask))

            if force_conv and ncp_conv:
                step_converged = True
                if show_progress:
                    print(
                        f"  Step {step}/{n_load_steps}, iter {it}, "
                        f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e}, "
                        f"||C_AC|| = {ncp_norm:.3e} "
                        f"(converged, {n_ncp_active} active)"
                    )
                break

            if show_progress and it % 5 == 0:
                print(
                    f"  Step {step}/{n_load_steps}, iter {it}, "
                    f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e}, "
                    f"||C_AC|| = {ncp_norm:.3e}, "
                    f"active={n_ncp_active}/{n_geom_active}"
                )

            # 8. 構造接線剛性
            K_T = assemble_tangent(u)

            # 9. NCP アクティブ制約行列を抽出
            if n_ncp_active > 0:
                active_row_indices = np.where(ncp_active_mask)[0]
                G_A = G_mat[active_row_indices, :]
                g_A = gaps[active_row_indices]
            else:
                G_A = sp.csr_matrix((0, ndof))
                g_A = np.array([])

            # 10. 鞍点系で解く
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

            # λ ≥ 0 射影
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
