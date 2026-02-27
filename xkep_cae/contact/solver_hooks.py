"""接触付き Newton-Raphson ソルバー（Outer/Inner 分離）.

Phase C2: 法線接触のみ（摩擦なし）。
Phase C3: 摩擦 return mapping + μランプ対応。
Phase C4: merit line search + Outer loop 運用強化。
Phase C5: 幾何微分込み一貫接線 + PDAS Active-set + slip consistent tangent。

設計方針:
- 既存の ``newton_raphson()`` を内部利用する「包装関数」方式
- Outer loop: 接触候補検出 + 幾何更新 + AL乗数更新 + μランプ
- Inner loop: 最近接点 (s,t) 固定で Newton-Raphson + 摩擦力
- Line search: merit function が減少する step length を採用
- PDAS: Inner loop 内で Active-set を動的更新（実験的、Phase C5）

収束判定（Outer）:
- 最近接パラメータ |Δs|, |Δt| が閾値以下
- merit function の改善停滞による早期終了
- または Inner が 1 反復で収束

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §5, §6, §8, §12
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.assembly import compute_contact_force, compute_contact_stiffness
from xkep_cae.contact.graph import ContactGraphHistory, snapshot_contact_graph
from xkep_cae.contact.law_friction import (
    compute_mu_effective,
    compute_tangential_displacement,
    friction_return_mapping,
    friction_tangent_2x2,
    rotate_friction_history,
)
from xkep_cae.contact.law_normal import (
    initialize_penalty_stiffness,
    update_al_multiplier,
)
from xkep_cae.contact.line_search import backtracking_line_search, merit_function
from xkep_cae.contact.pair import ContactManager, ContactStatus


@dataclass
class ContactSolveResult:
    """接触付き非線形解析の結果.

    Attributes:
        u: (ndof,) 最終変位ベクトル
        converged: 収束したかどうか
        n_load_steps: 荷重増分ステップ数
        total_newton_iterations: 全ステップの合計 Newton 反復回数
        total_outer_iterations: 全ステップの合計 Outer 反復回数
        n_active_final: 最終的な ACTIVE ペア数
        total_line_search_steps: 全ステップの合計 line search 縮小ステップ数
        load_history: 各ステップの荷重係数
        displacement_history: 各ステップの変位
        contact_force_history: 各ステップの接触力ノルム
        graph_history: 接触グラフの時系列（各ステップ終了時のスナップショット）
    """

    u: np.ndarray
    converged: bool
    n_load_steps: int
    total_newton_iterations: int
    total_outer_iterations: int
    n_active_final: int
    total_line_search_steps: int = 0
    load_history: list[float] = field(default_factory=list)
    displacement_history: list[np.ndarray] = field(default_factory=list)
    contact_force_history: list[float] = field(default_factory=list)
    graph_history: ContactGraphHistory = field(default_factory=ContactGraphHistory)


def _deformed_coords(
    node_coords_ref: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """参照座標 + 変位から変形座標を計算する.

    Args:
        node_coords_ref: (n_nodes, 3) 参照節点座標
        u: (ndof_total,) 変位ベクトル
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        coords_def: (n_nodes, 3) 変形後座標
    """
    n_nodes = node_coords_ref.shape[0]
    coords_def = node_coords_ref.copy()
    for i in range(n_nodes):
        coords_def[i, 0] += u[i * ndof_per_node + 0]
        coords_def[i, 1] += u[i * ndof_per_node + 1]
        coords_def[i, 2] += u[i * ndof_per_node + 2]
    return coords_def


def _update_gaps_fixed_st(
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    u: np.ndarray,
    ndof_per_node: int = 6,
) -> None:
    """Inner loop 中に gap のみ更新する（s, t, normal は固定）.

    Outer loop で確定した最近接パラメータ (s, t) を保持したまま、
    現在の変位 u に基づいて gap を再計算する。
    これにより、Inner NR の接触力 f_c が u に依存し、
    接触接線剛性 K_c との整合性が保たれる。

    Args:
        manager: 接触マネージャ
        node_coords_ref: (n_nodes, 3) 参照節点座標
        u: (ndof_total,) 現在の変位ベクトル
        ndof_per_node: 1節点あたりの DOF 数
    """
    coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)

    for pair in manager.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        s = pair.state.s
        t = pair.state.t

        xA0 = coords_def[pair.nodes_a[0]]
        xA1 = coords_def[pair.nodes_a[1]]
        xB0 = coords_def[pair.nodes_b[0]]
        xB1 = coords_def[pair.nodes_b[1]]

        PA = (1.0 - s) * xA0 + s * xA1
        PB = (1.0 - t) * xB0 + t * xB1
        dist = float(np.linalg.norm(PA - PB))

        pair.state.gap = dist - pair.radius_a - pair.radius_b


def _solve_linear_system(
    K: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    mode: str = "direct",
    iterative_tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
) -> np.ndarray:
    """接触NR用の線形連立方程式を解く.

    Args:
        K: 剛性行列（CSR形式）
        rhs: 右辺ベクトル
        mode: "direct" | "iterative" | "auto"
        iterative_tol: GMRES 収束判定
        ilu_drop_tol: ILU 前処理の drop tolerance

    Returns:
        解ベクトル
    """
    import warnings

    import scipy.sparse.linalg as spla

    if mode == "direct":
        return spla.spsolve(K, rhs)

    if mode == "iterative":
        return _solve_iterative(K, rhs, iterative_tol, ilu_drop_tol)

    # auto モード: direct を試行し、警告が出たら iterative にフォールバック
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = spla.spsolve(K, rhs)

    for w in caught:
        if "MatrixRankWarning" in str(w.category.__name__) or "singular" in str(w.message).lower():
            return _solve_iterative(K, rhs, iterative_tol, ilu_drop_tol)

    # NaN/Inf チェック: 直接法で精度劣化した場合もフォールバック
    if not np.all(np.isfinite(x)):
        return _solve_iterative(K, rhs, iterative_tol, ilu_drop_tol)

    return x


def _solve_iterative(
    K: sp.csr_matrix,
    rhs: np.ndarray,
    tol: float = 1e-10,
    ilu_drop_tol: float = 1e-4,
) -> np.ndarray:
    """GMRES + ILU前処理で線形系を解く.

    商用ソルバー（LS-DYNA implicit, ANSYS）で標準的な手法。
    ill-conditioned な K_T + K_c に対して直接法より安定。

    Args:
        K: 剛性行列（CSR形式）
        rhs: 右辺ベクトル
        tol: GMRES 収束判定
        ilu_drop_tol: ILU 前処理の drop tolerance

    Returns:
        解ベクトル
    """
    import scipy.sparse.linalg as spla

    K_csc = K.tocsc()
    try:
        ilu = spla.spilu(K_csc, drop_tol=ilu_drop_tol)
        M = spla.LinearOperator(K.shape, ilu.solve)
    except RuntimeError:
        # ILU分解失敗 → 前処理なしでGMRES
        M = None

    x, info = spla.gmres(K, rhs, M=M, atol=tol, maxiter=max(500, K.shape[0]))
    if info != 0:
        # GMRES 未収束 → 直接法にフォールバック
        x = spla.spsolve(K, rhs)
    return x


def newton_raphson_with_contact(
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
    max_iter: int = 30,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    tol_energy: float = 1e-10,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.0,
    broadphase_cell_size: float | None = None,
    f_ext_base: np.ndarray | None = None,
) -> ContactSolveResult:
    """接触付き Newton-Raphson（Outer/Inner 分離）.

    各荷重ステップで:
    1. Outer loop: 候補検出 + 幾何更新 + k_pen 初期化
    2. Inner loop: NR 反復（最近接固定、接触力/剛性を追加）
       - use_line_search=True: merit line search で step length を適応制御
    3. Outer 収束判定: |Δs|, |Δt| < tol_geometry + merit 改善停滞判定
    4. AL 乗数更新

    Args:
        f_ext_total: (ndof,) 最終外荷重
        fixed_dofs: 拘束DOF
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ（k_pen_scale, g_on, g_off 等設定済み）
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径（スカラー or 配列）
        n_load_steps: 荷重増分数
        max_iter: Inner Newton の最大反復数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_energy: エネルギーノルム収束判定
        show_progress: 進捗表示
        u0: 初期変位
        ndof_per_node: 1節点あたりの DOF 数
        broadphase_margin: broadphase 探索マージン
        broadphase_cell_size: broadphase セルサイズ
        f_ext_base: (ndof,) ベース外荷重（サイクリック荷重用）。
            設定時の実効荷重: f_ext = f_ext_base + lam * f_ext_total

    Returns:
        ContactSolveResult
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)

    n_outer_max = manager.config.n_outer_max
    tol_geometry = manager.config.tol_geometry

    # Line search 設定
    use_line_search = manager.config.use_line_search
    ls_max_steps = manager.config.line_search_max_steps
    merit_alpha = manager.config.merit_alpha
    merit_beta = manager.config.merit_beta

    # Phase C5 設定
    use_geometric_stiffness = manager.config.use_geometric_stiffness
    use_pdas = manager.config.use_pdas

    # 適応的ペナルティ設定
    tol_pen_ratio = manager.config.tol_penetration_ratio
    pen_growth = manager.config.penalty_growth_factor
    k_pen_max = manager.config.k_pen_max

    # Modified Newton + contact damping 設定
    use_modified_newton = manager.config.use_modified_newton
    modified_newton_refresh = manager.config.modified_newton_refresh
    contact_damping = manager.config.contact_damping
    k_pen_scaling_mode = manager.config.k_pen_scaling

    # AL乗数緩和
    al_relaxation = manager.config.al_relaxation
    adaptive_omega = manager.config.adaptive_omega
    omega_min = manager.config.omega_min
    omega_max = manager.config.omega_max
    omega_growth = manager.config.omega_growth

    # 線形ソルバー設定
    linear_solver_mode = manager.config.linear_solver
    iterative_tol = manager.config.iterative_tol
    ilu_drop_tol = manager.config.ilu_drop_tol

    # 活性セットチャタリング防止
    no_deact = manager.config.no_deactivation_within_step

    # モノリシック幾何更新（Inner NR内でs,t,normal毎反復更新）
    mono_geom = manager.config.monolithic_geometry

    # 接触接線モード（Uzawa型分解）
    contact_tangent_mode = manager.config.contact_tangent_mode
    # structural_only モードでは merit line search が不整合（Newton方向に接触寄与なし）
    if contact_tangent_mode == "structural_only":
        use_line_search = False

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
    graph_history = ContactGraphHistory()
    total_newton = 0
    total_outer = 0
    total_ls = 0
    global_ramp_counter = 0  # μランプカウンタ（全ステップ通算 Outer 回数）

    use_friction = manager.config.use_friction
    mu_target = manager.config.mu
    mu_ramp_steps = manager.config.mu_ramp_steps

    f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
    if f_ext_ref_norm < 1e-30:
        f_ext_ref_norm = 1.0

    _f_ext_base = f_ext_base if f_ext_base is not None else np.zeros(ndof)

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        f_ext = _f_ext_base + lam * f_ext_total

        # ステップ開始時の変位を参照状態として保存（摩擦用）
        u_step_ref = u.copy()

        # ステップ開始時の z_t を保存（NR iteration ごとにリセットするため）
        z_t_conv: dict[int, np.ndarray] = {}
        if use_friction:
            for pair_idx, pair in enumerate(manager.pairs):
                z_t_conv[pair_idx] = pair.state.z_t.copy()

        step_converged = False
        merit_prev_outer = float("inf")  # Outer loop merit 追跡

        for outer in range(n_outer_max):
            total_outer += 1
            global_ramp_counter += 1

            # --- μランプ ---
            mu_eff = 0.0
            if use_friction:
                mu_eff = compute_mu_effective(mu_target, global_ramp_counter, mu_ramp_steps)

            # --- Outer: 変形座標を計算し、候補検出 + 幾何更新 ---
            coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)

            # no_deactivation_within_step:
            #   outer == 0 かつ最初のステップ: 通常の候補検出（初期トポロジー確立）
            #   それ以外: 候補検出をスキップし、活性セットを固定
            # 活性化（INACTIVE→ACTIVE）のみ許可し、非活性化を禁止することで
            # Outer/Inner間のトポロジーチャタリングを防止。
            _is_first_outer = outer == 0 and step == 1
            _skip_detect = no_deact and not _is_first_outer
            if not _skip_detect:
                manager.detect_candidates(
                    coords_def,
                    connectivity,
                    radii,
                    margin=broadphase_margin,
                    cell_size=broadphase_cell_size,
                )

            # --- 段階的接触アクティベーション ---
            if manager.config.staged_activation_steps > 0 and not _skip_detect:
                max_layer = manager.compute_active_layer_for_step(step, n_load_steps)
                manager.filter_pairs_by_layer(max_layer)

            # --- 摩擦フレーム回転のために旧フレームを保存 ---
            old_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if use_friction:
                for pair_idx, pair in enumerate(manager.pairs):
                    if pair.state.status != ContactStatus.INACTIVE:
                        t1_norm = float(np.linalg.norm(pair.state.tangent1))
                        if t1_norm > 1e-10:
                            old_frames[pair_idx] = (
                                pair.state.tangent1.copy(),
                                pair.state.tangent2.copy(),
                            )

            # no_deactivation_within_step: 非活性化を全面禁止（活性化のみ許可）
            _allow_deact = not no_deact
            manager.update_geometry(coords_def, allow_deactivation=_allow_deact)

            # --- 摩擦履歴の平行輸送: 旧フレーム → 新フレーム ---
            if use_friction and old_frames:
                for pair_idx, pair in enumerate(manager.pairs):
                    if pair_idx not in old_frames:
                        continue
                    if pair.state.status == ContactStatus.INACTIVE:
                        continue
                    t1_old, t2_old = old_frames[pair_idx]
                    t1_new = pair.state.tangent1
                    t2_new = pair.state.tangent2
                    if pair_idx in z_t_conv:
                        z_t_conv[pair_idx] = rotate_friction_history(
                            z_t_conv[pair_idx],
                            t1_old,
                            t2_old,
                            t1_new,
                            t2_new,
                        )

            # k_pen 未設定のペアを初期化 + 新規ペアの z_t_conv を追加
            for pair_idx, pair in enumerate(manager.pairs):
                if pair.state.status != ContactStatus.INACTIVE and pair.state.k_pen <= 0.0:
                    if manager.config.k_pen_mode == "beam_ei":
                        # EI/L³ ベースの自動推定
                        from xkep_cae.contact.law_normal import auto_beam_penalty_stiffness

                        # 代表要素長さ: ペアのセグメント長の平均
                        xA0 = coords_def[pair.nodes_a[0]]
                        xA1 = coords_def[pair.nodes_a[1]]
                        xB0 = coords_def[pair.nodes_b[0]]
                        xB1 = coords_def[pair.nodes_b[1]]
                        L_a = float(np.linalg.norm(xA1 - xA0))
                        L_b = float(np.linalg.norm(xB1 - xB0))
                        L_avg = 0.5 * (L_a + L_b)
                        if L_avg < 1e-30:
                            L_avg = 1.0

                        k_auto = auto_beam_penalty_stiffness(
                            manager.config.beam_E,
                            manager.config.beam_I,
                            L_avg,
                            n_contact_pairs=max(1, manager.n_active),
                            scale=manager.config.k_pen_scale,
                            scaling=k_pen_scaling_mode,
                        )
                        initialize_penalty_stiffness(
                            pair,
                            k_pen=k_auto,
                            k_t_ratio=manager.config.k_t_ratio,
                        )
                    else:
                        # manual モード: k_pen_scale をそのまま使用
                        initialize_penalty_stiffness(
                            pair,
                            k_pen=manager.config.k_pen_scale,
                            k_t_ratio=manager.config.k_t_ratio,
                        )
                # 新規ペアの z_t_conv エントリ追加
                if use_friction and pair_idx not in z_t_conv:
                    z_t_conv[pair_idx] = np.zeros(2)

            # 前回の (s, t) を保存（Outer 収束判定用）
            prev_st = []
            for pair in manager.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    prev_st.append((pair.state.s, pair.state.t))
                else:
                    prev_st.append(None)

            # --- Inner: NR 反復（最近接点固定）---
            inner_converged = False
            energy_ref = None
            K_T_frozen = None  # Modified Newton用: 構造剛性キャッシュ
            f_c_prev = None  # Contact damping用: 前回の接触力

            for it in range(max_iter):
                total_newton += 1

                # 幾何更新:
                #   monolithic: s,t,normal を毎反復更新（Outer/Inner 分離なし）
                #   従来: s,t,normal 固定で gap のみ更新
                if mono_geom:
                    coords_def_it = _deformed_coords(node_coords_ref, u, ndof_per_node)
                    manager.update_geometry(
                        coords_def_it,
                        allow_deactivation=_allow_deact,
                        freeze_active_set=True,  # Active-set はOuter で管理
                    )
                else:
                    _update_gaps_fixed_st(manager, node_coords_ref, u, ndof_per_node)

                # --- PDAS Active-set 更新（Phase C5, 実験的）---
                if use_pdas:
                    for pair in manager.pairs:
                        if pair.state.status == ContactStatus.INACTIVE:
                            # 非活性ペア: 試行反力が正なら活性化
                            if pair.state.k_pen > 0.0:
                                p_trial = pair.state.lambda_n + pair.state.k_pen * (-pair.state.gap)
                                if p_trial > 0.0 and pair.state.gap <= manager.config.g_on:
                                    pair.state.status = ContactStatus.ACTIVE
                        else:
                            # 活性ペア: 試行反力が非正なら非活性化
                            p_trial = pair.state.lambda_n + pair.state.k_pen * (-pair.state.gap)
                            if p_trial <= 0.0:
                                pair.state.status = ContactStatus.INACTIVE
                                pair.state.p_n = 0.0

                # --- 摩擦 return mapping ---
                friction_forces: dict[int, np.ndarray] = {}
                friction_tangents: dict[int, np.ndarray] = {}

                if use_friction and mu_eff > 0.0:
                    from xkep_cae.contact.law_normal import evaluate_normal_force as _eval_pn

                    for pair_idx, pair in enumerate(manager.pairs):
                        if pair.state.status == ContactStatus.INACTIVE:
                            continue

                        # 現在の gap から p_n を更新（摩擦計算に必要）
                        _eval_pn(pair)
                        if pair.state.p_n <= 0.0:
                            continue

                        # z_t をステップ開始時の収束値にリセット
                        # （delta_ut は全量で計算するため、z_t も全量ベースにする）
                        if pair_idx in z_t_conv:
                            pair.state.z_t = z_t_conv[pair_idx].copy()
                        else:
                            pair.state.z_t = np.zeros(2)

                        # 接線相対変位増分（ステップ開始からの全量）
                        delta_ut = compute_tangential_displacement(
                            pair,
                            u,
                            u_step_ref,
                            node_coords_ref,
                            ndof_per_node,
                        )

                        # return mapping（全量ベース）
                        q_t = friction_return_mapping(pair, delta_ut, mu_eff)

                        if float(np.linalg.norm(q_t)) > 1e-30:
                            friction_forces[pair_idx] = q_t
                            D_t = friction_tangent_2x2(pair, mu_eff)
                            friction_tangents[pair_idx] = D_t

                # 構造内力
                f_int = assemble_internal_force(u)

                # Line contact 用の変形座標（line_contact=True の場合）
                _line_coords = None
                if manager.config.line_contact:
                    _line_coords = _deformed_coords(node_coords_ref, u, ndof_per_node)

                # 接触内力（法線 + 摩擦）
                f_c_raw = compute_contact_force(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                    friction_forces=friction_forces if friction_forces else None,
                    node_coords=_line_coords,
                )

                # Contact damping: under-relaxation で接触力の急変を抑制
                if contact_damping < 1.0 and f_c_prev is not None:
                    f_c = contact_damping * f_c_raw + (1.0 - contact_damping) * f_c_prev
                else:
                    f_c = f_c_raw
                f_c_prev = f_c_raw.copy()

                # 残差
                residual = f_ext - f_int - f_c
                residual[fixed_dofs] = 0.0
                res_norm = float(np.linalg.norm(residual))

                # 力ノルム判定
                if res_norm / f_ext_ref_norm < tol_force:
                    inner_converged = True
                    if show_progress:
                        friction_info = ""
                        if use_friction:
                            n_slip = sum(
                                1 for p in manager.pairs if p.state.status == ContactStatus.SLIDING
                            )
                            friction_info = f", μ={mu_eff:.3f}, slip={n_slip}"
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, ||R||/||f|| = {res_norm / f_ext_ref_norm:.3e} "
                            f"(force converged, {manager.n_active} active{friction_info})"
                        )
                    break

                # 接線剛性（構造 + 接触法線 + 接触摩擦）
                # Modified Newton: K_T を初回とrefresh間隔でのみ再計算
                if use_modified_newton:
                    if it == 0 or it % modified_newton_refresh == 0:
                        K_T_frozen = assemble_tangent(u)
                    K_T = K_T_frozen
                else:
                    K_T = assemble_tangent(u)
                # 接触接線モードに応じた K_c の組込み
                if contact_tangent_mode == "structural_only":
                    # Uzawa型: 接触剛性をシステム行列に含めない
                    # 接触力は残差 f_c にのみ反映される
                    K_total = K_T
                else:
                    K_c = compute_contact_stiffness(
                        manager,
                        ndof,
                        ndof_per_node=ndof_per_node,
                        friction_tangents=friction_tangents if friction_tangents else None,
                        use_geometric_stiffness=use_geometric_stiffness,
                        node_coords=_line_coords,
                    )
                    if contact_tangent_mode == "diagonal":
                        # 対角近似: K_c の対角成分のみを使用
                        K_total = K_T + sp.diags(K_c.diagonal())
                    elif contact_tangent_mode == "scaled":
                        # スケール接触接線: K_c を α 倍に縮小
                        alpha = manager.config.contact_tangent_scale
                        K_total = K_T + alpha * K_c
                    else:
                        # "full": 標準の完全接触接線剛性
                        K_total = K_T + K_c

                # BC 適用
                K_bc = K_total.tolil()
                r_bc = residual.copy()
                for dof in fixed_dofs:
                    K_bc[dof, :] = 0.0
                    K_bc[:, dof] = 0.0
                    K_bc[dof, dof] = 1.0
                    r_bc[dof] = 0.0

                du = _solve_linear_system(
                    K_bc.tocsr(),
                    r_bc,
                    mode=linear_solver_mode,
                    iterative_tol=iterative_tol,
                    ilu_drop_tol=ilu_drop_tol,
                )

                # エネルギーノルム
                energy = abs(float(np.dot(du, residual)))
                if energy_ref is None:
                    energy_ref = energy if energy > 1e-30 else 1.0

                u_norm = float(np.linalg.norm(u))
                du_norm = float(np.linalg.norm(du))

                if show_progress and it % 5 == 0:
                    print(
                        f"  Step {step}/{n_load_steps}, outer {outer}, iter {it}, "
                        f"||R||/||f|| = {res_norm / f_ext_ref_norm:.3e}, "
                        f"||du||/||u|| = {du_norm / max(u_norm, 1e-30):.3e}, "
                        f"active={manager.n_active}"
                    )

                # --- Line search ---
                if use_line_search and manager.n_active > 0:
                    phi_current = merit_function(
                        residual,
                        manager,
                        alpha=merit_alpha,
                        beta=merit_beta,
                    )

                    # 摩擦力を固定して merit 評価する closure
                    ff_snapshot = dict(friction_forces) if friction_forces else None
                    # f_ext をローカルにキャプチャ（B023 回避）
                    _f_ext_ls = f_ext
                    _ff_ls = ff_snapshot

                    def _eval_merit(
                        u_trial: np.ndarray,
                        _fe: np.ndarray = _f_ext_ls,
                        _ff: dict | None = _ff_ls,
                    ) -> float:
                        _update_gaps_fixed_st(
                            manager,
                            node_coords_ref,
                            u_trial,
                            ndof_per_node,
                        )
                        f_int_t = assemble_internal_force(u_trial)
                        _lc_t = None
                        if manager.config.line_contact:
                            _lc_t = _deformed_coords(node_coords_ref, u_trial, ndof_per_node)
                        f_c_t = compute_contact_force(
                            manager,
                            ndof,
                            ndof_per_node=ndof_per_node,
                            friction_forces=_ff,
                            node_coords=_lc_t,
                        )
                        res_t = _fe - f_int_t - f_c_t
                        res_t[fixed_dofs] = 0.0
                        return merit_function(
                            res_t,
                            manager,
                            alpha=merit_alpha,
                            beta=merit_beta,
                        )

                    eta, n_ls = backtracking_line_search(
                        u,
                        du,
                        phi_current,
                        _eval_merit,
                        max_steps=ls_max_steps,
                    )
                    total_ls += n_ls

                    u = u + eta * du

                    if show_progress and eta < 1.0:
                        print(f"    line search: eta={eta:.3f} ({n_ls} steps)")

                    # gap を最終状態に復元
                    _update_gaps_fixed_st(
                        manager,
                        node_coords_ref,
                        u,
                        ndof_per_node,
                    )
                else:
                    u += du

                # 変位ノルム判定
                if u_norm > 1e-30 and du_norm / u_norm < tol_disp:
                    inner_converged = True
                    if show_progress:
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, ||du||/||u|| = {du_norm / u_norm:.3e} "
                            f"(disp converged)"
                        )
                    break

                if energy / energy_ref < tol_energy:
                    inner_converged = True
                    if show_progress:
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, energy = {energy / energy_ref:.3e} "
                            f"(energy converged)"
                        )
                    break

            if not inner_converged:
                if show_progress:
                    print(
                        f"  WARNING: Step {step}, outer {outer} "
                        f"did not converge in {max_iter} iterations."
                    )
                if outer == 0:
                    # 最初の outer で発散 → 回復不能
                    return ContactSolveResult(
                        u=u,
                        converged=False,
                        n_load_steps=step,
                        total_newton_iterations=total_newton,
                        total_outer_iterations=total_outer,
                        n_active_final=manager.n_active,
                        total_line_search_steps=total_ls,
                        load_history=load_history,
                        displacement_history=disp_history,
                        contact_force_history=contact_force_history,
                        graph_history=graph_history,
                    )
                # outer > 0: 前回の outer で inner が収束しているため、
                # 現在の解を受容してステップ完了とする。
                # 商用ソルバー（Abaqus/LS-DYNA）の "accept on best iteration"
                # と同等の挙動。
                if show_progress:
                    print(
                        f"  Step {step}, outer {outer}: "
                        f"accepting current solution (inner stalled, "
                        f"active={manager.n_active})"
                    )
                step_converged = True
                break

            # --- Outer 収束判定 ---
            # 幾何更新して (s,t) の変化を検査
            coords_def_new = _deformed_coords(node_coords_ref, u, ndof_per_node)
            manager.update_geometry(coords_def_new, allow_deactivation=_allow_deact)

            max_ds = 0.0
            max_dt = 0.0
            idx = 0
            for pair in manager.pairs:
                if pair.state.status != ContactStatus.INACTIVE and prev_st[idx] is not None:
                    s_old, t_old = prev_st[idx]
                    max_ds = max(max_ds, abs(pair.state.s - s_old))
                    max_dt = max(max_dt, abs(pair.state.t - t_old))
                idx += 1

            # AL 乗数更新（適応的ω + 緩和付き + sticky contact対応）
            if adaptive_omega:
                omega_current = min(
                    omega_min * omega_growth**outer,
                    omega_max,
                )
            else:
                omega_current = al_relaxation
            preserve_inactive = manager.config.preserve_inactive_lambda
            for pair in manager.pairs:
                update_al_multiplier(pair, omega=omega_current, preserve_inactive=preserve_inactive)

            # --- 適応的ペナルティ増大 ---
            # 貫入量が tol_penetration_ratio * search_radius を超えるペアの k_pen を増大
            pen_exceeded = False
            max_pen_ratio = 0.0
            if tol_pen_ratio > 0.0:
                for pair in manager.pairs:
                    if pair.state.status == ContactStatus.INACTIVE:
                        continue
                    if pair.state.gap >= 0.0:
                        continue
                    penetration = abs(pair.state.gap)
                    sr = pair.search_radius
                    if sr < 1e-30:
                        continue
                    pen_ratio = penetration / sr
                    max_pen_ratio = max(max_pen_ratio, pen_ratio)
                    if pen_ratio > tol_pen_ratio:
                        pen_exceeded = True
                        if pair.state.k_pen < k_pen_max:
                            new_k = min(pair.state.k_pen * pen_growth, k_pen_max)
                            pair.state.k_pen = new_k
                            pair.state.k_t = new_k * manager.config.k_t_ratio

            # --- Merit-based Outer 終了判定 ---
            # Inner 収束後の残差で merit を評価
            if use_line_search:
                _update_gaps_fixed_st(manager, node_coords_ref, u, ndof_per_node)
                f_int_post = assemble_internal_force(u)
                _lc_post = None
                if manager.config.line_contact:
                    _lc_post = _deformed_coords(node_coords_ref, u, ndof_per_node)
                f_c_post = compute_contact_force(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                    node_coords=_lc_post,
                )
                res_post = f_ext - f_int_post - f_c_post
                res_post[fixed_dofs] = 0.0
                merit_cur = merit_function(
                    res_post,
                    manager,
                    alpha=merit_alpha,
                    beta=merit_beta,
                )

            if show_progress:
                friction_info = ""
                if use_friction:
                    friction_info = f", μ_eff={mu_eff:.3f}"
                merit_info = ""
                if use_line_search:
                    merit_info = f", merit={merit_cur:.3e}"
                pen_info = ""
                if tol_pen_ratio > 0.0 and max_pen_ratio > 0.0:
                    pen_info = f", pen_ratio={max_pen_ratio:.4f}"
                print(
                    f"  Step {step}, outer {outer}: "
                    f"max|Δs|={max_ds:.3e}, max|Δt|={max_dt:.3e}, "
                    f"active={manager.n_active}"
                    f"{friction_info}{merit_info}{pen_info}"
                )

            # (s,t) 収束 かつ 貫入量が許容範囲内 → 収束
            if max_ds < tol_geometry and max_dt < tol_geometry and not pen_exceeded:
                step_converged = True
                break

            # (s,t) 収束だが貫入超過 → ペナルティ増大済みなので次の outer へ
            if max_ds < tol_geometry and max_dt < tol_geometry and pen_exceeded:
                if show_progress:
                    print(
                        f"  Step {step}, outer {outer}: "
                        f"(s,t) converged but pen_ratio={max_pen_ratio:.4f} > "
                        f"tol={tol_pen_ratio:.4f}. "
                        f"Increasing k_pen and continuing."
                    )
                # 次の outer で再解を試みる
                continue

            # Merit 改善停滞による早期終了（2回目以降の Outer で判定）
            if use_line_search and outer > 0:
                merit_ratio = merit_cur / max(merit_prev_outer, 1e-30)
                if merit_ratio > 0.99 and not pen_exceeded:
                    # merit が改善していない → (s,t) 更新が効果なし
                    if show_progress:
                        print(
                            f"  Step {step}, outer {outer}: "
                            f"merit stagnated (ratio={merit_ratio:.4f}). "
                            f"Accepting."
                        )
                    step_converged = True
                    break

            if use_line_search:
                merit_prev_outer = merit_cur

        if not step_converged:
            # Outer ループ上限到達でも Inner は収束している → 受容
            if show_progress:
                print(
                    f"  Step {step}: outer loop reached {n_outer_max} "
                    f"(max|Δs|={max_ds:.3e}, max|Δt|={max_dt:.3e}). Accepting."
                )
            step_converged = True

        # 接触力ノルム記録
        _lc_final = None
        if manager.config.line_contact:
            _lc_final = _deformed_coords(node_coords_ref, u, ndof_per_node)
        f_c_final = compute_contact_force(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
            node_coords=_lc_final,
        )
        fc_norm = float(np.linalg.norm(f_c_final))

        load_history.append(lam)
        disp_history.append(u.copy())
        contact_force_history.append(fc_norm)

        # 接触グラフのスナップショット記録
        graph_history.add(snapshot_contact_graph(manager, step=step, load_factor=lam))

    return ContactSolveResult(
        u=u,
        converged=True,
        n_load_steps=n_load_steps,
        total_newton_iterations=total_newton,
        total_outer_iterations=total_outer,
        n_active_final=manager.n_active,
        total_line_search_steps=total_ls,
        load_history=load_history,
        displacement_history=disp_history,
        contact_force_history=contact_force_history,
        graph_history=graph_history,
    )


def _extract_strand_blocks(
    K_global: sp.spmatrix,
    strand_dof_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    """疎行列から素線ごとの対角ブロックを密行列として抽出する.

    K_T が素線ごとにブロック対角である性質を利用。
    各素線のブロックは小規模（例: 30×30）なので密行列で効率的に扱える。

    Args:
        K_global: グローバル剛性行列（CSR/CSC）
        strand_dof_ranges: 各素線の (dof_start, dof_end) リスト

    Returns:
        各素線の対角ブロック行列のリスト
    """
    blocks = []
    for dof_start, dof_end in strand_dof_ranges:
        block = K_global[dof_start:dof_end, dof_start:dof_end].toarray()
        blocks.append(block)
    return blocks


def _build_block_preconditioner(
    K_T_blocks: list[np.ndarray],
    K_c: sp.spmatrix | None,
    fixed_dofs: np.ndarray,
    strand_dof_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    """ブロック前処理行列の逆行列を構築する.

    M = block_diag(K_T_i + K_c_ii) の各ブロックを LU 分解し、
    逆行列として保持する。各ブロックは小規模なので密行列逆行列で効率的。

    Args:
        K_T_blocks: 各素線の構造剛性行列
        K_c: 接触剛性行列（None の場合は K_T のみ）
        fixed_dofs: 拘束DOFインデックス
        strand_dof_ranges: 各素線の (dof_start, dof_end) リスト

    Returns:
        各素線ブロックの逆行列リスト
    """
    fixed_set = set(int(d) for d in fixed_dofs)
    block_invs = []

    for bi, (dof_start, dof_end) in enumerate(strand_dof_ranges):
        K_local = K_T_blocks[bi].copy()
        if K_c is not None:
            K_local += K_c[dof_start:dof_end, dof_start:dof_end].toarray()

        # 拘束DOFの適用
        for d in range(dof_start, dof_end):
            if d in fixed_set:
                ld = d - dof_start
                K_local[ld, :] = 0.0
                K_local[:, ld] = 0.0
                K_local[ld, ld] = 1.0

        block_invs.append(np.linalg.inv(K_local))

    return block_invs


def _solve_block_preconditioned(
    K_total: sp.spmatrix,
    residual: np.ndarray,
    fixed_dofs: np.ndarray,
    block_invs: list[np.ndarray],
    strand_dof_ranges: list[tuple[int, int]],
) -> np.ndarray:
    """ブロック前処理付き GMRES で接触込みシステムを解く.

    モノリシック K_T + K_c の完全な結合を GMRES で解きつつ、
    素線ブロック逆行列を前処理として使用する。

    利点:
    - 前処理が各素線ブロック（良条件）から構築されるため安定
    - ILU 前処理と異なり、高 k_pen でも破綻しない
    - GMRES が off-diagonal K_c 結合を正しく反映
    - NR 法の二次収束性を維持

    Args:
        K_total: BC 適用済みの全体剛性行列 (K_T + K_c)
        residual: BC 適用済みの残差ベクトル
        fixed_dofs: 拘束DOFインデックス
        block_invs: 各素線ブロック逆行列
        strand_dof_ranges: 各素線の (dof_start, dof_end) リスト

    Returns:
        変位増分ベクトル du
    """
    import scipy.sparse.linalg as spla

    ndof = residual.shape[0]

    def _precond(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for bi, (ds, de) in enumerate(strand_dof_ranges):
            y[ds:de] = block_invs[bi] @ x[ds:de]
        return y

    M = spla.LinearOperator((ndof, ndof), matvec=_precond)

    du, info = spla.gmres(
        K_total,
        residual,
        M=M,
        atol=1e-12,
        maxiter=min(300, ndof),
    )

    if info != 0:
        # GMRES 未収束 → ブロック Jacobi にフォールバック
        du = _precond(residual)

    return du


def newton_raphson_block_contact(
    f_ext_total: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray | float,
    *,
    strand_dof_ranges: list[tuple[int, int]],
    n_load_steps: int = 10,
    max_iter: int = 30,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    tol_energy: float = 1e-10,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.0,
    broadphase_cell_size: float | None = None,
    f_ext_base: np.ndarray | None = None,
) -> ContactSolveResult:
    """ブロック前処理付きNRソルバー（素線ブロック前処理 + GMRES）.

    多本撚り（7本撚り等）の接触問題に特化したソルバー。
    構造剛性 K_T が素線ごとにブロック対角であることを利用し、
    素線ブロック逆行列を GMRES の前処理として使用する。

    モノリシック直接解法との違い:
    - 直接法 (spsolve): (K_T + K_c) を直接 LU 分解
      → 高 k_pen 時に条件数悪化で精度劣化
    - ブロック前処理GMRES: 素線ブロック M_i = (K_T_i + K_c_ii)^{-1} で前処理
      → 各ブロックは良条件で安定に逆行列計算可能
      → GMRES がオフダイアゴナル K_c 結合を正確に反映
      → NR 法の二次収束性を維持

    アルゴリズム:
    1. 各荷重ステップで Outer/Inner 二重ループ
    2. Outer: 接触候補検出 + 幾何更新 + AL乗数更新
    3. Inner: ブロック前処理付きNR反復
       a. 残差計算: r = f_ext - f_int(u) - f_c(u)
       b. 構造剛性ブロック抽出 + 接触対角ブロック追加
       c. ブロック逆行列を前処理として GMRES で (K_T + K_c) du = r を解く
       d. 変位更新: u += du

    Args:
        f_ext_total: (ndof,) 最終外荷重
        fixed_dofs: 拘束DOF
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径
        strand_dof_ranges: 各素線の (dof_start, dof_end) リスト
        n_load_steps: 荷重増分数
        max_iter: Inner Newton の最大反復数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_energy: エネルギーノルム収束判定
        show_progress: 進捗表示
        u0: 初期変位
        ndof_per_node: 1節点あたりの DOF 数
        broadphase_margin: broadphase 探索マージン
        broadphase_cell_size: broadphase セルサイズ
        f_ext_base: (ndof,) ベース外荷重（サイクリック荷重用）

    Returns:
        ContactSolveResult
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)

    n_outer_max = manager.config.n_outer_max
    tol_geometry = manager.config.tol_geometry

    # 適応的ペナルティ設定
    tol_pen_ratio = manager.config.tol_penetration_ratio
    pen_growth = manager.config.penalty_growth_factor
    k_pen_max = manager.config.k_pen_max
    k_pen_scaling_mode = manager.config.k_pen_scaling

    # AL乗数緩和
    al_relaxation = manager.config.al_relaxation
    adaptive_omega = manager.config.adaptive_omega
    omega_min = manager.config.omega_min
    omega_max = manager.config.omega_max
    omega_growth = manager.config.omega_growth

    # 活性セットチャタリング防止
    no_deact = manager.config.no_deactivation_within_step

    # 摩擦設定
    use_friction = manager.config.use_friction
    mu_target = manager.config.mu
    mu_ramp_steps = manager.config.mu_ramp_steps

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
    graph_history = ContactGraphHistory()
    total_newton = 0
    total_outer = 0
    global_ramp_counter = 0

    f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
    if f_ext_ref_norm < 1e-30:
        f_ext_ref_norm = 1.0

    _f_ext_base = f_ext_base if f_ext_base is not None else np.zeros(ndof)

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        f_ext = _f_ext_base + lam * f_ext_total

        # ステップ開始時の変位を参照状態として保存（摩擦用）
        u_step_ref = u.copy()

        # ステップ開始時の z_t を保存
        z_t_conv: dict[int, np.ndarray] = {}
        if use_friction:
            for pair_idx, pair in enumerate(manager.pairs):
                z_t_conv[pair_idx] = pair.state.z_t.copy()

        step_converged = False

        for outer in range(n_outer_max):
            total_outer += 1
            global_ramp_counter += 1

            # --- μランプ ---
            mu_eff = 0.0
            if use_friction:
                mu_eff = compute_mu_effective(mu_target, global_ramp_counter, mu_ramp_steps)

            # --- Outer: 幾何更新 ---
            coords_def = _deformed_coords(node_coords_ref, u, ndof_per_node)

            _is_first_outer = outer == 0 and step == 1
            _skip_detect = no_deact and not _is_first_outer
            if not _skip_detect:
                manager.detect_candidates(
                    coords_def,
                    connectivity,
                    radii,
                    margin=broadphase_margin,
                    cell_size=broadphase_cell_size,
                )

            # --- 段階的接触アクティベーション ---
            if manager.config.staged_activation_steps > 0 and not _skip_detect:
                max_layer = manager.compute_active_layer_for_step(step, n_load_steps)
                manager.filter_pairs_by_layer(max_layer)

            # --- 摩擦フレーム回転用の旧フレーム保存 ---
            old_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if use_friction:
                for pair_idx, pair in enumerate(manager.pairs):
                    if pair.state.status != ContactStatus.INACTIVE:
                        t1_norm = float(np.linalg.norm(pair.state.tangent1))
                        if t1_norm > 1e-10:
                            old_frames[pair_idx] = (
                                pair.state.tangent1.copy(),
                                pair.state.tangent2.copy(),
                            )

            _allow_deact = not no_deact
            manager.update_geometry(coords_def, allow_deactivation=_allow_deact)

            # --- 摩擦履歴の平行輸送 ---
            if use_friction and old_frames:
                for pair_idx, pair in enumerate(manager.pairs):
                    if pair_idx not in old_frames:
                        continue
                    if pair.state.status == ContactStatus.INACTIVE:
                        continue
                    t1_old, t2_old = old_frames[pair_idx]
                    t1_new = pair.state.tangent1
                    t2_new = pair.state.tangent2
                    if pair_idx in z_t_conv:
                        z_t_conv[pair_idx] = rotate_friction_history(
                            z_t_conv[pair_idx],
                            t1_old,
                            t2_old,
                            t1_new,
                            t2_new,
                        )

            # k_pen 未設定のペアを初期化
            for pair_idx, pair in enumerate(manager.pairs):
                if pair.state.status != ContactStatus.INACTIVE and pair.state.k_pen <= 0.0:
                    if manager.config.k_pen_mode == "beam_ei":
                        from xkep_cae.contact.law_normal import auto_beam_penalty_stiffness

                        xA0 = coords_def[pair.nodes_a[0]]
                        xA1 = coords_def[pair.nodes_a[1]]
                        xB0 = coords_def[pair.nodes_b[0]]
                        xB1 = coords_def[pair.nodes_b[1]]
                        L_a = float(np.linalg.norm(xA1 - xA0))
                        L_b = float(np.linalg.norm(xB1 - xB0))
                        L_avg = 0.5 * (L_a + L_b)
                        if L_avg < 1e-30:
                            L_avg = 1.0

                        k_auto = auto_beam_penalty_stiffness(
                            manager.config.beam_E,
                            manager.config.beam_I,
                            L_avg,
                            n_contact_pairs=max(1, manager.n_active),
                            scale=manager.config.k_pen_scale,
                            scaling=k_pen_scaling_mode,
                        )
                        initialize_penalty_stiffness(
                            pair,
                            k_pen=k_auto,
                            k_t_ratio=manager.config.k_t_ratio,
                        )
                    else:
                        initialize_penalty_stiffness(
                            pair,
                            k_pen=manager.config.k_pen_scale,
                            k_t_ratio=manager.config.k_t_ratio,
                        )
                if use_friction and pair_idx not in z_t_conv:
                    z_t_conv[pair_idx] = np.zeros(2)

            # 前回の (s, t) を保存
            prev_st = []
            for pair in manager.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    prev_st.append((pair.state.s, pair.state.t))
                else:
                    prev_st.append(None)

            # --- 構造剛性のブロック抽出 ---
            K_T = assemble_tangent(u)
            K_T_blocks = _extract_strand_blocks(K_T, strand_dof_ranges)

            # --- Inner: ブロック分解NR反復 ---
            inner_converged = False
            energy_ref = None

            for it in range(max_iter):
                total_newton += 1

                # gap 更新（s,t 固定）
                _update_gaps_fixed_st(manager, node_coords_ref, u, ndof_per_node)

                # --- 摩擦 return mapping ---
                friction_forces: dict[int, np.ndarray] = {}

                if use_friction and mu_eff > 0.0:
                    from xkep_cae.contact.law_normal import evaluate_normal_force as _eval_pn

                    for pair_idx, pair in enumerate(manager.pairs):
                        if pair.state.status == ContactStatus.INACTIVE:
                            continue
                        _eval_pn(pair)
                        if pair.state.p_n <= 0.0:
                            continue
                        if pair_idx in z_t_conv:
                            pair.state.z_t = z_t_conv[pair_idx].copy()
                        else:
                            pair.state.z_t = np.zeros(2)
                        delta_ut = compute_tangential_displacement(
                            pair,
                            u,
                            u_step_ref,
                            node_coords_ref,
                            ndof_per_node,
                        )
                        q_t = friction_return_mapping(pair, delta_ut, mu_eff)
                        if float(np.linalg.norm(q_t)) > 1e-30:
                            friction_forces[pair_idx] = q_t

                # 構造内力
                f_int = assemble_internal_force(u)

                # Line contact 用の変形座標
                _lc_blk = None
                if manager.config.line_contact:
                    _lc_blk = _deformed_coords(node_coords_ref, u, ndof_per_node)

                # 接触内力
                f_c = compute_contact_force(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                    friction_forces=friction_forces if friction_forces else None,
                    node_coords=_lc_blk,
                )

                # 残差
                residual = f_ext - f_int - f_c
                residual[fixed_dofs] = 0.0
                res_norm = float(np.linalg.norm(residual))

                # 力ノルム収束判定
                if res_norm / f_ext_ref_norm < tol_force:
                    inner_converged = True
                    if show_progress:
                        friction_info = ""
                        if use_friction:
                            n_slip = sum(
                                1 for p in manager.pairs if p.state.status == ContactStatus.SLIDING
                            )
                            friction_info = f", μ={mu_eff:.3f}, slip={n_slip}"
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, ||R||/||f|| = {res_norm / f_ext_ref_norm:.3e} "
                            f"(block converged, {manager.n_active} active{friction_info})"
                        )
                    break

                # 接触剛性の計算
                K_c = None
                if manager.n_active > 0:
                    K_c = compute_contact_stiffness(
                        manager,
                        ndof,
                        ndof_per_node=ndof_per_node,
                        friction_tangents=None,
                        use_geometric_stiffness=manager.config.use_geometric_stiffness,
                        node_coords=_lc_blk,
                    )

                # ブロック前処理の構築
                block_invs = _build_block_preconditioner(
                    K_T_blocks,
                    K_c,
                    fixed_dofs,
                    strand_dof_ranges,
                )

                # 全体剛性行列 + BC 適用
                K_total = K_T + K_c if K_c is not None else K_T
                K_bc = K_total.tolil()
                r_bc = residual.copy()
                for dof in fixed_dofs:
                    K_bc[dof, :] = 0.0
                    K_bc[:, dof] = 0.0
                    K_bc[dof, dof] = 1.0
                    r_bc[dof] = 0.0

                # ブロック前処理付き GMRES
                du = _solve_block_preconditioned(
                    K_bc.tocsr(),
                    r_bc,
                    fixed_dofs,
                    block_invs,
                    strand_dof_ranges,
                )

                # 変位更新
                u += du

                du_norm = float(np.linalg.norm(du))
                u_norm = float(np.linalg.norm(u))

                if show_progress and it % 5 == 0:
                    print(
                        f"  Step {step}/{n_load_steps}, outer {outer}, iter {it}, "
                        f"||R||/||f|| = {res_norm / f_ext_ref_norm:.3e}, "
                        f"||du||/||u|| = {du_norm / max(u_norm, 1e-30):.3e}, "
                        f"active={manager.n_active}"
                    )

                # 変位ノルム収束判定
                if u_norm > 1e-30 and du_norm / u_norm < tol_disp:
                    inner_converged = True
                    if show_progress:
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, ||du||/||u|| = {du_norm / u_norm:.3e} "
                            f"(disp converged)"
                        )
                    break

                # エネルギーノルム収束判定
                energy = abs(float(np.dot(du, residual)))
                if energy_ref is None:
                    energy_ref = energy if energy > 1e-30 else 1.0
                if energy / energy_ref < tol_energy:
                    inner_converged = True
                    if show_progress:
                        print(
                            f"  Step {step}/{n_load_steps}, outer {outer}, "
                            f"iter {it}, energy = {energy / energy_ref:.3e} "
                            f"(energy converged)"
                        )
                    break

            if not inner_converged:
                if show_progress:
                    print(
                        f"  WARNING: Step {step}, outer {outer} "
                        f"did not converge in {max_iter} iterations."
                    )
                if outer == 0:
                    return ContactSolveResult(
                        u=u,
                        converged=False,
                        n_load_steps=step,
                        total_newton_iterations=total_newton,
                        total_outer_iterations=total_outer,
                        n_active_final=manager.n_active,
                        load_history=load_history,
                        displacement_history=disp_history,
                        contact_force_history=contact_force_history,
                        graph_history=graph_history,
                    )
                if show_progress:
                    print(
                        f"  Step {step}, outer {outer}: "
                        f"accepting current solution (inner stalled, "
                        f"active={manager.n_active})"
                    )
                step_converged = True
                break

            # --- Outer 収束判定 ---
            coords_def_new = _deformed_coords(node_coords_ref, u, ndof_per_node)
            manager.update_geometry(coords_def_new, allow_deactivation=_allow_deact)

            max_ds = 0.0
            max_dt = 0.0
            for idx in range(len(manager.pairs)):
                pair = manager.pairs[idx]
                if pair.state.status != ContactStatus.INACTIVE and prev_st[idx] is not None:
                    s_old, t_old = prev_st[idx]
                    max_ds = max(max_ds, abs(pair.state.s - s_old))
                    max_dt = max(max_dt, abs(pair.state.t - t_old))

            # AL 乗数更新（adaptive omega: Outer反復ごとにωを段階的に増大）
            if adaptive_omega:
                omega_current = min(
                    omega_min * omega_growth**outer,
                    omega_max,
                )
            else:
                omega_current = al_relaxation
            preserve_inactive = manager.config.preserve_inactive_lambda
            for pair in manager.pairs:
                update_al_multiplier(
                    pair,
                    omega=omega_current,
                    preserve_inactive=preserve_inactive,
                )

            # --- 適応的ペナルティ増大 ---
            pen_exceeded = False
            max_pen_ratio = 0.0
            if tol_pen_ratio > 0.0:
                for pair in manager.pairs:
                    if pair.state.status == ContactStatus.INACTIVE:
                        continue
                    if pair.state.gap >= 0.0:
                        continue
                    penetration = abs(pair.state.gap)
                    sr = pair.search_radius
                    if sr < 1e-30:
                        continue
                    pen_ratio = penetration / sr
                    max_pen_ratio = max(max_pen_ratio, pen_ratio)
                    if pen_ratio > tol_pen_ratio:
                        pen_exceeded = True
                        if pair.state.k_pen < k_pen_max:
                            new_k = min(pair.state.k_pen * pen_growth, k_pen_max)
                            pair.state.k_pen = new_k
                            pair.state.k_t = new_k * manager.config.k_t_ratio

            if show_progress:
                friction_info = ""
                if use_friction:
                    friction_info = f", μ_eff={mu_eff:.3f}"
                pen_info = ""
                if tol_pen_ratio > 0.0 and max_pen_ratio > 0.0:
                    pen_info = f", pen_ratio={max_pen_ratio:.4f}"
                omega_info = ""
                if adaptive_omega:
                    omega_info = f", ω={omega_current:.4f}"
                print(
                    f"  Step {step}, outer {outer}: "
                    f"max|Δs|={max_ds:.3e}, max|Δt|={max_dt:.3e}, "
                    f"active={manager.n_active}"
                    f"{friction_info}{pen_info}{omega_info}"
                )

            if max_ds < tol_geometry and max_dt < tol_geometry and not pen_exceeded:
                step_converged = True
                break

            if max_ds < tol_geometry and max_dt < tol_geometry and pen_exceeded:
                if show_progress:
                    print(
                        f"  Step {step}, outer {outer}: "
                        f"(s,t) converged but pen_ratio={max_pen_ratio:.4f} > "
                        f"tol={tol_pen_ratio:.4f}. Increasing k_pen."
                    )
                continue

        if not step_converged:
            if show_progress:
                print(
                    f"  Step {step}: outer loop reached {n_outer_max} "
                    f"(max|Δs|={max_ds:.3e}, max|Δt|={max_dt:.3e}). Accepting."
                )
            step_converged = True

        # 記録
        _lc_blk_final = None
        if manager.config.line_contact:
            _lc_blk_final = _deformed_coords(node_coords_ref, u, ndof_per_node)
        f_c_final = compute_contact_force(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
            node_coords=_lc_blk_final,
        )
        fc_norm = float(np.linalg.norm(f_c_final))

        load_history.append(lam)
        disp_history.append(u.copy())
        contact_force_history.append(fc_norm)
        graph_history.add(snapshot_contact_graph(manager, step=step, load_factor=lam))

    return ContactSolveResult(
        u=u,
        converged=True,
        n_load_steps=n_load_steps,
        total_newton_iterations=total_newton,
        total_outer_iterations=total_outer,
        n_active_final=manager.n_active,
        load_history=load_history,
        displacement_history=disp_history,
        contact_force_history=contact_force_history,
        graph_history=graph_history,
    )


@dataclass
class CyclicContactResult:
    """サイクリック荷重解析の結果.

    Attributes:
        phases: 各フェーズの ContactSolveResult
        amplitudes: 各フェーズの目標荷重振幅
        load_factors: 全フェーズ通した荷重係数の時系列（f_ext_unit に対する倍率）
        displacements: 全フェーズ通した変位の時系列
        contact_forces: 全フェーズ通した接触力ノルムの時系列
        graph_history: 全フェーズ通した接触グラフ時系列
        converged: 全フェーズが収束したか
    """

    phases: list[ContactSolveResult]
    amplitudes: list[float]
    load_factors: list[float] = field(default_factory=list)
    displacements: list[np.ndarray] = field(default_factory=list)
    contact_forces: list[float] = field(default_factory=list)
    graph_history: ContactGraphHistory = field(default_factory=ContactGraphHistory)
    converged: bool = True

    @property
    def n_phases(self) -> int:
        """フェーズ数."""
        return len(self.phases)

    @property
    def n_total_steps(self) -> int:
        """全ステップ数."""
        return len(self.load_factors)


def run_contact_cyclic(
    f_ext_unit: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray | float,
    *,
    amplitudes: list[float],
    n_steps_per_phase: int = 10,
    max_iter: int = 30,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    tol_energy: float = 1e-10,
    show_progress: bool = True,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.0,
    broadphase_cell_size: float | None = None,
) -> CyclicContactResult:
    """サイクリック荷重解析（往復荷重による接触ヒステリシス観測）.

    amplitudes リストに従って荷重を段階的に変化させ、
    接触状態（摩擦履歴・AL乗数等）を各フェーズ間で引き継ぐ。

    例: amplitudes=[1.0, -1.0, 1.0] で 0→+F→-F→+F の往復荷重。

    各フェーズでの荷重:
        f_ext = amp_prev * f_ext_unit + lam * (amp - amp_prev) * f_ext_unit
        （lam は 0→1 で増分）

    Args:
        f_ext_unit: (ndof,) 単位荷重ベクトル（振幅1の荷重方向）
        fixed_dofs: 拘束DOF
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ（フェーズ間で状態を引き継ぐ）
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径
        amplitudes: 荷重振幅の列 [amp_1, amp_2, ...]
        n_steps_per_phase: 各フェーズの荷重増分数
        max_iter: Inner Newton の最大反復数
        tol_force: 力ノルム収束判定
        tol_disp: 変位ノルム収束判定
        tol_energy: エネルギーノルム収束判定
        show_progress: 進捗表示
        ndof_per_node: 1節点あたりの DOF 数
        broadphase_margin: broadphase 探索マージン
        broadphase_cell_size: broadphase セルサイズ

    Returns:
        CyclicContactResult
    """
    ndof = f_ext_unit.shape[0]
    u = np.zeros(ndof, dtype=float)
    current_amp = 0.0

    all_phases: list[ContactSolveResult] = []
    all_load_factors: list[float] = []
    all_disps: list[np.ndarray] = []
    all_cf: list[float] = []
    combined_graph = ContactGraphHistory()
    global_step = 0
    all_converged = True

    for phase_idx, amp in enumerate(amplitudes):
        delta_amp = amp - current_amp
        f_ext_total = delta_amp * f_ext_unit
        f_ext_base = current_amp * f_ext_unit

        if show_progress:
            print(
                f"\n=== Cyclic Phase {phase_idx + 1}/{len(amplitudes)}: "
                f"amp {current_amp:.3f} → {amp:.3f} ==="
            )

        result = newton_raphson_with_contact(
            f_ext_total,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            manager,
            node_coords_ref,
            connectivity,
            radii,
            n_load_steps=n_steps_per_phase,
            max_iter=max_iter,
            tol_force=tol_force,
            tol_disp=tol_disp,
            tol_energy=tol_energy,
            show_progress=show_progress,
            u0=u,
            ndof_per_node=ndof_per_node,
            broadphase_margin=broadphase_margin,
            broadphase_cell_size=broadphase_cell_size,
            f_ext_base=f_ext_base,
        )

        u = result.u.copy()
        all_phases.append(result)

        if not result.converged:
            all_converged = False

        # 荷重係数を絶対振幅に変換して記録
        for lam_local in result.load_history:
            abs_amp = current_amp + lam_local * delta_amp
            all_load_factors.append(abs_amp)

        all_disps.extend(result.displacement_history)
        all_cf.extend(result.contact_force_history)

        # グラフ時系列を統合（ステップ番号をグローバルに変換）
        for snap in result.graph_history.snapshots:
            global_step += 1
            from xkep_cae.contact.graph import ContactGraph

            merged = ContactGraph(
                step=global_step,
                load_factor=all_load_factors[-1] if all_load_factors else 0.0,
                nodes=snap.nodes,
                edges=snap.edges,
                n_total_pairs=snap.n_total_pairs,
            )
            combined_graph.add(merged)

        current_amp = amp

    return CyclicContactResult(
        phases=all_phases,
        amplitudes=amplitudes,
        load_factors=all_load_factors,
        displacements=all_disps,
        contact_forces=all_cf,
        graph_history=combined_graph,
        converged=all_converged,
    )
