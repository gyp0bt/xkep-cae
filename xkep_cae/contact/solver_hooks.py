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
from xkep_cae.contact.law_friction import (
    compute_mu_effective,
    compute_tangential_displacement,
    friction_return_mapping,
    friction_tangent_2x2,
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

    Returns:
        ContactSolveResult
    """
    import scipy.sparse.linalg as spla

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

    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
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

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        f_ext = lam * f_ext_total

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

            manager.detect_candidates(
                coords_def,
                connectivity,
                radii,
                margin=broadphase_margin,
                cell_size=broadphase_cell_size,
            )

            # --- 段階的接触アクティベーション ---
            if manager.config.staged_activation_steps > 0:
                max_layer = manager.compute_active_layer_for_step(step, n_load_steps)
                manager.filter_pairs_by_layer(max_layer)

            manager.update_geometry(coords_def)

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

            for it in range(max_iter):
                total_newton += 1

                # gap 更新（s, t, normal 固定で変位に基づく gap 再計算）
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

                # 接触内力（法線 + 摩擦）
                f_c = compute_contact_force(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                    friction_forces=friction_forces if friction_forces else None,
                )

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
                K_T = assemble_tangent(u)
                K_c = compute_contact_stiffness(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
                    friction_tangents=friction_tangents if friction_tangents else None,
                    use_geometric_stiffness=use_geometric_stiffness,
                )
                K_total = K_T + K_c

                # BC 適用
                K_bc = K_total.tolil()
                r_bc = residual.copy()
                for dof in fixed_dofs:
                    K_bc[dof, :] = 0.0
                    K_bc[:, dof] = 0.0
                    K_bc[dof, dof] = 1.0
                    r_bc[dof] = 0.0

                du = spla.spsolve(K_bc.tocsr(), r_bc)

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
                        f_c_t = compute_contact_force(
                            manager,
                            ndof,
                            ndof_per_node=ndof_per_node,
                            friction_forces=_ff,
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
                )

            # --- Outer 収束判定 ---
            # 幾何更新して (s,t) の変化を検査
            coords_def_new = _deformed_coords(node_coords_ref, u, ndof_per_node)
            manager.update_geometry(coords_def_new)

            max_ds = 0.0
            max_dt = 0.0
            idx = 0
            for pair in manager.pairs:
                if pair.state.status != ContactStatus.INACTIVE and prev_st[idx] is not None:
                    s_old, t_old = prev_st[idx]
                    max_ds = max(max_ds, abs(pair.state.s - s_old))
                    max_dt = max(max_dt, abs(pair.state.t - t_old))
                idx += 1

            # AL 乗数更新
            for pair in manager.pairs:
                update_al_multiplier(pair)

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
                f_c_post = compute_contact_force(
                    manager,
                    ndof,
                    ndof_per_node=ndof_per_node,
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
        f_c_final = compute_contact_force(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
        )
        fc_norm = float(np.linalg.norm(f_c_final))

        load_history.append(lam)
        disp_history.append(u.copy())
        contact_force_history.append(fc_norm)

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
    )
