"""Smooth Penalty + Uzawa ソルバー — Strategy 経由の王道構成.

固定構成:
- contact_mode = "smooth_penalty"（NCP鞍点系は摩擦接線剛性符号問題で発散: status-147）
- use_friction = True
- line_contact = True（ContactGeometryStrategy 経由）
- adaptive_timestepping = True

Strategy 必須（strategies=None フォールバックなし）。
solver_ncp.py の smooth_penalty パスを切り出し、Strategy 経由に書き換えたもの。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.diagnostics import ConvergenceDiagnostics, NCPSolveResult
from xkep_cae.contact.graph import ContactGraphHistory, snapshot_contact_graph
from xkep_cae.contact.pair import ContactManager
from xkep_cae.contact.utils import deformed_coords, ncp_line_search
from xkep_cae.process.data import SolverStrategies


def solve_smooth_penalty_friction(
    f_ext_total: np.ndarray,
    fixed_dofs: np.ndarray,
    assemble_tangent: Callable[[np.ndarray], sp.csr_matrix],
    assemble_internal_force: Callable[[np.ndarray], np.ndarray],
    manager: ContactManager,
    node_coords_ref: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray | float,
    *,
    strategies: SolverStrategies,
    core_radii: np.ndarray | float | None = None,
    max_iter: int = 50,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-8,
    show_progress: bool = True,
    u0: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.0,
    broadphase_cell_size: float | None = None,
    k_pen: float = 0.0,
    f_ext_base: np.ndarray | None = None,
    mu: float | None = None,
    use_line_search: bool = True,
    line_search_max_steps: int = 5,
    prescribed_dofs: np.ndarray | None = None,
    prescribed_values: np.ndarray | None = None,
    du_norm_cap: float = 0.0,
    dt_initial_fraction: float = 0.0,
    dt_grow_iter_threshold: int = 0,
    ul_assembler: object | None = None,
    lambda_init: np.ndarray | None = None,
) -> NCPSolveResult:
    """Smooth penalty + Uzawa + Coulomb 摩擦ソルバー.

    全ての接触力・摩擦力・時間積分は Strategy 経由で評価する。
    solver_ncp.py の smooth_penalty パスと同等の挙動を Strategy ベースで実現。

    Args:
        f_ext_total: (ndof,) 最終外荷重ベクトル
        fixed_dofs: 拘束DOF
        assemble_tangent: u → K_T(u) コールバック
        assemble_internal_force: u → f_int(u) コールバック
        manager: 接触マネージャ
        node_coords_ref: (n_nodes, 3) 参照節点座標
        connectivity: (n_elems, 2) 要素接続
        radii: 断面半径
        strategies: SolverStrategies（必須）
        k_pen: ペナルティ剛性（0 で PenaltyStrategy から自動決定）
        mu: 摩擦係数（None で manager.config.mu）
        ul_assembler: Updated Lagrangian アセンブラ

    Returns:
        NCPSolveResult
    """
    ndof = f_ext_total.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    u = u0.copy() if u0 is not None else np.zeros(ndof, dtype=float)

    # --- Strategy 取得（必須） ---
    _time_strategy = strategies.time_integration
    _penalty_strategy = strategies.penalty
    _friction_strategy = strategies.friction
    _contact_force_strategy = strategies.contact_force

    _dynamics = _time_strategy.is_dynamic
    _dt_sub = 0.0

    # 処方変位の設定
    _prescribed_dofs = (
        np.asarray(prescribed_dofs, dtype=int)
        if prescribed_dofs is not None
        else np.array([], dtype=int)
    )
    _prescribed_values = (
        np.asarray(prescribed_values, dtype=float)
        if prescribed_values is not None
        else np.array([])
    )
    has_prescribed = len(_prescribed_dofs) > 0
    if has_prescribed:
        fixed_dofs = np.unique(np.concatenate([fixed_dofs, _prescribed_dofs]))

    # k_pen の決定（PenaltyStrategy 経由）
    k_pen = _penalty_strategy.compute_k_pen(0, 1)
    if show_progress and k_pen > 0.0:
        print(f"  k_pen ({type(_penalty_strategy).__name__}): k_pen={k_pen:.2e}")

    # 摩擦設定
    _mu = mu if mu is not None else manager.config.mu

    # Uzawa パラメータ（ContactForceStrategy から取得）
    _n_uzawa_max = getattr(_contact_force_strategy, "_n_uzawa_max", 5)
    _tol_uzawa = getattr(_contact_force_strategy, "_tol_uzawa", 1e-6)

    # λ の初期化
    n_pairs = manager.n_pairs
    if lambda_init is not None and len(lambda_init) >= n_pairs:
        lam_all = lambda_init[:n_pairs].copy()
    elif lambda_init is not None:
        lam_all = np.zeros(n_pairs)
        lam_all[: len(lambda_init)] = lambda_init
    else:
        lam_all = np.zeros(n_pairs)

    # FrictionStrategy の k_pen 伝播
    if hasattr(_friction_strategy, "_k_pen"):
        _friction_strategy._k_pen = k_pen
    if hasattr(_friction_strategy, "_k_t_ratio"):
        _friction_strategy._k_t_ratio = manager.config.k_t_ratio

    # ContactForceStrategy の ndof 伝播
    if hasattr(_contact_force_strategy, "_ndof"):
        _contact_force_strategy._ndof = ndof

    # 結果蓄積
    load_history: list[float] = []
    disp_history: list[np.ndarray] = []
    contact_force_history: list[float] = []
    graph_history = ContactGraphHistory()
    total_newton = 0

    f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
    _dynamic_ref = f_ext_ref_norm < 1e-30
    if _dynamic_ref:
        f_ext_ref_norm = 1.0

    _f_ext_base = f_ext_base if f_ext_base is not None else np.zeros(ndof)
    u_ref = u.copy()

    # --- 適応時間増分制御 ---
    _dt_grow = manager.config.dt_grow_factor
    _dt_shrink = manager.config.dt_shrink_factor
    _dt_grow_iter = (
        dt_grow_iter_threshold
        if dt_grow_iter_threshold > 0
        else manager.config.dt_grow_iter_threshold
    )
    _dt_shrink_iter = manager.config.dt_shrink_iter_threshold
    _dt_contact_change_thr = manager.config.dt_contact_change_threshold
    _dt_min_frac = manager.config.dt_min_fraction
    _dt_max_frac = manager.config.dt_max_fraction

    _dt_initial = dt_initial_fraction if dt_initial_fraction > 0.0 else 0.0
    _effective_n = max(1, int(1.0 / _dt_initial) if _dt_initial > 0 else 1)
    if _dt_min_frac <= 0.0:
        _dt_min_frac = 1.0 / (_effective_n * 32)
    if _dt_max_frac <= 0.0:
        _dt_max_frac = min(8.0 / _effective_n, 1.0)

    # 被膜接触
    _use_coating = manager.config.coating_stiffness > 0.0

    step_queue: deque[float] = deque()
    _base_delta = _dt_initial if _dt_initial > 0.0 else 1.0
    step_queue.append(min(_base_delta, 1.0))

    load_frac_prev = 0.0
    step_display = 0
    _prev_n_active = 0
    _consecutive_good = 0

    # Updated Lagrangian
    _ul = ul_assembler is not None
    _ul_frac_base = 0.0

    # --- 初期貫入チェック ---
    manager.detect_candidates(
        node_coords_ref,
        connectivity,
        radii,
        margin=broadphase_margin,
        cell_size=broadphase_cell_size,
        core_radii=core_radii,
    )
    _use_adjust = manager.config.adjust_initial_penetration
    if _use_adjust and _ul:
        _use_adjust = False
    if _use_adjust:
        _pos_tol = manager.config.position_tolerance
        node_coords_ref, n_pen_fixed, n_gap_closed = manager.adjust_initial_positions(
            node_coords_ref, _pos_tol
        )
        if show_progress and (n_pen_fixed + n_gap_closed) > 0:
            print(f"  初期位置調整: 貫入修正={n_pen_fixed}ペア, ギャップ閉鎖={n_gap_closed}ペア")
        manager.detect_candidates(
            node_coords_ref,
            connectivity,
            radii,
            margin=broadphase_margin,
            cell_size=broadphase_cell_size,
            core_radii=core_radii,
        )

    _ul_has_accum = (
        _ul
        and hasattr(ul_assembler, "u_total_accum")
        and float(np.linalg.norm(ul_assembler.u_total_accum)) > 1e-15
    )
    n_initial_pen = manager.check_initial_penetration(node_coords_ref)
    if n_initial_pen > 0 and not _use_coating and not _ul_has_accum:
        raise ValueError(
            f"初期貫入が検出されました: {n_initial_pen}ペア。"
            f"メッシュ生成時のgapを増やしてください。"
        )

    if len(lam_all) < manager.n_pairs:
        lam_new = np.zeros(manager.n_pairs)
        lam_new[: len(lam_all)] = lam_all
        lam_all = lam_new

    # チェックポイント
    u_ckpt = u.copy()
    lam_ckpt = lam_all.copy()
    u_ref_ckpt = u_ref.copy()
    _ul_frac_base_ckpt = _ul_frac_base
    if _ul:
        ul_assembler.checkpoint()
    _time_strategy.checkpoint()

    u_prev_converged = u.copy()
    delta_frac_prev = 0.0

    # ====================================================================
    # 荷重ステップループ
    # ====================================================================
    while step_queue:
        load_frac = step_queue[0]
        if load_frac <= load_frac_prev + 1e-15:
            step_queue.popleft()
            continue
        step_display += 1
        f_ext = _f_ext_base + load_frac * f_ext_total

        step_converged = False
        energy_ref = None
        f_c = np.zeros(ndof)

        # 接線予測子
        delta_frac = load_frac - load_frac_prev
        if _dynamics:
            _dt_sub = getattr(_time_strategy, "_dt_physical", 0.0) * delta_frac
            if hasattr(_time_strategy, "predict"):
                u = _time_strategy.predict(u, _dt_sub)
        elif delta_frac_prev > 1e-30 and delta_frac > 1e-30:
            du_prev = u - u_prev_converged
            du_prev_norm = float(np.linalg.norm(du_prev))
            if du_prev_norm > 1e-30:
                ratio = min(delta_frac / delta_frac_prev, 2.0)
                u = u + ratio * du_prev

        if has_prescribed:
            u[_prescribed_dofs] = (load_frac - _ul_frac_base) * _prescribed_values

        # 候補検出
        coords_def = deformed_coords(node_coords_ref, u, ndof_per_node)
        manager.detect_candidates(
            coords_def,
            connectivity,
            radii,
            margin=broadphase_margin,
            cell_size=broadphase_cell_size,
            core_radii=core_radii,
        )
        manager.update_geometry(coords_def)

        # ペア数拡張
        if len(lam_all) < manager.n_pairs:
            old_n = len(lam_all)
            lam_new = np.zeros(manager.n_pairs)
            lam_new[:old_n] = lam_all
            lam_all = lam_new

        K_T = None
        _diag = ConvergenceDiagnostics(step=step_display, load_frac=load_frac)

        # ==============================================================
        # Smooth Penalty Newton + Uzawa
        # ==============================================================
        _uzawa_converged = False
        for _uzawa_iter in range(_n_uzawa_max):
            for it in range(max_iter):
                total_newton += 1

                # 1. 幾何更新
                coords_def = deformed_coords(node_coords_ref, u, ndof_per_node)
                manager.update_geometry(coords_def, freeze_active_set=True)

                # 2. 接触力（ContactForceStrategy 経由）
                f_c, _ = _contact_force_strategy.evaluate(u, lam_all, manager, k_pen)

                # 3. 摩擦力（FrictionStrategy 経由）
                if hasattr(_friction_strategy, "_mu_ramp_counter"):
                    _friction_strategy._mu_ramp_counter = step_display
                f_friction, _ = _friction_strategy.evaluate(
                    u,
                    manager.pairs,
                    _mu,
                    lambdas=lam_all,
                    u_ref=u_ref,
                    node_coords_ref=node_coords_ref,
                )
                f_c = f_c + f_friction

                # 4. 被膜力
                if _use_coating:
                    _coat_dt = max(load_frac - load_frac_prev, 1e-15)
                    f_coat = manager.compute_coating_forces(coords_def, dt=_coat_dt)
                    f_c = f_c + f_coat
                    if manager.config.coating_mu > 0.0:
                        f_coat_fric = manager.compute_coating_friction_forces(coords_def, u, u_ref)
                        f_c = f_c + f_coat_fric

                # 5. 力残差
                f_int = assemble_internal_force(u)
                R_u = f_int + f_c - f_ext

                # 動的解析: 慣性力・減衰力
                if _dynamics and _dt_sub > 1e-30:
                    _time_strategy.correct(u, np.zeros_like(u), _dt_sub)
                    R_u = _time_strategy.effective_residual(R_u, _dt_sub)

                R_u[fixed_dofs] = 0.0

                # 6. 収束判定
                res_u_norm = float(np.linalg.norm(R_u))
                if _dynamic_ref and it == 0 and res_u_norm > 1e-30:
                    f_ext_ref_norm = res_u_norm

                n_active = sum(
                    1 for p in manager.pairs if hasattr(p, "state") and p.state.p_n > 0.0
                )

                _diag.res_history.append(res_u_norm / f_ext_ref_norm)
                _diag.ncp_history.append(0.0)
                _diag.ncp_t_history.append(0.0)
                _diag.n_active_history.append(n_active)

                force_conv = res_u_norm / f_ext_ref_norm < tol_force
                if force_conv:
                    step_converged = True
                    if show_progress:
                        print(
                            f"  Incr {step_display} (frac={load_frac:.4f}), "
                            f"uzawa {_uzawa_iter}, iter {it}, "
                            f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e} "
                            f"(converged, {n_active} active)"
                        )
                    break

                if show_progress and it % 5 == 0:
                    print(
                        f"  Incr {step_display} (frac={load_frac:.4f}), "
                        f"uzawa {_uzawa_iter}, iter {it}, "
                        f"||R_u||/||f|| = {res_u_norm / f_ext_ref_norm:.3e}, "
                        f"active={n_active}"
                    )

                # 7. 接線剛性
                K_T = assemble_tangent(u)

                # 接触剛性（ContactForceStrategy 経由）
                K_c = _contact_force_strategy.tangent(u, lam_all, manager, k_pen)
                K_T = K_T + K_c

                # 被膜剛性
                if _use_coating:
                    _coat_dt = max(load_frac - load_frac_prev, 1e-15)
                    K_coat = manager.compute_coating_stiffness(coords_def, ndof, dt=_coat_dt)
                    K_T = K_T + K_coat
                    if manager.config.coating_mu > 0.0:
                        K_coat_fric = manager.compute_coating_friction_stiffness(coords_def, ndof)
                        K_T = K_T + K_coat_fric

                # 摩擦剛性（FrictionStrategy 経由）
                if _friction_strategy.friction_tangents:
                    K_fric = _friction_strategy.tangent(u, manager.pairs, _mu)
                    K_T = K_T + K_fric

                # 動的解析: 質量・減衰
                if _dynamics and _dt_sub > 1e-30:
                    K_T = _time_strategy.effective_stiffness(K_T, _dt_sub)

                # 8. 線形ソルブ
                K_eff = K_T.tocsc()
                _rhs = -R_u.copy()
                for d in fixed_dofs:
                    K_eff[d, :] = 0.0
                    K_eff[:, d] = 0.0
                    K_eff[d, d] = 1.0
                    _rhs[d] = 0.0
                K_eff.eliminate_zeros()

                try:
                    from scipy.sparse.linalg import spsolve

                    du = spsolve(K_eff.tocsc(), _rhs)
                except Exception:
                    if show_progress:
                        print(f"  WARNING: Linear solve failed at iter {it}")
                    break

                # 9. Line search + 更新
                _scale_factor = 1.0
                if use_line_search:
                    alpha = ncp_line_search(
                        u,
                        du,
                        f_ext,
                        fixed_dofs,
                        assemble_internal_force,
                        res_u_norm,
                        max_steps=line_search_max_steps,
                        f_c=f_c,
                        diverge_factor=3.0,
                    )
                    _scale_factor *= alpha
                if du_norm_cap > 0.0:
                    _du_n = float(np.linalg.norm(_scale_factor * du))
                    _u_ref_n = max(float(np.linalg.norm(u)), 1.0)
                    if _du_n > du_norm_cap * _u_ref_n:
                        _scale_factor *= du_norm_cap * _u_ref_n / _du_n
                du = _scale_factor * du
                u += du

                # 変位収束判定
                u_norm = float(np.linalg.norm(u))
                du_norm_val = float(np.linalg.norm(du))
                _diag.du_norm_history.append(du_norm_val)
                _diag.max_du_dof_history.append(
                    int(np.argmax(np.abs(du))) if du_norm_val > 0 else -1
                )

                if u_norm > 1e-30 and du_norm_val / u_norm < tol_disp:
                    step_converged = True
                    if show_progress:
                        print(
                            f"  Incr {step_display} (frac={load_frac:.4f}), "
                            f"uzawa {_uzawa_iter}, iter {it}, "
                            f"||du||/||u|| = {du_norm_val / u_norm:.3e} "
                            f"(disp converged, {n_active} active)"
                        )
                    break

                # エネルギー収束
                energy = abs(float(np.dot(du, R_u)))
                if energy_ref is None:
                    energy_ref = energy if energy > 1e-30 else 1.0
                if energy_ref > 1e-30 and energy / energy_ref < 1e-10:
                    step_converged = True
                    if show_progress:
                        print(
                            f"  Incr {step_display} (frac={load_frac:.4f}), "
                            f"uzawa {_uzawa_iter}, iter {it}, "
                            f"energy = {energy:.3e} (energy converged)"
                        )
                    break

            # --- Uzawa 乗数更新 ---
            if step_converged:
                coords_def = deformed_coords(node_coords_ref, u, ndof_per_node)
                manager.update_geometry(coords_def, freeze_active_set=True)

                lam_prev = lam_all.copy()
                for i, pair in enumerate(manager.pairs):
                    if i < len(lam_all):
                        lam_all[i] = max(0.0, lam_all[i] + k_pen * (-pair.state.gap))
                lam_change = float(np.linalg.norm(lam_all - lam_prev))
                lam_ref = max(float(np.linalg.norm(lam_all)), 1.0)

                if show_progress:
                    print(f"  Uzawa {_uzawa_iter}: ||Δλ||/||λ|| = {lam_change / lam_ref:.3e}")

                if lam_change / lam_ref < _tol_uzawa:
                    _uzawa_converged = True
                    break

                step_converged = False
                energy_ref = None
            else:
                break

        # ==============================================================
        # 不収束処理 + 適応時間増分リトライ
        # ==============================================================
        if not step_converged:
            delta = load_frac - load_frac_prev
            if delta > _dt_min_frac + 1e-15:
                # チェックポイント復元
                u = u_ckpt.copy()
                lam_all = lam_ckpt.copy()
                u_ref = u_ref_ckpt.copy()
                _ul_frac_base = _ul_frac_base_ckpt
                if _ul:
                    ul_assembler.rollback()
                    node_coords_ref = ul_assembler.coords_ref
                if _dynamics:
                    _time_strategy.restore_checkpoint()
                delta_frac_prev = 0.0
                _consecutive_good = 0
                mid_frac = load_frac_prev + delta * _dt_shrink
                step_queue.appendleft(load_frac)
                step_queue.appendleft(mid_frac)
                step_display -= 1
                if show_progress:
                    print(
                        f"  Adaptive dt retry: frac {load_frac:.4f} → "
                        f"sub-steps [{mid_frac:.4f}, {load_frac:.4f}]"
                    )
                continue
            else:
                if show_progress:
                    print(
                        f"  WARNING: Step {step_display} (frac={load_frac:.4f}) "
                        f"did not converge in {max_iter} iterations."
                    )
                    print(_diag.format_report(max_iter=max_iter))
                _u_fail = ul_assembler.u_total_accum + u if _ul else u
                return NCPSolveResult(
                    u=_u_fail,
                    lambdas=lam_all,
                    converged=False,
                    n_increments=step_display,
                    total_newton_iterations=total_newton,
                    n_active_final=manager.n_active,
                    load_history=load_history,
                    displacement_history=disp_history,
                    contact_force_history=contact_force_history,
                    graph_history=graph_history,
                    diagnostics=_diag,
                    velocity=_time_strategy.vel.copy() if _dynamics else None,
                    acceleration=_time_strategy.acc.copy() if _dynamics else None,
                )

        # ==============================================================
        # ステップ完了
        # ==============================================================
        step_queue.popleft()

        # 被膜圧縮量保存
        if _use_coating:
            for pair in manager.pairs:
                pair.state.coating_compression_prev = pair.state.coating_compression

        # 動的解析: 速度・加速度更新
        if _dynamics and _dt_sub > 1e-30:
            _time_strategy.correct(u, np.zeros_like(u), _dt_sub)

        # Updated Lagrangian: 参照配置更新
        if _ul:
            ul_assembler.update_reference(u)
            node_coords_ref = ul_assembler.coords_ref
            _ul_frac_base = load_frac
            u = np.zeros(ndof)
            u_ref = np.zeros(ndof)

        # 適応時間増分: 次ステップ幅決定
        if load_frac < 1.0 - 1e-12:
            current_delta = load_frac - load_frac_prev
            next_delta = current_delta

            step_iters = it + 1
            if step_iters <= _dt_grow_iter:
                _consecutive_good += 1
                if _consecutive_good <= 2:
                    next_delta = current_delta * _dt_grow
                else:
                    _damp = max(0.1, 1.0 / _consecutive_good)
                    next_delta = current_delta * (1.0 + (_dt_grow - 1.0) * _damp)
            elif step_iters >= _dt_shrink_iter:
                next_delta = current_delta * _dt_shrink
                _consecutive_good = 0
            else:
                _consecutive_good = 0

            _current_n_active = manager.n_active
            if _prev_n_active > 0:
                change_rate = abs(_current_n_active - _prev_n_active) / max(_prev_n_active, 1)
                if change_rate > _dt_contact_change_thr:
                    next_delta = min(next_delta, current_delta * _dt_shrink)

            next_delta = max(next_delta, _dt_min_frac)
            next_delta = min(next_delta, _dt_max_frac)
            next_frac = min(load_frac + next_delta, 1.0)
            if 1.0 - next_frac < _dt_min_frac * 0.5:
                next_frac = 1.0
            step_queue.appendleft(next_frac)

            if show_progress and abs(next_delta - current_delta) > 1e-10:
                print(
                    f"  Adaptive dt: delta {current_delta:.4f} → {next_delta:.4f} "
                    f"(iters={step_iters}, n_active={_current_n_active})"
                )

        _prev_n_active = manager.n_active

        # k_pen continuation
        _k_pen_new = _penalty_strategy.compute_k_pen(step_display, step_display + 1)
        if abs(_k_pen_new - k_pen) > 1e-30:
            k_pen = _k_pen_new
            if show_progress:
                print(f"  k_pen continuation: k_pen → {k_pen:.2e}")

        delta_frac_prev = load_frac - load_frac_prev
        u_prev_converged = u.copy()
        load_frac_prev = load_frac
        for i, pair in enumerate(manager.pairs):
            if i < len(lam_all):
                pair.state.lambda_n = lam_all[i]
                pair.state.p_n = max(0.0, lam_all[i] + k_pen * (-pair.state.gap))

        u_ref = u.copy()

        # チェックポイント保存
        u_ckpt = u.copy()
        lam_ckpt = lam_all.copy()
        u_ref_ckpt = u_ref.copy()
        _ul_frac_base_ckpt = _ul_frac_base
        if _ul:
            ul_assembler.checkpoint()
        if _dynamics:
            _time_strategy.checkpoint()

        load_history.append(load_frac)
        _u_hist = ul_assembler.u_total_accum + u if _ul else u.copy()
        disp_history.append(_u_hist.copy() if _ul else _u_hist)
        contact_force_history.append(float(np.linalg.norm(f_c)))

        try:
            graph = snapshot_contact_graph(manager, step_index=step_display - 1)
            graph_history.add_snapshot(graph)
        except Exception:
            pass

    _u_out = ul_assembler.u_total_accum + u if _ul else u
    return NCPSolveResult(
        u=_u_out,
        lambdas=lam_all,
        converged=True,
        n_increments=step_display,
        total_newton_iterations=total_newton,
        n_active_final=manager.n_active,
        load_history=load_history,
        displacement_history=disp_history,
        contact_force_history=contact_force_history,
        graph_history=graph_history,
        velocity=_time_strategy.vel.copy() if _dynamics else None,
        acceleration=_time_strategy.acc.copy() if _dynamics else None,
    )
