"""ContactFrictionProcess — 摩擦接触ソルバー（SolverProcess）.

設計仕様: docs/contact_friction.md

内部構成:
- SolverStateOutput: 全可変状態（frozen dataclass）
- NewtonUzawaProcess: 1荷重増分の NR + Uzawa
- AdaptiveSteppingProcess: 適応荷重増分制御（QUERY/SUCCESS/FAILURE）
- Strategy 5軸 + default_strategies()
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state, _n_pairs
from xkep_cae.contact._manager_process import (
    DetectCandidatesInput,
    DetectCandidatesProcess,
    UpdateGeometryInput,
    UpdateGeometryProcess,
)
from xkep_cae.contact.solver._adaptive_stepping import (
    AdaptiveStepInput,
    AdaptiveSteppingInput,
    AdaptiveSteppingProcess,
    StepAction,
)
from xkep_cae.contact.solver._contact_graph import (
    ContactGraphInput,
    ContactGraphProcess,
)
from xkep_cae.contact.solver._diagnostics import (
    DiagnosticsInput,
    DiagnosticsReportProcess,
)
from xkep_cae.contact.solver._initial_penetration import (
    InitialPenetrationInput,
    InitialPenetrationProcess,
)
from xkep_cae.contact.solver._newton_uzawa_dynamic import (
    NewtonUzawaDynamicInput,
    NewtonUzawaDynamicProcess,
    NewtonUzawaDynamicStepInput,
)
from xkep_cae.contact.solver._newton_uzawa_static import (
    NewtonUzawaStaticInput,
    NewtonUzawaStaticProcess,
    NewtonUzawaStaticStepInput,
)
from xkep_cae.contact.solver._solver_state import (
    SolverStateOutput,
    _build_u_output,
    _ensure_lam_size,
    _restore_checkpoint,
    _save_checkpoint,
    _state_set,
)
from xkep_cae.contact.solver._utils import DeformedCoordsInput, DeformedCoordsProcess
from xkep_cae.core import (
    ContactFrictionInputData,
    ProcessMeta,
    SolverProcess,
    SolverResultData,
)
from xkep_cae.core.data import default_strategies as _default_strategies
from xkep_cae.core.slots import StrategySlot


class ContactFrictionProcess(
    SolverProcess[ContactFrictionInputData, SolverResultData],
):
    """統一摩擦接触ソルバー（smooth penalty + 自動時間積分選択）.

    入力の mass_matrix / dt_physical の有無で動的/準静的を自動判定。
    - 動的: Generalized-α 時間積分
    - 準静的: 荷重制御 or 変位制御
    """

    meta = ProcessMeta(
        name="ContactFriction",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_friction.md",
    )
    uses = [
        NewtonUzawaStaticProcess,
        NewtonUzawaDynamicProcess,
        AdaptiveSteppingProcess,
        InitialPenetrationProcess,
        ContactGraphProcess,
        DiagnosticsReportProcess,
        DeformedCoordsProcess,
        DetectCandidatesProcess,
        UpdateGeometryProcess,
    ]

    # StrategySlot 宣言（Protocol は importlib 経由で取得するため object 型）
    penalty_slot = StrategySlot(object)
    friction_slot = StrategySlot(object)
    time_integration_slot = StrategySlot(object)
    contact_force_slot = StrategySlot(object, required=False)
    contact_geometry_slot = StrategySlot(object, required=False)

    def __init__(self, strategies: object | None = None) -> None:
        if strategies is None:
            strategies = _default_strategies()
        else:
            from xkep_cae.core.diagnostics import NonDefaultStrategyWarning

            warnings.warn(
                "ContactFrictionProcess: デフォルトではない Strategy 構成が指定されました。"
                " default_strategies() で生成されていない Strategy を使用しています。",
                NonDefaultStrategyWarning,
                stacklevel=2,
            )
        self.strategies = strategies

        self.penalty_slot = self.strategies.penalty
        self.friction_slot = self.strategies.friction
        self.time_integration_slot = self.strategies.time_integration
        if self.strategies.contact_force is not None:
            self.contact_force_slot = self.strategies.contact_force
        if self.strategies.contact_geometry is not None:
            self.contact_geometry_slot = self.strategies.contact_geometry

    def process(self, input_data: ContactFrictionInputData) -> SolverResultData:
        """ContactFrictionInputData → NR+Uzawa+適応荷重増分 → SolverResultData."""
        t0 = time.perf_counter()

        ndof = len(input_data.boundary.f_ext_total)
        f_ext_total = input_data.boundary.f_ext_total
        manager = input_data.contact.manager
        ul_assembler = input_data.callbacks.ul_assembler

        # --- Strategy 生成（deprecated 版: Phase 7-8 で新パッケージに完全移行予定） ---
        # beam_L 推定: メッシュ平均要素長
        _conn = input_data.mesh.connectivity
        _nc = input_data.mesh.node_coords
        _beam_L = 0.0
        if len(_conn) > 0:
            _lens = np.array(
                [float(np.linalg.norm(_nc[int(c[1])] - _nc[int(c[0])])) for c in _conn]
            )
            _beam_L = float(np.mean(_lens))

        strategies = _default_strategies(
            ndof=ndof,
            mass_matrix=input_data.mass_matrix,
            damping_matrix=input_data.damping_matrix,
            dt_physical=input_data.dt_physical,
            rho_inf=input_data.rho_inf,
            velocity=input_data.velocity,
            acceleration=input_data.acceleration,
            k_pen=input_data.contact.k_pen,
            beam_E=manager.config.beam_E,
            beam_I=manager.config.beam_I,
            beam_L=_beam_L,
            use_friction=True,
            mu=input_data.contact.mu or 0.15,
            contact_mode="smooth_penalty",
            line_contact=True,
            smoothing_delta=manager.config.smoothing_delta,
            n_uzawa_max=manager.config.n_uzawa_max,
            tol_uzawa=manager.config.tol_uzawa,
        )
        _time_strategy = strategies.time_integration
        _penalty_strategy = strategies.penalty
        _friction_strategy = strategies.friction
        _contact_force_strategy = strategies.contact_force
        _dynamics = _time_strategy.is_dynamic

        # --- 静的ソルバー使用警告 ---
        if not _dynamics:
            from xkep_cae.core.diagnostics import StaticSolverWarning

            warnings.warn(
                "ContactFrictionProcess: 準静的ソルバー（NewtonUzawaStaticProcess）を使用。"
                " mass_matrix / dt_physical を指定すると動的ソルバー"
                "（NewtonUzawaDynamicProcess）に切り替わります。",
                StaticSolverWarning,
                stacklevel=2,
            )

        # --- 固定DOF + 処方変位 ---
        fixed_dofs = np.asarray(input_data.boundary.fixed_dofs, dtype=int)
        _prescribed_dofs = (
            np.asarray(input_data.boundary.prescribed_dofs, dtype=int)
            if input_data.boundary.prescribed_dofs is not None
            else np.array([], dtype=int)
        )
        _prescribed_values = (
            np.asarray(input_data.boundary.prescribed_values, dtype=float)
            if input_data.boundary.prescribed_values is not None
            else np.array([])
        )
        has_prescribed = len(_prescribed_dofs) > 0
        if has_prescribed:
            fixed_dofs = np.unique(np.concatenate([fixed_dofs, _prescribed_dofs]))

        # --- k_pen 決定 ---
        k_pen = _penalty_strategy.compute_k_pen(0, 1)

        # --- 摩擦設定 ---
        mu = input_data.contact.mu if input_data.contact.mu is not None else manager.config.mu

        # --- Strategy パラメータ伝播 ---
        if hasattr(_friction_strategy, "set_k_pen"):
            _friction_strategy.set_k_pen(k_pen)
        if hasattr(_friction_strategy, "set_k_t_ratio"):
            _friction_strategy.set_k_t_ratio(manager.config.k_t_ratio)
        if hasattr(_contact_force_strategy, "set_ndof"):
            _contact_force_strategy.set_ndof(ndof)

        # --- SolverStateOutput 初期化 ---
        u0 = input_data.u0.copy() if input_data.u0 is not None else np.zeros(ndof)
        node_coords_ref = input_data.mesh.node_coords.copy()
        connectivity = input_data.mesh.connectivity
        radii = input_data.mesh.radii
        core_radii = None

        state = SolverStateOutput(
            u=u0,
            lam_all=np.zeros(_n_pairs(manager)),
            u_ref=u0.copy(),
            node_coords_ref=node_coords_ref,
            u_prev_converged=u0.copy(),
        )

        # --- 参照荷重ノルム ---
        f_ext_ref_norm = float(np.linalg.norm(f_ext_total))
        dynamic_ref = f_ext_ref_norm < 1e-30
        if dynamic_ref:
            f_ext_ref_norm = 1.0
        f_ext_base = (
            input_data.boundary.f_ext_base
            if input_data.boundary.f_ext_base is not None
            else np.zeros(ndof)
        )

        # --- 被膜設定 ---
        use_coating = manager.config.coating_stiffness > 0.0

        # --- UL ---
        _ul = ul_assembler is not None

        # --- 初期貫入チェック ---
        broadphase_margin = 0.0
        broadphase_cell_size = None
        _detect_proc = DetectCandidatesProcess()
        _geom_proc = UpdateGeometryProcess()
        _dc_init = _detect_proc.process(
            DetectCandidatesInput(
                manager=manager,
                node_coords=node_coords_ref,
                connectivity=connectivity,
                radii=radii,
                margin=broadphase_margin,
                cell_size=broadphase_cell_size,
                core_radii=core_radii,
            )
        )
        manager = _dc_init.manager
        _pen_proc = InitialPenetrationProcess()
        _use_adjust = manager.config.adjust_initial_penetration
        if _use_adjust and _ul:
            _use_adjust = False
        if _use_adjust:
            _pos_tol = manager.config.position_tolerance
            pen_out = _pen_proc.process(
                InitialPenetrationInput(
                    pairs=manager.pairs,
                    node_coords=node_coords_ref,
                    position_tolerance=_pos_tol,
                    adjust=True,
                )
            )
            if pen_out.adjusted_coords is not None:
                node_coords_ref = pen_out.adjusted_coords
            if (pen_out.n_pen_fixed + pen_out.n_gap_closed) > 0:
                print(
                    f"  初期位置調整: 貫入修正={pen_out.n_pen_fixed}ペア, "
                    f"ギャップ閉鎖={pen_out.n_gap_closed}ペア"
                )
            _state_set(state, "node_coords_ref", node_coords_ref)
            _dc_adj = _detect_proc.process(
                DetectCandidatesInput(
                    manager=manager,
                    node_coords=node_coords_ref,
                    connectivity=connectivity,
                    radii=radii,
                    margin=broadphase_margin,
                    cell_size=broadphase_cell_size,
                    core_radii=core_radii,
                )
            )
            manager = _dc_adj.manager

        _ul_has_accum = (
            _ul
            and hasattr(ul_assembler, "u_total_accum")
            and float(np.linalg.norm(ul_assembler.u_total_accum)) > 1e-15
        )
        pen_check = _pen_proc.process(
            InitialPenetrationInput(
                pairs=manager.pairs,
                node_coords=node_coords_ref,
                coating_stiffness=manager.config.coating_stiffness,
            )
        )
        n_initial_pen = pen_check.n_penetrations
        if n_initial_pen > 0 and not use_coating and not _ul_has_accum:
            raise ValueError(
                f"初期貫入が検出されました: {n_initial_pen}ペア。"
                f"メッシュ生成時のgapを増やしてください。"
            )

        _ensure_lam_size(state, _n_pairs(manager))

        # --- チェックポイント初期化 ---
        _save_checkpoint(state)
        if _ul:
            ul_assembler.checkpoint()
        _time_strategy.checkpoint()

        # --- 適応荷重増分コントローラ ---
        dt_grow_iter = manager.config.dt_grow_iter_threshold
        stepping_config = AdaptiveSteppingInput(
            dt_initial_fraction=0.0,
            dt_grow_factor=manager.config.dt_grow_factor,
            dt_shrink_factor=manager.config.dt_shrink_factor,
            dt_grow_iter_threshold=dt_grow_iter if dt_grow_iter > 0 else 5,
            dt_shrink_iter_threshold=manager.config.dt_shrink_iter_threshold,
            dt_contact_change_threshold=manager.config.dt_contact_change_threshold,
            dt_min_fraction=manager.config.dt_min_fraction,
            dt_max_fraction=manager.config.dt_max_fraction,
        )
        stepping = AdaptiveSteppingProcess(stepping_config)

        # --- Newton-Uzawa プロセス（Static/Dynamic 完全分離） ---
        if _dynamics:
            nr_config_dyn = NewtonUzawaDynamicInput(show_progress=True)
            nr_process_dyn = NewtonUzawaDynamicProcess()
        else:
            nr_config_sta = NewtonUzawaStaticInput(show_progress=True)
            nr_process_sta = NewtonUzawaStaticProcess()

        # --- 最終診断 ---
        last_diag = None

        # ================================================================
        # 荷重ステップループ
        # ================================================================
        while True:
            query_out = stepping.process(
                AdaptiveStepInput(
                    action=StepAction.QUERY,
                    load_frac_prev=state.load_frac_prev,
                )
            )
            if not query_out.has_more_steps:
                break
            load_frac = query_out.next_load_frac

            _state_set(state, "step_display", state.step_display + 1)
            f_ext = f_ext_base + load_frac * f_ext_total

            # 接線予測子
            delta_frac = load_frac - state.load_frac_prev
            if _dynamics:
                dt_sub = getattr(_time_strategy, "_dt_physical", 0.0) * delta_frac
                if hasattr(_time_strategy, "predict"):
                    _state_set(state, "u", _time_strategy.predict(state.u, dt_sub))
            else:
                dt_sub = 0.0
                if (
                    state.delta_frac_prev > 1e-30
                    and delta_frac > 1e-30
                    and state.u_prev_converged is not None
                ):
                    du_prev = state.u - state.u_prev_converged
                    du_prev_norm = float(np.linalg.norm(du_prev))
                    if du_prev_norm > 1e-30:
                        ratio = min(delta_frac / state.delta_frac_prev, 2.0)
                        _state_set(state, "u", state.u + ratio * du_prev)

            # 処方変位
            if has_prescribed:
                state.u[_prescribed_dofs] = (load_frac - state.ul_frac_base) * _prescribed_values

            # 候補検出
            _dc_out = DeformedCoordsProcess().process(
                DeformedCoordsInput(
                    node_coords_ref=state.node_coords_ref,
                    u=state.u,
                    ndof_per_node=6,
                )
            )
            coords_def = _dc_out.coords
            _dc_step = _detect_proc.process(
                DetectCandidatesInput(
                    manager=manager,
                    node_coords=coords_def,
                    connectivity=connectivity,
                    radii=radii,
                    margin=broadphase_margin,
                    cell_size=broadphase_cell_size,
                    core_radii=core_radii,
                )
            )
            manager = _dc_step.manager
            _ug_step = _geom_proc.process(
                UpdateGeometryInput(manager=manager, node_coords=coords_def)
            )
            manager = _ug_step.manager
            _ensure_lam_size(state, _n_pairs(manager))

            # --- NR + Uzawa 実行（Static/Dynamic 完全分離） ---
            if _dynamics:
                step_input = NewtonUzawaDynamicStepInput(
                    config=nr_config_dyn,
                    u=state.u,
                    lam_all=state.lam_all,
                    f_ext=f_ext,
                    f_ext_ref_norm=f_ext_ref_norm,
                    fixed_dofs=fixed_dofs,
                    assemble_tangent=input_data.callbacks.assemble_tangent,
                    assemble_internal_force=input_data.callbacks.assemble_internal_force,
                    manager=manager,
                    node_coords_ref=state.node_coords_ref,
                    strategies=strategies,
                    k_pen=k_pen,
                    mu=mu,
                    u_ref=state.u_ref,
                    load_frac=load_frac,
                    load_frac_prev=state.load_frac_prev,
                    step_display=state.step_display,
                    dt_sub=dt_sub,
                    use_coating=use_coating,
                    dynamic_ref=dynamic_ref,
                )
                step_result = nr_process_dyn.process(step_input)
            else:
                step_input = NewtonUzawaStaticStepInput(
                    config=nr_config_sta,
                    u=state.u,
                    lam_all=state.lam_all,
                    f_ext=f_ext,
                    f_ext_ref_norm=f_ext_ref_norm,
                    fixed_dofs=fixed_dofs,
                    assemble_tangent=input_data.callbacks.assemble_tangent,
                    assemble_internal_force=input_data.callbacks.assemble_internal_force,
                    manager=manager,
                    node_coords_ref=state.node_coords_ref,
                    strategies=strategies,
                    k_pen=k_pen,
                    mu=mu,
                    u_ref=state.u_ref,
                    load_frac=load_frac,
                    load_frac_prev=state.load_frac_prev,
                    step_display=state.step_display,
                    use_coating=use_coating,
                    dynamic_ref=dynamic_ref,
                )
                step_result = nr_process_sta.process(step_input)
            _state_set(state, "total_newton", state.total_newton + step_result.n_newton_iters)
            last_diag = step_result.diagnostics

            # ==============================================================
            # 不収束処理
            # ==============================================================
            if not step_result.converged:
                fail_out = stepping.process(
                    AdaptiveStepInput(
                        action=StepAction.FAILURE,
                        load_frac=load_frac,
                        load_frac_prev=state.load_frac_prev,
                    )
                )
                if fail_out.can_retry:
                    _restore_checkpoint(state)
                    if _ul:
                        ul_assembler.rollback()
                        _state_set(state, "node_coords_ref", ul_assembler.coords_ref)
                    if _dynamics:
                        _time_strategy.restore_checkpoint()
                    _state_set(state, "step_display", state.step_display - 1)
                    print(f"  Adaptive dt retry: frac {load_frac:.4f} → sub-steps")
                    continue
                else:
                    print(
                        f"  WARNING: Step {state.step_display} "
                        f"(frac={load_frac:.4f}) did not converge."
                    )
                    _diag_report = DiagnosticsReportProcess().process(
                        DiagnosticsInput(diagnostics=last_diag)
                    )
                    print(_diag_report.report)
                    _u_out = _build_u_output(state, ul_assembler)
                    elapsed = time.perf_counter() - t0
                    return SolverResultData(
                        u=_u_out,
                        converged=False,
                        n_increments=state.step_display,
                        total_newton_iterations=state.total_newton,
                        displacement_history=state.disp_history,
                        contact_force_history=state.contact_force_history,
                        elapsed_seconds=elapsed,
                        diagnostics=last_diag,
                    )

            # ==============================================================
            # ステップ完了
            # ==============================================================

            # 被膜圧縮量保存
            if use_coating:
                for ci, pair in enumerate(manager.pairs):
                    manager.pairs[ci] = _evolve_pair(
                        pair,
                        state=_evolve_state(
                            pair.state,
                            coating_compression_prev=pair.state.coating_compression,
                        ),
                    )

            # 動的解析: 速度・加速度更新
            if _dynamics and dt_sub > 1e-30:
                _time_strategy.correct(state.u, np.zeros_like(state.u), dt_sub)

            # Updated Lagrangian: 参照配置更新
            if _ul:
                ul_assembler.update_reference(state.u)
                _state_set(state, "node_coords_ref", ul_assembler.coords_ref)
                _state_set(state, "ul_frac_base", load_frac)
                _state_set(state, "u", np.zeros(ndof))
                _state_set(state, "u_ref", np.zeros(ndof))

            # 適応時間増分: 次ステップ幅決定
            stepping.process(
                AdaptiveStepInput(
                    action=StepAction.SUCCESS,
                    load_frac=load_frac,
                    load_frac_prev=state.load_frac_prev,
                    n_iters=step_result.n_newton_iters,
                    n_active=step_result.n_active,
                    prev_n_active=state.prev_n_active,
                )
            )
            _state_set(state, "prev_n_active", step_result.n_active)

            # k_pen continuation
            k_pen_new = _penalty_strategy.compute_k_pen(state.step_display, state.step_display + 1)
            if abs(k_pen_new - k_pen) > 1e-30:
                k_pen = k_pen_new
                print(f"  k_pen continuation: k_pen → {k_pen:.2e}")

            # 状態更新
            _state_set(state, "delta_frac_prev", load_frac - state.load_frac_prev)
            _state_set(state, "u_prev_converged", state.u.copy())
            _state_set(state, "load_frac_prev", load_frac)
            for i, pair in enumerate(manager.pairs):
                if i < len(state.lam_all):
                    manager.pairs[i] = _evolve_pair(
                        pair,
                        state=_evolve_state(
                            pair.state,
                            lambda_n=state.lam_all[i],
                            p_n=max(0.0, state.lam_all[i] + k_pen * (-pair.state.gap)),
                        ),
                    )
            _state_set(state, "u_ref", state.u.copy())

            # チェックポイント保存
            _save_checkpoint(state)
            if _ul:
                ul_assembler.checkpoint()
            if _dynamics:
                _time_strategy.checkpoint()

            # 履歴記録
            state.load_history.append(load_frac)
            _u_hist = ul_assembler.u_total_accum + state.u if _ul else state.u.copy()
            state.disp_history.append(_u_hist.copy() if _ul else _u_hist)
            state.contact_force_history.append(float(np.linalg.norm(step_result.f_c)))
            try:
                _cg_out = ContactGraphProcess().process(
                    ContactGraphInput(manager=manager, step=state.step_display - 1)
                )
                state.graph_snapshots.append(_cg_out.graph)
            except Exception:
                pass

        # ================================================================
        # 正常終了
        # ================================================================
        _u_out = _build_u_output(state, ul_assembler)
        elapsed = time.perf_counter() - t0

        return SolverResultData(
            u=_u_out,
            converged=True,
            n_increments=state.step_display,
            total_newton_iterations=state.total_newton,
            displacement_history=state.disp_history,
            contact_force_history=state.contact_force_history,
            elapsed_seconds=elapsed,
            diagnostics=last_diag,
        )
