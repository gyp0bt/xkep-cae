"""ContactFrictionProcess — 摩擦接触ソルバー（SolverProcess）.

設計仕様: docs/contact_friction.md

内部構成:
- SolverStateOutput: 全可変状態（frozen dataclass）
- NewtonDynamicProcess: 1荷重増分の NR（動的のみ）
- AdaptiveSteppingProcess: 適応荷重増分制御（QUERY/SUCCESS/FAILURE）
- Strategy 5軸 + default_strategies()

status-222 で一本化:
- Uzawa 削除（純粋 Huber ペナルティ）
- 準静的ソルバー削除（動的のみ）
- 摩擦必須（Coulomb return mapping）
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact._manager_process import (
    DetectCandidatesInput,
    DetectCandidatesProcess,
    UpdateGeometryInput,
    UpdateGeometryProcess,
)
from xkep_cae.contact.solver._adaptive_stepping import (
    StepAction,
)
from xkep_cae.contact.solver._contact_graph import (
    ContactGraphInput,
    ContactGraphProcess,
)
from xkep_cae.contact.solver._diagnostics import (
    DiagnosticsInput,
    DiagnosticsReportProcess,
    IncrementDiagnosticsOutput,
)
from xkep_cae.contact.solver._energy_diagnostics import (
    EnergyHistory,
    EnergyHistoryEntry,
    StepEnergyDiagnosticsProcess,
    StepEnergyInput,
)
from xkep_cae.contact.solver._initial_penetration import (
    InitialPenetrationInput,
    InitialPenetrationProcess,
)
from xkep_cae.contact.solver._newton_uzawa_dynamic import (
    NewtonDynamicInput,
    NewtonDynamicProcess,
    NewtonDynamicStepInput,
)
from xkep_cae.contact.solver._solver_state import (
    SolverStateOutput,
    _build_u_output,
    _restore_checkpoint,
    _save_checkpoint,
    _state_set,
)
from xkep_cae.contact.solver._unified_time_controller import (
    TimeStepQueryInput,
    UnifiedTimeStepInput,
    UnifiedTimeStepProcess,
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
    """統一摩擦接触ソルバー（Huber ペナルティ + Coulomb 摩擦 + 動的のみ）.

    status-222 で一本化:
    - 動的ソルバーのみ（Generalized-α 時間積分）
    - Huber ペナルティ接触力
    - Coulomb 摩擦必須
    """

    meta = ProcessMeta(
        name="ContactFriction",
        module="solve",
        version="2.0.0",
        document_path="docs/contact_friction.md",
    )
    uses = [
        NewtonDynamicProcess,
        UnifiedTimeStepProcess,
        InitialPenetrationProcess,
        ContactGraphProcess,
        DiagnosticsReportProcess,
        StepEnergyDiagnosticsProcess,
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
            mu=input_data.contact.mu or 0.15,
            line_contact=True,
            smoothing_delta=manager.config.smoothing_delta,
        )
        _time_strategy = strategies.time_integration
        _penalty_strategy = strategies.penalty
        _friction_strategy = strategies.friction
        _contact_force_strategy = strategies.contact_force
        _dynamics = _time_strategy.is_dynamic

        if not _dynamics:
            raise ValueError(
                "ContactFrictionProcess: 動的ソルバーのみ対応（status-222）。"
                " mass_matrix / dt_physical を指定してください。"
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
        # contact_setup.k_pen が明示指定されている場合はそれを使用。
        # 動的解析では DynamicPenaltyEstimateProcess で c0*M_ii ベースの
        # k_pen を計算するため、ここで AutoBeamEIPenalty に上書きされないようにする。
        _setup_kpen = input_data.contact.k_pen
        if _setup_kpen is not None and _setup_kpen > 0.0:
            k_pen = _setup_kpen
        else:
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

        # --- チェックポイント初期化 ---
        _save_checkpoint(state)
        if _ul:
            ul_assembler.checkpoint()
        _time_strategy.checkpoint()

        # --- 適応荷重増分コントローラ ---
        _t_total = input_data.dt_physical if input_data.dt_physical else 1.0
        dt_grow_att = manager.config.dt_grow_attempt_threshold
        _dt_min_frac = manager.config.dt_min_fraction
        _dt_max_frac = manager.config.dt_max_fraction
        # dt_initial=t_total → fraction=1.0（全ステップを試み、dt_max で制限される）
        # これは元の AdaptiveSteppingInput(dt_initial_fraction=0.0) と同じ挙動
        _dt_initial = _t_total
        _dt_min = _dt_min_frac * _t_total if _dt_min_frac > 0 else _t_total / 32.0
        _dt_max = _dt_max_frac * _t_total if _dt_max_frac > 0 else _t_total
        stepping = UnifiedTimeStepProcess(
            UnifiedTimeStepInput(
                t_total=_t_total,
                dt_initial=_dt_initial,
                dt_min=_dt_min,
                dt_max=_dt_max,
                dt_grow_factor=manager.config.dt_grow_factor,
                dt_shrink_factor=manager.config.dt_shrink_factor,
                dt_grow_attempt_threshold=dt_grow_att if dt_grow_att > 0 else 5,
                dt_shrink_attempt_threshold=manager.config.dt_shrink_attempt_threshold,
                dt_contact_change_threshold=manager.config.dt_contact_change_threshold,
            )
        )

        # --- Newton プロセス（動的のみ） ---
        nr_config_dyn = NewtonDynamicInput(
            show_progress=True,
            max_attempts=input_data.max_nr_attempts,
            tol_force=input_data.tol_force,
            tol_disp=input_data.tol_disp,
            divergence_window=input_data.divergence_window,
            du_norm_cap=input_data.du_norm_cap,
        )
        nr_process_dyn = NewtonDynamicProcess()

        # --- 最終診断 ---
        last_diag = None
        _energy_history = EnergyHistory()
        _energy_proc = StepEnergyDiagnosticsProcess()
        _n_cutbacks = 0
        _increment_diag_list: list[IncrementDiagnosticsOutput] = []

        # ================================================================
        # 荷重ステップループ
        # ================================================================
        while True:
            query_out = stepping.process(
                TimeStepQueryInput(
                    action=StepAction.QUERY,
                    load_frac_prev=state.load_frac_prev,
                )
            )
            if not query_out.has_more_steps:
                break
            load_frac = query_out.load_frac

            _state_set(state, "increment_display", state.increment_display + 1)
            f_ext = f_ext_base + load_frac * f_ext_total

            # 接線予測子
            dt_sub = query_out.dt_sub
            if hasattr(_time_strategy, "predict"):
                _state_set(state, "u", _time_strategy.predict(state.u, dt_sub))

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

            # --- NR 実行（動的のみ） ---
            step_input = NewtonDynamicStepInput(
                config=nr_config_dyn,
                u=state.u,
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
                increment_display=state.increment_display,
                dt_sub=dt_sub,
                use_coating=use_coating,
                dynamic_ref=dynamic_ref,
            )
            step_result = nr_process_dyn.process(step_input)
            _state_set(state, "total_newton", state.total_attempts + step_result.n_attempts)
            last_diag = step_result.diagnostics

            # ==============================================================
            # 不収束処理
            # ==============================================================
            if not step_result.converged:
                _step_diverged = getattr(step_result, "diverged", False)
                fail_out = stepping.process(
                    TimeStepQueryInput(
                        action=StepAction.FAILURE,
                        load_frac=load_frac,
                        load_frac_prev=state.load_frac_prev,
                        diverged=_step_diverged,
                    )
                )
                _n_cutbacks = fail_out.n_cutbacks
                if fail_out.can_retry:
                    _restore_checkpoint(state)
                    if _ul:
                        ul_assembler.rollback()
                        _state_set(state, "node_coords_ref", ul_assembler.coords_ref)
                    _time_strategy.restore_checkpoint()
                    _state_set(state, "increment_display", state.increment_display - 1)
                    print(f"  Adaptive dt retry: frac {load_frac:.4f} → sub-steps")
                    continue
                else:
                    print(
                        f"  WARNING: Incr {state.increment_display} "
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
                        n_increments=state.increment_display,
                        total_attempts=state.total_attempts,
                        displacement_history=state.disp_history,
                        contact_force_history=state.contact_force_history,
                        load_history=list(state.load_history),
                        elapsed_seconds=elapsed,
                        diagnostics=last_diag,
                        energy_history=_energy_history,
                        n_cutbacks=_n_cutbacks,
                        increment_diagnostics=_increment_diag_list,
                    )

            # ==============================================================
            # ステップ完了
            # ==============================================================

            # エネルギー診断
                _f_int = input_data.callbacks.assemble_internal_force(state.u)
                _e_out = _energy_proc.process(
                    StepEnergyInput(
                        u=state.u,
                        velocity=_time_strategy.vel,
                        mass_matrix=_time_strategy.M,
                        f_int=_f_int,
                        f_ext=f_ext,
                        f_c=step_result.f_c,
                        dt=dt_sub,
                        step=state.increment_display,
                    )
                )
                _t_physical = load_frac * (input_data.dt_physical or 0.0)
                _energy_history.append(
                    EnergyHistoryEntry(
                        step=state.increment_display,
                        time=_t_physical,
                        kinetic_energy=_e_out.kinetic_energy,
                        strain_energy=_e_out.strain_energy,
                        external_work=_e_out.external_work,
                        contact_work=_e_out.contact_work,
                        total_energy=_e_out.total_energy,
                        energy_ratio=_e_out.energy_ratio,
                    )
                )

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

            # 速度・加速度更新
            if dt_sub > 1e-30:
                _time_strategy.correct(state.u, np.zeros_like(state.u), dt_sub)

            # 動的解析では UL 更新をスキップ — CR 梁の corotational 分解が
            # 大変形を処理するため、参照配置リセットは不要。

            # 適応時間増分: 次ステップ幅決定
            stepping.process(
                TimeStepQueryInput(
                    action=StepAction.SUCCESS,
                    load_frac=load_frac,
                    load_frac_prev=state.load_frac_prev,
                    n_attempts=step_result.n_attempts,
                    n_active=step_result.n_active,
                    prev_n_active=state.prev_n_active,
                )
            )
            _state_set(state, "prev_n_active", step_result.n_active)

            # k_pen continuation（明示指定されていない場合のみ）
            if not (_setup_kpen is not None and _setup_kpen > 0.0):
                k_pen_new = _penalty_strategy.compute_k_pen(
                    state.increment_display, state.increment_display + 1
                )
                if abs(k_pen_new - k_pen) > 1e-30:
                    k_pen = k_pen_new
                    print(f"  k_pen continuation: k_pen → {k_pen:.2e}")

            # 状態更新
            _state_set(state, "delta_frac_prev", load_frac - state.load_frac_prev)
            _state_set(state, "u_prev_converged", state.u.copy())
            _state_set(state, "load_frac_prev", load_frac)
            _state_set(state, "u_ref", state.u.copy())

            # チェックポイント保存
            _save_checkpoint(state)
            if _ul:
                ul_assembler.checkpoint()
            _time_strategy.checkpoint()

            # インクリメント診断生成
            _fc_norm = float(np.linalg.norm(step_result.f_c))
            _diag = last_diag
            _final_res = _diag.res_history[-1] if _diag and _diag.res_history else 0.0
            _conv_rate = 1.0
            if _diag and len(_diag.res_history) >= 2:
                _r_prev = _diag.res_history[-2]
                _r_curr = _diag.res_history[-1]
                _conv_rate = _r_curr / _r_prev if _r_prev > 1e-30 else 1.0
            _n_active_final = step_result.n_active
            _n_sliding = 0
            _n_sticking = 0
            if _diag and _diag.pair_snapshots:
                _last_snap = _diag.pair_snapshots[-1]
                _n_sliding = sum(1 for p in _last_snap if p.status == "sliding")
                _n_sticking = sum(1 for p in _last_snap if p.status not in ("inactive", "sliding"))
            _ke = 0.0
            _se = 0.0
            _te = 0.0
            _er = 1.0
            if _energy_history is not None and len(_energy_history.entries) > 0:
                _last_e = _energy_history.entries[-1]
                _ke = _last_e.kinetic_energy
                _se = _last_e.strain_energy
                _te = _last_e.total_energy
                _er = _last_e.energy_ratio
            _incr_diag = IncrementDiagnosticsOutput(
                step=state.increment_display,
                load_frac=load_frac,
                converged=True,
                n_attempts=step_result.n_attempts,
                n_active=_n_active_final,
                final_residual=_final_res,
                convergence_rate=_conv_rate,
                du_norm=_diag.du_norm_history[-1] if _diag and _diag.du_norm_history else 0.0,
                kinetic_energy=_ke,
                strain_energy=_se,
                total_energy=_te,
                energy_ratio=_er,
                n_active_pairs=_n_active_final,
                n_sliding_pairs=_n_sliding,
                n_sticking_pairs=_n_sticking,
                contact_force_norm=_fc_norm,
                cutback_count=_n_cutbacks,
                dt=dt_sub,
            )
            _increment_diag_list.append(_incr_diag)

            # 履歴記録
            state.load_history.append(load_frac)
            _u_hist = ul_assembler.u_total_accum + state.u if _ul else state.u.copy()
            state.disp_history.append(_u_hist.copy() if _ul else _u_hist)
            state.contact_force_history.append(_fc_norm)
            try:
                _cg_out = ContactGraphProcess().process(
                    ContactGraphInput(manager=manager, step=state.increment_display - 1)
                )
                state.graph_snapshots.append(_cg_out.graph)
            except Exception:
                pass

        # ================================================================
        # 正常終了
        # ================================================================
        _u_out = _build_u_output(state, ul_assembler)
        elapsed = time.perf_counter() - t0

        # エネルギー診断サマリ出力
        if _energy_history is not None and len(_energy_history.entries) > 0:
            print(_energy_history.summary())

        return SolverResultData(
            u=_u_out,
            converged=True,
            n_increments=state.increment_display,
            total_attempts=state.total_attempts,
            displacement_history=state.disp_history,
            contact_force_history=state.contact_force_history,
            load_history=list(state.load_history),
            elapsed_seconds=elapsed,
            diagnostics=last_diag,
            energy_history=_energy_history,
            n_cutbacks=_n_cutbacks,
            increment_diagnostics=_increment_diag_list,
        )
