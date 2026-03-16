"""ContactFrictionProcess — 摩擦接触ソルバー（SolverProcess）.

旧 xkep_cae_deprecated/process/concrete/solve_contact_friction.py の完全書き直し。
設計仕様: docs/contact_friction.md

内部構成:
- SolverState: 全可変状態の集約
- NewtonUzawaLoop: 1荷重増分の NR + Uzawa
- AdaptiveLoadController: 適応荷重増分制御
- Strategy 5軸 + default_strategies()
"""

from __future__ import annotations

import importlib
import time

import numpy as np

from xkep_cae.core import (
    ContactFrictionInputData,
    ProcessMeta,
    SolverProcess,
    SolverResultData,
)
from xkep_cae.core.slots import StrategySlot


def _import_module(module_path: str) -> object:
    """importlib 経由でモジュールをインポート."""
    return importlib.import_module(module_path)


class ContactFrictionProcess(
    SolverProcess[ContactFrictionInputData, SolverResultData],
):
    """統一摩擦接触ソルバー（smooth penalty + 自動時間積分選択）.

    入力の mass_matrix / dt_physical の有無で動的/準静的を自動判定。
    - 動的: Generalized-α 時間積分
    - 準静的: 荷重制御 or 変位制御

    内部構成:
    - SolverState: 全可変状態の集約
    - NewtonUzawaLoop: 1荷重増分のNR+Uzawa
    - AdaptiveLoadController: 適応荷重増分制御
    - Strategy 5軸: penalty, friction, time_integration, contact_force, coating
    """

    meta = ProcessMeta(
        name="ContactFriction",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_friction.md",
    )
    uses = []

    # StrategySlot 宣言（Protocol は importlib 経由で取得するため object 型）
    penalty_slot = StrategySlot(object)
    friction_slot = StrategySlot(object)
    time_integration_slot = StrategySlot(object)
    contact_force_slot = StrategySlot(object, required=False)
    contact_geometry_slot = StrategySlot(object, required=False)

    def __init__(self, strategies: object | None = None) -> None:
        # deprecated 版の SolverStrategies を使用（NewtonUzawaLoop との互換性のため）
        if strategies is None:
            _data_mod = importlib.import_module("xkep_cae.process.data")
            strategies = _data_mod.default_strategies()
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

        # deprecated モジュールの遅延インポート（C14 準拠）
        _utils_mod = importlib.import_module("xkep_cae.contact.utils")
        _ip_mod = importlib.import_module("xkep_cae.contact.initial_penetration")
        _graph_mod = importlib.import_module("xkep_cae.contact.graph")
        _state_mod = importlib.import_module("xkep_cae.process.strategies.solver_state")
        _nul_mod = importlib.import_module("xkep_cae.process.strategies.newton_uzawa")
        _asc_mod = importlib.import_module("xkep_cae.process.strategies.adaptive_stepping")

        deformed_coords = _utils_mod.deformed_coords
        check_initial_penetration = _ip_mod.check_initial_penetration
        adjust_initial_positions = _ip_mod.adjust_initial_positions
        snapshot_contact_graph = _graph_mod.snapshot_contact_graph
        SolverState = _state_mod.SolverState  # noqa: N806
        NewtonUzawaLoop = _nul_mod.NewtonUzawaLoop  # noqa: N806
        NewtonUzawaConfig = _nul_mod.NewtonUzawaConfig  # noqa: N806
        AdaptiveLoadController = _asc_mod.AdaptiveLoadController  # noqa: N806
        AdaptiveSteppingConfig = _asc_mod.AdaptiveSteppingConfig  # noqa: N806

        ndof = len(input_data.boundary.f_ext_total)
        f_ext_total = input_data.boundary.f_ext_total
        manager = input_data.contact.manager
        ul_assembler = input_data.callbacks.ul_assembler

        # --- Strategy 生成（deprecated 版: NewtonUzawaLoop 互換） ---
        _data_mod = importlib.import_module("xkep_cae.process.data")
        _default_strategies = _data_mod.default_strategies
        strategies = _default_strategies(
            ndof=ndof,
            mass_matrix=input_data.mass_matrix,
            damping_matrix=input_data.damping_matrix,
            dt_physical=input_data.dt_physical,
            rho_inf=input_data.rho_inf,
            velocity=input_data.velocity,
            acceleration=input_data.acceleration,
            k_pen=input_data.contact.k_pen,
            use_friction=True,
            mu=input_data.contact.mu or 0.15,
            contact_mode="smooth_penalty",
            line_contact=True,
        )
        _time_strategy = strategies.time_integration
        _penalty_strategy = strategies.penalty
        _friction_strategy = strategies.friction
        _contact_force_strategy = strategies.contact_force
        _dynamics = _time_strategy.is_dynamic

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

        # --- FrictionStrategy k_pen 伝播 ---
        if hasattr(_friction_strategy, "_k_pen"):
            _friction_strategy._k_pen = k_pen
        if hasattr(_friction_strategy, "_k_t_ratio"):
            _friction_strategy._k_t_ratio = manager.config.k_t_ratio
        if hasattr(_contact_force_strategy, "_ndof"):
            _contact_force_strategy._ndof = ndof

        # --- SolverState 初期化 ---
        u0 = input_data.u0.copy() if input_data.u0 is not None else np.zeros(ndof)
        node_coords_ref = input_data.mesh.node_coords.copy()
        connectivity = input_data.mesh.connectivity
        radii = input_data.mesh.radii
        core_radii = None

        state = SolverState(
            u=u0,
            lam_all=np.zeros(manager.n_pairs),
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
            node_coords_ref, n_pen_fixed, n_gap_closed = adjust_initial_positions(
                manager.pairs, node_coords_ref, _pos_tol
            )
            if (n_pen_fixed + n_gap_closed) > 0:
                print(
                    f"  初期位置調整: 貫入修正={n_pen_fixed}ペア, ギャップ閉鎖={n_gap_closed}ペア"
                )
            state.node_coords_ref = node_coords_ref
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
        n_initial_pen = check_initial_penetration(
            manager.pairs, node_coords_ref, manager.config.coating_stiffness
        )
        if n_initial_pen > 0 and not use_coating and not _ul_has_accum:
            raise ValueError(
                f"初期貫入が検出されました: {n_initial_pen}ペア。"
                f"メッシュ生成時のgapを増やしてください。"
            )

        state.ensure_lam_size(manager.n_pairs)

        # --- チェックポイント初期化 ---
        state.save_checkpoint()
        if _ul:
            ul_assembler.checkpoint()
        _time_strategy.checkpoint()

        # --- 適応荷重増分コントローラ ---
        dt_grow_iter = manager.config.dt_grow_iter_threshold
        stepping_config = AdaptiveSteppingConfig(
            dt_initial_fraction=0.0,
            dt_grow_factor=manager.config.dt_grow_factor,
            dt_shrink_factor=manager.config.dt_shrink_factor,
            dt_grow_iter_threshold=dt_grow_iter if dt_grow_iter > 0 else 5,
            dt_shrink_iter_threshold=manager.config.dt_shrink_iter_threshold,
            dt_contact_change_threshold=manager.config.dt_contact_change_threshold,
            dt_min_fraction=manager.config.dt_min_fraction,
            dt_max_fraction=manager.config.dt_max_fraction,
        )
        controller = AdaptiveLoadController(stepping_config)

        # --- Newton-Uzawa ループ ---
        nr_loop = NewtonUzawaLoop(NewtonUzawaConfig(show_progress=True))

        # --- 最終診断 ---
        last_diag = None

        # ================================================================
        # 荷重ステップループ
        # ================================================================
        while controller.has_steps:
            load_frac = controller.peek_next()

            if controller.should_skip(load_frac, state.load_frac_prev):
                controller.pop_step()
                continue

            state.step_display += 1
            f_ext = f_ext_base + load_frac * f_ext_total

            # 接線予測子
            delta_frac = load_frac - state.load_frac_prev
            if _dynamics:
                dt_sub = getattr(_time_strategy, "_dt_physical", 0.0) * delta_frac
                if hasattr(_time_strategy, "predict"):
                    state.u = _time_strategy.predict(state.u, dt_sub)
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
                        state.u = state.u + ratio * du_prev

            # 処方変位
            if has_prescribed:
                state.u[_prescribed_dofs] = (load_frac - state.ul_frac_base) * _prescribed_values

            # 候補検出
            coords_def = deformed_coords(state.node_coords_ref, state.u, 6)
            manager.detect_candidates(
                coords_def,
                connectivity,
                radii,
                margin=broadphase_margin,
                cell_size=broadphase_cell_size,
                core_radii=core_radii,
            )
            manager.update_geometry(coords_def)
            state.ensure_lam_size(manager.n_pairs)

            # --- NR + Uzawa 実行 ---
            step_result = nr_loop.run(
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
            state.total_newton += step_result.n_newton_iters
            last_diag = step_result.diagnostics

            # ==============================================================
            # 不収束処理
            # ==============================================================
            if not step_result.converged:
                can_retry = controller.on_failure(load_frac, state.load_frac_prev)
                if can_retry:
                    state.restore_checkpoint()
                    if _ul:
                        ul_assembler.rollback()
                        state.node_coords_ref = ul_assembler.coords_ref
                    if _dynamics:
                        _time_strategy.restore_checkpoint()
                    state.step_display -= 1
                    print(f"  Adaptive dt retry: frac {load_frac:.4f} → sub-steps")
                    continue
                else:
                    print(
                        f"  WARNING: Step {state.step_display} "
                        f"(frac={load_frac:.4f}) did not converge."
                    )
                    print(last_diag.format_report())
                    result = state.build_result(
                        converged=False,
                        ul_assembler=ul_assembler,
                        time_strategy=_time_strategy,
                        diagnostics=last_diag,
                    )
                    result.n_active_final = manager.n_active
                    elapsed = time.perf_counter() - t0
                    return SolverResultData(
                        u=result.u,
                        converged=False,
                        n_increments=result.n_increments,
                        total_newton_iterations=result.total_newton_iterations,
                        displacement_history=result.displacement_history,
                        contact_force_history=result.contact_force_history,
                        elapsed_seconds=elapsed,
                        diagnostics=result.diagnostics,
                    )

            # ==============================================================
            # ステップ完了
            # ==============================================================
            controller.pop_step()

            # 被膜圧縮量保存
            if use_coating:
                for pair in manager.pairs:
                    pair.state.coating_compression_prev = pair.state.coating_compression

            # 動的解析: 速度・加速度更新
            if _dynamics and dt_sub > 1e-30:
                _time_strategy.correct(state.u, np.zeros_like(state.u), dt_sub)

            # Updated Lagrangian: 参照配置更新
            if _ul:
                ul_assembler.update_reference(state.u)
                state.node_coords_ref = ul_assembler.coords_ref
                state.ul_frac_base = load_frac
                state.u = np.zeros(ndof)
                state.u_ref = np.zeros(ndof)

            # 適応時間増分: 次ステップ幅決定
            n_iters = step_result.n_newton_iters
            controller.on_success(
                load_frac,
                state.load_frac_prev,
                n_iters,
                step_result.n_active,
                state.prev_n_active,
            )
            state.prev_n_active = step_result.n_active

            # k_pen continuation
            k_pen_new = _penalty_strategy.compute_k_pen(state.step_display, state.step_display + 1)
            if abs(k_pen_new - k_pen) > 1e-30:
                k_pen = k_pen_new
                print(f"  k_pen continuation: k_pen → {k_pen:.2e}")

            # 状態更新
            state.delta_frac_prev = load_frac - state.load_frac_prev
            state.u_prev_converged = state.u.copy()
            state.load_frac_prev = load_frac
            for i, pair in enumerate(manager.pairs):
                if i < len(state.lam_all):
                    pair.state.lambda_n = state.lam_all[i]
                    pair.state.p_n = max(0.0, state.lam_all[i] + k_pen * (-pair.state.gap))
            state.u_ref = state.u.copy()

            # チェックポイント保存
            state.save_checkpoint()
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
                graph = snapshot_contact_graph(manager, step_index=state.step_display - 1)
                state.graph_history.add_snapshot(graph)
            except Exception:
                pass

        # ================================================================
        # 正常終了
        # ================================================================
        result = state.build_result(
            converged=True,
            ul_assembler=ul_assembler,
            time_strategy=_time_strategy,
        )
        result.n_active_final = manager.n_active
        elapsed = time.perf_counter() - t0

        return SolverResultData(
            u=result.u,
            converged=True,
            n_increments=result.n_increments,
            total_newton_iterations=result.total_newton_iterations,
            displacement_history=result.displacement_history,
            contact_force_history=result.contact_force_history,
            elapsed_seconds=elapsed,
            diagnostics=result.diagnostics,
        )
