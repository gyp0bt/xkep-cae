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

from xkep_cae.constraints.mpc_elimination import (
    RebuildMPCTransformInput,
    RebuildMPCTransformProcess,
)
from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact._manager_process import (
    DetectCandidatesInput,
    DetectCandidatesProcess,
    UpdateGeometryInput,
    UpdateGeometryProcess,
)
from xkep_cae.contact.contact_force.strategy import HuberContactForceProcess
from xkep_cae.contact.damping.strategy import ContactNormalDampingProcess
from xkep_cae.contact.friction.strategy import CoulombReturnMappingProcess
from xkep_cae.contact.geometry.strategy import LineToLineGaussProcess
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
from xkep_cae.contact.solver._newton_dynamic import (
    NewtonDynamicInput,
    NewtonDynamicProcess,
    NewtonDynamicStepInput,
)
from xkep_cae.contact.solver._solver_state import (
    SolverStateOutput,
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
from xkep_cae.time_integration.strategy import GeneralizedAlphaProcess


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
        RebuildMPCTransformProcess,
    ]

    # StrategySlot 宣言（Protocol は importlib 経由で取得するため object 型）.
    # status-320: `default_types` に `default_strategies()` が注入する具象 Process
    # 型を宣言することで、`ParameterSweepBenchmarkProcess._collect_uses_graph()` が
    # StrategySlot 経由の依存（ContactForceStStiffnessProcess 等の n² 成長プロセス）
    # もクラスレベルで到達できる。penalty_slot は同じ protocol を満たす複数の具象
    # （AutoBeamEIPenalty / ConstantPenalty）があり、どちらも `uses=[]` の葉なので
    # default_types 宣言は省略している。
    penalty_slot = StrategySlot(object)
    friction_slot = StrategySlot(
        object,
        default_types=(CoulombReturnMappingProcess,),
    )
    time_integration_slot = StrategySlot(
        object,
        default_types=(GeneralizedAlphaProcess,),
    )
    contact_force_slot = StrategySlot(
        object,
        required=False,
        default_types=(HuberContactForceProcess,),
    )
    contact_geometry_slot = StrategySlot(
        object,
        required=False,
        default_types=(LineToLineGaussProcess,),
    )
    # status-366 Phase 2: 接触法線減衰 escape hatch（候補 (e)、status-363 §4）.
    # default_types に ContactNormalDampingProcess を宣言して StrategySlot 経由で
    # `uses` グラフから到達可能化する。default OFF（c_n=0）なので
    # `ContactFrictionInputData.contact_damping_coefficient > 0` のときのみ
    # NR ループで f_damp / K_damp を加算する。
    damping_slot = StrategySlot(
        object,
        required=False,
        default_types=(ContactNormalDampingProcess,),
    )

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
            huber_delta_h=manager.config.huber_delta_h,
            penalty_exponent=input_data.penalty_exponent,
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
        _prescribed_func = input_data.boundary.prescribed_func
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
        # 被膜厚さ > 0 の場合、芯線半径を計算（status-301）
        # core_radii = radius - coating_thickness で被膜圧縮量が正しく計算される
        _coat_thick = manager.config.coating_thickness
        if _coat_thick > 0.0 and manager.config.coating_stiffness > 0.0:
            if np.isscalar(radii):
                core_radii = float(radii) - _coat_thick
            else:
                core_radii = np.asarray(radii, dtype=float) - _coat_thick
        else:
            core_radii = None

        state = SolverStateOutput(
            u=u0,
            u_ref=u0.copy(),
            node_coords_ref=node_coords_ref,
            u_prev_converged=u0.copy(),
        )

        # チェックポイント途中再開（status-279）
        _frac_start = input_data.load_frac_start
        if _frac_start > 0.0:
            _state_set(state, "load_frac_prev", _frac_start)
            # ul_frac_base は 0.0 のまま: 動的解析ではUL更新しないため、
            # 処方変位は load_frac * prescribed_values で計算される。
            # ul_frac_base = frac_start にすると処方変位がリセットされるバグ。
            print(f"  [RESUME] load_frac_start={_frac_start:.4f}")

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

        # --- UL参照配置更新（status-281） ---
        # CR梁は大変形を処理できるが、90°超のヘリカル素線曲げでは
        # 全累積変位からの接線剛性精度が低下する。
        # update_reference()で参照配置を更新し、各ステップの増分変位を
        # 小さく保つことで二次収束を維持する。
        # checkpoint復元時は u0 を���準にする（初回UL更新で増分=0を保証）
        _is_resume = getattr(input_data, "skip_initial_detection", False)
        _ul_ref_base = u0.copy() if (_ul and _is_resume) else (np.zeros(ndof) if _ul else None)
        _ul_ref_base_ckpt: np.ndarray | None = None  # チェックポイント用

        def _ul_tangent_wrapper(u_total: np.ndarray) -> object:
            """ULアセンブラに増分変位を渡すラッパー."""
            u_incr = u_total - _ul_ref_base
            return input_data.callbacks.assemble_tangent(u_incr)

        def _ul_internal_force_wrapper(u_total: np.ndarray) -> np.ndarray:
            """ULアセンブラに増分変位を渡すラッパー."""
            u_incr = u_total - _ul_ref_base
            return input_data.callbacks.assemble_internal_force(u_incr)

        # ULありの場合はラッパー経由、なしの場合は直接コールバック
        _asm_tangent = _ul_tangent_wrapper if _ul else input_data.callbacks.assemble_tangent
        _asm_internal_force = (
            _ul_internal_force_wrapper if _ul else input_data.callbacks.assemble_internal_force
        )

        # --- 初期貫入チェック ---
        broadphase_margin = 0.0
        broadphase_cell_size = None
        _detect_proc = DetectCandidatesProcess()
        _geom_proc = UpdateGeometryProcess()
        _skip_init_detect = getattr(input_data, "skip_initial_detection", False)
        if not _skip_init_detect:
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
        else:
            print("  [RESUME] 初期接触検出スキップ（checkpoint pairs使用）")
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
            _ul_ref_base_ckpt = _ul_ref_base.copy()
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

        # チェックポイント途中再開: steppingのキューをfrac_start以降に設定
        if _frac_start > 0.0:
            _asp = stepping._adaptivesteppingprocess
            _asp._queue.clear()
            _dt_frac = _dt_max / _t_total if _t_total > 0 else 0.025
            _asp._queue.append(min(_frac_start + _dt_frac, 1.0))
            stepping._last_load_frac_prev = _frac_start

        # --- Newton プロセス（動的のみ） ---
        _compute_cond = getattr(input_data, "compute_condition_number", False)
        nr_config_dyn = NewtonDynamicInput(
            show_progress=True,
            max_attempts=input_data.max_nr_attempts,
            tol_force=input_data.tol_force,
            tol_disp=input_data.tol_disp,
            divergence_window=input_data.divergence_window,
            du_norm_cap=input_data.du_norm_cap,
            compute_condition_number=_compute_cond,
            char_length=_beam_L,
            dof_scale_rot=getattr(input_data, "dof_scale_rot", 1.0),
            contact_relax_omega=getattr(input_data, "contact_relax_omega", 0.5),
            stall_window=getattr(input_data, "stall_window", 4),
            tangent_fd_diagnostic=getattr(input_data, "tangent_fd_diagnostic", False),
            kc_component_fd_diagnostic=getattr(input_data, "kc_component_fd_diagnostic", False),
            chattering_delta_h_boost=getattr(input_data, "chattering_delta_h_boost", 4.0),
            chattering_extra_attempts=getattr(input_data, "chattering_extra_attempts", 20),
            nr_min_restore=getattr(input_data, "nr_min_restore", False),
            nr_min_restore_window=getattr(input_data, "nr_min_restore_window", 3),
            chattering_freeze_enabled=getattr(input_data, "chattering_freeze_enabled", True),
            chattering_freeze_max_cycles=getattr(input_data, "chattering_freeze_max_cycles", 5),
            chattering_freeze_nr_max=getattr(input_data, "chattering_freeze_nr_max", 15),
            chattering_freeze_tol_factor=getattr(input_data, "chattering_freeze_tol_factor", 10.0),
            # 接触 backtracking line search（status-362）
            contact_backtracking_enabled=getattr(input_data, "contact_backtracking_enabled", False),
            contact_backtracking_max_steps=getattr(input_data, "contact_backtracking_max_steps", 4),
            contact_backtracking_active_flip_threshold=getattr(
                input_data, "contact_backtracking_active_flip_threshold", 3
            ),
            contact_backtracking_active_flip_ratio=getattr(
                input_data, "contact_backtracking_active_flip_ratio", 0.3
            ),
            contact_backtracking_residual_ratio=getattr(
                input_data, "contact_backtracking_residual_ratio", 2.0
            ),
            contact_backtracking_alpha_decay=getattr(
                input_data, "contact_backtracking_alpha_decay", 0.5
            ),
            contact_backtracking_min_alpha=getattr(
                input_data, "contact_backtracking_min_alpha", 0.0625
            ),
            contact_backtracking_mixed_only=getattr(
                input_data, "contact_backtracking_mixed_only", True
            ),
            contact_backtracking_rate_threshold=getattr(
                input_data, "contact_backtracking_rate_threshold", 0.85
            ),
            # 接触法線減衰 escape hatch（status-366 Phase 2、候補 (e)）
            contact_damping_coefficient=getattr(input_data, "contact_damping_coefficient", 0.0),
        )
        nr_process_dyn = NewtonDynamicProcess()

        # --- 最終診断 ---
        last_diag = None
        _energy_history = EnergyHistory()
        _energy_proc = StepEnergyDiagnosticsProcess()
        _n_cutbacks = 0
        _increment_diag_list: list[IncrementDiagnosticsOutput] = []
        # 履歴アキュムレータ（SolverStateOutput から分離: status-251 S1修正）
        _load_history: list[float] = []
        _disp_history: list[np.ndarray] = []
        _contact_force_history: list[float] = []
        _graph_snapshots: list[object] = []
        # 接触法線減衰エネルギー履歴（status-366 Phase 2、候補 (e)）
        # 成功インクリメント毎に (load_frac, E_damp_cumulative) を追記。
        _damping_energy_history: list[tuple[float, float]] = []
        _damping_energy_cumulative: float = 0.0
        # 微小dt対策: 成功インクリメントのf_refを追跡し、下限値として使用（status-297）
        _global_f_ref: float = 0.0
        # status-307: 収束型インクリメント統計
        _conv_type_counts: dict[str, int] = {"force": 0, "disp": 0, "energy": 0}
        # status-333: M-κ追跡 + 接触ペアスナップショット
        _track_mk = getattr(input_data, "track_mk", False)
        _mk_moment_dofs = getattr(input_data, "mk_moment_dofs", ())
        _mk_curvature_func = getattr(input_data, "mk_curvature_func", None)
        _mk_history: list[tuple[float, float]] = []
        _track_pairs = getattr(input_data, "track_contact_pairs", False)
        _pair_history: list[tuple[float, tuple]] = []

        # ================================================================
        # 荷重ステップループ
        # ================================================================
        _mpc_current = input_data.boundary.mpc_transform
        _mpc_current_ckpt = _mpc_current  # MPC Tチェックポイント（status-283）
        _mpc_groups = input_data.boundary.mpc_groups  # T再構築用（status-283）
        _max_incr = input_data.max_increments
        _incr_count = 0
        while True:
            if _max_incr > 0 and _incr_count >= _max_incr:
                break
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
                if _prescribed_func is not None:
                    # prescribed_func(frac) は state.u に書き込む絶対値を返す
                    state.u[_prescribed_dofs] = _prescribed_func(load_frac)
                else:
                    state.u[_prescribed_dofs] = (
                        load_frac - state.ul_frac_base
                    ) * _prescribed_values

            # MPC制約をuに伝搬: u_full = T @ u_red（slave DOFをmaster値から再計算）
            if _mpc_current is not None:
                _mpc = _mpc_current
                _u_red = state.u[_mpc.independent_dofs]
                _u_proj = _mpc.T @ _u_red
                if hasattr(_u_proj, "toarray"):
                    _u_proj = _u_proj.toarray().ravel()
                state.u[:] = np.asarray(_u_proj).ravel()
                # time integrator の予測子もMPC射影（status-255）:
                # correct() で acc = c0*(u - u_pred) を計算するため、
                # u_pred もMPC整合でないと slave DOF の加速度が不正になる。
                if hasattr(_time_strategy, "_u_pred"):
                    _up_red = _time_strategy._u_pred[_mpc.independent_dofs]
                    _up_proj = _mpc.T @ _up_red
                    if hasattr(_up_proj, "toarray"):
                        _up_proj = _up_proj.toarray().ravel()
                    _time_strategy._u_pred[:] = np.asarray(_up_proj).ravel()

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
                UpdateGeometryInput(
                    manager=manager,
                    node_coords=coords_def,
                    connectivity=connectivity,
                )
            )
            manager = _ug_step.manager

            # --- NR 実行（動的のみ） ---
            step_input = NewtonDynamicStepInput(
                config=nr_config_dyn,
                u=state.u,
                f_ext=f_ext,
                f_ext_ref_norm=f_ext_ref_norm,
                fixed_dofs=fixed_dofs,
                assemble_tangent=_asm_tangent,
                assemble_internal_force=_asm_internal_force,
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
                connectivity=connectivity,
                mpc_transform=_mpc_current,
                # status-297: 通常インクリメントの力収束水準を絶対許容値として渡す
                # atol = global_f_ref × tol_force → 微小dtでも通常スケールの収束水準で判定
                atol_force=_global_f_ref * nr_config_dyn.tol_force,
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
                        if _ul_ref_base_ckpt is not None:
                            _ul_ref_base[:] = _ul_ref_base_ckpt
                        if _mpc_current_ckpt is not None:
                            _mpc_current = _mpc_current_ckpt
                    _time_strategy.restore_checkpoint()
                    _state_set(state, "increment_display", state.increment_display - 1)
                    # status-307: カットバック原因タグ + dt値出力
                    _cb_reason = getattr(step_result, "failure_reason", "unknown")
                    _cb_tag = f"[CUTBACK:{_cb_reason}]" if _cb_reason else "[CUTBACK]"
                    print(
                        f"  {_cb_tag} frac {load_frac:.4f}, "
                        f"dt={dt_sub:.4e} → sub-steps "
                        f"(cutback #{_n_cutbacks})"
                    )
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
                    _u_out = state.u.copy()  # state.uは初期配置からの全累積変位
                    elapsed = time.perf_counter() - t0
                    return SolverResultData(
                        u=_u_out,
                        converged=False,
                        n_increments=state.increment_display,
                        total_attempts=state.total_attempts,
                        displacement_history=tuple(_disp_history),
                        contact_force_history=tuple(_contact_force_history),
                        load_history=tuple(_load_history),
                        elapsed_seconds=elapsed,
                        diagnostics=last_diag,
                        energy_history=_energy_history,
                        n_cutbacks=_n_cutbacks,
                        increment_diagnostics=tuple(_increment_diag_list),
                        final_contact_manager=manager,
                        final_ul_ref_base=_ul_ref_base.copy() if _ul_ref_base is not None else None,
                        final_node_coords_ref=state.node_coords_ref.copy(),
                        moment_curvature_history=tuple(_mk_history),
                        contact_pair_history=tuple(_pair_history),
                        damping_energy_history=tuple(_damping_energy_history),
                    )

            # ==============================================================
            # ステップ完了
            # ==============================================================

            # status-307: 収束型統計
            _ct = getattr(step_result, "convergence_type", "")
            if _ct in _conv_type_counts:
                _conv_type_counts[_ct] += 1

            # status-307: 被膜圧縮統計（被膜あり時のみ）
            if use_coating and manager.pairs:
                _coat_comps = [
                    p.state.coating_compression
                    for p in manager.pairs
                    if p.state.coating_compression > 0
                ]
                if _coat_comps:
                    _coat_maxes = [
                        (p.radius_a - p.core_radius_a) + (p.radius_b - p.core_radius_b)
                        for p in manager.pairs
                        if p.state.coating_compression > 0
                    ]
                    _coat_ratios = [
                        c / max(m, 1e-30) for c, m in zip(_coat_comps, _coat_maxes, strict=True)
                    ]
                    _n_pen = sum(1 for r in _coat_ratios if r >= 1.0)
                    # 50ステップごとに出力（毎ステップは冗長）
                    if state.increment_display % 50 == 0 or _n_pen > 0:
                        print(
                            f"  [coat] incr={state.increment_display}: "
                            f"n_active={len(_coat_comps)}, "
                            f"mean={sum(_coat_ratios) / len(_coat_ratios) * 100:.0f}%, "
                            f"max={max(_coat_ratios) * 100:.0f}%, "
                            f"n_penetrated={_n_pen}"
                        )

            # status-333: M-κ追跡（f_intから曲げモーメント、curvature_funcから曲率）
            _f_int = _asm_internal_force(state.u)
            if _track_mk and _mk_moment_dofs and _mk_curvature_func is not None:
                _mk_moment = sum(float(_f_int[d]) for d in _mk_moment_dofs)
                _mk_kappa = _mk_curvature_func(load_frac)
                _mk_history.append((_mk_kappa, _mk_moment))
            _coat_energy = 0.0
            if use_coating:
                _coat_energy = strategies.coating.energy(manager.pairs, manager.config)
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
                    coating_energy=_coat_energy,
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
                    coating_energy=_e_out.coating_energy,
                )
            )

            # f_ref追跡更新（status-297: 微小dt対策）
            # 収束成功時のみ更新（failure時は continue or return で到達しない）
            _step_f_ref = getattr(step_result, "f_ref_used", 0.0)
            if step_result.converged and _step_f_ref > 1e-30:
                if _global_f_ref < 1e-30:
                    _global_f_ref = _step_f_ref
                else:
                    # 指数移動平均（α=0.3）で平滑化
                    _global_f_ref = 0.7 * _global_f_ref + 0.3 * _step_f_ref

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

            # UL参照配置更新（status-281: 大変形ヘリカル素線対応）
            # 各収束後にupdate_reference()で参照配置を更新し、増分変位を小さく保つ。
            if _ul and hasattr(ul_assembler, "update_reference"):
                _u_incr_ul = state.u - _ul_ref_base
                ul_assembler.update_reference(_u_incr_ul)
                _ul_ref_base[:] = state.u

                # MPC変換行列T再構築（status-283: 大回転時の線形化破綻対策）
                # UL参照配置更新後、変形後座標でMPCの相対位置ベクトルrを再計算。
                if _mpc_current is not None and _mpc_groups is not None:
                    _n_nodes = len(state.node_coords_ref)
                    _coords_current = state.node_coords_ref.copy()
                    for _i_n in range(_n_nodes):
                        _coords_current[_i_n] += state.u[_i_n * 6 : _i_n * 6 + 3]
                    _mpc_current = RebuildMPCTransformProcess().process(
                        RebuildMPCTransformInput(
                            mpc_groups=_mpc_groups,
                            node_coords=_coords_current,
                            ndof_total=ndof,
                            ndof_per_node=6,
                        )
                    )

            # 適応時間増分: 次ステップ幅決定（力ベース SDI 判定, status-233）
            _fc_norm = float(np.linalg.norm(step_result.f_c))
            stepping.process(
                TimeStepQueryInput(
                    action=StepAction.SUCCESS,
                    load_frac=load_frac,
                    load_frac_prev=state.load_frac_prev,
                    n_attempts=step_result.n_attempts,
                    n_active=step_result.n_active,
                    prev_n_active=state.prev_n_active,
                    contact_force_norm=_fc_norm,
                    prev_contact_force_norm=state.prev_contact_force_norm,
                )
            )
            _state_set(state, "prev_n_active", step_result.n_active)
            _state_set(state, "prev_contact_force_norm", _fc_norm)

            # 成功した dt ステップのみカウント（カットバックは含めない）
            _incr_count += 1

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
                _ul_ref_base_ckpt = _ul_ref_base.copy()
                _mpc_current_ckpt = _mpc_current  # MPC T チェックポイント（status-283）
            _time_strategy.checkpoint()

            # ── チェックポイント保存（status-286: API化 + 環境変数互換） ──
            # API: checkpoint_path / checkpoint_frac で指定（優先）
            # 環境変数: XKEP_CHECKPOINT_FRAC / XKEP_CHECKPOINT_PATH（後方互換）
            import os as _os

            _ckpt_path = input_data.checkpoint_path or _os.environ.get("XKEP_CHECKPOINT_PATH", "")
            _ckpt_frac_str = _os.environ.get("XKEP_CHECKPOINT_FRAC", "")
            _ckpt_frac = input_data.checkpoint_frac
            if _ckpt_frac_str and not input_data.checkpoint_path:
                _ckpt_frac = float(_ckpt_frac_str)
            if _ckpt_path and load_frac >= _ckpt_frac and not hasattr(self, "_ckpt_saved"):
                import pickle as _pickle

                _ckpt_data = {
                    "state": state,
                    "time_vel": _time_strategy.vel.copy(),
                    "time_acc": _time_strategy.acc.copy(),
                    "time_vel_old": _time_strategy._vel_old.copy()
                    if hasattr(_time_strategy, "_vel_old")
                    else None,
                    "time_acc_old": _time_strategy._acc_old.copy()
                    if hasattr(_time_strategy, "_acc_old")
                    else None,
                    "time_u_pred": _time_strategy._u_pred.copy()
                    if hasattr(_time_strategy, "_u_pred")
                    else None,
                    "manager_pairs": manager.pairs[:],
                    "manager_config": manager.config,
                    "load_frac": load_frac,
                    "k_pen": k_pen,
                    "stepping_state": stepping._state if hasattr(stepping, "_state") else None,
                    "dt_sub": dt_sub,
                    "incr_count": _incr_count,
                    "cutback_count": _n_cutbacks,
                    "node_coords_ref": state.node_coords_ref.copy(),
                }
                # ULアセンブラの完全状態保存（自工程保証: 次工程で
                # クリーンに開始できる状態を保証）
                _ul_asm = None
                if _ul and hasattr(ul_assembler, "_u_total_accum"):
                    _ul_asm = ul_assembler
                elif _ul and hasattr(ul_assembler, "_asm"):
                    _ul_asm = ul_assembler._asm
                if _ul_asm is not None:
                    _ckpt_data["ul_u_total_accum"] = _ul_asm._u_total_accum.copy()
                    if hasattr(_ul_asm, "coords_ref"):
                        _ckpt_data["ul_coords_ref"] = _ul_asm.coords_ref.copy()
                    if hasattr(_ul_asm, "R_ref"):
                        _ckpt_data["ul_R_ref"] = _ul_asm.R_ref.copy()
                if _ul_ref_base is not None:
                    _ckpt_data["ul_ref_base"] = _ul_ref_base.copy()
                if hasattr(manager, "connectivity"):
                    _ckpt_data["connectivity"] = manager.connectivity
                with open(_ckpt_path, "wb") as _f:
                    _pickle.dump(_ckpt_data, _f)
                self._ckpt_saved = True
                print(f"  [CHECKPOINT] frac={load_frac:.4f} → {_ckpt_path} (incr={_incr_count})")

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
            _load_history.append(load_frac)
            _u_hist = ul_assembler.u_total_accum + state.u if _ul else state.u.copy()
            _disp_history.append(_u_hist.copy() if _ul else _u_hist)
            _contact_force_history.append(_fc_norm)
            # 接触法線減衰エネルギー累積（status-366 Phase 2、候補 (e)）
            # DynamicStepOutput.damping_energy_rate は NR 最終反復の瞬時消散率。
            # dt 乗算で増分あたりの消散エネルギーに換算し累積する。
            _damp_rate = getattr(step_result, "damping_energy_rate", 0.0)
            if _damp_rate > 0.0 and dt_sub > 0.0:
                _damping_energy_cumulative += _damp_rate * dt_sub
                _damping_energy_history.append((load_frac, _damping_energy_cumulative))
            # status-333: 接触ペアスナップショット記録
            if _track_pairs and manager.pairs:
                from xkep_cae.core.data import ContactPairSnapshotEntry

                _snap_entries = tuple(
                    ContactPairSnapshotEntry(
                        elem_a=p.elem_a,
                        elem_b=p.elem_b,
                        p_n=p.state.p_n,
                        gap=p.state.gap,
                        slip_s=float(p.state.z_t[0]),
                        slip_t=float(p.state.z_t[1]),
                        stick=p.state.stick,
                        dissipation=p.state.dissipation,
                    )
                    for p in manager.pairs
                    if p.state.p_n > 0
                )
                _pair_history.append((load_frac, _snap_entries))
            try:
                _cg_out = ContactGraphProcess().process(
                    ContactGraphInput(manager=manager, step=state.increment_display - 1)
                )
                _graph_snapshots.append(_cg_out.graph)
            except Exception:
                pass

        # ================================================================
        # 正常終了
        # ================================================================
        _u_out = state.u.copy()  # state.uは初期配置からの全累積変位
        _final_vel = _time_strategy.vel.copy() if _dynamics else None
        _final_acc = _time_strategy.acc.copy() if _dynamics else None
        elapsed = time.perf_counter() - t0

        # エネルギー診断サマリ出力
        if _energy_history is not None and len(_energy_history.entries) > 0:
            print(_energy_history.summary())

        # status-307: 収束型統計サマリ
        _total_conv = sum(_conv_type_counts.values())
        if _total_conv > 0:
            _parts = []
            for _ckey in ("force", "disp", "energy"):
                _cn = _conv_type_counts[_ckey]
                if _cn > 0:
                    _parts.append(f"{_ckey}={_cn}({100 * _cn // _total_conv}%)")
            print(f"  [収束型統計] {', '.join(_parts)}, total={_total_conv}")

        return SolverResultData(
            u=_u_out,
            converged=True,
            n_increments=state.increment_display,
            total_attempts=state.total_attempts,
            displacement_history=_disp_history,
            contact_force_history=_contact_force_history,
            load_history=list(_load_history),
            elapsed_seconds=elapsed,
            diagnostics=last_diag,
            energy_history=_energy_history,
            n_cutbacks=_n_cutbacks,
            increment_diagnostics=_increment_diag_list,
            final_contact_manager=manager,
            final_velocity=_final_vel,
            final_acceleration=_final_acc,
            final_ul_ref_base=_ul_ref_base.copy() if _ul_ref_base is not None else None,
            final_node_coords_ref=state.node_coords_ref.copy(),
            moment_curvature_history=tuple(_mk_history),
            contact_pair_history=tuple(_pair_history),
            damping_energy_history=tuple(_damping_energy_history),
        )
