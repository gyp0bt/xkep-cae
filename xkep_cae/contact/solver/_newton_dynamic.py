"""Newton-Raphson イテレーション（動的）.

Generalized-α 時間積分による慣性力・減衰力を含む。
内部ステップは全てサブプロセスに委譲。

status-222 で Uzawa ループを削除。純粋 Huber ペナルティ + Coulomb 摩擦。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.contact._assembly_utils import _contact_dofs
from xkep_cae.contact.solver._diagnostics import (
    ConvergenceDiagnosticsOutput,
    NRIterationSnapshotOutput,
    PairDiagnosticsOutput,
    classify_chattering_type,
)
from xkep_cae.contact.solver._newton_steps import (
    ContactBacktrackingLineSearchInput,
    ContactBacktrackingLineSearchProcess,
    ContactForceAssemblyInput,
    ContactForceAssemblyProcess,
    ConvergenceCheckInput,
    ConvergenceCheckProcess,
    ConvergenceType,
    LinearSolveInput,
    LinearSolveProcess,
    LineSearchUpdateInput,
    LineSearchUpdateProcess,
    TangentAssemblyInput,
    TangentAssemblyProcess,
    TangentFDDiagnosticInput,
    TangentFDDiagnosticProcess,
)
from xkep_cae.core import ProcessMeta, SolverProcess
from xkep_cae.verify.kc_component_fd import (
    ContactKcComponentFDDiagnosticInput,
    ContactKcComponentFDDiagnosticProcess,
)


@dataclass(frozen=True)
class DynamicStepOutput:
    """1荷重増分の結果（動的）."""

    converged: bool
    n_attempts: int
    n_active: int
    f_c: np.ndarray
    diagnostics: ConvergenceDiagnosticsOutput
    diverged: bool = False
    f_ref_used: float = 0.0  # 実際に使用されたf_ref（status-297: global追跡用）
    convergence_type: str = ""  # "force"/"disp"/"energy"/""（status-307: 収束型追跡）
    failure_reason: str = ""  # "nr_limit"/"diverged"/"relax_fail"/"solve_fail"/""（status-307）


@dataclass(frozen=True)
class NewtonDynamicInput:
    """Newton ループの設定（動的）."""

    max_attempts: int = 50
    tol_force: float = 1e-8
    tol_disp: float = 1e-8
    use_line_search: bool = True
    line_search_max_steps: int = 5
    du_norm_cap: float = 0.0
    show_progress: bool = True
    ndof_per_node: int = 6
    divergence_window: int = 5
    compute_condition_number: bool = False  # 条件数診断（低速: toarray() 使用）
    char_length: float = 0.0  # 代表長さ [mm]（重み付きノルム用、status-241）
    dof_scale_rot: float = 1.0  # 回転 DOF の NR 更新スケーリング（status-241）
    # 接触力リラクゼーション（status-247: NR 2-サイクル対策）
    contact_relax_omega: float = 0.5  # リラクゼーション係数（0.5 = 半分ブレンド）
    stall_window: int = 4  # 残差プラトーをストールと判定するまでの反復数
    relax_max_iter: int = 25  # リラクゼーション有効後の最大反復数（超過で早期打切り）
    tangent_fd_diagnostic: bool = False  # ストール時にFD接線診断を実行（status-256）
    # チャタリング時 Huber delta_h ブースト（status-268）
    # チャタリング検知時に delta_h を boost 倍に拡大し、ペナルティ関数を平滑化。
    # 力ブレンド（残差-Jacobian不整合）を回避し、NR二次収束を維持。
    chattering_delta_h_boost: float = 4.0
    chattering_extra_attempts: int = 20  # ブースト時の追加NR反復上限（status-268）
    # NR残差最小値リストア（status-269: 過修正防止）
    # NR反復中の残差最小値を追跡し、発散検知時に最小残差の状態にリストアして
    # インクリメント成功とする。frozen_hermite_tangent=False の過修正発散を回避。
    nr_min_restore: bool = (
        False  # status-277: 残差最小値リストアOFF（不正確な状態を次incrに持ち越す問題）
    )
    nr_min_restore_window: int = 3  # 最小値からN回連続増加でリストア発動
    # NRインナー活性ペア凍結（status-276: 大量同時活性化対策）
    # att=0 で p_n > 0 のペアDOFを記録し、以降の反復で新規活性化ペアの
    # 接触力をゼロマスクする。活性集合変動による正のフィードバックを遮断。
    # 注: ベンチマーク検証で逆効果（物理的不整合誘発）のためデフォルトFalse
    freeze_contact_dofs_in_nr: bool = False
    # チャタリング検知→接触凍結モード（status-284: 陽解法スイッチ）
    # チャタリング検知後、接触力を凍結（外力扱い）+ K_c除外で構造系のみ収束させ、
    # 収束後に接触を再評価。活性集合が安定するまで凍結→再評価サイクルを繰り返す。
    # delta_hブーストより優先（チャタリングの根本原因=活性集合振動を直接抑制）。
    chattering_freeze_enabled: bool = True  # 接触凍結モード有効化
    chattering_freeze_max_cycles: int = 5  # 凍結→再評価の最大サイクル数
    chattering_freeze_nr_max: int = 15  # 凍結中の構造NR最大反復数
    chattering_freeze_tol_factor: float = 10.0  # 凍結中の収束判定緩和倍率
    # Type D 対策（status-288: 接線剛性不整合への自動応答）
    # 連続Type D検知時にFD接線診断を自動トリガーし、不整合の程度を定量化。
    # 不整合が大きい場合は接線スケーリングで補正し、線形収束率を改善。
    type_d_auto_fd: bool = True  # Type D連続検知時のFD診断自動トリガー
    type_d_consecutive_threshold: int = 5  # FD診断トリガーまでの連続Type D回数
    type_d_tangent_refresh_rate: float = 0.85  # この収束率を超えたら接線リフレッシュ考慮
    type_d_extra_attempts: int = 15  # Type D検知時の追加NR反復上限
    # K_c 成分分解 FD 診断（status-343/344: 仮説 A 最終検証）
    # tangent_fd_diagnostic/type_d_auto_fd と同じトリガー契機で
    # K_mat/K_geo/K_st を個別取得し、4 組み合わせで FD 相対誤差を報告。
    # x 成分 68% 不整合（status-342）の部分行列由来を切り分ける。
    kc_component_fd_diagnostic: bool = False  # True なら K_c 成分分解 FD 診断も出力
    # 接触残差 / active flip backtracking line search（status-362: 仮説 C 候補 (c)）
    # 既存 NCPLineSearch は ||R_u|| 全体でしか発散判定しないため、mixed (C+D)
    # 領域（active flip + tangent 不整合同時発生）の step を抑制できない。
    # status-361 の Type 分布実測で 19 本重荷重のみ mixed 16.6% 突出を受けた
    # 仮説 C (c) line search 強化。`ContactBacktrackingLineSearchProcess` が
    # u_try = u + α·du で接触残差・active 数を再評価し、過剰増加時に α 半減。
    # デフォルト OFF（実験的機能、7 本撚線回帰なしを前提に opt-in）。
    contact_backtracking_enabled: bool = False
    contact_backtracking_max_steps: int = 4
    contact_backtracking_active_flip_threshold: int = 3
    contact_backtracking_active_flip_ratio: float = 0.3
    contact_backtracking_residual_ratio: float = 2.0
    contact_backtracking_alpha_decay: float = 0.5
    contact_backtracking_min_alpha: float = 0.0625
    # mixed (C+D) 検知時のみ発動（Falseなら毎反復発動、コスト約 2x）
    contact_backtracking_mixed_only: bool = True
    # 発動判定閾値: active_set_changed かつ 収束率 > rate_threshold なら mixed 候補
    contact_backtracking_rate_threshold: float = 0.85


@dataclass(frozen=True)
class NewtonDynamicStepInput:
    """1荷重増分の NR 入力（動的）."""

    config: NewtonDynamicInput
    u: np.ndarray
    f_ext: np.ndarray
    f_ext_ref_norm: float
    fixed_dofs: np.ndarray
    assemble_tangent: object
    assemble_internal_force: object
    manager: object
    node_coords_ref: np.ndarray
    strategies: object
    k_pen: float
    mu: float
    u_ref: np.ndarray
    load_frac: float
    load_frac_prev: float
    increment_display: int
    dt_sub: float
    use_coating: bool
    dynamic_ref: bool
    connectivity: np.ndarray | None = None  # Hermite 中心線補間用
    mpc_transform: object | None = None  # MPCEliminationResult（status-253）
    atol_force: float = 0.0  # 力残差の絶対許容値 [N]（status-297: 微小dt対策）


# 後方互換エイリアス（呼び出し側の段階的移行用）
NewtonUzawaDynamicInput = NewtonDynamicInput
NewtonUzawaDynamicStepInput = NewtonDynamicStepInput


class NewtonDynamicProcess(
    SolverProcess[NewtonDynamicStepInput, DynamicStepOutput],
):
    """1荷重増分の Newton-Raphson イテレーション（動的）.

    Generalized-α 時間積分による慣性力・減衰力を含む。
    status-222 で Uzawa ループを除去し、純粋 Newton-Raphson に簡素化。
    """

    meta = ProcessMeta(
        name="NewtonDynamic",
        module="solve",
        version="2.0.0",
        document_path="docs/newton_solver.md",
    )
    uses = [
        ContactForceAssemblyProcess,
        ConvergenceCheckProcess,
        TangentAssemblyProcess,
        LinearSolveProcess,
        LineSearchUpdateProcess,
        ContactBacktrackingLineSearchProcess,
        TangentFDDiagnosticProcess,
        ContactKcComponentFDDiagnosticProcess,
    ]

    def process(  # noqa: C901, PLR0912, PLR0915
        self,
        input_data: NewtonDynamicStepInput,
    ) -> DynamicStepOutput:
        """1荷重増分のNRを実行（動的）.

        input_data.u は in-place で更新される。
        """
        cfg = input_data.config
        u = input_data.u
        f_ext = input_data.f_ext
        manager = input_data.manager
        strategies = input_data.strategies
        k_pen = input_data.k_pen
        mu = input_data.mu
        u_ref = input_data.u_ref
        load_frac = input_data.load_frac
        load_frac_prev = input_data.load_frac_prev
        increment_display = input_data.increment_display
        dt_sub = input_data.dt_sub
        ndof = len(f_ext)

        _time_strategy = strategies.time_integration
        _contact_force_strategy = strategies.contact_force
        _friction_strategy = strategies.friction
        _coating_strategy = strategies.coating

        # サブプロセスインスタンス
        _force_proc = ContactForceAssemblyProcess()
        _conv_proc = ConvergenceCheckProcess()
        _tangent_proc = TangentAssemblyProcess()
        _solve_proc = LinearSolveProcess()
        _linesearch_proc = LineSearchUpdateProcess()
        _bt_proc = ContactBacktrackingLineSearchProcess()

        diag = ConvergenceDiagnosticsOutput(step=increment_display, load_frac=load_frac)
        total_attempts = 0
        f_c = np.zeros(ndof)
        energy_ref = None
        step_converged = False
        n_active = 0
        _diverged = False
        _convergence_type = ""  # status-307: "force"/"disp"/"energy"
        _failure_reason = ""  # status-307: "nr_limit"/"diverged"/"relax_fail"/"solve_fail"
        _consecutive_increase = 0
        _prev_res_ratio = float("inf")
        _incr_f_ref: float = 0.0  # インクリメント内の参照残差（初回で設定）

        # 接触力リラクゼーション用トラッキング（status-247）
        _f_c_prev = np.zeros(ndof)
        _consecutive_stall = 0
        _relax_active = False
        _relax_iter = 0  # リラクゼーション有効化後の反復数
        _current_omega = 1.0  # 現在のリラクゼーション係数
        _prev_n_active = -1
        _delta_h_boosted = False  # status-268: delta_hブースト適用中フラグ
        _effective_max = cfg.max_attempts  # チャタリング時に動的拡張（status-268）
        # NR 2サイクル振動検知用（status-278）
        _u_prev2: np.ndarray | None = None  # 2反復前のu
        _u_prev1: np.ndarray | None = None  # 1反復前のu
        # NRリミットサイクル検知用（status-279: 汎用N-サイクル対応）
        _u_history_ring: list[np.ndarray] = []  # 直近の u 履歴（最大6個保持）
        _limit_cycle_window = 6  # 検査窓サイズ
        # NR残差最小値リストア用トラッキング（status-269）
        _min_res_ratio = float("inf")
        _min_res_u: np.ndarray | None = None
        _min_res_f_c: np.ndarray | None = None
        _min_res_att = 0
        _min_res_increase_count = 0  # 最小値からの連続増加カウント
        # チャタリング接触凍結モード（status-284）
        _freeze_active = False  # 凍結モード有効中フラグ
        _freeze_f_c: np.ndarray | None = None  # 凍結接触力ベクトル
        _freeze_cycle = 0  # 凍結→再評価サイクルカウント
        _freeze_n_active_prev = -1  # 前回凍結時のactive数
        # NRインナー活性ペア凍結（status-276）
        _frozen_contact_dofs: np.ndarray | None = None  # att=0で記録
        # チャタリング内訳分析用: 前反復のペア状態追跡（status-287）
        _prev_pair_statuses: dict[int, str] = {}  # pair_id -> status文字列
        # Type D 連続検知カウンタ（status-288）
        _consecutive_type_d = 0
        _type_d_fd_triggered = False  # FD診断を既にトリガー済みか
        _type_d_fd_result: str | None = None  # FD診断結果サマリ

        att = -1
        while att + 1 < _effective_max:
            att += 1
            total_attempts += 1

            # ── ステップ 2〜5: 接触力アセンブリ + 残差 ──
            force_out = _force_proc.process(
                ContactForceAssemblyInput(
                    u=u,
                    f_ext=f_ext,
                    fixed_dofs=input_data.fixed_dofs,
                    manager=manager,
                    node_coords_ref=input_data.node_coords_ref,
                    contact_force_strategy=_contact_force_strategy,
                    friction_strategy=_friction_strategy,
                    coating_strategy=_coating_strategy,
                    k_pen=k_pen,
                    mu=mu,
                    u_ref=u_ref,
                    load_frac=load_frac,
                    load_frac_prev=load_frac_prev,
                    increment_display=increment_display,
                    ndof_per_node=cfg.ndof_per_node,
                    use_coating=input_data.use_coating,
                    assemble_internal_force=input_data.assemble_internal_force,
                    connectivity=input_data.connectivity,
                )
            )
            f_c = force_out.f_c
            R_u = force_out.R_u

            # ── NRインナー活性ペア凍結（status-276） ──
            # att=0 で p_n > 0 のペアDOFを記録。
            # att > 0 では、初回になかった接触力をマスクして正のフィードバックを遮断。
            if cfg.freeze_contact_dofs_in_nr:
                if att == 0:
                    _frozen_contact_dofs = np.abs(f_c) > 1e-30
                elif _frozen_contact_dofs is not None:
                    _new_contact = (np.abs(f_c) > 1e-30) & ~_frozen_contact_dofs
                    if _new_contact.any():
                        f_c_orig = f_c.copy()
                        f_c = f_c.copy()
                        f_c[_new_contact] = 0.0
                        # R_u は f_c を含む形で計算済み → 差し替え
                        R_u = R_u - f_c_orig + f_c
                        R_u[input_data.fixed_dofs] = 0.0

            # ── 接触凍結モード（status-284: 陽解法スイッチ） ──
            # 凍結中: 実際の接触力 f_c を凍結値 _freeze_f_c で差し替え。
            # 接触力を外力として固定し、構造系のみでNR収束を求める。
            if _freeze_active and _freeze_f_c is not None:
                R_u = R_u - f_c + _freeze_f_c
                R_u[input_data.fixed_dofs] = 0.0
                f_c = _freeze_f_c.copy()
                _current_omega = 0.0  # K_c除外のため接線スケール=0
            # ── 接触チャタリング対策（status-268: delta_hブースト） ──
            # 凍結モードでない場合のフォールバック
            elif att > 0 and _relax_active:
                _relax_iter += 1
                if _delta_h_boosted:
                    pass
                elif cfg.contact_relax_omega < 1.0:
                    _current_omega = max(
                        0.05,
                        cfg.contact_relax_omega * (0.7 ** (_relax_iter // 2)),
                    )
                    f_c_blend = _current_omega * f_c + (1.0 - _current_omega) * _f_c_prev
                    R_u = R_u - f_c + f_c_blend
                    R_u[input_data.fixed_dofs] = 0.0
                    f_c = f_c_blend
            _f_c_prev = f_c.copy()

            # 動的: 慣性力・減衰力を残差に加算
            if dt_sub > 1e-30:
                _time_strategy.correct(u, np.zeros_like(u), dt_sub)
                R_u = _time_strategy.effective_residual(R_u, dt_sub)
                R_u[input_data.fixed_dofs] = 0.0

            coords_def = force_out.coords_def

            # ── ステップ 6: 力収束判定 ──
            # 変位制御問題: 初回反復の残差ノルムをインクリメント内参照値として保存
            _eff_ref = _incr_f_ref if _incr_f_ref > 1e-30 else input_data.f_ext_ref_norm
            conv_out = _conv_proc.process(
                ConvergenceCheckInput(
                    R_u=R_u,
                    du=None,
                    u=u,
                    f_ext_ref_norm=_eff_ref,
                    tol_force=cfg.tol_force,
                    tol_disp=cfg.tol_disp,
                    dynamic_ref=input_data.dynamic_ref,
                    is_first_attempt=(att == 0),
                    energy_ref=energy_ref,
                    manager=manager,
                    ndof_per_node=cfg.ndof_per_node,
                    char_length=cfg.char_length,
                    mpc_transform=input_data.mpc_transform,
                    atol_force=input_data.atol_force,
                )
            )
            # 初回反復で参照残差を保存
            if att == 0 and input_data.dynamic_ref:
                _incr_f_ref = conv_out.f_ref
            # status-307: f_ref値と判定モードの出力（初回のみ）
            if att == 0 and cfg.show_progress:
                _ref_src = "dynamic_ref" if input_data.dynamic_ref else "f_ext"
                _atol_str = ""
                if input_data.atol_force > 0:
                    _atol_str = f", atol={input_data.atol_force:.2e}"
                print(
                    f"  [f_ref] Incr {increment_display}: "
                    f"f_ref={conv_out.f_ref:.3e} ({_ref_src})"
                    f"{_atol_str}"
                )
            n_active = conv_out.n_active

            _res_ratio = conv_out.res_trans_norm / conv_out.f_ref
            diag.res_history.append(_res_ratio)
            diag.ncp_history.append(0.0)
            diag.ncp_t_history.append(0.0)
            diag.n_active_history.append(n_active)
            if len(diag.res_history) >= 2:
                _prev = diag.res_history[-2]
                diag.convergence_rate_history.append(_res_ratio / _prev if _prev > 1e-30 else 1.0)
            else:
                diag.convergence_rate_history.append(1.0)

            # ペア別診断スナップショット
            _pair_snap: list[PairDiagnosticsOutput] = []
            if hasattr(manager, "pairs"):
                for _pi, _pair in enumerate(manager.pairs):
                    if hasattr(_pair, "state"):
                        _st = _pair.state
                        _pair_snap.append(
                            PairDiagnosticsOutput(
                                pair_id=_pi,
                                elem_a=int(_pair.elem_a),
                                elem_b=int(_pair.elem_b),
                                gap=float(_st.gap),
                                p_n=float(_st.p_n),
                                status=str(_st.status.name).lower()
                                if hasattr(_st.status, "name")
                                else str(_st.status),
                            )
                        )
            diag.pair_snapshots.append(_pair_snap)

            # ── チャタリング内訳分析: NR反復スナップショット（status-287） ──
            # 接触DOF vs 構造DOFの残差分離 + ペア状態遷移追跡
            _cur_pair_statuses: dict[int, str] = {}
            _n_sliding_snap = 0
            _n_sticking_snap = 0
            for _ps in _pair_snap:
                _cur_pair_statuses[_ps.pair_id] = _ps.status
                if _ps.status == "sliding":
                    _n_sliding_snap += 1
                elif _ps.status == "active":
                    _n_sticking_snap += 1

            _pairs_activated = 0
            _pairs_deactivated = 0
            _pairs_s2s = 0  # stick→slide
            _pairs_sl2st = 0  # slide→stick
            _friction_changed = False
            for _pid, _cur_st in _cur_pair_statuses.items():
                _prev_st = _prev_pair_statuses.get(_pid, "inactive")
                if _prev_st == "inactive" and _cur_st != "inactive":
                    _pairs_activated += 1
                elif _prev_st != "inactive" and _cur_st == "inactive":
                    _pairs_deactivated += 1
                if _prev_st == "active" and _cur_st == "sliding":
                    _pairs_s2s += 1
                    _friction_changed = True
                elif _prev_st == "sliding" and _cur_st == "active":
                    _pairs_sl2st += 1
                    _friction_changed = True
            # 前反復に存在して今回消えたペアをdeactivated扱い
            for _pid in _prev_pair_statuses:
                if _pid not in _cur_pair_statuses:
                    _prev_st2 = _prev_pair_statuses[_pid]
                    if _prev_st2 != "inactive":
                        _pairs_deactivated += 1

            _active_set_chg = (_pairs_activated + _pairs_deactivated) > 0

            # 接触DOF vs 構造DOFの残差分離
            _contact_res = 0.0
            _structural_res = 0.0
            if hasattr(manager, "pairs") and len(R_u) > 0:
                _c_mask = np.zeros(len(R_u), dtype=bool)
                for _pair_obj in manager.pairs:
                    if hasattr(_pair_obj, "state") and _pair_obj.state.p_n > 0.0:
                        _cdofs = _contact_dofs(_pair_obj, cfg.ndof_per_node)
                        _valid = _cdofs[_cdofs < len(R_u)]
                        _c_mask[_valid] = True
                _contact_res = float(np.linalg.norm(R_u[_c_mask])) if _c_mask.any() else 0.0
                _structural_res = float(np.linalg.norm(R_u[~_c_mask]))

            _conv_rate = diag.convergence_rate_history[-1] if diag.convergence_rate_history else 1.0

            # comp別残差ノルム（status-290: x,y,z,θx,θy,θz）
            _ndpn = cfg.ndof_per_node
            _comp_norms: tuple[float, ...] = ()
            if _ndpn > 0 and len(R_u) >= _ndpn:
                _cn = []
                for _c in range(_ndpn):
                    _cdof_arr = np.arange(_c, len(R_u), _ndpn)
                    _cn.append(float(np.linalg.norm(R_u[_cdof_arr])))
                _comp_norms = tuple(_cn)

            _nr_snap = NRIterationSnapshotOutput(
                att=att,
                res_ratio=_res_ratio,
                n_active=n_active,
                n_sliding=_n_sliding_snap,
                n_sticking=_n_sticking_snap,
                contact_res_norm=_contact_res,
                structural_res_norm=_structural_res,
                active_set_changed=_active_set_chg,
                friction_state_changed=_friction_changed,
                pairs_activated=_pairs_activated,
                pairs_deactivated=_pairs_deactivated,
                pairs_stick_to_slide=_pairs_s2s,
                pairs_slide_to_stick=_pairs_sl2st,
                convergence_rate=_conv_rate,
                comp_res_norms=_comp_norms,
            )
            diag.nr_iteration_snapshots.append(_nr_snap)
            _prev_pair_statuses = _cur_pair_statuses

            # ── Type D 連続検知追跡（status-288） ──
            _iter_type_d = classify_chattering_type(_nr_snap)
            if "D" in _iter_type_d and not _nr_snap.active_set_changed:
                _consecutive_type_d += 1
            else:
                _consecutive_type_d = 0

            if conv_out.converged:
                # ── 凍結モード中の再評価（status-284） ──
                # 凍結中に構造系が収束した場合、凍結を解除して接触を再評価。
                # 新しい残差が tol_force * freeze_tol_factor 以下なら成功。
                # そうでなければ、新しい接触力で再凍結してNR継続。
                if _freeze_active and _freeze_f_c is not None:
                    # 実際の接触力で残差を再計算
                    _reeval_out = _force_proc.process(
                        ContactForceAssemblyInput(
                            u=u,
                            f_ext=f_ext,
                            fixed_dofs=input_data.fixed_dofs,
                            manager=manager,
                            node_coords_ref=input_data.node_coords_ref,
                            contact_force_strategy=_contact_force_strategy,
                            friction_strategy=_friction_strategy,
                            coating_strategy=_coating_strategy,
                            k_pen=k_pen,
                            mu=mu,
                            u_ref=u_ref,
                            load_frac=load_frac,
                            load_frac_prev=load_frac_prev,
                            increment_display=increment_display,
                            ndof_per_node=cfg.ndof_per_node,
                            use_coating=input_data.use_coating,
                            assemble_internal_force=input_data.assemble_internal_force,
                            connectivity=input_data.connectivity,
                        )
                    )
                    _reeval_R = _reeval_out.R_u
                    if dt_sub > 1e-30:
                        _reeval_R = _time_strategy.effective_residual(_reeval_R, dt_sub)
                        _reeval_R[input_data.fixed_dofs] = 0.0
                    _reeval_f_c = _reeval_out.f_c
                    _reeval_n_active = sum(
                        1 for _p in manager.pairs if hasattr(_p, "state") and _p.state.gap < 0.0
                    )
                    # 並進残差で判定
                    _reeval_trans = float(
                        np.linalg.norm(_reeval_R.reshape(-1, cfg.ndof_per_node)[:, :3].ravel())
                    )
                    _reeval_ratio = _reeval_trans / max(conv_out.f_ref, 1e-30)
                    _freeze_tol = cfg.tol_force * cfg.chattering_freeze_tol_factor

                    if _reeval_ratio < _freeze_tol:
                        # 再評価成功: 凍結解除して収束
                        _freeze_active = False
                        _current_omega = 1.0
                        f_c = _reeval_f_c
                        n_active = _reeval_n_active
                        step_converged = True
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"凍結解除→再評価成功 "
                                f"||R_t||/||f||={_reeval_ratio:.3e} < {_freeze_tol:.1e} "
                                f"({_reeval_n_active} active)"
                            )
                        break
                    elif _freeze_cycle < cfg.chattering_freeze_max_cycles:
                        # 再凍結: 新しい接触力で凍結を更新してNR継続
                        _freeze_f_c = _reeval_f_c.copy()
                        _freeze_cycle += 1
                        _consecutive_stall = 0
                        _relax_active = True  # 凍結モード維持
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"再評価||R_t||/||f||={_reeval_ratio:.3e} > {_freeze_tol:.1e} "
                                f"→ 再凍結 (cycle={_freeze_cycle}, "
                                f"active={_reeval_n_active})"
                            )
                        continue
                    else:
                        # 凍結サイクル上限: 緩和判定で最終チェック
                        if _reeval_ratio < _freeze_tol * 10.0:
                            _freeze_active = False
                            _current_omega = 1.0
                            f_c = _reeval_f_c
                            n_active = _reeval_n_active
                            step_converged = True
                            if cfg.show_progress:
                                print(
                                    f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                    f"attempt {att}, "
                                    f"凍結サイクル上限→緩和判定で収束 "
                                    f"||R_t||/||f||={_reeval_ratio:.3e}"
                                )
                            break
                        # 凍結失敗: 通常NRに戻す
                        _freeze_active = False
                        _current_omega = 1.0
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"凍結サイクル上限超過→通常NR復帰 "
                                f"(||R_t||/||f||={_reeval_ratio:.3e})"
                            )
                        continue
                step_converged = True
                _convergence_type = "force"
                if cfg.show_progress:
                    print(
                        f"  Incr {increment_display} (frac={load_frac:.4f}), "
                        f"attempt {att}, "
                        f"||R_t||/||f|| = {conv_out.res_trans_norm / conv_out.f_ref:.3e}, "
                        f"||R_r|| = {conv_out.res_rot_norm:.3e} "
                        f"(force converged, {n_active} active)"
                    )
                break

            # ── 発散早期検知 ──
            # status-264: 力収束判定と同じ res_trans_norm ベースに統一。
            # res_u_norm は回転残差を含むが f_ref は並進のみで不整合だった。
            _cur_ratio = conv_out.res_trans_norm / conv_out.f_ref

            # ── NR残差最小値追跡（status-269） ──
            if cfg.nr_min_restore and att >= 1 and _cur_ratio < _min_res_ratio:
                _min_res_ratio = _cur_ratio
                _min_res_u = u.copy()
                _min_res_f_c = f_c.copy()
                _min_res_att = att
                _min_res_increase_count = 0
            elif cfg.nr_min_restore and att >= 1 and _cur_ratio > _min_res_ratio:
                _min_res_increase_count += 1

            if att > 0 and _cur_ratio > _prev_res_ratio * 1.01:
                _consecutive_increase += 1
            else:
                _consecutive_increase = 0

            # ストール検知（status-247 + status-255 拡張 + status-284 凍結モード）:
            # (a) 高残差ストール (_cur_ratio > 0.5): delta_hブースト or リラクゼーション
            # (b) 低残差チャタリング (tol未満に収束せず多反復): 接触凍結モード
            _active_changed = n_active != _prev_n_active

            # --- 低残差チャタリング検知（status-284） ---
            # 残差が tol に到達せず多反復で振動/停滞するパターンを直接検知。
            # 典型: ||R||/||f|| ≈ 3.5e-4 ↔ 8.1e-4 で2サイクル（active数一定でも発生）。
            # active set の数ではなく残差の振動パターンで判定する。
            if (
                att >= 15
                and not _relax_active
                and not _freeze_active
                and cfg.chattering_freeze_enabled
                and _freeze_cycle < cfg.chattering_freeze_max_cycles
                and _cur_ratio > cfg.tol_force
                and n_active > 0
                and len(diag.res_history) >= 6
            ):
                # 直近6反復の残差で振動/停滞を検知
                _recent6 = diag.res_history[-6:]
                _r_max6 = max(_recent6)
                _r_min6 = min(_recent6)
                _r_mean6 = sum(_recent6) / 6.0
                # 条件: 平均残差 > tol かつ max/min 比が小さい（振動域が狭い）
                _ratio_spread = _r_max6 / max(_r_min6, 1e-30)
                if _r_mean6 > cfg.tol_force * 10.0 and _ratio_spread < 100.0:
                    # チャタリングタイプ分類（status-287）
                    _chatter_type = (
                        classify_chattering_type(_nr_snap) if diag.nr_iteration_snapshots else "-"
                    )
                    # status-288: Type D支配的（活性集合安定 + 接線不整合）の場合、
                    # 凍結モード（活性集合変化を抑制）は原理的に効かない。
                    # 代わりにNR上限拡張で線形収束を許容する。
                    _is_type_d_dominant = (
                        "D" in _chatter_type
                        and not _nr_snap.active_set_changed
                        and _consecutive_type_d >= 3
                    )
                    if _is_type_d_dominant:
                        # Type D対策: 凍結せずNR上限拡張
                        _relax_active = True
                        _effective_max = max(_effective_max, att + cfg.type_d_extra_attempts)
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"低残差停滞[{_chatter_type}] → Type D対策: "
                            f"NR上限拡張→{_effective_max} "
                            f"(att={att}, ||R||/||f||={_cur_ratio:.3e}, "
                            f"rate={_conv_rate:.3f}, "
                            f"R_c={_contact_res:.2e}, R_s={_structural_res:.2e}, "
                            f"active={n_active})"
                        )
                    else:
                        # 従来の凍結モード（Type A/B/E向け）
                        _relax_active = True
                        _freeze_active = True
                        _freeze_f_c = f_c.copy()
                        _freeze_n_active_prev = n_active
                        _freeze_cycle += 1
                        _effective_max = att + cfg.chattering_freeze_nr_max + 5
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"低残差チャタリング検知[{_chatter_type}] → 接触凍結モード "
                                f"(att={att}, ||R||/||f||={_cur_ratio:.3e}, "
                                f"R_c={_contact_res:.2e}, R_s={_structural_res:.2e}, "
                                f"cycle={_freeze_cycle}/{cfg.chattering_freeze_max_cycles}, "
                                f"active={n_active})"
                            )

            # --- 高残差ストール検知（従来） ---
            if att > 0 and _cur_ratio > 0.5:
                _ratio_change = abs(_cur_ratio - _prev_res_ratio) / max(_prev_res_ratio, 1e-30)
                if _ratio_change < 0.05:
                    _consecutive_stall += 1
                else:
                    _consecutive_stall = max(0, _consecutive_stall - 1)
                if _consecutive_stall >= cfg.stall_window and not _relax_active:
                    _relax_active = True
                    _stall_type = "チャタリング" if _active_changed else "残差停滞"
                    # チャタリングタイプ分類（status-287）
                    _chatter_type = (
                        classify_chattering_type(_nr_snap) if diag.nr_iteration_snapshots else "-"
                    )
                    # status-284: 接触凍結モード（最優先）
                    if (
                        cfg.chattering_freeze_enabled
                        and _active_changed
                        and _freeze_cycle < cfg.chattering_freeze_max_cycles
                    ):
                        _freeze_active = True
                        _freeze_f_c = f_c.copy()
                        _freeze_n_active_prev = n_active
                        _freeze_cycle += 1
                        _effective_max = att + cfg.chattering_freeze_nr_max + 5
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"接触{_stall_type}検知[{_chatter_type}] → 接触凍結モード "
                                f"(cycle={_freeze_cycle}/{cfg.chattering_freeze_max_cycles}, "
                                f"R_c={_contact_res:.2e}, R_s={_structural_res:.2e}, "
                                f"active={n_active})"
                            )
                    # status-268: delta_hブースト（凍結モード無効時のフォールバック）
                    elif cfg.chattering_delta_h_boost > 1.0 and hasattr(
                        _contact_force_strategy, "set_delta_h_boost"
                    ):
                        _delta_h_boosted = True
                        _contact_force_strategy.set_delta_h_boost(cfg.chattering_delta_h_boost)
                        _effective_max = cfg.max_attempts + cfg.chattering_extra_attempts
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"接触{_stall_type}検知[{_chatter_type}] → delta_hブースト "
                                f"(×{cfg.chattering_delta_h_boost}, "
                                f"max_att={_effective_max})"
                            )
                    elif cfg.show_progress:
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"接触{_stall_type}検知[{_chatter_type}] → リラクゼーション有効化 "
                            f"(ω={cfg.contact_relax_omega}, "
                            f"R_c={_contact_res:.2e}, R_s={_structural_res:.2e})"
                        )
            # リラクゼーション早期打切り（status-248: 無駄な反復を削減）
            # status-268: delta_hブースト時・凍結モード時は早期打切りをバイパス。
            if (
                _relax_active
                and _relax_iter >= cfg.relax_max_iter
                and not _delta_h_boosted
                and not _freeze_active
            ):
                _diverged = True  # status-277: 積極的dt縮小で小dt回復を促進
                _failure_reason = "relax_fail"
                if cfg.show_progress:
                    print(
                        f"  Incr {increment_display} (frac={load_frac:.4f}), "
                        f"リラクゼーション {_relax_iter} 反復で未収束 → early abort"
                    )
                break

            _prev_n_active = n_active
            _prev_res_ratio = _cur_ratio

            _diverge_detected = False
            _near_floor = _cur_ratio < 0.01
            if _consecutive_increase >= cfg.divergence_window and not _near_floor:
                _diverge_detected = True
                _reason = f"残差 {cfg.divergence_window} 回連続増加"
            elif att >= 10 and _cur_ratio > 100.0:
                _diverge_detected = True
                _reason = f"残差爆発 (||R||/||f|| = {_cur_ratio:.1e} > 100)"
            if _diverge_detected:
                # status-269: 残差最小値リストア（過修正防止）
                # 発散検知時に、NR反復中に到達した最小残差の状態にリストアして
                # インクリメント成功とする。条件:
                #  - nr_min_restore 有効
                #  - 最小残差が十分小さい（< nr_min_restore_threshold）
                #  - 最小値到達後に nr_min_restore_window 回以上増加
                #    （一時的な振動ではなく本当の過修正であることを確認）
                _can_restore = (
                    cfg.nr_min_restore
                    and _min_res_u is not None
                    and _min_res_ratio < 0.1
                    and _min_res_increase_count >= cfg.nr_min_restore_window
                )
                if _can_restore:
                    u[:] = _min_res_u
                    f_c = _min_res_f_c
                    step_converged = True
                    _diverged = False
                    if cfg.show_progress:
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"発散検知 ({_reason}) → "
                            f"最小残差リストア (att={_min_res_att}, "
                            f"||R||/||f||={_min_res_ratio:.3e})"
                        )
                    break
                _diverged = True
                _failure_reason = "diverged"
                if cfg.show_progress:
                    print(
                        f"  Incr {increment_display} (frac={load_frac:.4f}), "
                        f"発散検知 ({_reason}) → early abort"
                    )
                break

            # ── NR反復ごとのType分類ログ（status-288 + status-290 拡張） ──
            _iter_type = classify_chattering_type(_nr_snap, detailed_d=True)
            # status-307: 5反復ごと + 残差急変動時（前回比10倍以上増加）に出力
            _spike_detected = (
                att >= 1 and _prev_res_ratio > 1e-30 and _cur_ratio > _prev_res_ratio * 10.0
            )
            if cfg.show_progress and (att % 5 == 0 or _spike_detected):
                _wn_info = (
                    f", ||R_w||/||f||={conv_out.res_weighted_norm / conv_out.f_ref:.3e}"
                    if cfg.char_length > 0
                    else ""
                )
                _rate_info = f", rate={_conv_rate:.3f}" if att >= 2 else ""
                _type_info = f" [{_iter_type}]" if _iter_type != "-" else ""
                if _spike_detected:
                    _type_info += " [SPIKE]"
                # comp別残差（status-290: x,y,z成分のみ簡潔に出力）
                _comp_info = ""
                if _comp_norms and len(_comp_norms) >= 3:
                    _comp_total = max(sum(c * c for c in _comp_norms) ** 0.5, 1e-30)
                    _comp_pct = [100.0 * c / _comp_total for c in _comp_norms[:3]]
                    _comp_info = (
                        f", R(x={_comp_norms[0]:.1e}/{_comp_pct[0]:.0f}%"
                        f",y={_comp_norms[1]:.1e}/{_comp_pct[1]:.0f}%"
                        f",z={_comp_norms[2]:.1e}/{_comp_pct[2]:.0f}%)"
                    )
                print(
                    f"  Incr {increment_display} (frac={load_frac:.4f}), "
                    f"attempt {att}{_type_info}, "
                    f"||R_t||/||f|| = {conv_out.res_trans_norm / conv_out.f_ref:.3e}, "
                    f"||R_r|| = {conv_out.res_rot_norm:.3e}"
                    f"{_wn_info}{_rate_info}, "
                    f"active={n_active}"
                    f"{_comp_info}"
                )

            # ── ステップ 7: 接線剛性組立 ──
            tangent_out = _tangent_proc.process(
                TangentAssemblyInput(
                    u=u,
                    manager=manager,
                    contact_force_strategy=_contact_force_strategy,
                    friction_strategy=_friction_strategy,
                    coating_strategy=_coating_strategy,
                    assemble_tangent=input_data.assemble_tangent,
                    k_pen=k_pen,
                    mu=mu,
                    ndof=ndof,
                    coords_def=coords_def,
                    load_frac=load_frac,
                    load_frac_prev=load_frac_prev,
                    use_coating=input_data.use_coating,
                    # status-277: チャタリング時は接線スケーリングで安定化。
                    # status-260でfrac=0.59達成の要因（リラクゼーションとの整合）。
                    contact_tangent_scale=_current_omega if _relax_active else 1.0,
                )
            )
            K_T = tangent_out.K_T

            # 動的: 質量・減衰を接線剛性に加算
            if dt_sub > 1e-30:
                K_T = _time_strategy.effective_stiffness(K_T, dt_sub)

            # ── 条件数診断（オプション） ──
            if cfg.compute_condition_number:
                try:
                    _K_bc = K_T.tocsc().copy()
                    _free = np.setdiff1d(np.arange(_K_bc.shape[0]), input_data.fixed_dofs)
                    _K_free = _K_bc[np.ix_(_free, _free)].toarray()
                    _eigs = np.linalg.eigvalsh(_K_free)
                    _eig_min = float(np.min(np.abs(_eigs)))
                    _eig_max = float(np.max(np.abs(_eigs)))
                    _cond = _eig_max / max(_eig_min, 1e-30)
                    _n_negative = int(np.sum(_eigs < 0))
                    diag.condition_number_history.append(_cond)
                    diag.min_eigenvalue_history.append(float(np.min(_eigs)))
                    diag.max_eigenvalue_history.append(_eig_max)
                    if cfg.show_progress and att % 5 == 0:
                        print(
                            f"    [spectral] cond={_cond:.2e}, "
                            f"λ_min={float(np.min(_eigs)):.2e}, "
                            f"λ_max={_eig_max:.2e}, "
                            f"n_neg={_n_negative}"
                        )
                except Exception as _e:
                    if cfg.show_progress:
                        print(f"    [spectral] 計算失敗: {_e}")

            # ── ステップ 8: 線形ソルブ（LM 正則化付き） ──
            solve_out = _solve_proc.process(
                LinearSolveInput(
                    K_T=K_T,
                    R_u=R_u,
                    fixed_dofs=input_data.fixed_dofs,
                    mpc_transform=input_data.mpc_transform,
                )
            )
            if not solve_out.success:
                _failure_reason = "solve_fail"
                if cfg.show_progress:
                    print(f"  WARNING: Linear solve failed at attempt {att}")
                break

            du = solve_out.du

            # ── FD接線診断（status-288拡張: Type D連続検知でも自動トリガー） ──
            # トリガー条件:
            # (1) tangent_fd_diagnostic=True + ストール検知（従来）
            # (2) type_d_auto_fd=True + 連続Type D >= threshold（新規）
            _fd_trigger_stall = cfg.tangent_fd_diagnostic and _relax_active and _relax_iter == 0
            _fd_trigger_type_d = (
                cfg.type_d_auto_fd
                and _consecutive_type_d >= cfg.type_d_consecutive_threshold
                and not _type_d_fd_triggered
                and not _nr_snap.active_set_changed
                and att >= 5
            )
            if _fd_trigger_stall or _fd_trigger_type_d:
                _fd_diag_proc = TangentFDDiagnosticProcess()
                _mpc_T = None
                if input_data.mpc_transform is not None:
                    _mpc_T = input_data.mpc_transform.T

                # compute_residual クロージャ: 任意の u_eval で残差 R(u_eval) を再計算
                # FD方向検証に必要（status-257）
                def _compute_residual_at(u_eval: np.ndarray) -> np.ndarray:
                    _fr_out = _force_proc.process(
                        ContactForceAssemblyInput(
                            u=u_eval,
                            f_ext=f_ext,
                            fixed_dofs=input_data.fixed_dofs,
                            manager=manager,
                            node_coords_ref=input_data.node_coords_ref,
                            contact_force_strategy=_contact_force_strategy,
                            friction_strategy=_friction_strategy,
                            coating_strategy=_coating_strategy,
                            k_pen=k_pen,
                            mu=mu,
                            u_ref=u_ref,
                            load_frac=load_frac,
                            load_frac_prev=load_frac_prev,
                            increment_display=increment_display,
                            ndof_per_node=cfg.ndof_per_node,
                            use_coating=input_data.use_coating,
                            assemble_internal_force=input_data.assemble_internal_force,
                            connectivity=input_data.connectivity,
                        )
                    )
                    _R_eval = _fr_out.R_u
                    # 動的項を加算（慣性力・減衰力）
                    if dt_sub > 1e-30:
                        _R_eval = _time_strategy.effective_residual(_R_eval, dt_sub)
                        _R_eval[input_data.fixed_dofs] = 0.0
                    return _R_eval

                # gap<0 ペアの関連DOFを収集（status-261）
                _active_dofs_set: set[int] = set()
                if hasattr(manager, "pairs"):
                    for _pair in manager.pairs:
                        if hasattr(_pair, "state") and _pair.state.gap < 0.0:
                            _active_dofs_set.update(
                                _contact_dofs(_pair, cfg.ndof_per_node).tolist()
                            )
                _active_dofs_arr = (
                    np.array(sorted(_active_dofs_set), dtype=int) if _active_dofs_set else None
                )

                # compute_contact_force クロージャ: f_c のみを返す（status-290）
                def _compute_fc_at(u_eval: np.ndarray) -> np.ndarray:
                    _fc_out = _force_proc.process(
                        ContactForceAssemblyInput(
                            u=u_eval,
                            f_ext=f_ext,
                            fixed_dofs=input_data.fixed_dofs,
                            manager=manager,
                            node_coords_ref=input_data.node_coords_ref,
                            contact_force_strategy=_contact_force_strategy,
                            friction_strategy=_friction_strategy,
                            coating_strategy=_coating_strategy,
                            k_pen=k_pen,
                            mu=mu,
                            u_ref=u_ref,
                            load_frac=load_frac,
                            load_frac_prev=load_frac_prev,
                            increment_display=increment_display,
                            ndof_per_node=cfg.ndof_per_node,
                            use_coating=input_data.use_coating,
                            assemble_internal_force=input_data.assemble_internal_force,
                            connectivity=input_data.connectivity,
                        )
                    )
                    return _fc_out.f_c

                # K_c 取得: tangent_out から接触接線のみ抽出
                _K_c = tangent_out.K_T - input_data.assemble_tangent(u)
                # 動的項はK_Tに含まれるがK_cには不要 → K_T - K_struct で近似
                # （K_frictionやK_coatingも含まれるが主要成分はK_c）

                _fd_out = _fd_diag_proc.process(
                    TangentFDDiagnosticInput(
                        u=u,
                        du=du,
                        R_u=R_u,
                        K_T=K_T,
                        mpc_transform=_mpc_T,
                        fixed_dofs=input_data.fixed_dofs,
                        compute_residual=_compute_residual_at,
                        active_contact_dofs=_active_dofs_arr,
                        compute_contact_force=_compute_fc_at,
                        K_c=_K_c,
                    )
                )
                if cfg.show_progress:
                    print(_fd_out.report)

                # ── K_c 成分分解 FD 診断（status-343/344: 仮説 A 最終検証） ──
                # K_c を K_mat / K_geo / K_st に分解し、4 組み合わせで FD と突合。
                # x 成分 68% 不整合の由来を部分行列レベルで特定。
                if cfg.kc_component_fd_diagnostic and hasattr(
                    _contact_force_strategy, "tangent_components"
                ):
                    try:
                        _K_mat_c, _K_geo_c, _K_st_c = _contact_force_strategy.tangent_components(
                            u,
                            manager,
                            k_pen,
                            node_coords=input_data.node_coords_ref,
                        )
                        _kc_comp_proc = ContactKcComponentFDDiagnosticProcess()
                        _kc_out = _kc_comp_proc.process(
                            ContactKcComponentFDDiagnosticInput(
                                u=u,
                                du=du,
                                compute_contact_force=_compute_fc_at,
                                K_mat=_K_mat_c,
                                K_geo=_K_geo_c,
                                K_st=_K_st_c,
                                eps=1e-7,
                                label=f"incr={increment_display} att={att}",
                            )
                        )
                        if cfg.show_progress:
                            print("[K_c成分FD]")
                            print(_kc_out.report)
                    except Exception as _kc_err:  # noqa: BLE001
                        if cfg.show_progress:
                            print(f"[K_c成分FD] skipped: {_kc_err}")

                # Type D トリガー時の追加応答（status-288）
                if _fd_trigger_type_d:
                    _type_d_fd_triggered = True
                    _type_d_fd_result = (
                        f"att={att}, dir_ratio={_fd_out.directional_ratio:.3f}, "
                        f"deriv_agree={_fd_out.deriv_agreement:.3e}"
                    )
                    diag.type_d_fd_reports.append(_type_d_fd_result)
                    # FD方向比 > 1.0: Newton方向が残差を増加させている
                    # → NR反復上限を拡張して線形収束を許容
                    if _fd_out.directional_ratio > 0.95:
                        _effective_max = max(_effective_max, att + cfg.type_d_extra_attempts)
                        print(
                            f"  [Type D対策] FD診断: Newton方向精度不良 "
                            f"(dir_ratio={_fd_out.directional_ratio:.3f}), "
                            f"NR上限拡張→{_effective_max}"
                        )
                    else:
                        # Newton方向は有効だが収束率が悪い
                        # → 線形収束を許容しNR上限を拡張
                        _effective_max = max(_effective_max, att + cfg.type_d_extra_attempts)
                        print(
                            f"  [Type D対策] FD診断: 接線方向は有効 "
                            f"(dir_ratio={_fd_out.directional_ratio:.3f}), "
                            f"線形収束許容→NR上限{_effective_max}"
                        )

            # ── DOF スケーリング: 回転 DOF の更新を減衰（status-241） ──
            _sr = cfg.dof_scale_rot
            if _sr != 1.0 and cfg.ndof_per_node >= 6:
                _n_nd = len(du) // cfg.ndof_per_node
                for _ni in range(_n_nd):
                    _base = _ni * cfg.ndof_per_node
                    du[_base + 3 : _base + 6] *= _sr

            # ── ステップ 9: Line search + 更新 ──
            ls_out = _linesearch_proc.process(
                LineSearchUpdateInput(
                    u=u,
                    du=du,
                    f_ext=f_ext,
                    fixed_dofs=input_data.fixed_dofs,
                    assemble_internal_force=input_data.assemble_internal_force,
                    res_u_norm=conv_out.res_u_norm,
                    f_c=f_c,
                    use_line_search=cfg.use_line_search,
                    line_search_max_steps=cfg.line_search_max_steps,
                    du_norm_cap=cfg.du_norm_cap,
                )
            )
            du = ls_out.du_scaled

            # ── ステップ 9b: 接触 backtracking line search（status-362: 仮説 C (c)） ──
            # 既存 line search 後の du に対し、接触残差・active ペア数の過剰増加を
            # 再評価し、mixed (C+D) 領域の暴走 step を抑制する。
            # 発動時はペア状態をスナップショットし、復元後に本採択 du を適用する
            # （次 NR 反復冒頭の force_proc 呼び出しで manager 状態は再構築される）。
            #
            # トリガー条件（mixed (C+D) の狭義検知）:
            #   (1) att >= 2: att=0/1 は収束率の履歴が短いため除外
            #   (2) n_active >= 1: 接触未活性化の初期侵入 step は除外
            #   (3) active_set_changed: 今回の NR 反復で active 集合が変化
            #   (4) _conv_rate > rate_threshold: 収束率が悪い（Type D 気味）
            if cfg.contact_backtracking_enabled and not _freeze_active:
                _bt_trigger = True
                if cfg.contact_backtracking_mixed_only:
                    _bt_trigger = (
                        att >= 2
                        and n_active >= 1
                        and _nr_snap.active_set_changed
                        and _conv_rate > cfg.contact_backtracking_rate_threshold
                    )
                if _bt_trigger and hasattr(manager, "pairs"):
                    _pairs_snapshot = list(manager.pairs)

                    def _compute_contact_metrics(
                        u_eval: np.ndarray,
                    ) -> tuple[float, int]:
                        _trial_fr = _force_proc.process(
                            ContactForceAssemblyInput(
                                u=u_eval,
                                f_ext=f_ext,
                                fixed_dofs=input_data.fixed_dofs,
                                manager=manager,
                                node_coords_ref=input_data.node_coords_ref,
                                contact_force_strategy=_contact_force_strategy,
                                friction_strategy=_friction_strategy,
                                coating_strategy=_coating_strategy,
                                k_pen=k_pen,
                                mu=mu,
                                u_ref=u_ref,
                                load_frac=load_frac,
                                load_frac_prev=load_frac_prev,
                                increment_display=increment_display,
                                ndof_per_node=cfg.ndof_per_node,
                                use_coating=input_data.use_coating,
                                assemble_internal_force=input_data.assemble_internal_force,
                                connectivity=input_data.connectivity,
                            )
                        )
                        _R_trial = _trial_fr.R_u
                        if dt_sub > 1e-30:
                            _R_trial = _time_strategy.effective_residual(_R_trial, dt_sub)
                            _R_trial[input_data.fixed_dofs] = 0.0
                        _c_mask_tr = np.zeros(len(_R_trial), dtype=bool)
                        _n_act_tr = 0
                        for _p_tr in manager.pairs:
                            if hasattr(_p_tr, "state") and _p_tr.state.p_n > 0.0:
                                _n_act_tr += 1
                                _cd_tr = _contact_dofs(_p_tr, cfg.ndof_per_node)
                                _vd_tr = _cd_tr[_cd_tr < len(_R_trial)]
                                _c_mask_tr[_vd_tr] = True
                        _cres_tr = (
                            float(np.linalg.norm(_R_trial[_c_mask_tr])) if _c_mask_tr.any() else 0.0
                        )
                        return _cres_tr, _n_act_tr

                    _bt_out = _bt_proc.process(
                        ContactBacktrackingLineSearchInput(
                            u=u,
                            du=du,
                            n_active_pre=n_active,
                            contact_res_pre=_contact_res,
                            compute_trial=_compute_contact_metrics,
                            max_steps=cfg.contact_backtracking_max_steps,
                            active_flip_threshold=cfg.contact_backtracking_active_flip_threshold,
                            active_flip_ratio=cfg.contact_backtracking_active_flip_ratio,
                            residual_ratio_threshold=cfg.contact_backtracking_residual_ratio,
                            alpha_decay=cfg.contact_backtracking_alpha_decay,
                            min_alpha=cfg.contact_backtracking_min_alpha,
                        )
                    )
                    du = _bt_out.du_accepted
                    manager.pairs[:] = _pairs_snapshot
                    if cfg.show_progress and _bt_out.alpha < 1.0:
                        print(
                            f"  [BT:{_bt_out.reason}] Incr {increment_display} "
                            f"att={att}, α={_bt_out.alpha:.3f}, "
                            f"n_iter={_bt_out.n_bt_iter}, n_active_pre={n_active}"
                        )
            u += du

            # MPC制約をNR更新後のuに再射影（slave DOFの整合性維持）
            _mpc_nr = input_data.mpc_transform
            if _mpc_nr is not None:
                _u_red_nr = u[_mpc_nr.independent_dofs]
                _u_proj_nr = _mpc_nr.T @ _u_red_nr
                if hasattr(_u_proj_nr, "toarray"):
                    _u_proj_nr = _u_proj_nr.toarray().ravel()
                u[:] = np.asarray(_u_proj_nr).ravel()

            # ── NR リミットサイクル検知 + 平均化収束（status-278, status-279拡張） ──
            # NRがリミットサイクル（周期2〜6の振動）に入った場合、
            # 直近N個の平均状態を採用して収束とする。
            # 条件A（早期, status-278）: att>=4, 2サイクル, ||f_c||/||R||<5%
            # 条件B（後期, status-279）: att>=15, N-サイクル, ||R_t||/||f||<2.0
            _cycle_converged = False
            _tol_cycle = 1e-6  # リミットサイクルの判定閾値
            _contact_filter_threshold = 0.05  # 接触力が残差の5%以下なら微小
            _late_cycle_att = 15  # 後期判定の最小反復数
            _late_cycle_res_max = 3.0  # 後期判定の残差上限

            # u履歴を更新（最大_limit_cycle_window個保持）
            _u_history_ring.append(u.copy())
            if len(_u_history_ring) > _limit_cycle_window:
                _u_history_ring.pop(0)

            if att >= 4 and len(_u_history_ring) >= 3:
                _u_norm_ref = max(float(np.linalg.norm(u)), 1e-30)
                _fc_norm = float(np.linalg.norm(f_c))
                _R_norm = float(np.linalg.norm(R_u))
                _fc_ratio = _fc_norm / max(_R_norm, 1e-30)

                # 周期2〜を検査: u[att] ≈ u[att-period] は
                # _u_history_ring[-1] ≈ _u_history_ring[-(period+1)]
                _detected_period = 0
                for _period in range(2, len(_u_history_ring)):
                    _idx = -(_period + 1)
                    if abs(_idx) > len(_u_history_ring):
                        break
                    _cycle_diff = float(np.linalg.norm(_u_history_ring[-1] - _u_history_ring[_idx]))
                    if _cycle_diff / _u_norm_ref < _tol_cycle:
                        _detected_period = _period
                        break

                if _detected_period > 0:
                    # 条件A: 微小接触力（周期2のみ、status-278互換）
                    # status-279検証: 周期3以上を許可すると不正確な状態が蓄積し悪化
                    _early_ok = _detected_period == 2 and _fc_ratio < _contact_filter_threshold
                    # 条件B: 後期リミットサイクル（status-279で検証→逆効果で無効化）
                    # 残差の大きい状態を収束判定すると次インクリメントで発散する
                    _late_ok = False

                    if _early_ok or _late_ok:
                        # 直近_detected_period個の平均状態を採用
                        _avg_u = np.mean(_u_history_ring[-_detected_period:], axis=0)
                        u[:] = _avg_u
                        _mpc_avg = input_data.mpc_transform
                        if _mpc_avg is not None:
                            _u_red_avg = u[_mpc_avg.independent_dofs]
                            _u_proj_avg = _mpc_avg.T @ _u_red_avg
                            if hasattr(_u_proj_avg, "toarray"):
                                _u_proj_avg = _u_proj_avg.toarray().ravel()
                            u[:] = np.asarray(_u_proj_avg).ravel()
                        _cycle_converged = True
                        _reason_tag = (
                            "微小接触" if _early_ok else f"後期リミットサイクル(R={_cur_ratio:.2f})"
                        )
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"{_reason_tag}{_detected_period}サイクル検知 "
                                f"(||f_c||/||R||={_fc_ratio:.2e}) → 平均化収束"
                            )

            # 旧2サイクル用変数も更新（後方互換）
            _u_prev2 = _u_prev1
            _u_prev1 = u.copy()

            if _cycle_converged:
                step_converged = True
                break

            # ── 変位・エネルギー収束判定 ──
            _eff_ref2 = _incr_f_ref if _incr_f_ref > 1e-30 else input_data.f_ext_ref_norm
            conv_out2 = _conv_proc.process(
                ConvergenceCheckInput(
                    R_u=R_u,
                    du=du,
                    u=u,
                    f_ext_ref_norm=_eff_ref2,
                    tol_force=cfg.tol_force,
                    tol_disp=cfg.tol_disp,
                    dynamic_ref=input_data.dynamic_ref,
                    is_first_attempt=False,
                    energy_ref=energy_ref,
                    manager=manager,
                    ndof_per_node=cfg.ndof_per_node,
                    char_length=cfg.char_length,
                    mpc_transform=input_data.mpc_transform,
                    atol_force=input_data.atol_force,
                )
            )

            du_norm_val = conv_out2.du_norm
            diag.du_norm_history.append(du_norm_val)
            diag.max_du_dof_history.append(int(np.argmax(np.abs(du))) if du_norm_val > 0 else -1)
            energy_ref = conv_out2.energy_ref

            if conv_out2.converged:
                # 凍結モード中の変位収束→再評価（力収束と同じロジック）
                if _freeze_active and _freeze_f_c is not None:
                    _reeval_out2 = _force_proc.process(
                        ContactForceAssemblyInput(
                            u=u,
                            f_ext=f_ext,
                            fixed_dofs=input_data.fixed_dofs,
                            manager=manager,
                            node_coords_ref=input_data.node_coords_ref,
                            contact_force_strategy=_contact_force_strategy,
                            friction_strategy=_friction_strategy,
                            coating_strategy=_coating_strategy,
                            k_pen=k_pen,
                            mu=mu,
                            u_ref=u_ref,
                            load_frac=load_frac,
                            load_frac_prev=load_frac_prev,
                            increment_display=increment_display,
                            ndof_per_node=cfg.ndof_per_node,
                            use_coating=input_data.use_coating,
                            assemble_internal_force=input_data.assemble_internal_force,
                            connectivity=input_data.connectivity,
                        )
                    )
                    _re_R2 = _reeval_out2.R_u
                    if dt_sub > 1e-30:
                        _re_R2 = _time_strategy.effective_residual(_re_R2, dt_sub)
                        _re_R2[input_data.fixed_dofs] = 0.0
                    _re_trans2 = float(
                        np.linalg.norm(_re_R2.reshape(-1, cfg.ndof_per_node)[:, :3].ravel())
                    )
                    _re_ratio2 = _re_trans2 / max(conv_out.f_ref, 1e-30)
                    _freeze_tol2 = cfg.tol_force * cfg.chattering_freeze_tol_factor

                    if _re_ratio2 < _freeze_tol2:
                        _freeze_active = False
                        _current_omega = 1.0
                        f_c = _reeval_out2.f_c
                        step_converged = True
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"凍結解除(disp)→再評価成功 "
                                f"||R_t||/||f||={_re_ratio2:.3e}"
                            )
                        break
                    elif _freeze_cycle < cfg.chattering_freeze_max_cycles:
                        _freeze_f_c = _reeval_out2.f_c.copy()
                        _freeze_cycle += 1
                        _consecutive_stall = 0
                        _relax_active = True
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"凍結解除(disp)→再凍結 "
                                f"||R_t||/||f||={_re_ratio2:.3e} "
                                f"(cycle={_freeze_cycle})"
                            )
                        continue
                    else:
                        _freeze_active = False
                        _current_omega = 1.0
                        # サイクル上限→緩和判定
                        if _re_ratio2 < _freeze_tol2 * 10.0:
                            f_c = _reeval_out2.f_c
                            step_converged = True
                            if cfg.show_progress:
                                print(
                                    f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                    f"attempt {att}, "
                                    f"凍結上限→緩和収束(disp) "
                                    f"||R_t||/||f||={_re_ratio2:.3e}"
                                )
                            break
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"凍結上限超過(disp)→通常NR復帰 "
                                f"||R_t||/||f||={_re_ratio2:.3e}"
                            )
                        continue
                step_converged = True
                ctype = conv_out2.convergence_type
                if ctype == ConvergenceType.DISPLACEMENT:
                    _convergence_type = "disp"
                else:
                    _convergence_type = "energy"
                if cfg.show_progress:
                    if ctype == ConvergenceType.DISPLACEMENT:
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"attempt {att}, "
                            f"||du||/||u|| = {du_norm_val / max(float(np.linalg.norm(u)), 1e-30):.3e} "
                            f"(disp converged, {n_active} active)"
                        )
                    else:
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"attempt {att}, "
                            f"energy = {conv_out2.energy:.3e} (energy converged)"
                        )
                break

        # NR上限到達で不収束の場合
        if not step_converged and not _failure_reason:
            _failure_reason = "nr_limit"

        # status-268: delta_hブーストを解除（次インクリメントに影響させない）
        if _delta_h_boosted and hasattr(_contact_force_strategy, "set_delta_h_boost"):
            _contact_force_strategy.set_delta_h_boost(1.0)

        # ── チャタリング内訳サマリ出力（status-287 + status-288 構造化） ──
        # 全インクリメント（収束・不収束問わず）でNRスナップショットからType分布を記録。
        # show_progressとは独立に常にprint出力し、ログファイルに確実に残す。
        if diag.nr_iteration_snapshots:
            _all_snaps = diag.nr_iteration_snapshots
            _n_total_snaps = len(_all_snaps)
            # Type Dサブタイプ付き分布（status-290）
            _type_counts_d: dict[str, int] = {}
            for _snap in _all_snaps:
                _t = classify_chattering_type(_snap, detailed_d=True)
                _type_counts_d[_t] = _type_counts_d.get(_t, 0) + 1
            _type_str_full = ", ".join(
                f"{k}:{v}({100.0 * v / _n_total_snaps:.0f}%)"
                for k, v in sorted(_type_counts_d.items())
            )
            # 直近10反復の分布（サブタイプ付き）
            _recent10 = _all_snaps[-10:]
            _rc10: dict[str, int] = {}
            for _snap in _recent10:
                _t = classify_chattering_type(_snap, detailed_d=True)
                _rc10[_t] = _rc10.get(_t, 0) + 1
            _type_str_recent = ", ".join(f"{k}:{v}" for k, v in sorted(_rc10.items()))
            _last = _all_snaps[-1]
            _status_tag = "収束" if step_converged else "不収束"
            # comp別残差サマリ（最終反復）
            _comp_summary = ""
            if _last.comp_res_norms and len(_last.comp_res_norms) >= 3:
                _cnames = ["x", "y", "z", "θx", "θy", "θz"]
                _cn = _last.comp_res_norms
                _ct = max(sum(c * c for c in _cn) ** 0.5, 1e-30)
                _comp_parts = [
                    f"{_cnames[i]}={_cn[i]:.1e}/{100 * _cn[i] / _ct:.0f}%"
                    for i in range(min(len(_cn), 6))
                ]
                _comp_summary = f", comp({','.join(_comp_parts)})"
            # 不収束時は常に出力、収束時は多反復(>15)の場合のみ
            if not step_converged or total_attempts > 15:
                print(
                    f"  [NR診断] Incr {increment_display} (frac={load_frac:.4f}), "
                    f"{_status_tag} att={total_attempts}, "
                    f"Type分布[{_type_str_full}], "
                    f"直近10[{_type_str_recent}], "
                    f"R_c={_last.contact_res_norm:.2e}, R_s={_last.structural_res_norm:.2e}, "
                    f"active={_last.n_active}, sliding={_last.n_sliding}, "
                    f"activated={_last.pairs_activated}, deactivated={_last.pairs_deactivated}, "
                    f"stick→slide={_last.pairs_stick_to_slide}, "
                    f"slide→stick={_last.pairs_slide_to_stick}"
                    f"{_comp_summary}"
                )

        return DynamicStepOutput(
            converged=step_converged,
            n_attempts=total_attempts,
            n_active=n_active,
            f_c=f_c,
            diagnostics=diag,
            diverged=_diverged,
            f_ref_used=_incr_f_ref,
            convergence_type=_convergence_type,
            failure_reason=_failure_reason,
        )


# 後方互換エイリアス
NewtonUzawaDynamicProcess = NewtonDynamicProcess
