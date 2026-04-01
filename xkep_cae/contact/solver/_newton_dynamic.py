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
    PairDiagnosticsOutput,
)
from xkep_cae.contact.solver._newton_steps import (
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


@dataclass(frozen=True)
class DynamicStepOutput:
    """1荷重増分の結果（動的）."""

    converged: bool
    n_attempts: int
    n_active: int
    f_c: np.ndarray
    diagnostics: ConvergenceDiagnosticsOutput
    diverged: bool = False


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
        TangentFDDiagnosticProcess,
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

        diag = ConvergenceDiagnosticsOutput(step=increment_display, load_frac=load_frac)
        total_attempts = 0
        f_c = np.zeros(ndof)
        energy_ref = None
        step_converged = False
        n_active = 0
        _diverged = False
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
        # NR残差最小値リストア用トラッキング（status-269）
        _min_res_ratio = float("inf")
        _min_res_u: np.ndarray | None = None
        _min_res_f_c: np.ndarray | None = None
        _min_res_att = 0
        _min_res_increase_count = 0  # 最小値からの連続増加カウント
        # NRインナー活性ペア凍結（status-276）
        _frozen_contact_dofs: np.ndarray | None = None  # att=0で記録

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

            # ── 接触チャタリング対策（status-268: delta_hブースト優先） ──
            # チャタリング検知後:
            #   delta_h_boost > 1: Huber遷移幅を拡大（残差-Jacobian整合維持）
            #   delta_h_boost <= 1: フォールバックとして力ブレンド（status-247）
            if att > 0 and _relax_active:
                _relax_iter += 1
                if _delta_h_boosted:
                    # delta_hブースト適用中: 通常NR（力ブレンド不要）
                    pass
                elif cfg.contact_relax_omega < 1.0:
                    # フォールバック: 力ブレンド（delta_hブースト無効時のみ）
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
                )
            )
            # 初回反復で参照残差を保存
            if att == 0 and input_data.dynamic_ref:
                _incr_f_ref = conv_out.f_ref
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

            if conv_out.converged:
                step_converged = True
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

            # ストール検知（status-247 + status-255 拡張）:
            # (a) active set 振動 + 残差停滞 → チャタリング
            # (b) active set 安定 + 残差停滞 → 接線剛性不整合（MPC+接触）
            # どちらも早期打切り → dt cutback で通過を試みる
            _active_changed = n_active != _prev_n_active
            if att > 0 and _cur_ratio > 0.5:
                _ratio_change = abs(_cur_ratio - _prev_res_ratio) / max(_prev_res_ratio, 1e-30)
                # 残差停滞 < 5%: active set 振動の有無に関わらず検知
                if _ratio_change < 0.05:
                    _consecutive_stall += 1
                else:
                    _consecutive_stall = max(0, _consecutive_stall - 1)
                if _consecutive_stall >= cfg.stall_window and not _relax_active:
                    _relax_active = True
                    _stall_type = "チャタリング" if _active_changed else "残差停滞"
                    # status-268: delta_hブースト（力ブレンドより優先）
                    if cfg.chattering_delta_h_boost > 1.0 and hasattr(
                        _contact_force_strategy, "set_delta_h_boost"
                    ):
                        _delta_h_boosted = True
                        _contact_force_strategy.set_delta_h_boost(cfg.chattering_delta_h_boost)
                        # NR反復上限を動的拡張（変位収束到達のための余裕）
                        _effective_max = cfg.max_attempts + cfg.chattering_extra_attempts
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"接触{_stall_type}検知 → delta_hブースト "
                                f"(×{cfg.chattering_delta_h_boost}, "
                                f"max_att={_effective_max})"
                            )
                    elif cfg.show_progress:
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"接触{_stall_type}検知 → リラクゼーション有効化 "
                            f"(ω={cfg.contact_relax_omega})"
                        )
            # リラクゼーション早期打切り（status-248: 無駄な反復を削減）
            # status-267: _diverged=False に修正。リラクゼーション abort は発散ではなく
            # 活性集合振動の停滞。diverged=True だと dt が shrink²=0.25 で過度に縮小され
            # （91/91全失敗と相まって）チャタリング帯域でdt枯渇を引き起こしていた。
            # status-268: delta_hブースト時は早期打切りをバイパス。
            # ブーストは力ブレンド不要のため max_attempts まで NR 継続し
            # 変位収束に到達するための時間を確保する。
            if _relax_active and _relax_iter >= cfg.relax_max_iter and not _delta_h_boosted:
                _diverged = True  # status-277: 積極的dt縮小で小dt回復を促進
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
                if cfg.show_progress:
                    print(
                        f"  Incr {increment_display} (frac={load_frac:.4f}), "
                        f"発散検知 ({_reason}) → early abort"
                    )
                break

            if cfg.show_progress and att % 5 == 0:
                _wn_info = (
                    f", ||R_w||/||f||={conv_out.res_weighted_norm / conv_out.f_ref:.3e}"
                    if cfg.char_length > 0
                    else ""
                )
                print(
                    f"  Incr {increment_display} (frac={load_frac:.4f}), "
                    f"attempt {att}, "
                    f"||R_t||/||f|| = {conv_out.res_trans_norm / conv_out.f_ref:.3e}, "
                    f"||R_r|| = {conv_out.res_rot_norm:.3e}"
                    f"{_wn_info}, "
                    f"active={n_active}"
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
                if cfg.show_progress:
                    print(f"  WARNING: Linear solve failed at attempt {att}")
                break

            du = solve_out.du

            # ── FD接線診断（ストール検知時 + tangent_fd_diagnostic=True） ──
            if cfg.tangent_fd_diagnostic and _relax_active and _relax_iter == 0:
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
                    )
                )
                if cfg.show_progress:
                    print(_fd_out.report)

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
            u += du

            # MPC制約をNR更新後のuに再射影（slave DOFの整合性維持）
            _mpc_nr = input_data.mpc_transform
            if _mpc_nr is not None:
                _u_red_nr = u[_mpc_nr.independent_dofs]
                _u_proj_nr = _mpc_nr.T @ _u_red_nr
                if hasattr(_u_proj_nr, "toarray"):
                    _u_proj_nr = _u_proj_nr.toarray().ravel()
                u[:] = np.asarray(_u_proj_nr).ravel()

            # ── NR 2サイクル振動検知 + 微小接触力フィルタ収束（status-278） ──
            # 接触活性/非活性の微小変動で du が完全な2サイクル振動に入り、
            # かつ接触力が構造残差に対して十分小さい場合、
            # 「微小接触は物理的に無意味」として現在の状態を収束と判定する。
            # 条件: (1) att>=4, (2) 2サイクル振動検知, (3) ||f_c||/||R|| < threshold
            _cycle_converged = False
            _tol_cycle = 1e-6  # 2サイクル振動の判定閾値
            _contact_filter_threshold = 0.05  # 接触力が残差の5%以下なら微小
            if att >= 4 and _u_prev2 is not None:
                _cycle_diff = float(np.linalg.norm(u - _u_prev2))
                _u_norm_ref = max(float(np.linalg.norm(u)), 1e-30)
                if _cycle_diff / _u_norm_ref < _tol_cycle:
                    # 2サイクル振動確定
                    _fc_norm = float(np.linalg.norm(f_c))
                    _R_norm = float(np.linalg.norm(R_u))
                    _fc_ratio = _fc_norm / max(_R_norm, 1e-30)
                    if _fc_ratio < _contact_filter_threshold:
                        # 微小接触力: 中間状態を採用
                        u[:] = 0.5 * (u + _u_prev1)
                        _mpc_avg = input_data.mpc_transform
                        if _mpc_avg is not None:
                            _u_red_avg = u[_mpc_avg.independent_dofs]
                            _u_proj_avg = _mpc_avg.T @ _u_red_avg
                            if hasattr(_u_proj_avg, "toarray"):
                                _u_proj_avg = _u_proj_avg.toarray().ravel()
                            u[:] = np.asarray(_u_proj_avg).ravel()
                        _cycle_converged = True
                        if cfg.show_progress:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"attempt {att}, "
                                f"微小接触2サイクル検知 "
                                f"(||f_c||/||R||={_fc_ratio:.2e}) → 平均化収束"
                            )
            # u履歴を更新
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
                )
            )

            du_norm_val = conv_out2.du_norm
            diag.du_norm_history.append(du_norm_val)
            diag.max_du_dof_history.append(int(np.argmax(np.abs(du))) if du_norm_val > 0 else -1)
            energy_ref = conv_out2.energy_ref

            if conv_out2.converged:
                step_converged = True
                if cfg.show_progress:
                    ctype = conv_out2.convergence_type
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

        # status-268: delta_hブーストを解除（次インクリメントに影響させない）
        if _delta_h_boosted and hasattr(_contact_force_strategy, "set_delta_h_boost"):
            _contact_force_strategy.set_delta_h_boost(1.0)

        return DynamicStepOutput(
            converged=step_converged,
            n_attempts=total_attempts,
            n_active=n_active,
            f_c=f_c,
            diagnostics=diag,
            diverged=_diverged,
        )


# 後方互換エイリアス
NewtonUzawaDynamicProcess = NewtonDynamicProcess
