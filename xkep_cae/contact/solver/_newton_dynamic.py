"""Newton-Raphson イテレーション（動的）.

Generalized-α 時間積分による慣性力・減衰力を含む。
内部ステップは全てサブプロセスに委譲。

status-222 で Uzawa ループを削除。純粋 Huber ペナルティ + Coulomb 摩擦。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

        for att in range(cfg.max_attempts):
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

            # ── 接触力リラクゼーション（status-247: NR 2-サイクル対策） ──
            # 残差プラトー + active set 振動を検知し、接触力をブレンドして安定化。
            # 収束時は f_c ≈ f_c_prev なので収束解は不変。
            # omega は漸進的に低下: ω₀ * 0.7^(iter//2)、下限 0.05
            if att > 0 and _relax_active and cfg.contact_relax_omega < 1.0:
                _relax_iter += 1
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
            _cur_ratio = conv_out.res_u_norm / conv_out.f_ref
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
                    if cfg.show_progress:
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"接触{_stall_type}検知 → リラクゼーション有効化 "
                            f"(ω={cfg.contact_relax_omega})"
                        )
            # リラクゼーション早期打切り（status-248: 無駄な反復を削減）
            if _relax_active and _relax_iter >= cfg.relax_max_iter:
                _diverged = True
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
