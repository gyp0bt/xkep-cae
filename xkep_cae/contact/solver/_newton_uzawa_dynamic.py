"""Newton-Raphson + Uzawa イテレーション（動的）.

NewtonUzawaStatic と完全分離された動的版。
Generalized-α 時間積分による慣性力・減衰力を含む。
内部ステップは全てサブプロセスに委譲。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.contact.solver._diagnostics import ConvergenceDiagnosticsOutput
from xkep_cae.contact.solver._nuzawa_steps import (
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
    UzawaUpdateInput,
    UzawaUpdateProcess,
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


@dataclass(frozen=True)
class NewtonUzawaDynamicInput:
    """Newton-Uzawa ループの設定（動的）."""

    max_attempts: int = 50
    tol_force: float = 1e-8
    tol_disp: float = 1e-8
    use_line_search: bool = True
    line_search_max_steps: int = 5
    du_norm_cap: float = 0.0
    show_progress: bool = True
    ndof_per_node: int = 6


@dataclass(frozen=True)
class NewtonUzawaDynamicStepInput:
    """1荷重増分の NR+Uzawa 入力（動的）."""

    config: NewtonUzawaDynamicInput
    u: np.ndarray
    lam_all: np.ndarray
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


class NewtonUzawaDynamicProcess(
    SolverProcess[NewtonUzawaDynamicStepInput, DynamicStepOutput],
):
    """1荷重増分の Newton-Raphson + Uzawa イテレーション（動的）.

    Generalized-α 時間積分による慣性力・減衰力を含む動的版。
    内部ステップは全てサブプロセスに委譲する。
    """

    meta = ProcessMeta(
        name="NewtonUzawaDynamic",
        module="solve",
        version="1.0.0",
        document_path="docs/newton_uzawa.md",
    )
    uses = [
        ContactForceAssemblyProcess,
        ConvergenceCheckProcess,
        TangentAssemblyProcess,
        LinearSolveProcess,
        LineSearchUpdateProcess,
        UzawaUpdateProcess,
    ]

    def process(  # noqa: C901, PLR0912, PLR0915
        self,
        input_data: NewtonUzawaDynamicStepInput,
    ) -> DynamicStepOutput:
        """1荷重増分のNR+Uzawaを実行（動的）.

        input_data.u, input_data.lam_all は in-place で更新される。
        """
        cfg = input_data.config
        u = input_data.u
        lam_all = input_data.lam_all
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

        n_uzawa_max = getattr(_contact_force_strategy, "n_uzawa_max", 5)
        tol_uzawa = getattr(_contact_force_strategy, "tol_uzawa", 1e-6)

        # サブプロセスインスタンス
        _force_proc = ContactForceAssemblyProcess()
        _conv_proc = ConvergenceCheckProcess()
        _tangent_proc = TangentAssemblyProcess()
        _solve_proc = LinearSolveProcess()
        _linesearch_proc = LineSearchUpdateProcess()
        _uzawa_proc = UzawaUpdateProcess()

        diag = ConvergenceDiagnosticsOutput(step=increment_display, load_frac=load_frac)
        total_attempts = 0
        f_c = np.zeros(ndof)
        energy_ref = None
        step_converged = False
        n_active = 0

        for _uzawa_iter in range(n_uzawa_max):
            for att in range(cfg.max_attempts):
                total_attempts += 1

                # ── ステップ 2〜5: 接触力アセンブリ + 残差 ──
                force_out = _force_proc.process(
                    ContactForceAssemblyInput(
                        u=u,
                        lam_all=lam_all,
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
                    )
                )
                f_c = force_out.f_c
                R_u = force_out.R_u

                # 動的: 慣性力・減衰力を残差に加算
                if dt_sub > 1e-30:
                    _time_strategy.correct(u, np.zeros_like(u), dt_sub)
                    R_u = _time_strategy.effective_residual(R_u, dt_sub)
                    R_u[input_data.fixed_dofs] = 0.0

                coords_def = force_out.coords_def

                # ── ステップ 6: 力収束判定 ──
                conv_out = _conv_proc.process(
                    ConvergenceCheckInput(
                        R_u=R_u,
                        du=None,
                        u=u,
                        f_ext_ref_norm=input_data.f_ext_ref_norm,
                        tol_force=cfg.tol_force,
                        tol_disp=cfg.tol_disp,
                        dynamic_ref=input_data.dynamic_ref,
                        is_first_attempt=(att == 0),
                        energy_ref=energy_ref,
                        manager=manager,
                    )
                )
                n_active = conv_out.n_active

                diag.res_history.append(conv_out.res_u_norm / conv_out.f_ref)
                diag.ncp_history.append(0.0)
                diag.ncp_t_history.append(0.0)
                diag.n_active_history.append(n_active)

                if conv_out.converged:
                    step_converged = True
                    if cfg.show_progress:
                        print(
                            f"  Incr {increment_display} (frac={load_frac:.4f}), "
                            f"uzawa {_uzawa_iter}, attempt {att}, "
                            f"||R_u||/||f|| = {conv_out.res_u_norm / conv_out.f_ref:.3e} "
                            f"(converged, {n_active} active)"
                        )
                    break

                if cfg.show_progress and att % 5 == 0:
                    print(
                        f"  Incr {increment_display} (frac={load_frac:.4f}), "
                        f"uzawa {_uzawa_iter}, attempt {att}, "
                        f"||R_u||/||f|| = {conv_out.res_u_norm / conv_out.f_ref:.3e}, "
                        f"active={n_active}"
                    )

                # ── ステップ 7: 接線剛性組立 ──
                tangent_out = _tangent_proc.process(
                    TangentAssemblyInput(
                        u=u,
                        lam_all=lam_all,
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
                    )
                )
                K_T = tangent_out.K_T

                # 動的: 質量・減衰を接線剛性に加算
                if dt_sub > 1e-30:
                    K_T = _time_strategy.effective_stiffness(K_T, dt_sub)

                # ── ステップ 8: 線形ソルブ ──
                solve_out = _solve_proc.process(
                    LinearSolveInput(
                        K_T=K_T,
                        R_u=R_u,
                        fixed_dofs=input_data.fixed_dofs,
                    )
                )
                if not solve_out.success:
                    if cfg.show_progress:
                        print(f"  WARNING: Linear solve failed at attempt {att}")
                    break

                du = solve_out.du

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

                # ── 変位・エネルギー収束判定 ──
                conv_out2 = _conv_proc.process(
                    ConvergenceCheckInput(
                        R_u=R_u,
                        du=du,
                        u=u,
                        f_ext_ref_norm=input_data.f_ext_ref_norm,
                        tol_force=cfg.tol_force,
                        tol_disp=cfg.tol_disp,
                        dynamic_ref=input_data.dynamic_ref,
                        is_first_attempt=False,
                        energy_ref=energy_ref,
                        manager=manager,
                    )
                )

                du_norm_val = conv_out2.du_norm
                diag.du_norm_history.append(du_norm_val)
                diag.max_du_dof_history.append(
                    int(np.argmax(np.abs(du))) if du_norm_val > 0 else -1
                )
                energy_ref = conv_out2.energy_ref

                if conv_out2.converged:
                    step_converged = True
                    if cfg.show_progress:
                        ctype = conv_out2.convergence_type
                        if ctype == ConvergenceType.DISPLACEMENT:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"uzawa {_uzawa_iter}, attempt {att}, "
                                f"||du||/||u|| = {du_norm_val / max(float(np.linalg.norm(u)), 1e-30):.3e} "
                                f"(disp converged, {n_active} active)"
                            )
                        else:
                            print(
                                f"  Incr {increment_display} (frac={load_frac:.4f}), "
                                f"uzawa {_uzawa_iter}, attempt {att}, "
                                f"energy = {conv_out2.energy:.3e} (energy converged)"
                            )
                    break

            # ── Uzawa 乗数更新 ──
            if step_converged:
                uzawa_out = _uzawa_proc.process(
                    UzawaUpdateInput(
                        lam_all=lam_all,
                        manager=manager,
                        k_pen=k_pen,
                        node_coords_ref=input_data.node_coords_ref,
                        u=u,
                        ndof_per_node=cfg.ndof_per_node,
                        tol_uzawa=tol_uzawa,
                    )
                )

                if cfg.show_progress:
                    print(f"  Uzawa {_uzawa_iter}: ||Δλ||/||λ|| = {uzawa_out.lam_change_ratio:.3e}")

                if uzawa_out.converged:
                    break  # Uzawa converged

                step_converged = False
                energy_ref = None
            else:
                break

        return DynamicStepOutput(
            converged=step_converged,
            n_attempts=total_attempts,
            n_active=n_active,
            f_c=f_c,
            diagnostics=diag,
        )
