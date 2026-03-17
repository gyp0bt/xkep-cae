"""Newton-Raphson + Uzawa イテレーション.

solver_smooth_penalty.py の内側ループ（NR反復 + Uzawa乗数更新）を分離。
1荷重増分に対してNR収束 → Uzawa収束を達成するクラス。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from __xkep_cae_deprecated.contact.diagnostics import ConvergenceDiagnosticsOutput
from __xkep_cae_deprecated.contact.pair import ContactManager
from __xkep_cae_deprecated.contact.utils import deformed_coords, ncp_line_search
from __xkep_cae_deprecated.process.data import SolverStrategies


@dataclass
class StepResult:
    """1荷重増分の結果."""

    converged: bool
    n_newton_iters: int
    n_active: int
    f_c: np.ndarray  # 最終接触力
    diagnostics: ConvergenceDiagnosticsOutput


@dataclass
class NewtonUzawaConfig:
    """Newton-Uzawa ループの設定."""

    max_iter: int = 50
    tol_force: float = 1e-8
    tol_disp: float = 1e-8
    use_line_search: bool = True
    line_search_max_steps: int = 5
    du_norm_cap: float = 0.0
    show_progress: bool = True
    ndof_per_node: int = 6


class NewtonUzawaLoop:
    """1荷重増分のNewton-Raphson + Uzawaイテレーション.

    solver_smooth_penalty.py 315-523行の分離。
    Strategy 経由で接触力・摩擦力・被膜力・時間積分を評価する。
    """

    def __init__(self, config: NewtonUzawaConfig | None = None) -> None:
        self.config = config or NewtonUzawaConfig()

    def run(
        self,
        *,
        u: np.ndarray,
        lam_all: np.ndarray,
        f_ext: np.ndarray,
        f_ext_ref_norm: float,
        fixed_dofs: np.ndarray,
        assemble_tangent,
        assemble_internal_force,
        manager: ContactManager,
        node_coords_ref: np.ndarray,
        strategies: SolverStrategies,
        k_pen: float,
        mu: float,
        u_ref: np.ndarray,
        load_frac: float,
        load_frac_prev: float,
        step_display: int,
        dt_sub: float,
        use_coating: bool,
        dynamic_ref: bool,
    ) -> StepResult:
        """1荷重増分のNR+Uzawaを実行.

        u, lam_all は in-place で更新される。

        Returns:
            StepResult
        """
        cfg = self.config
        ndof = len(f_ext)

        _time_strategy = strategies.time_integration
        _contact_force_strategy = strategies.contact_force
        _friction_strategy = strategies.friction
        _coating_strategy = strategies.coating
        _dynamics = _time_strategy.is_dynamic

        n_uzawa_max = getattr(_contact_force_strategy, "_n_uzawa_max", 5)
        tol_uzawa = getattr(_contact_force_strategy, "_tol_uzawa", 1e-6)

        diag = ConvergenceDiagnosticsOutput(step=step_display, load_frac=load_frac)
        total_newton = 0
        f_c = np.zeros(ndof)
        energy_ref = None
        step_converged = False
        n_active = 0

        for _uzawa_iter in range(n_uzawa_max):
            for it in range(cfg.max_iter):
                total_newton += 1

                # 1. 幾何更新
                coords_def = deformed_coords(node_coords_ref, u, cfg.ndof_per_node)
                manager.update_geometry(coords_def, freeze_active_set=True)

                # 2. 接触力（ContactForceStrategy 経由）
                f_c, _ = _contact_force_strategy.evaluate(u, lam_all, manager, k_pen)

                # 3. 摩擦力（FrictionStrategy 経由）
                if hasattr(_friction_strategy, "_mu_ramp_counter"):
                    _friction_strategy._mu_ramp_counter = step_display
                f_friction, _ = _friction_strategy.evaluate(
                    u,
                    manager.pairs,
                    mu,
                    lambdas=lam_all,
                    u_ref=u_ref,
                    node_coords_ref=node_coords_ref,
                )
                f_c = f_c + f_friction

                # 4. 被膜力（CoatingStrategy 経由）
                if use_coating and _coating_strategy is not None:
                    coat_dt = max(load_frac - load_frac_prev, 1e-15)
                    f_coat = _coating_strategy.forces(
                        manager.pairs, coords_def, manager.config, coat_dt
                    )
                    f_c = f_c + f_coat
                    if manager.config.coating_mu > 0.0:
                        f_coat_fric = _coating_strategy.friction_forces(
                            manager.pairs, coords_def, manager.config, u, u_ref
                        )
                        f_c = f_c + f_coat_fric

                # 5. 力残差
                f_int = assemble_internal_force(u)
                R_u = f_int + f_c - f_ext

                # 動的解析: 慣性力・減衰力
                if _dynamics and dt_sub > 1e-30:
                    _time_strategy.correct(u, np.zeros_like(u), dt_sub)
                    R_u = _time_strategy.effective_residual(R_u, dt_sub)

                R_u[fixed_dofs] = 0.0

                # 6. 収束判定
                res_u_norm = float(np.linalg.norm(R_u))
                _f_ref = f_ext_ref_norm
                if dynamic_ref and it == 0 and res_u_norm > 1e-30:
                    _f_ref = res_u_norm

                n_active = sum(
                    1 for p in manager.pairs if hasattr(p, "state") and p.state.p_n > 0.0
                )

                diag.res_history.append(res_u_norm / _f_ref)
                diag.ncp_history.append(0.0)
                diag.ncp_t_history.append(0.0)
                diag.n_active_history.append(n_active)

                force_conv = res_u_norm / _f_ref < cfg.tol_force
                if force_conv:
                    step_converged = True
                    if cfg.show_progress:
                        print(
                            f"  Incr {step_display} (frac={load_frac:.4f}), "
                            f"uzawa {_uzawa_iter}, iter {it}, "
                            f"||R_u||/||f|| = {res_u_norm / _f_ref:.3e} "
                            f"(converged, {n_active} active)"
                        )
                    break

                if cfg.show_progress and it % 5 == 0:
                    print(
                        f"  Incr {step_display} (frac={load_frac:.4f}), "
                        f"uzawa {_uzawa_iter}, iter {it}, "
                        f"||R_u||/||f|| = {res_u_norm / _f_ref:.3e}, "
                        f"active={n_active}"
                    )

                # 7. 接線剛性
                K_T = assemble_tangent(u)

                K_c = _contact_force_strategy.tangent(u, lam_all, manager, k_pen)
                K_T = K_T + K_c

                # 被膜剛性
                if use_coating and _coating_strategy is not None:
                    coat_dt = max(load_frac - load_frac_prev, 1e-15)
                    K_coat = _coating_strategy.stiffness(
                        manager.pairs, coords_def, manager.config, ndof, coat_dt
                    )
                    K_T = K_T + K_coat
                    if manager.config.coating_mu > 0.0:
                        K_coat_fric = _coating_strategy.friction_stiffness(
                            manager.pairs, coords_def, manager.config, ndof
                        )
                        K_T = K_T + K_coat_fric

                # 摩擦剛性
                if _friction_strategy.friction_tangents:
                    K_fric = _friction_strategy.tangent(u, manager.pairs, mu)
                    K_T = K_T + K_fric

                # 動的解析: 質量・減衰
                if _dynamics and dt_sub > 1e-30:
                    K_T = _time_strategy.effective_stiffness(K_T, dt_sub)

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
                    if cfg.show_progress:
                        print(f"  WARNING: Linear solve failed at iter {it}")
                    break

                # 9. Line search + 更新
                _scale_factor = 1.0
                if cfg.use_line_search:
                    alpha = ncp_line_search(
                        u,
                        du,
                        f_ext,
                        fixed_dofs,
                        assemble_internal_force,
                        res_u_norm,
                        max_steps=cfg.line_search_max_steps,
                        f_c=f_c,
                        diverge_factor=3.0,
                    )
                    _scale_factor *= alpha
                if cfg.du_norm_cap > 0.0:
                    _du_n = float(np.linalg.norm(_scale_factor * du))
                    _u_ref_n = max(float(np.linalg.norm(u)), 1.0)
                    if _du_n > cfg.du_norm_cap * _u_ref_n:
                        _scale_factor *= cfg.du_norm_cap * _u_ref_n / _du_n
                du = _scale_factor * du
                u += du

                # 変位収束判定
                u_norm = float(np.linalg.norm(u))
                du_norm_val = float(np.linalg.norm(du))
                diag.du_norm_history.append(du_norm_val)
                diag.max_du_dof_history.append(
                    int(np.argmax(np.abs(du))) if du_norm_val > 0 else -1
                )

                if u_norm > 1e-30 and du_norm_val / u_norm < cfg.tol_disp:
                    step_converged = True
                    if cfg.show_progress:
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
                    if cfg.show_progress:
                        print(
                            f"  Incr {step_display} (frac={load_frac:.4f}), "
                            f"uzawa {_uzawa_iter}, iter {it}, "
                            f"energy = {energy:.3e} (energy converged)"
                        )
                    break

            # --- Uzawa 乗数更新 ---
            if step_converged:
                coords_def = deformed_coords(node_coords_ref, u, cfg.ndof_per_node)
                manager.update_geometry(coords_def, freeze_active_set=True)

                lam_prev = lam_all.copy()
                for i, pair in enumerate(manager.pairs):
                    if i < len(lam_all):
                        lam_all[i] = max(0.0, lam_all[i] + k_pen * (-pair.state.gap))
                lam_change = float(np.linalg.norm(lam_all - lam_prev))
                lam_ref = max(float(np.linalg.norm(lam_all)), 1.0)

                if cfg.show_progress:
                    print(f"  Uzawa {_uzawa_iter}: ||Δλ||/||λ|| = {lam_change / lam_ref:.3e}")

                if lam_change / lam_ref < tol_uzawa:
                    break  # Uzawa converged

                step_converged = False
                energy_ref = None
            else:
                break

        return StepResult(
            converged=step_converged,
            n_newton_iters=total_newton,
            n_active=n_active,
            f_c=f_c,
            diagnostics=diag,
        )
