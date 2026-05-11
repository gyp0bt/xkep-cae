"""work/beam_element_validation/51_gamma3_multi_element_explicit.py — Phase γ-3 多要素 explicit + TL.

[← README](README.md) | [← project README](../../README.md)

status-394 後継: implicit 凍結方針下で次の foundation 確認として、Phase γ-1
(multi-element implicit static, status-392) を **explicit + TL モード** に拡張し、
β-2 (1 要素 explicit, status-391) を多要素規模に拡げて circular arc 解への収束を
status-388 透明性ルール（独立 3 指標 AND gate）で確認する。

**判定設計**:

- **γ-3 PASS** (n=2,4,8,16) → 多要素 explicit + TL の foundation 健全性確定。
  続く (z3) 「explicit モード TL 固定 API + 19 本撚線適用」へ進める準備が整う。
- **γ-3 FAIL** → β-2 (1 要素) と γ-1 (多要素 implicit) は PASS していたので、
  multi-element + explicit + TL の組合せ固有に問題がある（mass 配置 / dt_c 推定 /
  damping 配分など）。19 本撚線適用前にここで局在化必須。

**実装パス**:

- ``ChainedBeamSection`` (status-392) を採用して n_elements ∈ {1,2,4,8,16} を駆動
- ``assemble_internal_force`` (TL モード — coords_init は不変、UL update_reference 不要)
- 集中質量を要素ごと ``timo_beam3d_lumped_mass_local`` で組み立て、global に accumulate
- β-2 と同じ slow ramp (5 T_FE_1) + hold (5 T_FE_1) + 質量比例減衰 (ζ=2 過減衰)
- ω_max は `K_aa φ = ω² M_aa φ` を u=0 で解いた dominant eigenvalue で推定
- 3 指標 AND gate: |u_x_tip| / |u_z_tip| / L_chord_endpoints (vs circular arc)

実行::

    uv run --extra dev python work/beam_element_validation/51_gamma3_multi_element_explicit.py \\
        2>&1 | tee /tmp/gamma3_$(date +%s).log
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from work.beam_element_validation._alpha_common import (
    MetricRow,
    evaluate_three_metric_gate,
)
from work.beam_element_validation._gamma_common import (
    ChainedBeamSection,
    assemble_internal_force,
    assemble_tangent,
    compute_chord_total,
    compute_polyline_length,
)
from xkep_cae.elements._beam_cr import timo_beam3d_lumped_mass_local

THETA_PRESCRIBED = 0.15  # rad (α-3 / β-2 / γ-1 と同じ)
RHO = 8.96e-9  # ton/mm³

N_ELEMENTS_LIST = [1, 2, 4, 8, 16]


# =============================================================================
# 多要素チェーン用ヘルパ — TL explicit central diff
# =============================================================================


def assemble_lumped_mass_diag(section: ChainedBeamSection, rho: float) -> np.ndarray:
    """各要素の ``timo_beam3d_lumped_mass_local`` を global diag に accumulate."""
    diag = np.zeros(section.n_dofs, dtype=float)
    for e in range(section.n_elements):
        Me_diag = timo_beam3d_lumped_mass_local(
            rho,
            section.A,
            section.I,
            section.I,
            section.L_elem,
        )
        dof_e = section.element_dofs(e)
        for k in range(12):
            diag[dof_e[k]] += Me_diag[k, k]
    return diag


def compute_natural_frequencies_chain(
    section: ChainedBeamSection,
    rho: float,
    fixed_dofs: list[int],
    prescribed_dofs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """K_aa φ = ω² M_aa φ を u=0 で解いて (ω 配列, modes) を返す（多要素対応）.

    ``compute_natural_frequencies_fe`` (1 要素版) の chain 拡張.
    集中質量 (lumped) を使用 — explicit central diff の安定性推定と整合.
    """
    fixed_set = set(fixed_dofs)
    presc_set = set(prescribed_dofs or [])
    if fixed_set & presc_set:
        raise ValueError(f"DOF {fixed_set & presc_set} は fixed と prescribed 両方に指定")
    a_idx = np.array(sorted(set(range(section.n_dofs)) - fixed_set - presc_set), dtype=int)

    u0 = np.zeros(section.n_dofs)
    K = assemble_tangent(section, u0)
    M_diag = assemble_lumped_mass_diag(section, rho)

    K_aa = K[np.ix_(a_idx, a_idx)]
    M_aa_diag = M_diag[a_idx]
    # 集中質量で M_aa は対角 → eigh(K_aa, diag(M_aa)) を直接解く
    M_aa = np.diag(M_aa_diag)
    eigvals, eigvecs = eigh(K_aa, M_aa)
    omegas = np.sqrt(np.maximum(eigvals, 0.0))
    return omegas, eigvecs


@dataclass
class ExplicitChainResult:
    u_final: np.ndarray
    KE_final: float
    SE_strain_proxy: float  # 0.5 u·f_int (大変形では参考値)
    L_chord_total: float
    L_polyline: float
    res_norm: float
    dt: float
    n_steps: int
    omega_1: float
    omega_max: float


def solve_explicit_chain(
    section: ChainedBeamSection,
    rho: float,
    fixed_dofs: list[int],
    prescribed_disp_funcs: dict[int, callable],
    t_total: float,
    n_steps: int,
    damping_alpha: float,
) -> ExplicitChainResult:
    """multi-element TL explicit central diff (leap-frog Verlet) + lumped mass + Rayleigh damping.

    Args:
        section: ``ChainedBeamSection``.
        rho: 密度 [ton/mm³].
        fixed_dofs: u=0 固定 DOF list.
        prescribed_disp_funcs: {DOF index: t -> (u, v) 関数}.
        t_total: 解析総時間 [s].
        n_steps: 時間ステップ数. dt = t_total / n_steps.
        damping_alpha: 質量比例 Rayleigh α.
    """
    fixed_set = set(fixed_dofs)
    presc_set = set(prescribed_disp_funcs.keys())
    a_idx = np.array(sorted(set(range(section.n_dofs)) - fixed_set - presc_set), dtype=int)

    M_diag = assemble_lumped_mass_diag(section, rho)
    M_aa = M_diag[a_idx]

    dt = t_total / n_steps
    u = np.zeros(section.n_dofs)
    v = np.zeros(section.n_dofs)
    a = np.zeros(section.n_dofs)

    # 初期: prescribed at t=0
    for d, ramp in prescribed_disp_funcs.items():
        u_p, v_p = ramp(0.0)
        u[d] = u_p
        v[d] = v_p

    f_int = assemble_internal_force(section, u)
    a[a_idx] = -f_int[a_idx] / M_aa - damping_alpha * v[a_idx]
    v_half = v.copy()
    v_half[a_idx] = v[a_idx] + 0.5 * dt * a[a_idx]

    for k in range(n_steps):
        t_k1 = (k + 1) * dt
        u[a_idx] = u[a_idx] + dt * v_half[a_idx]
        for d, ramp in prescribed_disp_funcs.items():
            u_p, v_p = ramp(t_k1)
            u[d] = u_p
            v[d] = v_p

        f_int = assemble_internal_force(section, u)
        a[a_idx] = -f_int[a_idx] / M_aa - damping_alpha * v_half[a_idx]
        v_half[a_idx] = v_half[a_idx] + dt * a[a_idx]
        v[a_idx] = v_half[a_idx] - 0.5 * dt * a[a_idx]

    # 最終診断
    KE_final = 0.5 * float(np.sum(M_diag * v * v))
    SE_proxy = 0.5 * float(u @ f_int)
    L_chord_total = compute_chord_total(section, u)
    L_polyline = compute_polyline_length(section, u)
    res_norm = float(np.linalg.norm(f_int[a_idx]))

    return ExplicitChainResult(
        u_final=u.copy(),
        KE_final=KE_final,
        SE_strain_proxy=SE_proxy,
        L_chord_total=L_chord_total,
        L_polyline=L_polyline,
        res_norm=res_norm,
        dt=dt,
        n_steps=n_steps,
        omega_1=float("nan"),  # 後で fill
        omega_max=float("nan"),
    )


# =============================================================================
# 解析解（γ-1 と同じ closed forms）
# =============================================================================


def cr_closed_form(L_total: float, theta: float, n: int) -> tuple[float, float, float]:
    """n 要素 CR closed form (u_x_tip, u_z_tip, L_chord_endpoints)."""
    L_elem = L_total / n
    s_half_n = math.sin(theta / (2.0 * n))
    s_half = math.sin(theta / 2.0)
    c_half = math.cos(theta / 2.0)
    if s_half_n < 1e-30:
        x_n = L_total * math.sin(theta) / theta
        z_n = L_total * (1.0 - math.cos(theta)) / theta
    else:
        x_n = L_elem * s_half * c_half / s_half_n
        z_n = L_elem * s_half * s_half / s_half_n
    u_x = x_n - L_total
    u_z = z_n
    L_chord = math.hypot(x_n, z_n)
    return u_x, u_z, L_chord


def arc_form(L_total: float, theta: float) -> tuple[float, float, float]:
    """Circular arc 解 (u_x_tip, u_z_tip, L_chord_endpoints)."""
    R = L_total / theta
    u_x = R * math.sin(theta) - L_total
    u_z = R * (1.0 - math.cos(theta))
    L_chord = 2.0 * R * math.sin(theta / 2.0)
    return u_x, u_z, L_chord


# =============================================================================
# 1 ケース実行
# =============================================================================


@dataclass
class CaseSummary:
    n_elements: int
    u_x_tip: float
    u_z_tip: float
    L_chord_total: float
    L_polyline: float
    KE_SE_ratio: float
    res_norm: float
    omega_1: float
    omega_max: float
    n_steps: int
    err_u_x_arc: float
    err_u_z_arc: float
    err_L_chord_arc: float
    gate_passed: bool


def run_one_case(n_elements: int) -> CaseSummary:
    section = ChainedBeamSection(L_total=10.0, n_elements=n_elements, r=0.5, E=130.0e3, nu=0.3)
    fixed_dofs = list(range(6))  # 左端 6 DOF 固定
    last_node = section.n_nodes - 1
    presc_dof = 6 * last_node + 4  # 右端 θ_y

    # FE 系 ω_1 / ω_max（damping 設計と dt_c 推定）
    omegas, _ = compute_natural_frequencies_chain(
        section, RHO, fixed_dofs=fixed_dofs, prescribed_dofs=[presc_dof]
    )
    omega_1 = float(omegas[0])
    omega_max = float(omegas[-1])
    T_1 = 2.0 * math.pi / omega_1

    n_ramp_periods = 5.0
    n_hold_periods = 5.0
    t_ramp = n_ramp_periods * T_1
    t_hold = n_hold_periods * T_1
    t_total = t_ramp + t_hold

    zeta = 2.0  # β-2 と同じ
    damping_alpha = 2.0 * zeta * omega_1
    zeta_high = damping_alpha / (2.0 * omega_max)
    dt_critical_damped = 2.0 / omega_max * (math.sqrt(1.0 + zeta_high**2) - zeta_high)
    dt_target = 0.5 * dt_critical_damped
    n_steps = int(math.ceil(t_total / dt_target))
    dt = t_total / n_steps

    def theta_ramp(t: float) -> tuple[float, float]:
        if t <= 0.0:
            return 0.0, 0.0
        if t <= t_ramp:
            return THETA_PRESCRIBED * (t / t_ramp), THETA_PRESCRIBED / t_ramp
        return THETA_PRESCRIBED, 0.0

    print(f"\n{'=' * 80}")
    print(f"  Phase γ-3: n_elements={n_elements}, θ_y(tip)={THETA_PRESCRIBED} rad")
    print(f"{'=' * 80}")
    print(
        f"  L_total={section.L_total} mm, L_elem={section.L_elem:.4f} mm, "
        f"n_dofs={section.n_dofs}, presc_dof={presc_dof}"
    )
    print(
        f"  ω_1={omega_1:.4e} rad/s, T_1={T_1:.4e} s, ω_max={omega_max:.4e} rad/s "
        f"(ω_max/ω_1={omega_max / max(omega_1, 1e-30):.1f})"
    )
    print(f"  damping α={damping_alpha:.4e} 1/s (ζ@1={zeta:.2f}, ζ@max={zeta_high:.4e})")
    print(f"  t_ramp={t_ramp:.4e} s, t_hold={t_hold:.4e} s, t_total={t_total:.4e} s")
    print(
        f"  dt_critical_damped={dt_critical_damped:.4e}, dt={dt:.4e} "
        f"(dt/dt_crit={dt / dt_critical_damped:.3f}), n_steps={n_steps}"
    )

    res = solve_explicit_chain(
        section=section,
        rho=RHO,
        fixed_dofs=fixed_dofs,
        prescribed_disp_funcs={presc_dof: theta_ramp},
        t_total=t_total,
        n_steps=n_steps,
        damping_alpha=damping_alpha,
    )
    res.omega_1 = omega_1
    res.omega_max = omega_max

    u = res.u_final
    u_x_tip = float(u[6 * last_node + 0])
    u_z_tip = float(u[6 * last_node + 2])

    u_x_arc, u_z_arc, L_chord_arc = arc_form(section.L_total, THETA_PRESCRIBED)
    u_x_cr, u_z_cr, L_chord_cr = cr_closed_form(section.L_total, THETA_PRESCRIBED, n_elements)

    rows = [
        # gate 対象 3 指標 — circular arc 解への収束（γ-1 と同じ判定基準）
        MetricRow(
            "|u_x_tip| [mm] (arc)",
            u_x_arc,
            u_x_tip,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        MetricRow(
            "|u_z_tip| [mm] (arc)",
            u_z_arc,
            u_z_tip,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        MetricRow(
            "L_chord_endpoints [mm] (arc)",
            L_chord_arc,
            res.L_chord_total,
            gate_threshold=0.10,
            in_gate=True,
        ),
        # 診断 — n 要素 CR closed form 一致（実装健全性、機械精度〜数値誤差期待）
        MetricRow(
            "|u_x_tip| [mm] (CR closed, 診断)",
            u_x_cr,
            u_x_tip,
            in_gate=False,
            compare_abs=True,
        ),
        MetricRow(
            "|u_z_tip| [mm] (CR closed, 診断)",
            u_z_cr,
            u_z_tip,
            in_gate=False,
            compare_abs=True,
        ),
        MetricRow(
            "L_chord_endpoints [mm] (CR closed, 診断)",
            L_chord_cr,
            res.L_chord_total,
            in_gate=False,
        ),
        MetricRow(
            "Σ L_elem [mm] (polyline, 診断)",
            section.L_total,
            res.L_polyline,
            in_gate=False,
        ),
        MetricRow(
            "θ_y_tip [rad] (=処方値, 診断)",
            THETA_PRESCRIBED,
            float(u[6 * last_node + 4]),
            in_gate=False,
        ),
        MetricRow(
            "KE/SE_proxy [-] (quasi-static, 診断)",
            0.0,
            res.KE_final / max(abs(res.SE_strain_proxy), 1e-30),
            in_gate=False,
            abs_zero_ref=1.0,
        ),
        MetricRow(
            "||f_int_a|| [N] (settle 残差, 診断)",
            0.0,
            res.res_norm,
            in_gate=False,
            abs_zero_ref=section.EI / section.L_total**2,
        ),
    ]

    gate_passed = evaluate_three_metric_gate(
        case_name=f"γ-3 n_elements={n_elements} (explicit + TL)",
        rows=rows,
        converged=True,
        n_iters=res.n_steps,
        n_steps=res.n_steps,
    )

    def _rel_err(a: float, n_: float, abs_compare: bool = True) -> float:
        ax = abs(a) if abs_compare else a
        nx = abs(n_) if abs_compare else n_
        return abs(nx - ax) / max(abs(ax), 1e-30)

    return CaseSummary(
        n_elements=n_elements,
        u_x_tip=u_x_tip,
        u_z_tip=u_z_tip,
        L_chord_total=res.L_chord_total,
        L_polyline=res.L_polyline,
        KE_SE_ratio=res.KE_final / max(abs(res.SE_strain_proxy), 1e-30),
        res_norm=res.res_norm,
        omega_1=omega_1,
        omega_max=omega_max,
        n_steps=res.n_steps,
        err_u_x_arc=_rel_err(u_x_arc, u_x_tip),
        err_u_z_arc=_rel_err(u_z_arc, u_z_tip),
        err_L_chord_arc=_rel_err(L_chord_arc, res.L_chord_total, abs_compare=False),
        gate_passed=gate_passed,
    )


# =============================================================================
# 収束表 + 総合判定
# =============================================================================


def print_convergence_table(summaries: list[CaseSummary]) -> None:
    print()
    print("=" * 80)
    print("  Phase γ-3 収束表（vs circular arc 解）")
    print("=" * 80)
    print(
        f"  {'n':>3} | {'err |u_x| [%]':>14} | {'err |u_z| [%]':>14} | "
        f"{'err L_chord [%]':>16} | {'n_steps':>8} | {'KE/SE':>10} | gate"
    )
    print("  " + "-" * 88)
    for s in summaries:
        flag = "PASS" if s.gate_passed else "FAIL"
        print(
            f"  {s.n_elements:>3d} | {s.err_u_x_arc * 100:>14.4f} | "
            f"{s.err_u_z_arc * 100:>14.4f} | {s.err_L_chord_arc * 100:>16.4f} | "
            f"{s.n_steps:>8d} | {s.KE_SE_ratio:>10.3e} | {flag}"
        )

    # log-log slope of u_x (n≥2) で O(1/n²) 確認（γ-1 と同じ手順）
    sums_n2 = [s for s in summaries if s.n_elements >= 2]
    if len(sums_n2) >= 2:
        errs_ux = np.array([max(s.err_u_x_arc, 1e-30) for s in sums_n2])
        ns = np.array([s.n_elements for s in sums_n2], dtype=float)
        log_n = np.log(ns)
        log_err = np.log(errs_ux)
        slope, intercept = np.polyfit(log_n, log_err, 1)
        print()
        print(f"  log-log slope of err(|u_x|) vs n (n≥2): {slope:.3f} (理論値 -2.0 で O(1/n²))")


def main() -> int:
    summaries: list[CaseSummary] = []
    for n in N_ELEMENTS_LIST:
        s = run_one_case(n)
        summaries.append(s)

    print_convergence_table(summaries)

    # 総合判定: n=1 は α-3 / γ-1 と同じ chord 長保存制約で FAIL 期待。n≥2 は PASS 期待。
    all_n2_pass = all(s.gate_passed for s in summaries if s.n_elements >= 2)
    n1_summary = next((s for s in summaries if s.n_elements == 1), None)

    print()
    print("=" * 80)
    print("  総合判定（多要素 explicit + TL の foundation 健全性）")
    print("=" * 80)
    if all_n2_pass:
        print("  n_elements ∈ {2,4,8,16} で **3 指標 AND gate 全 PASS**.")
        print("  → 多要素 explicit + TL モードで CR foundation は健全.")
        print("    続く (z3) explicit-TL 固定 API + 19 本撚線適用へ進む準備が整う.")
        if n1_summary is not None and not n1_summary.gate_passed:
            print(
                f"  n=1 FAIL (err |u_x|={n1_summary.err_u_x_arc * 100:.2f}%) は α-3 / γ-1 同様の"
                " chord 長保存制約による既知の離散化誤差で期待通り."
            )
        return 0
    else:
        print("  n_elements ≥ 2 で 1 件以上 FAIL.")
        print("  → 多要素 explicit + TL に固有の問題あり、19 本適用前に局在化必須.")
        for s in summaries:
            if s.n_elements >= 2 and not s.gate_passed:
                print(
                    f"    n={s.n_elements}: err |u_x|={s.err_u_x_arc * 100:.2f}%, "
                    f"err |u_z|={s.err_u_z_arc * 100:.2f}%, "
                    f"err L_chord={s.err_L_chord_arc * 100:.2f}%"
                )
        return 1


if __name__ == "__main__":
    sys.exit(main())
