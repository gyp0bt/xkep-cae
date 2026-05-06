"""work/beam_element_validation/46_beta2_explicit_quasistatic.py — Phase β-2 explicit quasi-static.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase β-2: α-3 と同 BC（θ_y=0.15 rad 処方）を **explicit + slow ramp +
sufficient damping** で実施し、α-3 (implicit) Hermite 解との 3 指標 10% 以内一致を
検証する。これは「explicit + 大回転」が 1 要素規模で本質的に成立するかの最終 gate.

**判定設計**:

- **β-2 PASS** → CR foundation 健全 + (z2) Cosserat の主目的は explicit + 大回転
  robust 化 (UL 凍結回避) に絞れる. 既存 CR 実装は 1 要素 explicit dynamics で
  問題なく動作する → status-381〜387 explicit + UL の精度問題は assembler レベル
  もしくは UL formulation 由来であり、CR 要素自体は健全.
- **β-2 FAIL** → (z2) Cosserat 移行根拠 absolute 確定. 1 要素 explicit でも CR が
  large rotation 領域で精度を出せないなら、撚線規模では絶望的.

**設定**:

- L=10 mm、r=0.5 mm、E=130 GPa、ν=0.3、ρ=8.96e-9 ton/mm³
- BC: 左端 (DOF 0–5) 固定、右端 θ_y(DOF 10) を **slow ramp で 0 → 0.15 rad**
- damping: 質量比例 Rayleigh ``α = 4·ω_1`` (ζ ≈ 2 で過減衰、振動なく擬似静的)
- ramp 時間: t_ramp = 5·T_FE_1（5 周期かけて緩やか）
- hold 時間: t_hold = 5·T_FE_1（最終状態が α-3 Hermite に整合するまで settle）
- explicit 中央差分（leap-frog Verlet）+ 集中質量

**3 指標 AND gate**（α-3 と同じ Hermite 解、status-388 透明性ルール）:

1. ``|u_x_tip|`` ≈ Hermite 解 ``L·(cos(θ/2) − 1)`` (gate 10%)
2. ``|u_z_tip|`` ≈ Hermite 解 ``L·sin(θ/2)`` (gate 10%)
3. ``L_chord`` ≈ ``L`` (chord 保存 1 要素 CR、geometric, gate 10%)

**診断**:

- α-3 implicit (foundation) との直接比較
- KE/SE 比（quasi-static であれば KE << SE）
- final 残差 ``||f_int_a||`` (settle が完了したか)

実行::

    uv run --extra dev python work/beam_element_validation/46_beta2_explicit_quasistatic.py \\
        2>&1 | tee /tmp/beta2_$(date +%s).log
"""

from __future__ import annotations

import math
import sys

import numpy as np

from work.beam_element_validation._alpha_common import (
    BeamSection,
    MetricRow,
    compute_chord_arc_length,
    evaluate_three_metric_gate,
)
from work.beam_element_validation._beta_common import (
    compute_natural_frequencies_fe,
    solve_explicit_central_diff,
)

THETA_PRESCRIBED = 0.15  # rad (α-3 と同じ)


def main() -> int:
    section = BeamSection(L=10.0, r=0.5, E=130.0e3, nu=0.3)
    rho = 8.96e-9  # ton/mm³

    # FE 系第 1 モード周期（damping 設計と t_ramp 基準）
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    omegas, _ = compute_natural_frequencies_fe(section, rho, fixed_dofs=fixed_dofs, use_lumped=True)
    omega_1 = float(omegas[0])
    omega_max = float(omegas[-1])
    T_1 = 2.0 * math.pi / omega_1

    # ramp と hold 時間
    n_ramp_periods = 5.0
    n_hold_periods = 5.0
    t_ramp = n_ramp_periods * T_1
    t_hold = n_hold_periods * T_1
    t_total = t_ramp + t_hold

    # 質量比例 Rayleigh damping: ζ = α / (2 ω) → α = 2 ζ ω。ζ=2 で過減衰
    zeta = 2.0
    damping_alpha = 2.0 * zeta * omega_1

    # dt は damping を含む安定条件: dt ≤ 2/(ω_max·(1+ζ_max))
    # damping_alpha (一様) は最低モードに ζ_low = α/(2 ω_low)、最高モードに
    # ζ_high = α/(2 ω_max) を与える。ζ_high = α/(2 ω_max) = 2·ω_1/ω_max.
    # 中央差分 with damping: dt ≤ 2/ω_max · (sqrt(1+ζ²) - ζ) (Belytschko 6-178)
    zeta_high = damping_alpha / (2.0 * omega_max)
    dt_critical_damped = 2.0 / omega_max * (math.sqrt(1.0 + zeta_high**2) - zeta_high)

    # safety factor
    dt_target = 0.5 * dt_critical_damped
    n_steps = int(math.ceil(t_total / dt_target))
    dt = t_total / n_steps

    # ramp 関数: 0 → THETA_PRESCRIBED, t_ramp 後 hold
    def theta_ramp(t: float) -> tuple[float, float]:
        if t <= 0.0:
            return 0.0, 0.0
        if t <= t_ramp:
            u_p = THETA_PRESCRIBED * (t / t_ramp)
            v_p = THETA_PRESCRIBED / t_ramp
            return u_p, v_p
        return THETA_PRESCRIBED, 0.0

    print()
    print("=" * 80)
    print("  Phase β-2: 1 要素 explicit + slow ramp で α-3 Hermite 解と一致するか検証")
    print("=" * 80)
    print(f"  L={section.L} mm, r={section.r} mm, E={section.E} MPa, EI={section.EI:.4e} N·mm²")
    print(f"  FE ω_1={omega_1:.4e} rad/s, T_FE_1={T_1:.4e} s, ω_max={omega_max:.4e} rad/s")
    print(
        f"  damping: α={damping_alpha:.4e} 1/s (ζ@mode_1 = {zeta:.2f}, "
        f"ζ@mode_max = {zeta_high:.4e})"
    )
    print(
        f"  時間: t_ramp={t_ramp:.4e} s ({n_ramp_periods}·T_1), "
        f"t_hold={t_hold:.4e} s ({n_hold_periods}·T_1), t_total={t_total:.4e} s"
    )
    print(
        f"  dt_critical_damped={dt_critical_damped:.4e} s, dt={dt:.4e} s, "
        f"dt/dt_crit={dt / dt_critical_damped:.3f}, n_steps={n_steps}"
    )
    print(f"  BC: 左端 fix, 右端 θ_y ramp 0 → {THETA_PRESCRIBED} rad")
    print()

    # explicit 中央差分実行（log_every で履歴サンプリング）
    log_every = max(1, n_steps // 2000)
    res = solve_explicit_central_diff(
        section=section,
        rho=rho,
        fixed_dofs=fixed_dofs,
        t_total=t_total,
        n_steps=n_steps,
        prescribed_disp={10: theta_ramp},
        damping_alpha=damping_alpha,
        use_lumped=True,
        log_every=log_every,
    )

    # 最終状態
    u_final = res.u_history[-1]
    L_chord_final = compute_chord_arc_length(section, u_final)

    u_x_num = float(u_final[6])
    u_z_num = float(u_final[8])
    theta_y_num = float(u_final[10])

    # α-3 Hermite 解析解（chord 保存型, α=θ_R/2）
    theta = THETA_PRESCRIBED
    alpha_chord = theta / 2.0
    u_x_hermite = section.L * (math.cos(alpha_chord) - 1.0)
    u_z_hermite = section.L * math.sin(alpha_chord)

    # 残差 (settle 完了確認)
    from xkep_cae.elements._beam_cr import timo_beam3d_cr_internal_force

    coords = section.coords_2node()
    bk = section.beam_kwargs()
    f_int_final = timo_beam3d_cr_internal_force(coords, u_final, **bk)
    a_idx = np.array(sorted(set(range(12)) - set(fixed_dofs) - {10}), dtype=int)
    res_norm = float(np.linalg.norm(f_int_final[a_idx]))

    KE_final = float(res.KE_history[-1])
    SE_final = float(res.SE_history[-1])

    print()
    print(f"  最終 θ_y_tip = {theta_y_num:.6e} rad (処方 {theta:.6e})")
    print(f"  最終 u_x_tip = {u_x_num:.6e} mm (Hermite {u_x_hermite:.6e})")
    print(f"  最終 u_z_tip = {u_z_num:.6e} mm (Hermite {u_z_hermite:.6e})")
    print(f"  最終 L_chord = {L_chord_final:.6f} mm (L_0={section.L:.6f})")
    print(
        f"  KE_final={KE_final:.4e}, SE_final={SE_final:.4e}, "
        f"KE/SE={KE_final / max(SE_final, 1e-30):.4e}"
    )
    print(f"  ||f_int_a|| = {res_norm:.4e} N (settle 残差)")

    # --- 3 指標 AND gate ---
    rows = [
        # gate 対象 3 指標（α-3 と同じ Hermite 解との一致）
        MetricRow(
            name="|u_x_tip| [mm] (Hermite)",
            analytical=u_x_hermite,
            numerical=u_x_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        MetricRow(
            name="|u_z_tip| [mm] (Hermite)",
            analytical=u_z_hermite,
            numerical=u_z_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        MetricRow(
            name="L_chord [mm] (geometric)",
            analytical=section.L,
            numerical=L_chord_final,
            gate_threshold=0.10,
            in_gate=True,
        ),
        # 診断列
        MetricRow(
            name="θ_y_tip [rad] (=処方値, 診断)",
            analytical=theta,
            numerical=theta_y_num,
            in_gate=False,
        ),
        MetricRow(
            name="KE/SE [-] (quasi-static, 診断)",
            analytical=0.0,
            numerical=KE_final / max(SE_final, 1e-30),
            in_gate=False,
            abs_zero_ref=1.0,
        ),
        MetricRow(
            name="||f_int_a|| [N] (settle 残差, 診断)",
            analytical=0.0,
            numerical=res_norm,
            in_gate=False,
            abs_zero_ref=section.EI / section.L**2,
        ),
    ]

    passed = evaluate_three_metric_gate(
        case_name=f"β-2 explicit quasi-static (θ_y={THETA_PRESCRIBED} rad, ramp+hold)",
        rows=rows,
        converged=True,
        n_iters=n_steps,
        n_steps=n_steps,
    )

    print()
    print("=" * 80)
    if passed:
        print("[β-2 結果] **PASS** — 1 要素 explicit + slow ramp + 質量比例減衰 は")
        print("  α-3 implicit Hermite 解と 3 指標一致.")
        print("  → CR foundation 健全 + (z2) Cosserat の主目的は")
        print("    explicit + 大回転 robust 化（assembler／UL 由来の問題）に絞れる.")
        print("  status-381〜387 explicit + UL の精度問題は CR 要素自体ではなく")
        print("  上位層（assembler / UL formulation / mass scaling 戦略）に局在.")
    else:
        print("[β-2 結果] **FAIL** — 3 指標 AND gate 未達.")
        print("  → (z2) Cosserat 移行根拠 absolute 確定.")
        print("  CR 1 要素でさえ explicit + 大回転で精度が出ない場合、撚線規模では")
        print("  絶望的であり、SO(3) 直接積分の Cosserat 路線が必須となる.")
    print("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
