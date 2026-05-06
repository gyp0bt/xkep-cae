"""work/beam_element_validation/45_beta1_free_vibration.py — Phase β-1 自由振動.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase β-1: CR Timoshenko 3D 梁要素 1 つの cantilever（左端固定）に
**初期速度 v_z(tip) = 1 mm/s** を与え、explicit 中央差分 + 集中質量で **5 周期**
解析、3 指標 AND gate で 1 要素 explicit dynamics の foundation 健全性を検証。

**設定**:

- L=10 mm、r=0.5 mm（A=0.7854 mm²、I=4.909×10⁻² mm⁴、EI=6.381×10³ N·mm²）
- E=130 GPa、ν=0.3、ρ=8.96e-9 ton/mm³（Cu, status-330 系）
- BC: 左端 (DOF 0–5) 完全固定
- IC: u=0、v_z(DOF 8) = 1 mm/s
- 集中質量（lumped）、Rayleigh damping α=0
- explicit 中央差分（leap-frog Verlet）
- t_total = 5·T_FE_1（FE 系第 1 モード周期で 5 周期）
- dt = T_FE_1 / 500（Courant 安全率 0.5 を確実に下回る）

**3 指標 AND gate**（status-388 透明性ルール、kinematic 2 + energetics-or-geometric 1）:

1. **周期 T_meas** ≈ T_FE_1 (FE 系 K_aa φ = ω² M_aa φ から計算): gate ≤ 5%
   FE-consistent 第 1 モード周期は時間積分とは独立な kinematics 解析解
   （特性方程式の解、線形振動論）。
2. **最大振幅 |u_z_max|** ≈ v_0/ω_1 (エネルギー保存則による線形振動)
   1-mode 線形振動: u_z(t) = (v_0/ω_1)·sin(ω_1 t) → |u_z|_max = v_0/ω_1
   gate ≤ 10%（多モード混入＋central diff の数値減衰／増幅で多少誤差あり）.
3. **エネルギー保存** |ΔE/E_0| < 10% (5 周期通算、E = KE+SE)
   undamped central diff は厳密保存ではないが low-frequency 振動で 5 周期程度なら drift 小.

**診断**:

- T_Bernoulli_continuous = 2π/ω_cont（連続体解析解, 1 要素 lumped FE は ~30%
  低周波側にずれる、Phase γ で n_elements 増加と共に解消）.
- L_chord 保存（max |L_chord - L|/L、参考値）.

実行::

    uv run --extra dev python work/beam_element_validation/45_beta1_free_vibration.py \\
        2>&1 | tee /tmp/beta1_$(date +%s).log
"""

from __future__ import annotations

import math
import sys

import numpy as np

from work.beam_element_validation._alpha_common import (
    BeamSection,
    MetricRow,
    evaluate_three_metric_gate,
)
from work.beam_element_validation._beta_common import (
    compute_natural_frequencies_fe,
    measure_period_zero_crossings,
    solve_explicit_central_diff,
)


def main() -> int:
    section = BeamSection(L=10.0, r=0.5, E=130.0e3, nu=0.3)
    rho = 8.96e-9  # ton/mm³ (Cu)

    # --- FE 固有値解析（lumped mass、active=右端 6 DOF）---
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    omegas, _ = compute_natural_frequencies_fe(section, rho, fixed_dofs=fixed_dofs, use_lumped=True)
    omega_1_fe = float(omegas[0])
    omega_max_fe = float(omegas[-1])
    T_1_fe = 2.0 * math.pi / omega_1_fe

    # 連続体 Bernoulli cantilever 第 1 モード（診断、参考）
    rho_A = rho * section.A
    omega_1_cont = (1.875104) ** 2 * math.sqrt(section.EI / (rho_A * section.L**4))
    T_1_cont = 2.0 * math.pi / omega_1_cont

    # --- 解析設定 ---
    v0 = 1.0  # mm/s, initial velocity at tip u_z (DOF 8)
    n_periods = 5.0
    t_total = n_periods * T_1_fe
    n_steps = int(round(n_periods * 500))  # 500 steps per period

    dt = t_total / n_steps
    dt_critical = 2.0 / omega_max_fe  # central diff stability
    courant_ratio = dt / dt_critical

    print()
    print("=" * 80)
    print("  Phase β-1: 1 要素 cantilever 自由振動 (explicit central difference)")
    print("=" * 80)
    print(
        f"  L={section.L} mm, r={section.r} mm, E={section.E} MPa, ν={section.nu}, "
        f"ρ={rho:.3e} ton/mm³"
    )
    print(f"  EI={section.EI:.4e} N·mm², ρA={rho_A:.4e} ton/mm, A={section.A:.4f} mm²")
    print(
        f"  FE 固有値 ω_1={omega_1_fe:.4e} rad/s, T_FE_1={T_1_fe:.4e} s, "
        f"ω_max={omega_max_fe:.4e} rad/s"
    )
    print(
        f"  連続体 ω_1_cont={omega_1_cont:.4e} rad/s, T_cont={T_1_cont:.4e} s "
        f"(参考、1 要素 lumped FE は ~30% 低周波側)"
    )
    print(
        f"  解析: t_total={t_total:.4e} s ({n_periods} 周期), "
        f"n_steps={n_steps}, dt={dt:.4e} s, dt/dt_crit={courant_ratio:.3f}"
    )
    print(f"  IC: v_z(tip, DOF 8) = {v0} mm/s")
    print()

    if courant_ratio >= 1.0:
        print(f"  [WARN] Courant 比 {courant_ratio:.3f} ≥ 1.0 — 不安定の可能性")

    # --- 解析実行 ---
    res = solve_explicit_central_diff(
        section=section,
        rho=rho,
        fixed_dofs=fixed_dofs,
        t_total=t_total,
        n_steps=n_steps,
        initial_velocity={8: v0},
        damping_alpha=0.0,
        use_lumped=True,
    )

    # --- 指標計算 ---
    u_z_tip = res.u_history[:, 8]
    u_z_max_meas = float(np.max(np.abs(u_z_tip)))

    # 線形振動の解析解（1 モード仮定、エネルギー保存）
    u_z_max_anal = v0 / omega_1_fe

    # 周期計測（u_z の zero crossing）
    period_meas, n_zc = measure_period_zero_crossings(res.t, u_z_tip, skip_initial=0)

    # エネルギー保存
    E_total = res.total_energy
    E_0 = float(E_total[0])
    if E_0 < 1e-30:
        # KE_0 only (u=0, SE_0=0)
        E_0 = 0.5 * rho_A * section.L * v0**2 / 2.0  # rough
    E_max = float(np.max(E_total))
    E_min = float(np.min(E_total))
    E_drift = (E_max - E_min) / max(abs(E_0), 1e-30)

    # L_chord 保存
    L_chord_max = float(np.max(res.L_chord_history))
    L_chord_min = float(np.min(res.L_chord_history))
    L_chord_drift = max(abs(L_chord_max - section.L), abs(L_chord_min - section.L))

    print()
    print(f"  測定: u_z_max={u_z_max_meas:.4e} mm (解析={u_z_max_anal:.4e})")
    print(f"  測定: T_meas={period_meas:.4e} s ({n_zc} zero crossings 使用)")
    print(
        f"  測定: E_0={E_0:.4e} N·mm, E_max={E_max:.4e}, E_min={E_min:.4e}, "
        f"drift={E_drift * 100:.3f}%"
    )
    print(
        f"  測定: L_chord max drift={L_chord_drift:.4e} mm ({L_chord_drift / section.L * 100:.4f}%)"
    )

    # --- 3 指標 AND gate 行 ---
    rows = [
        MetricRow(
            name="T_period [s] (FE 第 1 モード)",
            analytical=T_1_fe,
            numerical=period_meas,
            gate_threshold=0.05,
            in_gate=True,
        ),
        MetricRow(
            name="|u_z_max| [mm] (v_0/ω_1)",
            analytical=u_z_max_anal,
            numerical=u_z_max_meas,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        MetricRow(
            name="E drift / E_0 [-] (5 周期)",
            analytical=0.0,
            numerical=E_drift,
            gate_threshold=0.10,
            in_gate=True,
            abs_zero_ref=1.0,
        ),
        # 診断
        MetricRow(
            name="T_period [s] (連続体 Bernoulli, 診断)",
            analytical=T_1_cont,
            numerical=period_meas,
            in_gate=False,
        ),
        MetricRow(
            name="L_chord drift [mm] (geometric, 診断)",
            analytical=0.0,
            numerical=L_chord_drift,
            in_gate=False,
            abs_zero_ref=section.L,
        ),
    ]

    passed = evaluate_three_metric_gate(
        case_name=f"β-1 自由振動 (cantilever, v_z={v0} mm/s, {n_periods} 周期)",
        rows=rows,
        converged=True,
        n_iters=n_steps,
        n_steps=n_steps,
    )

    print()
    print("=" * 80)
    if passed:
        print("[β-1 結果] **PASS** — 1 要素 cantilever explicit 自由振動は")
        print("  FE 第 1 モード周期 + エネルギー保存 + 線形振動振幅と 3 指標一致.")
        print("  → CR foundation + 中央差分 + 集中質量 + 質量行列の組合せが健全.")
        print(
            f"  注: 連続体 Bernoulli ω_1 との差は 1 要素 lumped FE 離散化の本質的"
            f" {abs(T_1_fe - T_1_cont) / T_1_cont * 100:.1f}% で、Phase γ で n_elements"
        )
        print("       増加と共に縮小するはず（参考、診断列）.")
    else:
        print("[β-1 結果] **FAIL** — 3 指標 AND gate 未達.")
        print("  explicit central diff + lumped mass の foundation を疑う必要あり.")
    print("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
