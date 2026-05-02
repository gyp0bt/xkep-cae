"""work/beam_element_validation/42_alpha2_pure_bending_small.py — Phase α-2 純粋曲げ small κ.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase α-2: CR Timoshenko 3D 梁要素 1 つに純粋曲げ（end moment）を
**implicit static** で適用、3 指標 AND gate で線形小曲げ領域の foundation 健全性を検証。

**設定**:
- L=10 mm、r=0.5 mm（A=0.7854 mm²、I=4.909×10⁻² mm⁴）
- E=130 GPa、ν=0.3 → EI = E·I = 6.381×10³ N·mm²
- BC: 左端 (DOF 0–5) 完全固定、右端 M_y = 10 N·mm（DOF 10）

  status-389 §2 表は M=0.01 N·mm + EI=6.382 N·mm² と書いているが、EI = E·I =
  130000 · 0.04909 = **6381.4 N·mm²** で 10³ 単位ミス。plan が表に書いた観察値
  (u_z=0.07835 mm, θ=0.01567 rad) を再現するには **M=10 N·mm** が必要。
  本スクリプトは plan の意図（small κ Euler-Bernoulli linear bending）を尊重し
  M=10 N·mm を採用、解析解は数式から動的に計算する（透明性ルール）。

**解析解（線形 Euler-Bernoulli end moment loading）**:

| 指標 | 式 |
|------|---|
| `u_z_tip` | M·L² / (2·EI) ≈ 7.835×10⁻² mm |
| `θ_y_tip` | M·L / EI ≈ 1.567×10⁻² rad |
| `u_x_tip` | 0（二次小、線形理論で無視） |
| `SE` | (1/2) M θ ≈ 7.835×10⁻² N·mm |
| `L_arc` | ≈ 10.000 mm（不伸長性、二次小） |

**3 指標 AND gate**: `u_z_tip` / `θ_y_tip` / `f_int_M_y` の 3 個 — 全 10% 以内で PASS。
L_arc は二次小（10⁻⁶ オーダー）で線形領域では gate しても trivial に通るため診断のみ。

実行:
    uv run --extra dev python work/beam_element_validation/42_alpha2_pure_bending_small.py \\
        2>&1 | tee /tmp/alpha2_$(date +%s).log
"""

from __future__ import annotations

import sys

import numpy as np

from work.beam_element_validation._alpha_common import (
    BeamSection,
    MetricRow,
    NRResult,
    compute_chord_arc_length,
    numerical_strain_energy,
    run_case,
)

M_APPLIED = 10.0  # N·mm — small κ 領域で意味のあるスケール


def _alpha2_metrics(section: BeamSection, res: NRResult) -> list[MetricRow]:
    """α-2 純粋曲げ small κ の 3 指標 + 診断列."""
    EI = section.EI
    L = section.L

    # 解析解（Euler-Bernoulli linear, end moment）
    u_z_anal = M_APPLIED * L**2 / (2.0 * EI)
    theta_y_anal = M_APPLIED * L / EI
    u_x_anal = 0.0
    se_anal = 0.5 * M_APPLIED * theta_y_anal
    L_arc_anal = L  # 二次小、線形理論で不伸長

    # 数値解
    u = res.u
    u_x_num = float(u[6])
    u_z_num = float(u[8])
    theta_y_num = float(u[10])
    L_arc_num = compute_chord_arc_length(section, u)
    se_num = numerical_strain_energy(u, res.f_int_final)

    rows = [
        # gate 対象 3 指標（kinematic 2 + reaction 1） — XZ 平面 M_y 規約は実装ごとに
        # 符号が異なるため `compare_abs=True` で吸収（status-388 透明性ルール準拠）.
        MetricRow(
            "|u_z_tip| [mm]",
            u_z_anal,
            u_z_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        MetricRow(
            "|θ_y_tip| [rad]",
            theta_y_anal,
            theta_y_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        # 処方外力との釣り合い: f_int[10] = M_applied
        MetricRow(
            "|f_int M_y_tip| [N·mm]",
            M_APPLIED,
            float(res.f_int_final[10]),
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        # 診断
        MetricRow(
            "u_x_tip [mm] (=0, 診断)",
            u_x_anal,
            u_x_num,
            in_gate=False,
            abs_zero_ref=u_z_anal,
        ),
        MetricRow("L_arc [mm] (二次小, 診断)", L_arc_anal, L_arc_num, in_gate=False),
        MetricRow("SE [N·mm] (診断)", se_anal, se_num, in_gate=False),
    ]
    return rows


def main() -> int:
    section = BeamSection(L=10.0, r=0.5, E=130.0e3, nu=0.3)

    # BC: 左端完全固定、右端 M_y = M_APPLIED [N·mm]
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    F_ext = np.zeros(12)
    F_ext[10] = M_APPLIED  # 右端 θ_y モーメント

    passed = run_case(
        case_name=f"α-2 純粋曲げ small κ（implicit static, M_y={M_APPLIED} N·mm）",
        section=section,
        fixed_dofs=fixed_dofs,
        F_ext=F_ext,
        analytical_metrics=_alpha2_metrics,
        n_load_steps=1,
        tol_rel=1.0e-9,
        max_iter=20,
        verbose=False,
    )

    print()
    print("=" * 80)
    if passed:
        print("[α-2 結果] **PASS** — CR Timoshenko 1 要素 implicit static は")
        print("  純粋曲げ small κ で 3 指標一致、線形 Euler-Bernoulli foundation 健全。")
    else:
        print("[α-2 結果] **FAIL** — 3 指標 AND gate 未達。")
    print("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
