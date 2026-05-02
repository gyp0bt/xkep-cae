"""work/beam_element_validation/41_alpha1_axial_tension.py — Phase α-1 純軸引張.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase α-1: CR Timoshenko 3D 梁要素 1 つに純軸引張を **implicit static**
で適用、3 指標 AND gate（status-388 透明性ルール）で foundation 健全性を検証する。

**設定**:
- L=10 mm、r=0.5 mm（A=0.7854 mm²、I=0.04909 mm⁴）
- E=130 GPa、ν=0.3
- BC: 左端 (DOF 0–5) 完全固定、右端 F_x=100 N（DOF 6 軸方向引張）
- n_elements = 1（節点 2 個）

**解析解（線形弾性）**:

| 指標 | 式 |
|------|---|
| `u_x_tip` | F·L / (E·A) ≈ 9.794×10⁻³ mm |
| `u_z_tip` | 0（軸対称） |
| `θ_y_tip` | 0（軸対称） |
| `SE` | (1/2) F u_x ≈ 4.897×10⁻⁴ N·mm |
| `L_arc` | L + u_x_tip ≈ 10.009794 mm |

**3 指標 AND gate**: `u_x_tip` / `u_z_tip` / `L_arc` の 3 個 — 全 10% 以内で PASS。
`u_z_tip = 0` は絶対誤差 1e-10 mm 等で評価する（相対誤差は 0/0 で不定）。

実行:
    uv run --extra dev python work/beam_element_validation/41_alpha1_axial_tension.py \\
        2>&1 | tee /tmp/alpha1_$(date +%s).log
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


def _alpha1_metrics(section: BeamSection, res: NRResult) -> list[MetricRow]:
    """α-1 純軸引張の 3 指標 + 診断列."""
    F_axial = 100.0  # 入力外力（main で同値を使用）

    # 解析解（線形）
    u_x_anal = F_axial * section.L / section.EA
    u_z_anal = 0.0
    se_anal = 0.5 * F_axial * u_x_anal
    L_arc_anal = section.L + u_x_anal

    # 数値解
    u = res.u
    u_x_num = float(u[6])
    u_z_num = float(u[8])
    L_arc_num = compute_chord_arc_length(section, u)
    se_num = numerical_strain_energy(u, res.f_int_final)

    rows = [
        # gate 対象 3 指標（kinematic 1 + geometric 1 + reaction 1）
        MetricRow(
            "u_x_tip [mm]",
            u_x_anal,
            u_x_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        # u_z = 0 — 絶対誤差を u_x_anal スケールで正規化
        MetricRow(
            "u_z_tip [mm] (=0)",
            u_z_anal,
            u_z_num,
            gate_threshold=0.10,
            in_gate=True,
            abs_zero_ref=u_x_anal,
        ),
        # 不伸長性: L_arc = L + u_x_tip — 軸引張で chord 長は伸びる
        MetricRow("L_arc [mm]", L_arc_anal, L_arc_num, gate_threshold=0.10, in_gate=True),
        # 診断
        MetricRow(
            "f_int axial [N] (診断)",
            F_axial,
            float(res.f_int_final[6]),
            in_gate=False,
        ),
        MetricRow("SE [N·mm] (診断)", se_anal, se_num, in_gate=False),
        MetricRow(
            "u_y_tip [mm] (=0, 診断)",
            0.0,
            float(u[7]),
            in_gate=False,
            abs_zero_ref=u_x_anal,
        ),
        MetricRow(
            "θ_y_tip [rad] (=0, 診断)",
            0.0,
            float(u[10]),
            in_gate=False,
            abs_zero_ref=1.0,
        ),
    ]
    return rows


def main() -> int:
    section = BeamSection(L=10.0, r=0.5, E=130.0e3, nu=0.3)

    # BC: 左端完全固定、右端 F_x = 100 N
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    F_ext = np.zeros(12)
    F_ext[6] = 100.0  # 右端軸方向力

    passed = run_case(
        case_name="α-1 純軸引張（implicit static, F_x=100 N）",
        section=section,
        fixed_dofs=fixed_dofs,
        F_ext=F_ext,
        analytical_metrics=_alpha1_metrics,
        n_load_steps=1,
        tol_rel=1.0e-9,
        max_iter=20,
        verbose=False,
    )

    print()
    print("=" * 80)
    if passed:
        print("[α-1 結果] **PASS** — CR Timoshenko 1 要素 implicit static は")
        print("  純軸引張で 3 指標一致、線形弾性 foundation 健全。")
    else:
        print("[α-1 結果] **FAIL** — 3 指標 AND gate 未達。")
        print("  status-389 §4 (z2) Cosserat 検討前に CR foundation を疑う必要あり。")
    print("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
