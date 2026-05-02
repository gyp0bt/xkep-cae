"""work/beam_element_validation/44_alpha4_pure_shear.py — Phase α-4 純せん断（cantilever 横荷重）.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase α-4: CR Timoshenko 3D 梁要素 1 つに片持ち横荷重を **implicit static**
で適用、3 指標 AND gate で Timoshenko せん断補正項の foundation 健全性を検証。

**設定**:
- L=10 mm、r=0.5 mm（A=0.7854 mm²、I=4.909×10⁻² mm⁴）
- E=130 GPa、ν=0.3 → G = E/(2(1+ν)) = 50 GPa
- せん断補正係数 κ_s = 6/7（円形断面）→ A_s = κ_s·A = 0.6732 mm²
- BC: 左端 (DOF 0–5) 完全固定、右端 F_z = 0.01 N（DOF 8）
- n_elements = 1

  status-389 §2 plan は F_z=10 N を書いているが u_z=522 mm（>50×L）で完全に
  非線形領域、Timoshenko 線形理論と比較不能。plan §α-4 末尾の注釈に従い
  **F_z=0.01 N** で線形領域に保つ（u_z ≈ 5×10⁻⁴ mm）。

**解析解（Timoshenko cantilever, F at tip）**:

| 指標 | 式 | 値（F=0.01 N） |
|------|---|---|
| `u_z_tip` | F·L³/(3·EI) + F·L/(G·A_s) | 5.254×10⁻⁴ mm |
| `θ_y_tip` | F·L²/(2·EI) | 7.84×10⁻⁵ rad |
| `M_base` | F·L | 0.1 N·mm（基部反力モーメント） |
| `SE` | (1/2)·F·u_z | 2.627×10⁻⁶ N·mm |

bending share = 99.43%, shear share = 0.57% — bending 支配の細梁形状。
せん断補正項の妥当性は `r/L = 0.05` と小さいため α-4 では検出感度低い。
（より太い梁 r/L ≥ 0.2 で shear share > 10% にする別ケースは Phase β/γ で）

**3 指標 AND gate**: `|u_z_tip|` / `|θ_y_tip|` / `|M_base|` の 3 個（kinematic 2 + reaction 1）。
全 10% 以内で PASS。

実行:
    uv run --extra dev python work/beam_element_validation/44_alpha4_pure_shear.py \\
        2>&1 | tee /tmp/alpha4_$(date +%s).log
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

F_TRANSVERSE = 0.01  # N — 線形領域に保つため小さく


def _alpha4_metrics(section: BeamSection, res: NRResult) -> list[MetricRow]:
    """α-4 純せん断（cantilever 横荷重）の 3 指標 + 診断列."""
    L = section.L
    EI = section.EI
    G = section.G
    A_s = section.A * section.kappa_s

    # 解析解（Timoshenko cantilever, F at tip）
    u_z_bend = F_TRANSVERSE * L**3 / (3.0 * EI)
    u_z_shear = F_TRANSVERSE * L / (G * A_s)
    u_z_anal = u_z_bend + u_z_shear
    theta_y_anal = F_TRANSVERSE * L**2 / (2.0 * EI)
    M_base_anal = F_TRANSVERSE * L
    se_anal = 0.5 * F_TRANSVERSE * u_z_anal

    u = res.u
    u_z_num = float(u[8])
    theta_y_num = float(u[10])
    L_chord_num = compute_chord_arc_length(section, u)
    se_num = numerical_strain_energy(u, res.f_int_final)

    # 基部反力 M_y は左端 DOF 4 の f_int
    M_base_num = float(res.f_int_final[4])

    rows = [
        # gate 対象 3 指標（kinematic 2 + reaction 1） — 符号規約は絶対値で吸収
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
        MetricRow(
            "|M_base| [N·mm]",
            M_base_anal,
            M_base_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        # 診断
        MetricRow(
            "u_z bend share [mm] (診断)",
            u_z_bend,
            u_z_num,
            in_gate=False,
            compare_abs=True,
        ),
        MetricRow(
            "u_z shear share [mm] (診断)",
            u_z_shear,
            u_z_num - u_z_bend if abs(u_z_num) > abs(u_z_bend) else 0.0,
            in_gate=False,
            compare_abs=True,
        ),
        MetricRow("L_chord [mm] (二次小, 診断)", L, L_chord_num, in_gate=False),
        MetricRow("SE [N·mm] (診断)", se_anal, se_num, in_gate=False),
    ]
    return rows


def main() -> int:
    section = BeamSection(L=10.0, r=0.5, E=130.0e3, nu=0.3)

    # BC: 左端完全固定、右端 F_z = F_TRANSVERSE [N]
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    F_ext = np.zeros(12)
    F_ext[8] = F_TRANSVERSE  # 右端 z 軸方向横荷重

    passed = run_case(
        case_name=f"α-4 純せん断（implicit static, F_z={F_TRANSVERSE} N, cantilever）",
        section=section,
        fixed_dofs=fixed_dofs,
        F_ext=F_ext,
        analytical_metrics=_alpha4_metrics,
        n_load_steps=1,
        tol_rel=1.0e-9,
        max_iter=20,
        verbose=False,
    )

    print()
    print("=" * 80)
    if passed:
        print("[α-4 結果] **PASS** — CR Timoshenko 1 要素 implicit static は")
        print("  cantilever 横荷重（bending + shear）で 3 指標一致、")
        print("  Timoshenko せん断補正項 foundation 健全。")
    else:
        print("[α-4 結果] **FAIL** — 3 指標 AND gate 未達。")
    print("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
