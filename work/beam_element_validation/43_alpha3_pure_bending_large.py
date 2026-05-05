"""work/beam_element_validation/43_alpha3_pure_bending_large.py — Phase α-3 純粋曲げ large κ.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase α-3: CR Timoshenko 3D 梁要素 1 つに **prescribed θ_y=0.15 rad**
の large rotation BC を **implicit static** で適用、3 指標 AND gate で large
rotation 領域の foundation（CR 定式化 + Battini-Pacoste 接線剛性）健全性を検証。

**設定**:
- L=10 mm、r=0.5 mm（A=0.7854 mm²、I=4.909×10⁻² mm⁴、EI=6.381×10³ N·mm²）
- E=130 GPa、ν=0.3
- BC: 左端 (DOF 0–5) 完全固定、右端 θ_y = 0.15 rad 処方（DOF 10）
- n_elements = 1

**解析解 — 1 要素 CR Hermite 解 vs circular arc**:

1 要素では `chord 長 = L` 保存（CR の linear local frame + chord-aligned 性質より）。
左端 fix + 右端 θ_R 処方の 1 要素 CR 解は対称性により **chord が θ_R/2 だけ回転**:

    α = θ_R / 2 = 0.0750 rad (chord rotation)
    u_x_tip = L·(cos α − 1) = -0.0281 mm  (chord-preserving)
    u_z_tip = L·sin α       = +0.7493 mm
    L_chord = L = 10.000 mm (保存)

一方 circular arc 解（uniform κ、curve length L 保存）は:

    R = L/θ = 66.667 mm
    u_x_tip = R·sin(θ) − L  = -0.0374 mm  (curve-length-preserving)
    u_z_tip = R·(1 − cos θ) = +0.7486 mm
    L_chord = 2R·sin(θ/2)   = 9.9813 mm

**両者の差は 1 要素の Hermite 補間が curve length を保存できないことに由来する
本質的な離散化誤差**。Phase γ で n_elements を増やすと circular arc に収束する。

→ 本 α-3 では **1 要素 CR Hermite 解** を gate の解析解に採用する。
   foundation = 「CR 局所剛性 + Battini-Pacoste 接線が 1 要素実装として正しく動作」
   の確認が目的。circular arc 解は「Phase γ multi-element 収束目標」として診断列に記載。

**3 指標 AND gate**: `|u_x_tip|` / `|u_z_tip|` / `L_chord` の 3 個。全 10% 以内で PASS。
SE = (1/2)·EI·θ²/L は Hermite 解でも circular arc 解でも同値（curvature integral
は不変）のため診断指標として併記。

実行:
    uv run --extra dev python work/beam_element_validation/43_alpha3_pure_bending_large.py \\
        2>&1 | tee /tmp/alpha3_$(date +%s).log
"""

from __future__ import annotations

import math
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

THETA_PRESCRIBED = 0.15  # rad


def _alpha3_metrics(section: BeamSection, res: NRResult) -> list[MetricRow]:
    """α-3 純粋曲げ large κ の 3 指標 + 診断列.

    解析解は **1 要素 CR Hermite 解** (chord 保存型 α=θ_R/2 chord rotation).
    circular arc 解（curve length 保存）は診断列のみで併記。
    """
    L = section.L
    EI = section.EI

    theta = THETA_PRESCRIBED
    # 1 要素 CR Hermite 解（chord 保存）
    alpha = theta / 2.0
    u_x_hermite = L * (math.cos(alpha) - 1.0)
    u_z_hermite = L * math.sin(alpha)
    L_chord_hermite = L  # 保存

    # circular arc 解（curve length 保存、Phase γ multi-element 収束目標）
    R = L / theta
    u_x_arc = R * math.sin(theta) - L
    u_z_arc = R * (1.0 - math.cos(theta))
    L_chord_arc = 2.0 * R * math.sin(theta / 2.0)

    se_anal = 0.5 * EI * theta**2 / L  # 両解で同値（curvature integral 不変）

    u = res.u
    u_x_num = float(u[6])
    u_z_num = float(u[8])
    theta_y_num = float(u[10])
    L_chord_num = compute_chord_arc_length(section, u)
    se_num = numerical_strain_energy(u, res.f_int_final)

    rows = [
        # gate 対象 3 指標（kinematic 2 + geometric 1） — 1 要素 CR Hermite 解との一致
        MetricRow(
            "|u_x_tip| [mm] (Hermite)",
            u_x_hermite,
            u_x_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        MetricRow(
            "|u_z_tip| [mm] (Hermite)",
            u_z_hermite,
            u_z_num,
            gate_threshold=0.10,
            in_gate=True,
            compare_abs=True,
        ),
        # L_chord = L (1 要素 CR は chord 保存)
        MetricRow(
            "L_chord [mm] (Hermite)",
            L_chord_hermite,
            L_chord_num,
            gate_threshold=0.10,
            in_gate=True,
        ),
        # 診断 — circular arc 解は Phase γ multi-element 収束目標として併記
        MetricRow(
            "|u_x_tip| [mm] (arc 解、診断)",
            u_x_arc,
            u_x_num,
            in_gate=False,
            compare_abs=True,
        ),
        MetricRow(
            "|u_z_tip| [mm] (arc 解、診断)",
            u_z_arc,
            u_z_num,
            in_gate=False,
            compare_abs=True,
        ),
        MetricRow(
            "L_chord [mm] (arc 解, 診断)",
            L_chord_arc,
            L_chord_num,
            in_gate=False,
        ),
        MetricRow(
            "θ_y_tip [rad] (=処方値, 診断)",
            theta,
            theta_y_num,
            in_gate=False,
        ),
        MetricRow(
            "SE [N·mm] (両解共通, 診断)",
            se_anal,
            se_num,
            in_gate=False,
        ),
    ]
    return rows


def main() -> int:
    section = BeamSection(L=10.0, r=0.5, E=130.0e3, nu=0.3)

    # BC: 左端完全固定、右端 θ_y = 0.15 rad 処方
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    F_ext = np.zeros(12)  # 外力なし（処方変位のみ）
    prescribed_disp = {10: THETA_PRESCRIBED}

    # large rotation のため load stepping (10 段階) を用いて Newton 収束を確実化
    passed = run_case(
        case_name=f"α-3 純粋曲げ large κ（implicit static, θ_y={THETA_PRESCRIBED} rad ≈ {math.degrees(THETA_PRESCRIBED):.1f}°）",
        section=section,
        fixed_dofs=fixed_dofs,
        F_ext=F_ext,
        analytical_metrics=_alpha3_metrics,
        prescribed_disp=prescribed_disp,
        n_load_steps=10,
        tol_rel=1.0e-9,
        max_iter=30,
        verbose=False,
    )

    print()
    print("=" * 80)
    if passed:
        print("[α-3 結果] **PASS** — CR Timoshenko 1 要素 implicit static は")
        print("  Hermite chord 保存解と 3 指標一致、CR 定式化健全（large rotation OK）.")
        print()
        print("  注: 1 要素は curve length 保存できないため circular arc 解とは")
        print("      u_x で ~25% 差。これは Phase γ で n_elements 増加と共に解消する")
        print("      離散化誤差であり、CR の局所剛性 + Battini-Pacoste 接線そのものは")
        print("      想定通りに動作している（foundation 健全）。")
        print("  → status-389 §4 のシナリオ「foundation 健全 → (z2) Cosserat は")
        print("     explicit + 大回転 robust 化に主目的を絞れる」を支持。")
    else:
        print("[α-3 結果] **FAIL** — 3 指標 AND gate 未達。")
        print("  CR 定式化 1 要素で Hermite 解にも一致しない場合は実装バグ可能性。")
    print("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
