"""work/beam_element_validation/49_beta2_with_assembler_ul.py — assembler / UL 1 要素再現実験.

[← README](README.md) | [← project README](../../README.md)

status-393 §6.1 「次セッション最優先候補」: assembler / UL update_reference の 1 要素規模
再現実験。Phase β-2 が **直接駆動（``timo_beam3d_cr_*`` 直接呼出）** で機械精度
0.000% の Hermite 解一致を出した（status-391）一方で、status-381〜387 の精度問題は
**assembler 経由 + UL ``update_reference`` 有効化**でのみ顕在化したと推定される
（status-382 §3 解析）。本スクリプトは α-3 / β-2 と同 BC（θ_y=0.15 rad 処方）を
4 通りの実装パスで実行し、**改修対象を 1 要素規模で局在化**する。

**判定設計**:

| モード | 実装パス | UL update_reference | 期待結果 | 仮説 |
|---|---|---|---|---|
| A | implicit + assembler | なし（TL モード） | PASS（直接駆動と機械精度一致） | assembler 自体は健全 |
| B | implicit + assembler | あり（増分ごと） | PASS or FAIL | UL が implicit でも問題か検証 |
| C | explicit  + assembler | なし（TL モード） | PASS（β-2 直接駆動と機械精度一致） | assembler explicit パスは健全 |
| D | explicit  + assembler | あり（増分ごと） | FAIL（status-381〜387 再現） | UL update_reference + explicit が真の根本 |

**判定マトリクス**:

- A=PASS, B=PASS, C=PASS, D=FAIL → **UL update_reference + explicit のみが問題**.
  改修対象は assembler の UL 機構そのものではなく、explicit との結合方式. (z2)
  Cosserat ではなく、explicit + UL 更新タイミングの再設計に絞れる.
- A=PASS, B=FAIL → UL update_reference が implicit でも壊している（rotation
  composition の bug 等）. assembler の UL 機構自体に bug.
- A=FAIL → assembler が静的でも壊している（CR 形式と差異）. これは β-2 直接
  駆動 PASS と矛盾し、想定外の状況.
- C=FAIL → explicit + assembler 自体に問題（mass matrix 等）.

**3 指標 AND gate**（α-3 と同じ Hermite 解、status-388 透明性ルール）:

1. ``|u_x_tip|`` ≈ Hermite 解 ``L·(cos(θ/2) − 1)`` (gate 10%)
2. ``|u_z_tip|`` ≈ Hermite 解 ``L·sin(θ/2)`` (gate 10%)
3. ``L_chord`` ≈ ``L`` (chord 保存 1 要素 CR、geometric, gate 10%)

実行::

    uv run --extra dev python work/beam_element_validation/49_beta2_with_assembler_ul.py \\
        2>&1 | tee /tmp/beta2_assembler_ul_$(date +%s).log
"""

from __future__ import annotations

import math
import sys

import numpy as np
import scipy.sparse as sp_mod

from work.beam_element_validation._alpha_common import (
    BeamSection,
    MetricRow,
    evaluate_three_metric_gate,
)
from work.beam_element_validation._beta_common import (
    compute_natural_frequencies_fe,
)
from xkep_cae.elements._beam_assembler import ULCRBeamAssembler

THETA_PRESCRIBED = 0.15  # rad (α-3 / β-2 と同じ)
PRESCRIBED_DOF = 10  # node 2 θ_y
FIXED_DOFS = [0, 1, 2, 3, 4, 5]  # node 1 完全固定


def _make_assembler(section: BeamSection) -> ULCRBeamAssembler:
    """1 要素 12 DOF assembler を生成."""
    coords = section.coords_2node()
    conn = np.array([[0, 1]], dtype=np.int64)
    bk = section.beam_kwargs()
    return ULCRBeamAssembler(
        node_coords=coords,
        connectivity=conn,
        E=bk["E"],
        G=bk["G"],
        A=bk["A"],
        Iy=bk["Iy"],
        Iz=bk["Iz"],
        J=bk["J"],
        kappa_y=bk["kappa_y"],
        kappa_z=bk["kappa_z"],
    )


def _solve_dense(K_sparse: object, r: np.ndarray) -> np.ndarray:
    """sparse K_T を dense 化して NR 線形ソルブ（1 要素 12 DOF 想定で OK）."""
    K_dense = K_sparse.toarray() if sp_mod.issparse(K_sparse) else np.asarray(K_sparse)
    return np.linalg.solve(K_dense, r)


def _compute_tip_position(
    section: BeamSection,
    asm: ULCRBeamAssembler,
    u_incr: np.ndarray,
) -> tuple[float, float, float]:
    """tip 絶対位置 (x, y, z) [mm].

    UL モードでは coords_ref が更新されているので、現時点の絶対位置は
    ``asm.coords_ref[1] + u_incr[6:9]`` で求まる。
    TL モードでは coords_ref は initial のまま、``u_incr`` は累積総変位.
    """
    tip = asm.coords_ref[1] + u_incr[6:9]
    return float(tip[0]), float(tip[1]), float(tip[2])


def _compute_chord_length_assembler(
    asm: ULCRBeamAssembler,
    u_incr: np.ndarray,
) -> float:
    """変形後 chord 長 = ``|coords_ref[1] + u_incr[6:9] − coords_ref[0] − u_incr[0:3]|``."""
    p1 = asm.coords_ref[0] + u_incr[0:3]
    p2 = asm.coords_ref[1] + u_incr[6:9]
    return float(np.linalg.norm(p2 - p1))


# =============================================================================
# Mode A & B: implicit static
# =============================================================================


def run_implicit(
    section: BeamSection,
    *,
    use_ul: bool,
    n_load_steps: int = 10,
    tol_rel: float = 1.0e-9,
    max_iter: int = 50,
    verbose: bool = False,
) -> tuple[float, float, float, bool, int]:
    """implicit static + assembler 駆動.

    Args:
        use_ul: True なら各 load step 収束後に ``asm.update_reference(u_incr)`` を
            呼び出し、u_incr を 0 にリセット（UL モード）.
            False なら update_reference を呼ばず u_incr を累積（TL モード）.

    Returns:
        (u_x_tip, u_z_tip, L_chord, converged, n_iters_total)
    """
    asm = _make_assembler(section)
    fixed_set = set(FIXED_DOFS)
    presc_set = {PRESCRIBED_DOF}
    a_idx = np.array(sorted(set(range(12)) - fixed_set - presc_set), dtype=int)

    u_incr = np.zeros(12)  # UL では「現在の reference からの増分」、TL では「累積」
    n_iters_total = 0
    converged_all = True

    for k in range(1, n_load_steps + 1):
        # 処方値: UL では「この step の Δθ_y」、TL では「累積 θ_y」
        if use_ul:
            target_step_value = THETA_PRESCRIBED / n_load_steps
        else:
            target_step_value = THETA_PRESCRIBED * k / n_load_steps
        u_incr[PRESCRIBED_DOF] = target_step_value

        # NR 反復
        last_resid = 0.0
        step_converged = False
        for _it in range(max_iter):
            f_int = asm.assemble_internal_force(u_incr)
            r_a = -f_int[a_idx]
            f_ref = max(np.linalg.norm(f_int), 1.0e-30)
            resid_norm = float(np.linalg.norm(r_a))
            last_resid = resid_norm
            if resid_norm / f_ref < tol_rel or resid_norm < 1.0e-14:
                step_converged = True
                break
            K_T = asm.assemble_tangent(u_incr)
            K_T_dense = K_T.toarray() if sp_mod.issparse(K_T) else np.asarray(K_T)
            K_aa = K_T_dense[np.ix_(a_idx, a_idx)]
            du_a = np.linalg.solve(K_aa, r_a)
            u_incr[a_idx] += du_a
            n_iters_total += 1

        if verbose:
            mode = "UL" if use_ul else "TL"
            print(
                f"  [{mode}] step {k}/{n_load_steps}: "
                f"target_θ_y={u_incr[PRESCRIBED_DOF]:.6e} rad, "
                f"converged={step_converged}, ||r||={last_resid:.3e}"
            )

        if not step_converged:
            converged_all = False
            break

        # UL: 収束後に reference 更新 + u_incr リセット
        if use_ul:
            asm.update_reference(u_incr)
            u_incr = np.zeros(12)

    # tip 絶対位置（TL/UL 両方で coords_ref + u_incr の組合せで一意）
    tip = asm.coords_ref[1] + u_incr[6:9]
    p1 = asm.coords_ref[0] + u_incr[0:3]
    L_chord = float(np.linalg.norm(tip - p1))

    # tip displacement = tip − initial_tip
    init_tip = section.coords_2node()[1]
    u_x_tip = float(tip[0] - init_tip[0])
    u_z_tip = float(tip[2] - init_tip[2])

    return u_x_tip, u_z_tip, L_chord, converged_all, n_iters_total


# =============================================================================
# Mode C & D: explicit central difference
# =============================================================================


def run_explicit(
    section: BeamSection,
    rho: float,
    *,
    use_ul: bool,
    update_per_step: bool = True,
    verbose: bool = False,
) -> tuple[float, float, float, dict]:
    """explicit central diff (leap-frog Verlet) + assembler 駆動.

    β-2 と同じ slow ramp + hold + 質量比例減衰.

    Args:
        use_ul: True なら ``update_reference`` を呼び出す（UL モード）.
            False なら呼ばない（TL モード）.
        update_per_step: use_ul=True のとき毎 step update_reference を呼ぶ
            （= ``explicit_ul_update_interval=1``、status-383 既定挙動）.
            False なら ramp 終了後に 1 度だけ呼ぶ（参考）.

    Returns:
        (u_x_tip, u_z_tip, L_chord, diag_dict)
    """
    asm = _make_assembler(section)
    fixed_set = set(FIXED_DOFS)
    presc_set = {PRESCRIBED_DOF}
    a_idx = np.array(sorted(set(range(12)) - fixed_set - presc_set), dtype=int)

    # FE 1 周期から ramp / hold / damping を決める（β-2 と同じ）
    omegas, _ = compute_natural_frequencies_fe(section, rho, fixed_dofs=FIXED_DOFS, use_lumped=True)
    omega_1 = float(omegas[0])
    omega_max = float(omegas[-1])
    T_1 = 2.0 * math.pi / omega_1
    n_ramp_periods = 5.0
    n_hold_periods = 5.0
    t_ramp = n_ramp_periods * T_1
    t_hold = n_hold_periods * T_1
    t_total = t_ramp + t_hold
    zeta = 2.0
    damping_alpha = 2.0 * zeta * omega_1
    zeta_high = damping_alpha / (2.0 * omega_max)
    dt_critical_damped = 2.0 / omega_max * (math.sqrt(1.0 + zeta_high**2) - zeta_high)
    dt_target = 0.5 * dt_critical_damped
    n_steps = int(math.ceil(t_total / dt_target))
    dt = t_total / n_steps

    # 集中質量（assembler 経由）
    M_csr = asm.assemble_mass(rho, lumped=True)
    M_full_diag = np.asarray(M_csr.diagonal()).copy()

    # ramp 関数
    def theta_total_at(t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t <= t_ramp:
            return THETA_PRESCRIBED * (t / t_ramp)
        return THETA_PRESCRIBED

    # 初期化
    u_incr = np.zeros(12)
    v = np.zeros(12)
    a = np.zeros(12)

    f_int = asm.assemble_internal_force(u_incr)
    a[a_idx] = (-f_int[a_idx] - damping_alpha * M_full_diag[a_idx] * v[a_idx]) / M_full_diag[a_idx]
    v_half = v.copy()
    v_half[a_idx] = v[a_idx] + 0.5 * dt * a[a_idx]

    # 直前 step の処方角度（UL 増分計算用）
    theta_prev = 0.0

    if verbose:
        mode = ("UL/per-step" if update_per_step else "UL/post-ramp") if use_ul else "TL"
        print(
            f"  [{mode}] dt={dt:.4e} s, n_steps={n_steps}, T_1={T_1:.4e} s, α={damping_alpha:.4e}"
        )

    for k in range(n_steps):
        t_k1 = (k + 1) * dt
        theta_total_target = theta_total_at(t_k1)

        # 処方変位（DOF 10）の現 step 値
        if use_ul:
            # UL: u_incr[10] = θ_total_target − accumulated_θ
            # accumulated_θ は assembler の R_ref から取り出すべきだが、
            # 1 要素 1 軸回転なので theta_prev で十分
            u_incr[PRESCRIBED_DOF] = theta_total_target - theta_prev
        else:
            u_incr[PRESCRIBED_DOF] = theta_total_target

        # u 更新
        u_incr[a_idx] = u_incr[a_idx] + dt * v_half[a_idx]

        # 内力評価
        f_int = asm.assemble_internal_force(u_incr)

        # 加速度
        a[a_idx] = (-f_int[a_idx] / M_full_diag[a_idx]) - damping_alpha * v_half[a_idx]

        # 半 step 速度更新
        v_half[a_idx] = v_half[a_idx] + dt * a[a_idx]
        v[a_idx] = v_half[a_idx] - 0.5 * dt * a[a_idx]

        # UL モード: per-step なら毎 step update_reference
        if use_ul and update_per_step:
            asm.update_reference(u_incr)
            theta_prev = theta_total_target  # 累積 θ_y を bookkeep
            u_incr = np.zeros(12)

    # use_ul + post-ramp の場合は最後に 1 度だけ update_reference
    if use_ul and not update_per_step:
        asm.update_reference(u_incr)
        u_incr_final = np.zeros(12)
    else:
        u_incr_final = u_incr

    tip = asm.coords_ref[1] + u_incr_final[6:9]
    p1 = asm.coords_ref[0] + u_incr_final[0:3]
    L_chord = float(np.linalg.norm(tip - p1))

    init_tip = section.coords_2node()[1]
    u_x_tip = float(tip[0] - init_tip[0])
    u_z_tip = float(tip[2] - init_tip[2])

    diag = {
        "n_steps": n_steps,
        "dt": dt,
        "T_1": T_1,
        "damping_alpha": damping_alpha,
    }
    return u_x_tip, u_z_tip, L_chord, diag


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    section = BeamSection(L=10.0, r=0.5, E=130.0e3, nu=0.3)
    rho = 8.96e-9  # ton/mm³

    theta = THETA_PRESCRIBED
    alpha_chord = theta / 2.0
    u_x_hermite = section.L * (math.cos(alpha_chord) - 1.0)
    u_z_hermite = section.L * math.sin(alpha_chord)

    print()
    print("=" * 80)
    print("  assembler / UL update_reference 1 要素再現実験 (status-393 §6.1)")
    print("=" * 80)
    print(f"  L={section.L} mm, r={section.r} mm, E={section.E} MPa")
    print(f"  EI={section.EI:.4e} N·mm², EA={section.EA:.4e} N")
    print(f"  θ_y_prescribed={theta} rad ({math.degrees(theta):.2f}°)")
    print(f"  Hermite 解析解: u_x={u_x_hermite:.6e} mm, u_z={u_z_hermite:.6e} mm")
    print()

    # ----- Mode A: implicit + assembler + no UL (TL) -----
    print("\n[Mode A] implicit + assembler + no_UL (TL モード)")
    u_x_A, u_z_A, L_A, conv_A, it_A = run_implicit(
        section, use_ul=False, n_load_steps=10, verbose=True
    )
    rows_A = [
        MetricRow("|u_x_tip| [mm]", u_x_hermite, u_x_A, in_gate=True, compare_abs=True),
        MetricRow("|u_z_tip| [mm]", u_z_hermite, u_z_A, in_gate=True, compare_abs=True),
        MetricRow("L_chord [mm]", section.L, L_A, in_gate=True),
    ]
    pass_A = evaluate_three_metric_gate(
        "Mode A (implicit + assembler + TL)",
        rows_A,
        converged=conv_A,
        n_iters=it_A,
        n_steps=10,
    )

    # ----- Mode B: implicit + assembler + UL (each increment) -----
    print("\n[Mode B] implicit + assembler + UL (各 increment 毎に update_reference)")
    u_x_B, u_z_B, L_B, conv_B, it_B = run_implicit(
        section, use_ul=True, n_load_steps=10, verbose=True
    )
    rows_B = [
        MetricRow("|u_x_tip| [mm]", u_x_hermite, u_x_B, in_gate=True, compare_abs=True),
        MetricRow("|u_z_tip| [mm]", u_z_hermite, u_z_B, in_gate=True, compare_abs=True),
        MetricRow("L_chord [mm]", section.L, L_B, in_gate=True),
    ]
    pass_B = evaluate_three_metric_gate(
        "Mode B (implicit + assembler + UL)",
        rows_B,
        converged=conv_B,
        n_iters=it_B,
        n_steps=10,
    )

    # ----- Mode C: explicit + assembler + no UL (TL) -----
    print("\n[Mode C] explicit + assembler + no_UL (TL モード)")
    u_x_C, u_z_C, L_C, diag_C = run_explicit(section, rho, use_ul=False, verbose=True)
    rows_C = [
        MetricRow("|u_x_tip| [mm]", u_x_hermite, u_x_C, in_gate=True, compare_abs=True),
        MetricRow("|u_z_tip| [mm]", u_z_hermite, u_z_C, in_gate=True, compare_abs=True),
        MetricRow("L_chord [mm]", section.L, L_C, in_gate=True),
    ]
    pass_C = evaluate_three_metric_gate(
        "Mode C (explicit + assembler + TL)",
        rows_C,
        converged=True,
        n_iters=diag_C["n_steps"],
        n_steps=diag_C["n_steps"],
    )

    # ----- Mode D: explicit + assembler + UL (per step) -----
    print("\n[Mode D] explicit + assembler + UL (毎 step update_reference, status-381〜387 再現)")
    u_x_D, u_z_D, L_D, diag_D = run_explicit(
        section, rho, use_ul=True, update_per_step=True, verbose=True
    )
    rows_D = [
        MetricRow("|u_x_tip| [mm]", u_x_hermite, u_x_D, in_gate=True, compare_abs=True),
        MetricRow("|u_z_tip| [mm]", u_z_hermite, u_z_D, in_gate=True, compare_abs=True),
        MetricRow("L_chord [mm]", section.L, L_D, in_gate=True),
    ]
    pass_D = evaluate_three_metric_gate(
        "Mode D (explicit + assembler + UL per step)",
        rows_D,
        converged=True,
        n_iters=diag_D["n_steps"],
        n_steps=diag_D["n_steps"],
    )

    # ----- 総合判定 -----
    print()
    print("=" * 80)
    print("  4 モード総合判定")
    print("=" * 80)
    summary = [
        ("A: implicit + assembler + TL", pass_A, u_x_A, u_z_A, L_A),
        ("B: implicit + assembler + UL", pass_B, u_x_B, u_z_B, L_B),
        ("C: explicit  + assembler + TL", pass_C, u_x_C, u_z_C, L_C),
        ("D: explicit  + assembler + UL", pass_D, u_x_D, u_z_D, L_D),
    ]
    print(f"  {'モード':32s} {'gate':>6s} {'u_x [mm]':>13s} {'u_z [mm]':>13s} {'L_chord':>10s}")
    print("  " + "-" * 78)
    for name, p, ux, uz, lc in summary:
        flag = "PASS" if p else "FAIL"
        print(f"  {name:32s} {flag:>6s} {ux:>13.6e} {uz:>13.6e} {lc:>10.4f}")
    print(
        f"  {'(Hermite 解析解)':32s} {'-':>6s} {u_x_hermite:>13.6e} {u_z_hermite:>13.6e} "
        f"{section.L:>10.4f}"
    )
    print()

    # ----- 改修対象局在化の所見 -----
    print("=" * 80)
    print("  改修対象局在化の所見")
    print("=" * 80)
    if pass_A and pass_B and pass_C and pass_D:
        print("  ALL PASS — 1 要素規模では assembler / UL / explicit いずれも問題なく動作.")
        print("  status-381〜387 の精度問題は撚線規模・接触統合・mass scaling 戦略・")
        print("  multi-element UL 蓄積などの **より上位の組合せ** に局在.")
    elif pass_A and pass_B and pass_C and not pass_D:
        print("  → **UL update_reference + explicit のみ FAIL**.")
        print("    status-381〜387 の精度問題は 1 要素規模で再現可能.")
        print("    改修対象は assembler の UL 機構そのものではなく explicit との")
        print("    結合方式（update タイミング、運動学的整合性）.")
        print("    (z2) Cosserat 不要、explicit + UL update 設計の見直しに絞れる.")
    elif pass_A and not pass_B:
        print("  → **UL update_reference が implicit でも FAIL**.")
        print("    assembler の UL 機構そのものに bug. rotation composition 等の見直し必要.")
    elif not pass_A:
        print("  → **assembler が implicit static でも FAIL** — β-2 直接駆動 PASS と矛盾.")
        print("    想定外の状況、調査必須.")
    else:
        print("  → 複合的失敗. 各モードの数値を個別に分析する必要あり.")
    print("=" * 80)

    return 0 if (pass_A and pass_B and pass_C and pass_D) else 1


if __name__ == "__main__":
    sys.exit(main())
