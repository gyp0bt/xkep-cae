"""work/beam_hysteresis/40_explicit_n_inc_sweep.py — `n_increments` 大化掃引（status-388 訂正版）.

[← README](README.md) | [← project README](../../README.md)

**status-388 訂正**: 当初 status-387 用に作成した本スクリプトは、解析解として
**90° 曲げ (u=73.30mm)** を使っていたが、実 BC は `bending_curvature=0.015` ×
`L=100mm` = 1.5 rad ≈ **86°** で、正しい解析解は |u|=70.44mm。さらに `max|u|`
単一指標一致は STA2 該当（偶然の交差を許容）のため、CLAUDE.md「妥当性テスト
の透明性ルール」に従い **独立 3 指標同時 10% 一致** に更新。

status-386 §5.4 副次「t_cycle 据え置き + n_increments 大」探索の再検証。
explicit + UL が precision gate を達成可能かを **kinematics + geometric** で
独立 3 指標として判定する。

**実 BC の解析解 3 指標** (κ=0.015 1/mm, L=100mm, θ=κ·L=1.5 rad ≈ 86°):

1. `|u_x_anal| = |R·sin(θ) − L| = 33.50 mm`（縮み方向、R = L/θ = 66.67 mm）
2. `|u_z_anal| = R·(1 − cos(θ)) = 61.96 mm`（曲げ方向）
3. `L_arc_anal = L = 100.00 mm`（不伸長性 — 変形後 chord 和、独立性最強）

`|u| = sqrt(u_x² + u_z²) = 70.44 mm` は (1)(2) から導出されるため独立指標
としてカウントしない。SE = `0.5 u^T f_int` は MPC 拘束 DOF 消去で信頼できず
（implicit でも 2.69 N·mm で解析解 71.79 N·mm と桁違い）、本スクリプトでは
診断列のみで gate に組み込まない。

**Gate（3 指標 AND）**: 全 (|u_x|, |u_z|, L_arc) で `err < 10%` のみ
「精度達成」。1 指標 PASS / 2 指標 FAIL は「達成と装わない」（数値の捏造禁止）。

**符号規約**: 実装座標系で u_x, u_z の符号は処方回転向きに依存（通常 implicit
で u_x≈+62, u_z≈-33 — analytical の (-33.50, +61.96) と x/z 役割が入れ替わる）。
本スクリプトは絶対値比較で吸収する。

実行:
    uv run --extra dev python work/beam_hysteresis/40_explicit_n_inc_sweep.py \\
        2>&1 | tee /tmp/n_inc_sweep_$(date +%s).log
"""

from __future__ import annotations

import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _make_single_beam_cfg() -> dict:
    """単梁 90° 曲げ baseline（接触なし、L=100 mm、E=130 GPa、`39` と同条件）."""
    return dict(
        n_strands=1,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=20000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=False,
    )


def _run_one(label: str, cfg_factory, **overrides) -> dict:
    base = cfg_factory()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] {label}")
    print("─" * 72)
    print(f"  solver_mode = {cfg.solver_mode}")
    print(f"  n_increments_per_cycle = {cfg.n_increments_per_cycle}")
    if cfg.solver_mode == "explicit":
        print(f"  explicit_mass_scaling_max_beta = {cfg.explicit_mass_scaling_max_beta:.2e}")
        print(f"  explicit_mass_scaling_selective = {cfg.explicit_mass_scaling_selective}")

    t0 = time.perf_counter()
    try:
        result = StrandBendingOscillationProcess().process(cfg)
        elapsed = time.perf_counter() - t0
        sr = result.solver_result
        frac = float(sr.load_history[-1]) if sr.load_history else 0.0
        u = sr.u
        n_total_nodes = u.shape[0] // 6
        u_trans = u.reshape(n_total_nodes, 6)[:, :3]
        u_norms = np.linalg.norm(u_trans, axis=1)
        # status-388: 3 指標同時一致テスト用に tip 節点の (u_x, u_z) を抽出
        # tip = max |u| ノード（曲げ先端、参照点ノード除外のため strand 節点のみ対象）
        n_strand = result.n_strand_nodes
        u_norms_strand = u_norms[:n_strand]
        tip_idx = int(np.argmax(u_norms_strand))
        tip_u_x = float(u_trans[tip_idx, 0])
        tip_u_z = float(u_trans[tip_idx, 2])
        max_u_trans = float(u_norms_strand[tip_idx])
        # status-388: 不伸長性チェック L_arc — strand 節点 chord 和を解析解 L=100 と比較
        node_coords_init = result.mesh.node_coords[:n_strand]  # (n_strand, 3)
        deformed = node_coords_init + u_trans[:n_strand]
        # 線形チェーン仮定（n_strands=1, 単梁の連続接続）。chord 和。
        diffs = np.diff(deformed, axis=0)
        L_arc = float(np.sum(np.linalg.norm(diffs, axis=1)))
        # 参考値: SE は MPC 拘束 DOF 消去で `0.5 u^T f_int` が解析解と桁違いになる
        # ため診断目的のみ（gate に組み込まない）
        se_final = float("nan")
        if sr.energy_history is not None:
            entries = getattr(sr.energy_history, "entries", None)
            if entries:
                se_final = float(entries[-1].strain_energy)
        diverged = False
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        frac = float("nan")
        max_u_trans = float("nan")
        tip_u_x = float("nan")
        tip_u_z = float("nan")
        L_arc = float("nan")
        se_final = float("nan")
        sr = None
        diverged = True
        print(f"  [DIVERGED] {type(exc).__name__}: {exc}")

    if sr is not None:
        print(
            f"  frac={frac:.4f}, conv={sr.converged}, "
            f"incr={sr.n_increments}, cb={sr.n_cutbacks}, t={elapsed:.2f}s, "
            f"|u|={max_u_trans:.3f}, u_x={tip_u_x:+.3f}, u_z={tip_u_z:+.3f}, "
            f"L_arc={L_arc:.3f}, SE(diag)={se_final:.3e}"
        )
    return dict(
        label=label,
        frac=frac,
        converged=bool(sr.converged) if sr is not None else False,
        n_increments=int(sr.n_increments) if sr is not None else 0,
        n_cutbacks=int(sr.n_cutbacks) if sr is not None else 0,
        elapsed=elapsed,
        max_u_trans=max_u_trans,
        tip_u_x=tip_u_x,
        tip_u_z=tip_u_z,
        L_arc=L_arc,
        se_final=se_final,
        diverged=diverged,
    )


def _analytical_circular_arc(L_strand: float, kappa: float, E: float, r_wire: float) -> dict:
    """status-388: 実 BC `bending_curvature=κ` での円弧解析解 3 指標.

    BC: 先端に θ = κ·L の処方回転を与え、梁が一様曲率の円弧を描く前提
    （quasi-static）. 解析解は kinematics + 歪エネルギーの 3 独立指標:

        u_x_anal = R · sin(θ) − L         （x 軸方向、負値：縮み）
        u_z_anal = R · (1 − cos(θ))       （z 軸方向、正値：曲げ方向）
        SE_anal  = (1/2) · EI · κ² · L    （uniform κ pure bending）

    `|u|` は u_x, u_z から導出されるため独立指標としてカウントしない。
    """
    theta = kappa * L_strand
    R = L_strand / theta if theta > 1e-12 else float("inf")
    u_x = R * np.sin(theta) - L_strand
    u_z = R * (1.0 - np.cos(theta))
    u_norm = float(np.sqrt(u_x**2 + u_z**2))
    # 円形断面 r=r_wire の二次モーメント
    I_y = np.pi * r_wire**4 / 4.0
    EI = E * I_y
    SE = 0.5 * EI * kappa**2 * L_strand
    return dict(
        theta=float(theta),
        R=float(R),
        u_x=float(u_x),
        u_z=float(u_z),
        u_norm=u_norm,
        SE=float(SE),
        EI=float(EI),
    )


def _summarize(runs: list[dict], anal: dict, L_strand: float, gate: float = 0.10) -> None:
    """status-388 透明性ルール: 3 指標 AND gate で判定.

    3 指標は **kinematic 2 + geometric 1**:

    1. tip |u_x|: x 成分絶対値（実装座標系で符号は処方回転向き依存）
    2. tip |u_z|: z 成分絶対値
    3. L_arc: 変形後 strand 節点 chord 和 ≈ 解析解 L_strand（不伸長性 gate）

    SE は `0.5 u^T f_int` が MPC 拘束 DOF 消去のため信頼できず（implicit でも
    解析解 71.79 と桁違い）、診断列のみで gate に組み込まない。

    `|u_x|` / `|u_z|` 比較は処方回転向きに対する符号依存を吸収するため絶対値で行う
    （analytical は ±33.50 / ±61.96 のいずれかで実装ごとに異なりうる）。
    """
    print()
    print("=" * 138)
    print(
        f"{'label':50s} | frac  | |u_x|/|u_z| 多重集合 [mm] (max err) | "
        f"L_arc [mm] (err) | SE(diag) | t [s] | gate"
    )
    # 解析解 multiset: {|u_x|, |u_z|} = {33.50, 61.96} を sort
    anal_set = sorted([abs(anal["u_x"]), abs(anal["u_z"])])
    anal_label = f"analytical (κ·L={anal['theta']:.4f} rad)"
    print(
        f"{anal_label:50s} | "
        f"  --  | {anal_set[0]:7.3f}, {anal_set[1]:7.3f}            | "
        f"{L_strand:7.3f}          | {anal['SE']:.3e} |  ---  |   --"
    )
    print("─" * 138)
    for r in runs:
        if r["diverged"] or not np.isfinite(r["max_u_trans"]):
            print(
                f"  {r['label']:48s} | {'-':>5s} | {'DIVERGED':>32s} | "
                f"{'':>16s} | {'':>8s} | {r['elapsed']:5.1f} | FAIL"
            )
            continue
        # 多重集合一致: {|u_x|, |u_z|} を sort して analytical sort と pair-wise 比較
        run_set = sorted([abs(r["tip_u_x"]), abs(r["tip_u_z"])])
        err_low = abs(run_set[0] - anal_set[0]) / max(anal_set[0], 1e-30)
        err_high = abs(run_set[1] - anal_set[1]) / max(anal_set[1], 1e-30)
        err_max_kin = max(err_low, err_high)
        err_L = abs(r["L_arc"] - L_strand) / L_strand
        gate_kin = err_max_kin < gate
        gate_L = err_L < gate
        gate_all = r["frac"] >= 0.999 and gate_kin and gate_L
        flag = "PASS" if gate_all else "FAIL"
        sk = "✓" if gate_kin else "✗"
        sL = "✓" if gate_L else "✗"
        print(
            f"  {r['label']:48s} | {r['frac']:5.3f} | "
            f"{run_set[0]:7.3f}, {run_set[1]:7.3f} ({err_max_kin * 100:5.1f}%{sk}) | "
            f"{r['L_arc']:7.3f} ({err_L * 100:5.1f}%{sL}) | "
            f"{r['se_final']:.2e} | {r['elapsed']:5.1f} | {flag}"
        )


def main() -> int:
    print("=" * 72)
    print("status-388 — 訂正版: 実 BC κ=0.015 の解析解 3 指標同時一致テスト")
    print("=" * 72)

    L_strand = 100.0
    kappa = 0.015
    E = 130.0e3
    r_wire = 0.5
    anal = _analytical_circular_arc(L_strand, kappa, E, r_wire)
    print(
        f"\n実 BC: κ={kappa} 1/mm, L={L_strand}mm → θ=κ·L={anal['theta']:.4f} rad "
        f"(≈ {np.degrees(anal['theta']):.2f}°)"
    )
    print("解析解 3 指標 (status-388 透明性ルール — 絶対値比較で符号規約吸収):")
    print(f"  |u_x|_anal = {abs(anal['u_x']):.4f} mm  (= |R·sin(θ) − L|、kinematic)")
    print(f"  |u_z|_anal = {abs(anal['u_z']):.4f} mm  (= R·(1 − cos(θ))、kinematic)")
    print(f"  L_arc_anal = {L_strand:.4f} mm    (= L、不伸長性 chord 和)")
    print(f"  SE_diag    = {anal['SE']:.4e} N·mm (= (1/2)·EI·κ²·L、EI={anal['EI']:.4e})")
    print("             — 診断列のみ、MPC 拘束消去で信頼できないため gate 外")
    print(f"  R_anal   = {anal['R']:.4f} mm   (|u| = {anal['u_norm']:.4f} 導出値)")
    print("\n（status-387 が誤って使った 90° 解析解: |u|=73.303mm — 実 BC は 86°）")
    print("\n3 指標 AND gate (10%): 全 (|u_x|, |u_z|, L_arc) PASS でのみ「精度達成」")

    print("\n" + "=" * 72)
    print("単梁 86° 曲げ — explicit uniform β² (selective=False) n_inc 掃引")
    print("=" * 72)

    runs: list[dict] = []

    # ── reference ──
    runs.append(
        _run_one(
            "implicit_baseline (reference, n_inc=20)",
            _make_single_beam_cfg,
            solver_mode="implicit",
        )
    )

    # status-386 #11 を再掲（n_inc=200）
    runs.append(
        _run_one(
            "exp_n_inc=200 (status-386 #11 再掲)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            n_increments_per_cycle=200,
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e4,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=False,
        )
    )

    # n_inc 段階的拡大
    for n_inc in (500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000):
        runs.append(
            _run_one(
                f"exp_n_inc={n_inc}",
                _make_single_beam_cfg,
                solver_mode="explicit",
                n_increments_per_cycle=n_inc,
                explicit_courant_safety=0.9,
                explicit_courant_check_interval=10,
                explicit_mass_scaling_auto=True,
                explicit_mass_scaling_max_beta=1.0e4,
                explicit_mass_scaling_max_growth_per_update=4.0,
                explicit_mass_scaling_selective=False,
            )
        )

    # ── damping + relax で overshoot 抑制を試す ──
    # n_inc=16000 で overshoot (max|u|=106mm, err=44.76%) が観測された場合、
    # 質量比例 damping + BC 完了後 relax phase で静的解析解 (73.30mm) に
    # 収束させられるか検証する。
    print("\n" + "=" * 72)
    print("damping + relax 併用 — sweet spot 周辺 + overshoot 抑制")
    print("=" * 72)
    for n_inc, alpha, relax in (
        (8000, 5.0, 200),
        (12000, 5.0, 200),
        (16000, 5.0, 500),
    ):
        runs.append(
            _run_one(
                f"exp_n_inc={n_inc}_damp{alpha}_relax{relax}",
                _make_single_beam_cfg,
                solver_mode="explicit",
                n_increments_per_cycle=n_inc,
                explicit_courant_safety=0.9,
                explicit_courant_check_interval=10,
                explicit_mass_scaling_auto=True,
                explicit_mass_scaling_max_beta=1.0e4,
                explicit_mass_scaling_max_growth_per_update=4.0,
                explicit_mass_scaling_selective=False,
                explicit_mass_proportional_damping_alpha=alpha,
                explicit_relax_steps=relax,
                explicit_relax_tol=1e-3,
            )
        )

    _summarize(runs, anal, L_strand)

    print()
    print("Gate (status-388 透明性ルール — 全 3 指標 AND):")
    print("  - (a) frac=1.0 完走")
    print(f"  - (b) ||u_x| − |u_x|_anal| / |u_x|_anal < 10%, |u_x|_anal={abs(anal['u_x']):.3f} mm")
    print(f"  - (c) ||u_z| − |u_z|_anal| / |u_z|_anal < 10%, |u_z|_anal={abs(anal['u_z']):.3f} mm")
    print(f"  - (d) |L_arc − L|/L < 10%, L={L_strand:.3f} mm（不伸長性）")
    print()
    print("評価ポイント:")
    print("  - max|u| が n_inc 増加で単調改善 → gate 達成可能性あり")
    print("  - max|u| がプラトー化 → UL 凍結 (status-382/383) の本質欠陥が支配")
    print("  - elapsed が n_inc にほぼ比例 → 1 substep/increment（β が target を吸収）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
