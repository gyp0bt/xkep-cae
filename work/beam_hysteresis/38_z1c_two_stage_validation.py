"""work/beam_hysteresis/38_z1c_two_stage_validation.py — z1c 効果検証.

[← README](README.md) | [← project README](../../README.md)

status-385 候補 (z1c) 「2 段階質量スケーリング」（β_stiff + β_outside）の効果を、
単梁 90° カンチレバー曲げ + 7 本撚線 90° 曲げで実測する。

status-384 では selective mass scaling (z1b) を実装し、stiff DOF（接触ペナルティ
等の高 K 局在）に β² 倍化を限定したが:

- **単梁**: K がほぼ一様で stiff DOF が検出されず、selective モードでは実質 β=1 で発散
- **7 本撚線**: 残り 84% の beam DOF が β=1 のまま dt 制限を支配し、stiff
  DOF 側の β を 1000 まで上げても Courant cap (8.8e6) に届かず未完走

(z1c) は **β_outside**（mask=False の梁 DOF 用 modest β）を導入し、
beam DOF を β_outside² 倍化して beam dt を拡大、stiff DOF はより aggressive に
β² 倍化する 2 段階アプローチ。

Gate (MCDD 凍結解除条件 (5)、status-381 §5):
- `|max|u_explicit| − max|u_analytical|| / max|u_analytical| < 0.10`
- 単梁 90° カンチレバー曲げ: 解析解 max|u| ≈ 73.30 mm

実行:
    uv run --extra dev python work/beam_hysteresis/38_z1c_two_stage_validation.py \\
        2>&1 | tee /tmp/z1c_$(date +%s).log
"""

from __future__ import annotations

import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _make_single_beam_cfg() -> dict:
    """単梁 90° 曲げ baseline（接触なし、L=100 mm、E=130 GPa）."""
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


def _make_7strand_cfg() -> dict:
    """7 本撚線 90° 曲げ baseline（接触あり）."""
    return dict(
        n_strands=7,
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
        max_increments=2000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=True,
    )


def _run_one(label: str, cfg_factory, **overrides) -> dict:
    base = cfg_factory()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] {label}")
    print("─" * 72)

    t0 = time.perf_counter()
    try:
        result = StrandBendingOscillationProcess().process(cfg)
        elapsed = time.perf_counter() - t0
        sr = result.solver_result
        frac = float(sr.load_history[-1]) if sr.load_history else 0.0
        u = sr.u
        n_total_nodes = u.shape[0] // 6
        u_trans = u.reshape(n_total_nodes, 6)[:, :3]
        max_u_trans = float(np.max(np.linalg.norm(u_trans, axis=1)))
        diverged = False
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        frac = float("nan")
        max_u_trans = float("nan")
        sr = None
        diverged = True
        print(f"  [DIVERGED] {type(exc).__name__}: {exc}")

    if sr is not None:
        print(
            f"  frac={frac:.4f}, conv={sr.converged}, "
            f"incr={sr.n_increments}, cb={sr.n_cutbacks}, t={elapsed:.2f}s, "
            f"max|u|={max_u_trans:.3e} mm"
        )
    return dict(
        label=label,
        frac=frac,
        converged=bool(sr.converged) if sr is not None else False,
        n_increments=int(sr.n_increments) if sr is not None else 0,
        n_cutbacks=int(sr.n_cutbacks) if sr is not None else 0,
        elapsed=elapsed,
        max_u_trans=max_u_trans,
        diverged=diverged,
    )


def _analytical_max_u(L_strand: float) -> float:
    """90° カンチレバー曲げの解析解 max|u_trans|（quarter circle radius from arc length）."""
    R = 2.0 * L_strand / np.pi
    return float(np.sqrt((L_strand - R) ** 2 + R**2))


def _summarize(runs: list[dict], u_analytical: float, L_strand: float) -> None:
    print()
    print("=" * 88)
    print(f"{'label':46s} | frac  |   max|u|     | err_anal | gate")
    print("─" * 88)
    for r in runs:
        if r["diverged"] or not np.isfinite(r["max_u_trans"]):
            print(
                f"  {r['label']:44s} | {'-':>5s} | {'DIVERGED':>11s} | "
                f"   ---  | FAIL"
            )
            continue
        err_anal = abs(r["max_u_trans"] - u_analytical) / u_analytical
        gate_disp = r["max_u_trans"] < L_strand * 10.0
        gate_acc = err_anal < 0.10
        gate = (
            "PASS" if (r["frac"] >= 0.999 and gate_disp and gate_acc) else "FAIL"
        )
        print(
            f"  {r['label']:44s} | {r['frac']:5.3f} | {r['max_u_trans']:.3e} | "
            f"{err_anal * 100:6.2f}% | {gate}"
        )


def main() -> int:
    print("=" * 72)
    print("status-385 候補 (z1c) 2 段階質量スケーリング検証")
    print("=" * 72)

    L_strand = 100.0
    u_analytical = _analytical_max_u(L_strand)
    print(f"\n解析解: 90° quarter circle, L={L_strand}mm → max|u| = {u_analytical:.3f} mm")

    # ============================================================================
    # ベンチマーク 1: 単梁 90° 曲げ
    # ============================================================================
    print("\n" + "=" * 72)
    print("Bench 1: 単梁 90° 曲げ (接触なし、K 一様 → stiff DOF 検出ゼロ)")
    print("=" * 72)
    print("単梁系では heterogeneous K がないため、selective モードでは全 DOF が")
    print("'mask=False' になる。よって β_outside² 倍化が全 DOF に作用し、selective")
    print("無効時の β² 倍化と数学的に等価（z1c の 2 段階効果は観察されない）。")
    print("ここでは β_outside による dt 拡大が単梁でも機能することを実証する。")

    runs1: list[dict] = []
    runs1.append(
        _run_one(
            "implicit_baseline",
            _make_single_beam_cfg,
            solver_mode="implicit",
        )
    )
    # selective + β_outside=1 (status-384 既存挙動、stiff 検出ゼロで実質 β=1 で発散)
    runs1.append(
        _run_one(
            "exp_z1b_selective_only",
            _make_single_beam_cfg,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e3,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=1.0,
        )
    )
    # z1c: selective + β_outside=10（mask=False 全 DOF を 100 倍化）
    runs1.append(
        _run_one(
            "exp_z1c_beta_outside_10",
            _make_single_beam_cfg,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e3,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=10.0,
        )
    )
    # z1c: selective + β_outside=100（aggressive、KE budget 違反の可能性）
    runs1.append(
        _run_one(
            "exp_z1c_beta_outside_100",
            _make_single_beam_cfg,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e3,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=100.0,
        )
    )

    _summarize(runs1, u_analytical, L_strand)

    # ============================================================================
    # ベンチマーク 2: 7 本撚線 90° 曲げ（接触あり、K heterogeneous）
    # ============================================================================
    print("\n" + "=" * 72)
    print("Bench 2: 7 本撚線 90° 曲げ (接触あり、stiff DOF 検出される)")
    print("=" * 72)
    print("status-384 の 7 本撚線で stiff DOF=15.7% 検出 → 残り 84% beam DOF が")
    print("β=1 のまま dt 制限を支配し、target β=8.8e6（cap 1000 を超過）で frac<<1.0。")
    print("(z1c) で β_outside を 10 に上げ beam DOF も modest にスケーリングする。")

    runs2: list[dict] = []
    runs2.append(
        _run_one(
            "implicit_baseline_7s",
            _make_7strand_cfg,
            solver_mode="implicit",
        )
    )
    # status-384 baseline (z1b 単独)
    runs2.append(
        _run_one(
            "exp_z1b_only_7s",
            _make_7strand_cfg,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e3,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=1.0,
        )
    )
    # z1c: β_outside=10
    runs2.append(
        _run_one(
            "exp_z1c_beta_out_10_7s",
            _make_7strand_cfg,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e3,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=10.0,
        )
    )
    # z1c: β_outside=100, β_stiff_max=1e4（より aggressive）
    runs2.append(
        _run_one(
            "exp_z1c_beta_out_100_betamax_1e4_7s",
            _make_7strand_cfg,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e4,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=100.0,
        )
    )
    # z1c: 大幅な aggressive スケーリング（β_outside=10, β_stiff_max=1e6）
    # 物理的に妥当な解が出るかの上限テスト（KE budget 違反警告は受容）
    runs2.append(
        _run_one(
            "exp_z1c_beta_out_10_betamax_1e6_7s",
            _make_7strand_cfg,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e6,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=10.0,
            explicit_mass_proportional_damping_alpha=10.0,  # 動的緩和
        )
    )

    _summarize(runs2, u_analytical, L_strand)

    print()
    print("Gate (MCDD 凍結解除条件 (5)):")
    print("  - frac=1.0 完走 + max|u| < L_strand × 10 + 精度 < 10%")
    print(f"  - 解析解 max|u| = {u_analytical:.3f} mm（quarter circle）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
