"""work/beam_hysteresis/39_z1d_t_cycle_validation.py — 候補 (z1d) 単梁中心検証.

[← README](README.md) | [← project README](../../README.md)

status-386 候補 (z1d) 「t_cycle 下限緩和」を **単梁** で検証する。

status-385 §6.1 が示した課題:

- implicit 7 本撚線は frac=1.0 を達成（70.45mm, 解析解 73.30mm の 96%, 誤差 3.9%）
- explicit 全ケースは **単梁ですら** DIVERGED または精度 gate (5) 未達
  - status-381: implicit 70mm（96%）vs explicit 40mm（48%）— **systematic 50% under**
  - status-385 Bench 1: 全 explicit ケース DIVERGED（initial target β=4.6e4 が
    β_stiff cap=1e3 を超過）

**status-385 で発見された真因**: target β = dt_sub/dt_c で、`t_cycle = max(10·T1,
1.0)` の **下限 1 秒** が dt_sub を 物理 T1 (~6.7ms) の ~150x に押し上げる。
loading rate が物理スケールから離れることで target β が β_stiff cap を超過する。

**(z1d) の解**: `t_cycle_min_seconds=0.0` を opt-in で指定し、`t_cycle = 10·T1` の
純粋物理スケールに切替える。dt_sub が 1/15 程度に縮小し、target β も 1/15 程度
に減少する。これで β_stiff cap=1e3 + β_outside=10 程度の組合せで
MCDD 凍結解除条件 (5) (`|u_explicit − u_analytical|/|u_analytical| < 0.10`)
達成可能と推定される（status-385 §6.1）。

**Gate** (MCDD 凍結解除条件):
1. frac = 1.0
2. max|u_trans| < L_strand × 10 = 1000 mm
3. **|max|u_explicit| − u_analytical| / u_analytical < 0.10**（単梁解析解 73.30mm）

実行:
    uv run --extra dev python work/beam_hysteresis/39_z1d_t_cycle_validation.py \\
        2>&1 | tee /tmp/z1d_$(date +%s).log
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
        bending_curvature=0.015,  # 0.015 × 100mm = 1.5 rad ≈ 86° → quarter circle 近似
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
    if "t_cycle_min_seconds" in overrides:
        print(f"  t_cycle_min_seconds = {overrides['t_cycle_min_seconds']:.4g} s")
    print(f"  solver_mode = {cfg.solver_mode}")

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
    """90° カンチレバー曲げの解析解 max|u_trans|（quarter circle radius from arc length）.

    曲率 κ = π/(2L) で梁が円弧を描くと R = 2L/π、先端変位は (L−R, 0, R) の point
    から (R, 0, 0) になるので |u| = sqrt((L−R)^2 + R^2)。
    """
    R = 2.0 * L_strand / np.pi
    return float(np.sqrt((L_strand - R) ** 2 + R**2))


def _summarize(runs: list[dict], u_analytical: float, L_strand: float) -> None:
    print()
    print("=" * 96)
    print(
        f"{'label':54s} | frac  |   max|u|     | err_anal | gate"
    )
    print("─" * 96)
    for r in runs:
        if r["diverged"] or not np.isfinite(r["max_u_trans"]):
            print(
                f"  {r['label']:52s} | {'-':>5s} | {'DIVERGED':>11s} | "
                f"   ---  | FAIL"
            )
            continue
        err_anal = abs(r["max_u_trans"] - u_analytical) / u_analytical
        gate_disp = r["max_u_trans"] < L_strand * 10.0
        gate_acc = err_anal < 0.10
        gate = "PASS" if (r["frac"] >= 0.999 and gate_disp and gate_acc) else "FAIL"
        print(
            f"  {r['label']:52s} | {r['frac']:5.3f} | {r['max_u_trans']:.3e} | "
            f"{err_anal * 100:6.2f}% | {gate}"
        )


def main() -> int:
    print("=" * 72)
    print("status-386 候補 (z1d) t_cycle 下限緩和 — 単梁中心検証")
    print("=" * 72)

    L_strand = 100.0
    u_analytical = _analytical_max_u(L_strand)
    print(f"\n単梁 90° 曲げ解析解: L={L_strand}mm → max|u| = {u_analytical:.3f} mm")
    print("(MCDD 条件 (5) gate: |max|u_explicit| − u_anal| / u_anal < 0.10)")

    # ─────────────────────────────────────────────────────────────────────────
    # Bench: 単梁 90° 曲げ — t_cycle_min_seconds 掃引
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("単梁 90° 曲げ — implicit baseline + explicit (z1c+z1d) sweep")
    print("=" * 72)

    runs: list[dict] = []

    # ── (1) implicit baseline（既存挙動 / status-385 と同条件） ──
    runs.append(
        _run_one(
            "implicit_baseline (default t_cycle_min=1.0)",
            _make_single_beam_cfg,
            solver_mode="implicit",
        )
    )

    # ── (2) implicit + t_cycle_min=0.0（z1d 単独効果 — implicit 側 regression） ──
    runs.append(
        _run_one(
            "implicit_z1d_only (t_cycle_min=0.0)",
            _make_single_beam_cfg,
            solver_mode="implicit",
            t_cycle_min_seconds=0.0,
        )
    )

    # ── (3) explicit 旧設定（status-385 Bench 1 再現、t_cycle_min=1.0 = 既存） ──
    runs.append(
        _run_one(
            "exp_z1c_only (t_cycle_min=1.0, status-385 baseline)",
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

    # ── (4) explicit + z1d 純粋物理 (t_cycle_min=0.0, β_outside=10, β_stiff_max=1e3) ──
    # status-385 §6.1 が予測した組合せ。
    runs.append(
        _run_one(
            "exp_z1c_z1d (t_cycle_min=0.0, β_out=10, β_stiff_max=1e3)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            t_cycle_min_seconds=0.0,
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

    # ── (5) explicit + z1d + smaller β_outside (t_cycle_min=0.0, β_outside=3) ──
    runs.append(
        _run_one(
            "exp_z1c_z1d_modest (t_cycle_min=0.0, β_out=3, β_stiff_max=1e3)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            t_cycle_min_seconds=0.0,
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e3,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=3.0,
        )
    )

    # ── (6) explicit + z1d 部分緩和 (t_cycle_min=0.1, β_out=10) ──
    runs.append(
        _run_one(
            "exp_z1c_z1d_partial (t_cycle_min=0.1, β_out=10)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            t_cycle_min_seconds=0.1,
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

    # ── (7) explicit + z1d + damping + relax phase ──
    # status-382 の (p1)+(p3) と組合せて精度 gate 達成を狙う
    runs.append(
        _run_one(
            "exp_z1c_z1d_damp_relax (t_cycle_min=0.0, α=10, relax=500)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            t_cycle_min_seconds=0.0,
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e3,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=10.0,
            explicit_mass_proportional_damping_alpha=10.0,
            explicit_relax_steps=500,
            explicit_relax_tol=1e-3,
        )
    )

    # ── (8) explicit non-selective + z1d (β² 一律スケール、単梁向け) ──
    # 単梁は K がほぼ一様、stiff DOF は誤検出ノイズ。selective=False で β² 全 DOF
    # 一律倍化すれば dt_c_beam も β でスケールし、target β を 1 段階で吸収可能。
    runs.append(
        _run_one(
            "exp_z1d_uniform_beta (t_cycle_min=0.0, selective=False, β_max=1e4)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            t_cycle_min_seconds=0.0,
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e4,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=False,
        )
    )

    # ── (9) explicit non-selective + z1d + damping + relax (gate 狙い) ──
    runs.append(
        _run_one(
            "exp_z1d_uniform_damp_relax (t_cycle_min=0.0, α=10, relax=500)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            t_cycle_min_seconds=0.0,
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e4,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=False,
            explicit_mass_proportional_damping_alpha=10.0,
            explicit_relax_steps=500,
            explicit_relax_tol=1e-3,
        )
    )

    # ── (10) explicit selective + z1d + 大 β_outside（梁 dt 緩和） ──
    # β_outside を 2000 程度まで上げれば dt_c_beam = 2000·1.64e-6 ≈ 3.3e-3 ≥ dt_sub
    runs.append(
        _run_one(
            "exp_z1c_z1d_large_outside (t_cycle_min=0.0, β_out=2000, β_stiff_max=1e4)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            t_cycle_min_seconds=0.0,
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e4,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=True,
            explicit_mass_scaling_stiff_threshold_ratio=10.0,
            explicit_mass_scaling_beta_outside=2000.0,
        )
    )

    # ── (11) z1d 反対方向: t_cycle 据え置き + n_increments 拡大 ──
    # 仮説: 解の精度は β_max （したがって 弾性波伝播時間 β·L/c）と t_cycle の比で決まる。
    # t_cycle=1.0s 据え置き + n_increments=200（dt_sub=5e-3、target β を 1/10 に）
    # で β を低く保ち、波伝播 ~26μs · β = 26ms ≪ t_cycle=1s で十分伝播。
    runs.append(
        _run_one(
            "exp_more_increments (t_cycle_min=1.0, n_inc=200, β_max=1e4)",
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

    _summarize(runs, u_analytical, L_strand)

    print()
    print("Gate (MCDD 凍結解除条件):")
    print("  - (1) frac=1.0 完走")
    print(f"  - (2) max|u| < L_strand × 10 = {L_strand * 10.0:.0f} mm")
    print(f"  - (5) 解析解誤差 < 10% （単梁 90° 解析解 max|u| = {u_analytical:.3f} mm）")
    print()
    print("評価ポイント:")
    print("  - (1) implicit_baseline と (2) implicit_z1d_only が同等収束 → z1d は")
    print("    implicit 側 regression を発生させない（default 変更安全性確認）")
    print("  - (3) exp_z1c_only DIVERGED の再現 → status-385 baseline 一致")
    print("  - (4)〜(7) のいずれかで PASS → MCDD 凍結解除条件 (5) 達成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
