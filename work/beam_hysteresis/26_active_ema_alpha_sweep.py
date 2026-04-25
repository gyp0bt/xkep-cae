"""work/beam_hysteresis/26_active_ema_alpha_sweep.py — 候補 (g1) active 履歴 EMA 平滑化 α 掃引.

[← README](README.md) | [← project README](../../README.md)

status-370 で結果 B 確定（K_c は active 境界でも FD 機械精度、19 本 Type D
stall は NR alg 側動力学）を受け、status-371 で候補 (g1) **active 履歴
平滑化** を実装。NR 反復間で

    p_n_eff = α·p_n_new + (1-α)·p_n_prev

を適用し、active 集合振動を直接低域通過化する escape hatch。

本スクリプトは α ∈ {0.1, 0.3, 0.5} を **7 本 / 19 本撚線 90° 曲げ** で
掃引し、frac / cutback / elapsed の変化を計測する。gate 基準:

- 7 本: frac=1.0 維持（status-336 baseline）
- 19 本: frac ≥ 0.6（status-357 baseline 0.3739 の +60%、status-339
  baseline 0.4839 比で 24% 改善目標）

7 本 baseline（status-336）: frac=1.0000, incr=607, cb=389, ~250s
19 本 baseline（status-339）: frac=0.4839, incr=271, cb=39, 534.68s

実行:

    uv run python work/beam_hysteresis/26_active_ema_alpha_sweep.py 2>&1 \\
        | tee /tmp/active_ema_alpha_sweep_$(date +%s).log

引数:
    --n-strands {7,19}     掃引対象の撚線本数（既定: 7）
    --alphas A1,A2,...     α 値の CSV（既定: 0.1,0.3,0.5）
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _run_one(n_strands: int, alpha: float) -> dict:
    """1 ケース実行し主要指標を返す."""
    cfg = StrandBendingOscillationConfig(
        n_strands=n_strands,
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
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        penalty_exponent=1.5,
        # ★ 候補 (g1): active 履歴平滑化 α
        active_ema_alpha=alpha,
    )
    print()
    print("-" * 70)
    print(
        f"ケース n_strands={n_strands}, α={alpha:.2f} "
        f"({'baseline' if alpha == 0.0 else 'EMA 有効'})"
    )
    print("-" * 70)

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")
    print(f"  max |u|:        {float(np.max(np.abs(sr.u))):.6e}")

    return {
        "alpha": alpha,
        "frac": frac,
        "converged": sr.converged,
        "n_inc": sr.n_increments,
        "n_cb": sr.n_cutbacks,
        "elapsed": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-strands", type=int, default=7, choices=[7, 19])
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.3,0.5")
    args = parser.parse_args(argv)

    alphas = [float(a) for a in args.alphas.split(",")]
    n_strands = args.n_strands

    print("=" * 70)
    print(
        f"候補 (g1) active 履歴 EMA 平滑化 α 掃引 — n_strands={n_strands}, "
        f"αs={alphas}"
    )
    print("=" * 70)
    if n_strands == 7:
        print("baseline (status-336): frac=1.0000, incr=607, cb=389, ~250s")
    elif n_strands == 19:
        print("baseline (status-339): frac=0.4839, incr=271, cb=39, 534.68s")

    rows = [_run_one(n_strands, a) for a in alphas]

    print()
    print("=" * 70)
    print(f"集約（n_strands={n_strands}）")
    print("=" * 70)
    print(
        f"  {'α':>6s} {'frac':>8s} {'conv':>6s} {'n_inc':>7s} "
        f"{'n_cb':>6s} {'elapsed[s]':>11s}"
    )
    for row in rows:
        print(
            f"  {row['alpha']:>6.2f} {row['frac']:>8.4f} "
            f"{'Y' if row['converged'] else 'N':>6s} {row['n_inc']:>7d} "
            f"{row['n_cb']:>6d} {row['elapsed']:>11.2f}"
        )

    # gate チェック
    base_frac = {7: 1.0, 19: 0.6}[n_strands]
    gate_label = {7: "frac=1.0 維持", 19: "frac ≥ 0.6"}[n_strands]
    print()
    print(f"gate: {gate_label}")
    any_pass = any(r["frac"] >= base_frac - 1e-6 for r in rows if r["alpha"] > 0)
    if any_pass:
        print("  → 1 ケース以上で gate 通過（採択方向）")
    else:
        print("  → 全ケース gate 未達（候補 (g1) 却下方向、(g3) pair-wise relaxation 検討）")

    print()
    print("=" * 70)
    print("完了（候補 (g1) α 掃引）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
