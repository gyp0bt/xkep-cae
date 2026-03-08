#!/usr/bin/env python3
"""NCP接触ソルバーのGeneralized-α動的解析統合の検証.

動的解析（質量行列+Rayleigh減衰）を有効にしたPhase2揺動の収束確認。
準静的解析との比較で、慣性効果による正則化を検証する。

使い方:
    python scripts/verify_dynamics_ncp.py 2>&1 | tee /tmp/log-dynamics-$(date +%s).log
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation


def run_comparison(n_strands: int = 7) -> None:
    """準静的 vs 動的解析の比較."""
    common_params = dict(
        n_strands=n_strands,
        pitch=40.0,
        n_elems_per_pitch=16,
        bend_angle_deg=45.0,
        oscillation_amplitude_mm=2.0,
        n_cycles=1,
        n_steps_per_quarter=3,
        max_iter=30,
        tol_force=1e-6,
        use_ncp=True,
        use_mortar=True,
        use_updated_lagrangian=True,
        show_progress=True,
        adaptive_timestepping=True,
    )

    # --- 1. 準静的解析（従来方式） ---
    print("=" * 70)
    print(f"  {n_strands}本撚線: 準静的解析（dynamics=False）")
    print("=" * 70)
    t0 = time.perf_counter()
    result_static = run_bending_oscillation(
        **common_params,
        dynamics=False,
    )
    t_static = time.perf_counter() - t0
    print(f"\n準静的解析完了: {t_static:.2f}s")
    print(f"  Phase1 収束: {result_static.phase1_converged}")
    print(f"  Phase2 収束: {result_static.phase2_converged}")
    print(f"  総NR反復: {sum(r.total_newton_iterations for r in result_static.phase2_results)}")

    # --- 2. 動的解析（Generalized-α + Rayleigh減衰） ---
    # 鋼線の物理的減衰: ξ ≈ 0.002（0.2%）
    # Rayleigh: α_R=0, β_R=2ξ/ω_1
    # ω_1 ≈ 2π × oscillation_frequency → β_R ≈ 2*0.002/(2π*1) ≈ 6.4e-4
    xi = 0.002  # 鋼の減衰比
    f_osc = 1.0  # Hz
    omega_1 = 2.0 * np.pi * f_osc
    beta_R = 2.0 * xi / omega_1  # 剛性比例Rayleigh減衰

    print("\n" + "=" * 70)
    print(f"  {n_strands}本撚線: 動的解析（dynamics=True, f={f_osc}Hz, ξ={xi}）")
    print("=" * 70)
    t0 = time.perf_counter()
    result_dynamic = run_bending_oscillation(
        **common_params,
        dynamics=True,
        oscillation_frequency_hz=f_osc,
        rho=7.85e-9,  # ton/mm³（鋼）
        rayleigh_alpha=0.0,
        rayleigh_beta=beta_R,
        rho_inf=1.0,  # エネルギー保存（数値減衰なし）
    )
    t_dynamic = time.perf_counter() - t0
    print(f"\n動的解析完了: {t_dynamic:.2f}s")
    print(f"  Phase1 収束: {result_dynamic.phase1_converged}")
    print(f"  Phase2 収束: {result_dynamic.phase2_converged}")
    print(f"  総NR反復: {sum(r.total_newton_iterations for r in result_dynamic.phase2_results)}")

    # --- 結果比較 ---
    print("\n" + "=" * 70)
    print("  比較結果")
    print("=" * 70)
    print(f"{'':20s} {'準静的':>12s} {'動的':>12s}")
    print("-" * 50)
    print(f"{'Phase2収束':20s} {str(result_static.phase2_converged):>12s} {str(result_dynamic.phase2_converged):>12s}")
    p2_nr_static = sum(r.total_newton_iterations for r in result_static.phase2_results)
    p2_nr_dynamic = sum(r.total_newton_iterations for r in result_dynamic.phase2_results)
    print(f"{'Phase2 NR反復':20s} {p2_nr_static:>12d} {p2_nr_dynamic:>12d}")
    print(f"{'計算時間 [s]':20s} {t_static:>12.2f} {t_dynamic:>12.2f}")


if __name__ == "__main__":
    run_comparison(n_strands=7)
