"""s_unclamped 修正の90度曲げ検証スクリプト.

status-291 TODO: s_unclamped修正がNR収束率に与える影響を検証。
Hertz型ペナルティ(α=1.5) + 接触あり90度曲げで:
  - frac 到達率
  - NR反復回数
  - カットバック回数
を計測し、status-285のベースラインと比較。

Usage:
    python contracts/verify_s_unclamped_90deg.py 2>&1 | tee /tmp/log-$(date +%s).log

[← README](../README.md)
"""

import math
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def run_90deg_bending():
    """Hertz型ペナルティ + 接触あり90度曲げ."""
    kappa_90 = math.pi / 2.0 / 100.0  # κ = π/(2*L), θ = κ*L = π/2
    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=kappa_90,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,  # status-285と同一（MPC不使用、直接処方変位）
        contact_enabled=True,
        penalty_exponent=1.5,  # Hertz型
    )

    print("=" * 70)
    print("90度曲げ検証: s_unclamped修正 + Hertz型ペナルティ(α=1.5)")
    print("=" * 70)
    print(f"  κ = {kappa_90:.6f} [1/mm]")
    print(f"  θ_target = {math.degrees(kappa_90 * 100):.1f}°")
    print(f"  n_strands = {cfg.n_strands}")
    print(f"  n_elements_per_pitch = {cfg.n_elements_per_pitch}")
    print(f"  penalty_exponent = {cfg.penalty_exponent}")
    print(f"  contact_enabled = {cfg.contact_enabled}")
    print(f"  max_nr_attempts = {cfg.max_nr_attempts}")
    print(f"  max_increments = {cfg.max_increments}")
    print()

    t0 = time.time()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.time() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    print()
    print("=" * 70)
    print("結果サマリ")
    print("=" * 70)
    print(f"  frac          = {frac:.4f}")
    print(f"  n_increments  = {sr.n_increments}")
    print(f"  n_cutbacks    = {sr.n_cutbacks}")
    print(f"  elapsed       = {elapsed:.1f} sec")
    print()
    print("--- status-285 ベースライン比較 ---")
    print("  ベースライン (status-285, α=1.5, s_unclamped修正前):")
    print("    frac=0.9981, incr=551, cutback=60")
    print()
    print(f"  今回 (s_unclamped修正後):")
    print(f"    frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}")
    print()

    # 改善判定
    if frac >= 0.9981:
        if sr.n_cutbacks < 60:
            print(f"  → カットバック改善: 60 → {sr.n_cutbacks} ({(60 - sr.n_cutbacks)/60*100:.0f}%削減)")
        if sr.n_increments < 551:
            print(f"  → インクリメント改善: 551 → {sr.n_increments} ({(551 - sr.n_increments)/551*100:.0f}%削減)")
        if frac > 0.9981:
            print(f"  → frac改善: 0.9981 → {frac:.4f}")
    else:
        print(f"  → frac低下: 0.9981 → {frac:.4f} (回帰の可能性)")

    print()
    print("検証完了")
    return frac, sr.n_increments, sr.n_cutbacks


if __name__ == "__main__":
    run_90deg_bending()
